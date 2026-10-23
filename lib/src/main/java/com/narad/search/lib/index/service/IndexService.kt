package com.narad.search.lib.index.service

import android.app.Service
import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.IBinder
import android.provider.MediaStore
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.core.app.ServiceCompat
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.narad.search.lib.index.provision.EmbeddingGemma2Manifest
import com.narad.search.lib.index.provision.EmbeddingGemma2VisionManifest
import com.narad.search.lib.index.provision.ModelProvisioner
import com.narad.search.lib.index.worker.IndexWorker
import androidx.work.WorkInfo
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

/**
 * Foreground [Service] that:
 *  1. Posts a persistent progress notification (channel created via [IndexNotifications]).
 *  2. Registers a [MediaStoreObserver] on Downloads / Files URIs.
 *  3. Delegates the long initial-build pass to [IndexWorker] via WorkManager
 *     (6-hour-cap resilience via checkpointing, KTD-8).
 *
 * User-initiated only (started by an explicit "Start Indexing" Intent from the
 * Settings screen, NOT auto-started on boot; the periodic delta job handles
 * that via [BootReceiver]).
 *
 * Foreground service type: `dataSync` (declared in lib's AndroidManifest).
 *
 * API-34 compliance: uses [ServiceCompat.startForeground] with explicit
 * [ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC] to avoid the 2-arg
 * [startForeground] crash on API 34+ (fix item 2).
 */
class IndexService : Service() {

    companion object {
        private const val TAG = "IndexService"

        const val ACTION_START_INDEXING = "com.narad.search.ACTION_START_INDEXING"
        const val ACTION_STOP_INDEXING = "com.narad.search.ACTION_STOP_INDEXING"

        /**
         * User-initiated reindex action.
         *
         * When received: calls [IndexRepository.clearAll] to wipe the index (on
         * [Dispatchers.IO] (not the main thread), then calls
         * [enqueueUserInitiatedIndex] which enqueues a fresh [IndexWorker] job with
         * [ExistingWorkPolicy.REPLACE], cancelling any running/pending job first.
         *
         * Both [ACTION_START_INDEXING] and [ACTION_REINDEX] ultimately delegate their
         * WorkManager enqueue to [enqueueUserInitiatedIndex] (REPLACE policy).
         * Automated producers ([BootReceiver], [MediaStoreObserver]) call
         * [WorkManager.enqueueUniqueWork] directly with [ExistingWorkPolicy.KEEP].
         */
        const val ACTION_REINDEX = "com.narad.search.ACTION_REINDEX"

        /**
         * The unique work name used for the initial index build job.
         *
         * Exposed so other callers (e.g. the Settings tab) can reference
         * it consistently when calling [WorkManager.cancelUniqueWork].
         */
        const val INITIAL_INDEX_WORK_NAME = "narad_initial_index"

        private const val LIB_PREFS = "narad_lib_prefs"
        private const val KEY_FIRST_INDEX_DONE = "first_index_done"

        /**
         * True once the first full index run has returned success (even with 0 files), set by
         * [IndexWorker] itself so it does not depend on this service observing the run. Until then
         * every delta enqueue is blocked ([BootReceiver.enqueueIfModelReady]) so the first full
         * read runs only as the foreground run. Reindex all does not reset it.
         */
        fun isFirstIndexDone(context: Context): Boolean =
            context.getSharedPreferences(LIB_PREFS, Context.MODE_PRIVATE).getBoolean(KEY_FIRST_INDEX_DONE, false)

        fun setFirstIndexDone(context: Context, done: Boolean) {
            context.getSharedPreferences(LIB_PREFS, Context.MODE_PRIVATE).edit()
                .putBoolean(KEY_FIRST_INDEX_DONE, done).commit()
        }

        /**
         * Returns the expected path to the EmbeddingGemma 2 text graph, read from the manifest.
         *
         * The model is downloaded to internal storage by [ModelProvisioner] on
         * first run (NaradApplication/the Settings tab trigger the download).
         * This method returns the target path that provision() writes to.
         */
        fun getDefaultModelPath(context: Context): String {
            val graph = EmbeddingGemma2Manifest.manifest.files.first { it.relativeDest.endsWith(".onnx") }
            return File(context.filesDir, graph.relativeDest).absolutePath
        }

        /**
         * Returns the expected path to the EmbeddingGemma 2 vision encoder ONNX.
         *
         * Mirrors [getDefaultModelPath]: the model is placed in internal storage
         * by [ModelProvisioner] (or manually pushed for testing).
         * [IndexWorker] uses this path to guard the image-indexing branch:
         * if the file is absent, image indexing is skipped gracefully.
         *
         * Expected layout:
         * ```
         * filesDir/models/embeddinggemma-2/vision_encoder_quantized.onnx
         * ```
         */
        fun getVisionEncoderPath(context: Context): String {
            val graph = EmbeddingGemma2VisionManifest.manifest.files.first { it.relativeDest.endsWith(".onnx") }
            return File(context.filesDir, graph.relativeDest).absolutePath
        }

        /**
         * Enqueue a user-initiated fresh full index build with [ExistingWorkPolicy.REPLACE].
         *
         * Use this from the Settings tab "Start Indexing" / [ACTION_REINDEX] so that
         * a prior empty-success job (model was absent) is replaced and re-runs.
         *
         * Automated producers ([BootReceiver], [MediaStoreObserver]) MUST use KEEP
         * so they do not accidentally interrupt a running index build.
         */
        /**
         * Pure function: returns true if [states] contains any ENQUEUED or RUNNING entry.
         *
         * Used by [startIndexing] to decide whether to re-enqueue or observe existing work.
         * Extracted as a pure function so JVM tests can verify the idempotency guard
         * without a WorkManager context (review-024).
         */
        fun isAlreadyActive(states: List<WorkInfo.State>): Boolean =
            states.any { it == WorkInfo.State.ENQUEUED || it == WorkInfo.State.RUNNING }

        fun canReindex(modelReady: Boolean, documentAccess: Boolean, photoAccess: Boolean): Boolean =
            modelReady && (documentAccess || photoAccess)

        fun isReindexReady(context: Context): Boolean {
            val documentAccess = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                Environment.isExternalStorageManager()
            } else {
                ContextCompat.checkSelfPermission(context, Manifest.permission.READ_EXTERNAL_STORAGE) ==
                    PackageManager.PERMISSION_GRANTED
            }
            val photoAccess = when {
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE ->
                    ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_IMAGES) ==
                        PackageManager.PERMISSION_GRANTED ||
                        ContextCompat.checkSelfPermission(
                            context,
                            Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED
                        ) == PackageManager.PERMISSION_GRANTED
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU ->
                    ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_IMAGES) ==
                        PackageManager.PERMISSION_GRANTED
                else ->
                    ContextCompat.checkSelfPermission(context, Manifest.permission.READ_EXTERNAL_STORAGE) ==
                        PackageManager.PERMISSION_GRANTED
            }
            return canReindex(
                modelReady = ModelProvisioner(EmbeddingGemma2Manifest.manifest)
                    .isProvisioned(context),
                documentAccess = documentAccess,
                photoAccess = photoAccess,
            )
        }

        /**
         * Returns the [UUID] of the enqueued request so the caller can observe it
         * by ID (not by unique-work name, which would include stale CANCELLED siblings
         * from prior REPLACE calls).
         */
        fun enqueueUserInitiatedIndex(context: Context): UUID {
            val request = OneTimeWorkRequestBuilder<IndexWorker>()
                .setInputData(workDataOf(IndexWorker.KEY_MODEL_PATH to getDefaultModelPath(context)))
                .build()

            WorkManager.getInstance(context).enqueueUniqueWork(
                INITIAL_INDEX_WORK_NAME,
                ExistingWorkPolicy.REPLACE,
                request
            )
            // Schedule overnight catch-up now, not only after the next reboot.
            BootReceiver.enqueuePeriodicDeltaWork(context)
            return request.id
        }
    }

    private var mediaStoreObserver: MediaStoreObserver? = null

    // ponytail: SupervisorJob so an observer failure doesn't cancel the whole scope
    private val serviceScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var observeJob: Job? = null

    // Serialises the review-024 query-then-enqueue in startIndexing().
    private val startMutex = Mutex()

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        IndexNotifications.createChannel(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_INDEXING -> startIndexing()
            ACTION_STOP_INDEXING  -> stopIndexing()
            ACTION_REINDEX        -> reindex()
            else                  -> startIndexing()
        }
        return START_STICKY
    }

    /**
     * Start the foreground service, register the MediaStore observer, and enqueue the
     * initial-build WorkManager job via [enqueueUserInitiatedIndex] (REPLACE policy).
     *
     * Fix 11 / Fix 13: the dead `useReplace=false/KEEP` branch has been removed.
     * All user-initiated starts from the Settings tab always use REPLACE so that a
     * prior empty-success job (model was absent) is replaced and re-runs.
     * Automated producers ([BootReceiver], [MediaStoreObserver]) call
     * [WorkManager.enqueueUniqueWork] directly with [ExistingWorkPolicy.KEEP].
     *
     * review-024: idempotent. If [INITIAL_INDEX_WORK_NAME] work is already ENQUEUED
     * or RUNNING (e.g. a second ACTION_START_INDEXING arrives before the first job
     * starts), observe the existing job without re-enqueueing. [isAlreadyActive] is
     * the pure testable gate; REPLACE is still used when the prior job is terminal.
     */
    private fun startIndexing() {
        // Post foreground notification (API 34 compliance via ServiceCompat 4-arg form).
        ServiceCompat.startForeground(
            this,
            IndexNotifications.NOTIFICATION_ID,
            IndexNotifications.buildProgressNotification(this),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        )

        // Register MediaStore observer for Downloads and Files
        val modelPath = getDefaultModelPath(this)
        unregisterMediaStoreObserver()
        mediaStoreObserver = MediaStoreObserver(this, modelPath).also { observer ->
            val cr = contentResolver
            cr.registerContentObserver(
                MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                true,
                observer
            )
            cr.registerContentObserver(
                MediaStore.Files.getContentUri("external"),
                true,
                observer
            )
        }

        observeJob?.cancel()
        observeJob = null

        // review-024: query plus enqueue run under startMutex. Without it, two close
        // ACTION_START_INDEXING calls both suspend at withContext(IO), both query before
        // either enqueues, both see no active work, and the second REPLACE restarts the
        // first job. With the lock, the second call queries after the first enqueued and
        // sees ENQUEUED.
        serviceScope.launch {
            startMutex.withLock {
                val wm = WorkManager.getInstance(this@IndexService)
                val infos = withContext(Dispatchers.IO) {
                    wm.getWorkInfosForUniqueWork(INITIAL_INDEX_WORK_NAME).get()
                }
                if (isAlreadyActive(infos.map { it.state })) {
                    // Already ENQUEUED or RUNNING: observe the existing job, skip re-enqueue.
                    val activeId = infos.first {
                        it.state == WorkInfo.State.ENQUEUED || it.state == WorkInfo.State.RUNNING
                    }.id
                    observeWork(activeId)
                } else {
                    // Not active (absent, SUCCEEDED, FAILED, CANCELLED): enqueue with REPLACE
                    // so a prior empty-success job (model was absent) is replaced and re-runs.
                    observeWork(enqueueUserInitiatedIndex(this@IndexService))
                }
            }
        }
    }

    /**
     * User-initiated "Reindex all": wipe the index and start a fresh full build.
     *
     * Fix 6: [IndexRepository.clearAll] runs on [Dispatchers.IO] (not the main thread)
     * so the brief SQL DELETE does not block the UI or the 5-second FGS window.
     * [ServiceCompat.startForeground] is called FIRST (on the main thread, within the
     * 5-second window) before the background work is dispatched.
     */
    private fun reindex() {
        if (!isReindexReady(this)) {
            stopSelf()
            return
        }

        // Must call startForeground on the main thread first (within 5-second FGS window).
        ServiceCompat.startForeground(
            this,
            IndexNotifications.NOTIFICATION_ID,
            IndexNotifications.buildProgressNotification(this),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        )

        observeJob?.cancel()
        observeJob = null

        // Fix 6: clearAll + re-enqueue on IO thread so the main thread is not blocked.
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val db = com.narad.search.lib.index.store.IndexDatabase(this@IndexService)
                com.narad.search.lib.index.store.IndexRepository(db).clearAll()
                db.close()
            } catch (e: Exception) {
                android.util.Log.w("IndexService", "clearAll() in reindex() threw", e)
            }
            // Fix 13: delegate enqueue through the canonical helper (REPLACE policy).
            val requestId = enqueueUserInitiatedIndex(this@IndexService)
            // Post observeWork to the main thread: it modifies observeJob (a Main-thread field)
            // and must not race with a replaced run's CANCELLED notification on IO.
            serviceScope.launch { observeWork(requestId) }
        }
    }

    private fun stopIndexing() {
        observeJob?.cancel()
        observeJob = null
        unregisterMediaStoreObserver()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    /**
     * Observe the enqueued work by its unique [UUID].
     *
     * Observing by UUID (not the unique-work name) avoids a false-positive teardown:
     * [enqueueUserInitiatedIndex] uses REPLACE, which leaves the replaced run as a
     * CANCELLED sibling in the unique-work list. Observing by name would see that
     * CANCELLED entry on the next REPLACE call and tear down the service immediately.
     *
     * Called on the main thread; [serviceScope] uses [Dispatchers.Main].
     */
    private fun observeWork(id: UUID) {
        observeJob?.cancel()
        observeJob = serviceScope.launch {
            WorkManager.getInstance(this@IndexService)
                .getWorkInfoByIdFlow(id)
                .collect { info ->
                    if (info != null && info.state.isFinished) {
                        onWorkFinished()
                    }
                }
        }
    }

    /**
     * Called when the observed [IndexWorker] run reaches a terminal state.
     *
     * Sequence (U25 guard for deferred U5):
     *  1. Enqueue the overnight periodic delta (KEEP - idempotent if already scheduled). The
     *     worker has already set first_index_done on success, so the guard lets it through.
     *  2. Unregister the MediaStore observer and cancel its debouncer.
     *  3. Remove the foreground notification.
     *  4. Stop the service.
     */
    private fun onWorkFinished() {
        Log.d(TAG, "IndexWorker finished - stopping foreground service")
        // 1. Schedule overnight catch-up (KEEP is a no-op if already enqueued).
        BootReceiver.enqueuePeriodicDeltaWork(this)
        // 2. Unregister MediaStore observer.
        unregisterMediaStoreObserver()
        // 3. Remove the foreground notification.
        stopForeground(STOP_FOREGROUND_REMOVE)
        // 4. Stop the service (triggers onDestroy which cancels serviceScope).
        stopSelf()
    }

    override fun onDestroy() {
        super.onDestroy()
        unregisterMediaStoreObserver()
        // Cancel the work observer coroutine and any other child jobs.
        serviceScope.cancel()
    }

    private fun unregisterMediaStoreObserver() {
        mediaStoreObserver?.let { observer ->
            contentResolver.unregisterContentObserver(observer)
            observer.cancel()
        }
        mediaStoreObserver = null
    }

}
