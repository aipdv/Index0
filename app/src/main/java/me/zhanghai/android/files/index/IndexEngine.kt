package me.zhanghai.android.files.index

import android.app.Activity
import android.app.Application
import android.content.Intent
import android.os.Bundle
import android.os.Environment
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.work.ExistingWorkPolicy
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.narad.search.lib.device.DeviceTierDetector
import com.narad.search.lib.index.extraction.DocumentExtractor
import com.narad.search.lib.index.provision.EmbeddingGemma2Manifest
import com.narad.search.lib.index.provision.EmbeddingGemma2VisionManifest
import com.narad.search.lib.index.provision.ModelProvisioner
import com.narad.search.lib.index.provision.ProvisionState
import com.narad.search.lib.index.service.BootReceiver
import com.narad.search.lib.index.service.IndexService
import com.narad.search.lib.index.service.MediaStoreObserver
import com.narad.search.lib.index.worker.IndexWorker
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import me.zhanghai.android.files.BuildConfig
import me.zhanghai.android.files.app.Analytics
import me.zhanghai.android.files.app.application
import java.util.concurrent.atomic.AtomicBoolean

internal const val FOREGROUND_DELTA_MIN_GAP_MS = 2 * 60 * 1000L

/** Pure: run the app-open delta at most once per [FOREGROUND_DELTA_MIN_GAP_MS] (from Narad). */
internal fun shouldRunForegroundDelta(lastFinishedAt: Long, now: Long): Boolean =
    lastFinishedAt <= 0L || lastFinishedAt > now || now - lastFinishedAt >= FOREGROUND_DELTA_MIN_GAP_MS

/**
 * Pure: fetch a missing photo model when the user opens the app. The terms-time download can fail
 * or the phone ran as Tier C then, and nothing else fetches it. Not while a download runs, and at
 * most once per process. [imagesOn] (tier check) runs last, only when everything else passes.
 */
internal fun shouldFetchPhotoModel(
    termsAccepted: Boolean,
    textReady: Boolean,
    visionState: ProvisionState,
    triedThisProcess: Boolean,
    imagesOn: () -> Boolean,
): Boolean =
    termsAccepted && textReady && !triedThisProcess &&
        (visionState is ProvisionState.NotStarted || visionState is ProvisionState.Failed) &&
        imagesOn()

/** App-wide wiring of the Narad engine (port of NaradApplication). */
object IndexEngine {
    private const val TAG = "IndexEngine"
    private const val PREFS = "index0_prefs"
    private const val KEY_TERMS_ACCEPTED = "gemma_terms_accepted"

    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    val provisioner: ModelProvisioner by lazy {
        ModelProvisioner(manifest = EmbeddingGemma2Manifest.manifest)
    }

    val visionProvisioner: ModelProvisioner by lazy {
        ModelProvisioner(manifest = EmbeddingGemma2VisionManifest.manifest)
    }

    val searchController: SearchController by lazy { RealSearchController(application) }

    val isTermsAccepted: Boolean
        get() = application.getSharedPreferences(PREFS, Application.MODE_PRIVATE)
            .getBoolean(KEY_TERMS_ACCEPTED, false)

    // ponytail: one try per process; add backoff if Failed rates show in analytics
    private val photoModelFetchTried = AtomicBoolean(false)

    private lateinit var diskRefresh: Job

    fun initialize() {
        // Without this every PDF extracts to null with no error.
        DocumentExtractor.init(application)
        diskRefresh = scope.launch(Dispatchers.IO) {
            provisioner.refreshFromDisk(application)
            visionProvisioner.refreshFromDisk(application)
        }
        application.registerActivityLifecycleCallbacks(object : Application.ActivityLifecycleCallbacks {
            override fun onActivityStarted(activity: Activity) {
                scope.launch {
                    if (shouldRunForegroundDelta(lastIndexFinishedAt(), System.currentTimeMillis())) {
                        BootReceiver.enqueueForegroundDeltaWork(application)
                    }
                }
                fetchPhotoModel()
            }
            override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) {}
            override fun onActivityResumed(activity: Activity) {}
            override fun onActivityPaused(activity: Activity) {}
            override fun onActivityStopped(activity: Activity) {}
            override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) {}
            override fun onActivityDestroyed(activity: Activity) {}
        })
    }

    /** The only entry to the model download: the user accepted the Gemma terms on screen. */
    fun acceptTermsAndProvision() {
        application.getSharedPreferences(PREFS, Application.MODE_PRIVATE).edit()
            .putBoolean(KEY_TERMS_ACCEPTED, true).apply()
        provisionThenIndex()
    }

    private fun provisionThenIndex() {
        if (!isTermsAccepted) return
        scope.launch {
            Analytics.capture("model_download_started")
            provisioner.provision(application, BuildConfig.MODEL_BASE_URL)
            val textReady = provisioner.stateFlow.value == ProvisionState.Ready
            Analytics.capture(if (textReady) "model_download_completed" else "model_download_failed")
            val imagesOn = withContext(Dispatchers.IO) {
                DeviceTierDetector.detect(application).imageEmbeddingEnabled
            }
            val visionState = if (textReady && imagesOn) {
                visionProvisioner.provision(application, BuildConfig.MODEL_BASE_URL)
                visionProvisioner.stateFlow.value
            } else {
                ProvisionState.Ready
            }
            if (textReady && (visionState is ProvisionState.Ready || visionState is ProvisionState.Failed) &&
                Environment.isExternalStorageManager()) {
                startIndexing()
            }
        }
    }

    /**
     * The one path that fetches a missing photo model after terms time. The activity-start hook
     * calls it at most once per process; a user tap ([userTap]) skips that limit, the user is present.
     */
    fun fetchPhotoModel(userTap: Boolean = false) {
        scope.launch {
            diskRefresh.join()
            val fetch = shouldFetchPhotoModel(
                termsAccepted = isTermsAccepted,
                textReady = provisioner.stateFlow.value == ProvisionState.Ready,
                visionState = visionProvisioner.stateFlow.value,
                triedThisProcess = !userTap && photoModelFetchTried.get(),
            ) { DeviceTierDetector.detect(application).imageEmbeddingEnabled }
            if (!fetch || (!userTap && !photoModelFetchTried.compareAndSet(false, true))) return@launch
            visionProvisioner.provision(application, BuildConfig.MODEL_BASE_URL)
            // A run that started during the download is waiting in retry backoff for this model
            // (or went text-only): REPLACE runs one for the photos now. KEEP would drop this
            // request behind it. WorkManager, not startIndexing(): the user may have left the
            // app, and a background foreground-service start throws. No-op before the first index.
            if (visionProvisioner.stateFlow.value == ProvisionState.Ready) {
                BootReceiver.enqueueForegroundDeltaWork(application, ExistingWorkPolicy.REPLACE)
            }
        }
    }

    /** Full pass, then prune. Also used after an exclusion change (never a wipe). */
    fun startIndexing() {
        try {
            ContextCompat.startForegroundService(
                application,
                Intent(application, IndexService::class.java).setAction(IndexService.ACTION_START_INDEXING)
            )
        } catch (e: IllegalStateException) {
            Log.w(TAG, "Cannot start indexing service from background", e)
        }
    }

    fun reindexAll(): Boolean {
        if (!IndexService.isReindexReady(application)) return false
        try {
            ContextCompat.startForegroundService(
                application,
                Intent(application, IndexService::class.java).setAction(IndexService.ACTION_REINDEX)
            )
        } catch (e: IllegalStateException) {
            Log.w(TAG, "Cannot start reindex service from background", e)
            return false
        }
        return true
    }

    private suspend fun lastIndexFinishedAt(): Long = try {
        val wm = WorkManager.getInstance(application)
        listOf(
            IndexService.INITIAL_INDEX_WORK_NAME,
            BootReceiver.FOREGROUND_DELTA_WORK_NAME,
            MediaStoreObserver.DELTA_WORK_NAME,
        ).flatMap { wm.getWorkInfosForUniqueWorkFlow(it).first() }
            .filter { it.state == WorkInfo.State.SUCCEEDED }
            .maxOfOrNull { it.outputData.getLong(IndexWorker.KEY_FINISHED_AT, 0L) } ?: 0L
    } catch (e: Exception) {
        Log.w(TAG, "Could not read the last index run, running the delta", e)
        0L
    }
}
