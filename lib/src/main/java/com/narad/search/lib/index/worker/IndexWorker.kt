package com.narad.search.lib.index.worker

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.net.Uri
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerFactory
import androidx.work.WorkerParameters
import com.narad.search.lib.device.DeviceTierDetector
import com.narad.search.lib.index.chunking.Chunker
import com.narad.search.lib.index.enumeration.MediaStoreEnumerator
import com.narad.search.lib.index.extraction.DocumentExtractor
import com.narad.search.lib.index.provision.EmbeddingGemma2Manifest
import com.narad.search.lib.index.provision.EmbeddingGemma2VisionManifest
import com.narad.search.lib.index.service.IndexNotifications
import com.narad.search.lib.index.service.IndexService
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository
import com.narad.search.lib.query.model.EmbeddingModel
import com.narad.search.lib.query.model.image.EG2BitmapImageEncoder
import com.narad.search.lib.query.model.image.VisionEncoder
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File

internal enum class VisionFilesState {
    ABSENT,
    INCOMPLETE,
    READY
}

internal enum class VisionIndexOutcome {
    COMPLETE,
    RETRY,
    TEXT_ONLY,
    /** Documents done; the photo pass waits for free memory (next run tries again). */
    DEFERRED
}

internal const val MAX_VISION_RETRIES = 3
const val PHOTOS_NOT_READY_STATUS = "Photos not ready"
const val PHOTOS_DEFERRED_STATUS = "Photos deferred"

// ponytail: calibration knob. The vision encoder adds ~450 MB on top of the text model (stress
// test: ~950 MB PSS for both). Tune from the overnight metrics build on real 4 GB phones.
internal const val PHOTO_PASS_HEADROOM_BYTES = 600L * 1024 * 1024

/** Pure: true when free memory above the low-memory line leaves room to load the vision encoder. */
internal fun photoPassFits(
    availBytes: Long,
    thresholdBytes: Long,
    lowMemory: Boolean,
    headroomBytes: Long,
): Boolean = !lowMemory && availBytes - thresholdBytes >= headroomBytes

internal fun visionUnavailableOutcome(runAttemptCount: Int): VisionIndexOutcome =
    if (runAttemptCount < MAX_VISION_RETRIES) {
        VisionIndexOutcome.RETRY
    } else {
        VisionIndexOutcome.TEXT_ONLY
    }

/** What a finished run reports: the Done stage and the [IndexWorker.KEY_STATUS] value when photos were skipped. */
internal data class TerminalOutput(val stage: IndexStage.Done, val status: String?)

/**
 * Pure: the terminal stage of a run, or null when the run must be retried.
 *
 * Computed and published INSIDE the run's try block (review-021) so a failing final
 * setProgress still reaches the ERROR event writer before the DB closes.
 */
internal fun terminalOutput(
    outcome: VisionIndexOutcome,
    totalIndexed: Int,
    photosIndexed: Int,
): TerminalOutput? {
    val count = totalIndexed + photosIndexed
    return when (outcome) {
        VisionIndexOutcome.COMPLETE ->
            TerminalOutput(
                IndexStage.Done(count, reason = if (count == 0) "Nothing to read yet" else null),
                status = null,
            )
        VisionIndexOutcome.RETRY -> null
        VisionIndexOutcome.TEXT_ONLY -> TerminalOutput(IndexStage.Done(count), PHOTOS_NOT_READY_STATUS)
        VisionIndexOutcome.DEFERRED -> TerminalOutput(IndexStage.Done(count), PHOTOS_DEFERRED_STATUS)
    }
}

/**
 * Holds the text embedder and builds it on first use. A delta where no document changed never
 * embeds, so it never pays the model load (about 490 MB of native memory). [get] is also the
 * shared-model entry point for the photo branch, so documents and photos share one build.
 *
 * Runs are sequential on one IO thread, so there is no lock.
 */
internal class LazyEmbedder(private val build: () -> Embedder) : Embedder {
    private var inner: Embedder? = null

    val isBuilt: Boolean get() = inner != null

    fun get(): Embedder = inner ?: build().also { inner = it }

    override fun embed(text: String): FloatArray = get().embed(text)

    override fun close() {
        inner?.close()
    }
}

/**
 * Thrown out of [LazyImageEncoder.encode] to end the photo pass with [outcome]. It is an Error,
 * not an Exception, because [ImageIndexer] catches Exception per photo and would carry on.
 */
private class PhotoPassAborted(val outcome: VisionIndexOutcome) : Error()

/** The text model failed to build. An Error for the same reason; [original] is rethrown as the run failure. */
private class TextModelBuildFailed(val original: Throwable) : Error()

/**
 * Builds the real encoder (and through it the shared text model and [VisionEncoder]) on the first
 * [encode] call. [ImageIndexer] only encodes after a photo decoded, so a delta whose pending
 * photos are all unreadable builds nothing. [build] throws [PhotoPassAborted] when it cannot.
 */
private class LazyImageEncoder(private val build: () -> ImageEncoder) : ImageEncoder {
    private var inner: ImageEncoder? = null

    override fun encode(bitmap: android.graphics.Bitmap): FloatArray =
        (inner ?: build().also { inner = it }).encode(bitmap)

    override fun close() {
        inner?.close()
    }
}

/** How much of the photo library the app can read. Only FULL may drive a prune. */
internal enum class PhotoAccess { NONE, PARTIAL, FULL }

/**
 * Pure: photo access level for an API level and the granted permissions.
 * API 34+ "selected photos" (READ_MEDIA_VISUAL_USER_SELECTED only) is PARTIAL: the app can read
 * the chosen subset but MediaStore lists only that subset, so it must never prune.
 */
internal fun photoAccess(
    sdk: Int,
    mediaImages: Boolean,
    userSelected: Boolean,
    readExternal: Boolean,
    // Index0: All files access (API 30+) lists every image in MediaStore, so it is FULL.
    manager: Boolean = false,
): PhotoAccess = when {
    sdk >= android.os.Build.VERSION_CODES.R && manager -> PhotoAccess.FULL
    sdk >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE ->
        if (mediaImages) PhotoAccess.FULL else if (userSelected) PhotoAccess.PARTIAL else PhotoAccess.NONE
    sdk >= android.os.Build.VERSION_CODES.TIRAMISU ->
        if (mediaImages) PhotoAccess.FULL else PhotoAccess.NONE
    else -> if (readExternal) PhotoAccess.FULL else PhotoAccess.NONE
}

/**
 * Pure prune gate, documents: access held when the listing was made AND still held after the
 * pass AND the whole pass ran (no stop). Checking both ends closes a mid-pass grant change.
 */
internal fun shouldPruneDocuments(
    atListing: Boolean,
    now: Boolean,
    passFinished: Boolean,
): Boolean = atListing && now && passFinished

/**
 * Pure prune gate, photos: FULL access (not user-selected-only) when the listing was made AND
 * after the pass, AND the whole pass ran. A PARTIAL listing followed by FULL (the user switched
 * "Select photos" to "Allow all" mid-pass) holds a subset, so it must not prune.
 */
internal fun shouldPrunePhotos(
    atListing: PhotoAccess,
    now: PhotoAccess,
    passFinished: Boolean,
): Boolean = atListing == PhotoAccess.FULL && now == PhotoAccess.FULL && passFinished

/**
 * Returns true when all three model artefacts exist on device.
 *
 * Extracted from [IndexWorker.doWork] so it can be tested without an Android
 * context. Existence implies completeness because [ModelProvisioner] renames
 * each file into place only after its SHA-256 check.
 */
internal fun modelFilesReady(modelPath: String): Boolean =
    listOf(
        File(modelPath),
        File(modelPath + "_data"),
        File(modelPath).resolveSibling("tokenizer.json")
    ).all { it.exists() }

/**
 * Returns 0-100 indicating how much of the model has been downloaded.
 *
 * Computes bytesPresent / bytesExpected from the [EmbeddingGemma2Manifest].
 * Returns 0 if the model directory does not exist. Returns 100 when all files
 * are on disk at their expected sizes (same condition as [modelFilesReady]).
 *
 * Extracted from [IndexWorker.doWork] for JVM testability.
 */
internal fun modelDownloadPercent(modelPath: String): Int {
    val modelDir = File(modelPath).parentFile ?: return 0
    val manifest = EmbeddingGemma2Manifest.manifest
    val totalBytes = manifest.files.sumOf { it.sizeBytes }
    if (totalBytes == 0L) return 0
    val presentBytes = manifest.files.sumOf { mf ->
        val f = File(modelDir, File(mf.relativeDest).name)
        if (f.exists()) minOf(f.length(), mf.sizeBytes) else 0L
    }
    return ((presentBytes * 100L) / totalBytes).toInt()
}

/**
 * Polls until the model is ready, publishing [IndexStage.WaitingForModel] each cycle.
 *
 * Returns true if the model became ready, false if stopped or timed out.
 *
 * Extracted for JVM testability - all side-effects are injected as lambdas:
 *  - [isReady]    - returns true when the model files exist
 *  - [getPercent] - returns 0-100 download percent
 *  - [onStage]    - called with each [IndexStage.WaitingForModel] update
 *  - [isStopped]  - returns true when the worker was cancelled
 *  - [maxMs]      - wall-clock cap (coroutine timeout, virtual time in tests)
 *  - [pollMs]     - delay between readiness checks
 */
internal suspend fun waitForModel(
    isReady: () -> Boolean,
    getPercent: () -> Int,
    onStage: (IndexStage) -> Unit,
    isStopped: () -> Boolean,
    maxMs: Long,
    pollMs: Long
): Boolean {
    if (isStopped()) return false
    if (isReady()) return true
    // Emit at least one stage before entering the wait loop.
    onStage(IndexStage.WaitingForModel(percent = getPercent().takeIf { it > 0 }))
    val became = withTimeoutOrNull(maxMs) {
        while (!isStopped()) {
            if (isReady()) return@withTimeoutOrNull true
            delay(pollMs)
            onStage(IndexStage.WaitingForModel(percent = getPercent().takeIf { it > 0 }))
        }
        null // stopped
    }
    return became == true
}

internal fun visionFilesState(visionModelPath: String): VisionFilesState {
    val modelDir = File(visionModelPath).parentFile ?: return VisionFilesState.INCOMPLETE
    val expectedFiles = EmbeddingGemma2VisionManifest.manifest.files.map { modelFile ->
        File(modelDir, File(modelFile.relativeDest).name) to modelFile.sizeBytes
    }
    if (expectedFiles.none { (file, _) -> file.exists() }) return VisionFilesState.ABSENT
    return if (expectedFiles.all { (file, size) -> file.isFile && file.length() == size }) {
        VisionFilesState.READY
    } else {
        VisionFilesState.INCOMPLETE
    }
}

/**
 * WorkManager [CoroutineWorker] that orchestrates the indexing pipeline.
 *
 * Input data:
 *  - [KEY_MODEL_PATH]     : absolute path to the ONNX model file
 *  - [KEY_FILE_PATHS]     : newline-separated (`"\n"`) list of specific files to index.
 *                           Newline is used (not comma) because file paths may contain commas.
 *                           Producer side must encode with `filePaths.joinToString("\n")`.
 *
 * Guard: if the model file is absent → return [Result.retry()] without attempting to load any model.
 *
 * KTD-8 compliance: checkpoints every 50 files via [Indexer.onProgress].
 *
 * Retry cap: permanent failures (non-model-absent exceptions) fail after [MAX_RETRIES] attempts
 * to avoid infinite retry loops (fix item 4).
 *
 * Embedder injection: use [Factory] (a [WorkerFactory]) to inject a custom [embedderFactory]
 * for testing. Production uses the default constructor which calls [EmbeddingModelFactory].
 *
 * Image encoder injection: [imageEncoderFactory] is an overridable factory for the
 * EG2 vision encoder. Default builds [EG2BitmapImageEncoder] from the shared [EmbeddingModel].
 * Inject a fake in tests to avoid the real ONNX model. The factory receives the vision-model
 * path (from [IndexService.getVisionEncoderPath]) and returns a nullable encoder.
 * Null means "skip image indexing" (graceful degradation: capability not available, or model absent).
 */
class IndexWorker @JvmOverloads constructor(
    context: Context,
    params: WorkerParameters,
    /**
     * Overridable factory for testing. Production default builds from model path.
     *
     * When non-null, the model-file-existence guard is bypassed (embedderFactory != null
     * signals "test path").  When null, production [EmbeddingModelFactory.build] is called.
     *
     * This is the LAST constructor parameter so that the existing trailing-lambda calling
     * convention `IndexWorker(ctx, params) { _ -> FakeEmbedder() }` continues to work.
     */
    private val embedderFactory: ((modelPath: String) -> Embedder)? = null
) : CoroutineWorker(context, params) {

    /**
     * Image encoder factory, set via [withImageEncoder] after construction.
     *
     * This is a mutable property rather than a constructor parameter to avoid
     * Kotlin trailing-lambda ambiguity: adding [imageEncoderFactory] as a 4th
     * constructor param with a default would cause `IndexWorker(ctx, params) { lambda }`
     * to route the lambda to `imageEncoderFactory` (the LAST param) rather than
     * `embedderFactory` (3rd param), breaking existing tests.
     *
     * Instead, callers that need both factories use [IndexWorker.withImageEncoder]:
     * ```
     * IndexWorker(ctx, params, embedderFactory = { _ -> FakeEmbedder() })
     *     .withImageEncoder { _ -> FakeImageEncoder() }
     * ```
     *
     * Null means "use production path" (DeviceTierDetector + model-file guard).
     */
    private var imageEncoderFactory: ((visionModelPath: String) -> ImageEncoder?)? = null

    /**
     * Controls whether the model-file-existence guard fires in doWork.
     *
     * Defaults to true when no embedderFactory is injected (production path) and false
     * when a fake embedder is injected (test path; model files not needed).
     *
     * Override to true in tests that want to exercise the WaitingForModel wait path
     * even while using a FakeEmbedder (e.g. u25_regression_tapStartDuringDownload).
     * ponytail: seam exists so the test can force the guard without a real ONNX model.
     */
    internal var checkModelFiles: Boolean = embedderFactory == null

    /**
     * Max milliseconds to wait for the model before giving up and retrying.
     * Settable in tests to avoid a 5-minute block.
     */
    internal var modelWaitMaxMs: Long = 5 * 60 * 1_000L

    /**
     * Poll interval while waiting for the model.
     * Settable in tests to use virtual-time delays.
     */
    internal var modelPollMs: Long = 5_000L

    /**
     * Set the image encoder factory and return [this] for fluent chaining.
     *
     * Use in tests that need to exercise the image branch:
     * ```kotlin
     * IndexWorker(ctx, params, embedderFactory = { _ -> FakeEmbedder() })
     *     .withImageEncoder { _ -> FakeImageEncoder() }
     * ```
     */
    internal fun withImageEncoder(factory: (visionModelPath: String) -> ImageEncoder?): IndexWorker {
        this.imageEncoderFactory = factory
        return this
    }

    /**
     * WorkManager [WorkerFactory] that injects custom factories into [IndexWorker].
     *
     * Use this in instrumented tests to avoid loading the ONNX models on the emulator.
     *
     * Usage (text-only fake, image branch skipped):
     * ```
     * val config = Configuration.Builder()
     *     .setWorkerFactory(IndexWorker.Factory(
     *         embedderFactory = { _ -> FakeEmbedder() },
     *         imageEncoderFactory = { _ -> null }  // skip image branch
     *     ))
     *     .build()
     * WorkManagerTestInitHelper.initializeTestWorkManager(context, config)
     * ```
     *
     * Usage (image branch with fake encoder):
     * ```
     * IndexWorker.Factory(
     *     embedderFactory = { _ -> FakeEmbedder() },
     *     imageEncoderFactory = { _ -> FakeImageEncoder() }
     * )
     * ```
     */
    class Factory private constructor(
        private val embedderFactory: (modelPath: String) -> Embedder,
        /** Optional image encoder factory. Null skips image branch. */
        private val imageEncoderFactory: ((visionModelPath: String) -> ImageEncoder?)?
    ) : WorkerFactory() {

        /**
         * Backward-compatible single-factory constructor.
         *
         * Preserves the original call syntax:
         * ```
         * IndexWorker.Factory { _ -> FakeEmbedder() }
         * ```
         *
         * Kotlin trailing-lambda syntax matches the LAST constructor parameter. If we had
         * `imageEncoderFactory` as a second parameter with a default, the trailing lambda
         * would route there instead of to `embedderFactory`. This single-param secondary
         * constructor avoids that ambiguity. Kotlin resolves the trailing-lambda call to
         * this overload because it is the best single-lambda match.
         */
        constructor(embedderFactory: (modelPath: String) -> Embedder)
            : this(embedderFactory, null)

        override fun createWorker(
            appContext: Context,
            workerClassName: String,
            workerParameters: WorkerParameters
        ): CoroutineWorker? {
            return if (workerClassName == IndexWorker::class.java.name) {
                val worker = IndexWorker(appContext, workerParameters, embedderFactory)
                val localFactory = imageEncoderFactory
                if (localFactory != null) {
                    worker.withImageEncoder(localFactory)
                } else {
                    worker
                }
            } else {
                null  // let WorkManager's default factory handle other workers
            }
        }

        companion object {
            /**
             * Create a [Factory] with BOTH text and image encoder factories.
             *
             * Use in tests that need to exercise the image branch end-to-end:
             * ```kotlin
             * val factory = IndexWorker.Factory.withBothEncoders(
             *     embedderFactory     = { _ -> FakeEmbedder() },
             *     imageEncoderFactory = { _ -> FakeImageEncoder() }
             * )
             * ```
             *
             * This static factory exists because the primary 2-param constructor is private
             * (to prevent Kotlin's trailing-lambda syntax from accidentally routing a single
             * lambda to `imageEncoderFactory` instead of `embedderFactory`).
             */
            fun withBothEncoders(
                embedderFactory: (modelPath: String) -> Embedder,
                imageEncoderFactory: (visionModelPath: String) -> ImageEncoder?
            ): Factory = Factory(embedderFactory, imageEncoderFactory)
        }
    }

    companion object {
        private const val TAG = "IndexWorker"

        const val KEY_MODEL_PATH = "model_path"

        /**
         * Newline-separated list of file paths to index.
         * IMPORTANT: paths are separated by `"\n"` (not comma) because file paths may contain commas.
         * Encode: `filePaths.joinToString("\n")`
         * Decode: `value.split("\n").filter { it.isNotBlank() }`
         */
        const val KEY_FILE_PATHS = "file_paths"

        const val KEY_PROGRESS_PROCESSED = "processed"
        const val KEY_PROGRESS_TOTAL = "total"
        const val KEY_STATUS = "status"
        const val KEY_FINISHED_AT = "finished_at"
        /** Epoch millis when the run started. Duration = [KEY_FINISHED_AT] - this. */
        const val KEY_STARTED_AT = "started_at"

        /**
         * WorkManager progress key for the current [IndexStage].
         *
         * Alias for [IndexStage.KEY_STAGE] - exposed here for call-site convenience.
         * Deserialize via [IndexStage.fromData].
         */
        const val KEY_STAGE = IndexStage.KEY_STAGE

        /** Maximum number of attempts before giving up on a permanent failure. */
        const val MAX_RETRIES = 3

        private val INDEX_RUN_LOCK = Mutex()
    }

    override suspend fun doWork(): Result {
        val modelPath = inputData.getString(KEY_MODEL_PATH)
            ?: return Result.retry()

        // Guard: model file must exist on device before we can embed anything.
        // When absent, publish WaitingForModel and wait up to 10 minutes for the
        // provisioner to finish. If the wait times out or the worker is stopped,
        // publish the stage and retry so WorkManager reschedules.
        // checkModelFiles is false when embedderFactory is injected (test path) unless
        // the test explicitly overrides it to exercise the WaitingForModel path.
        if (checkModelFiles && !modelFilesReady(modelPath)) {
            Log.d(TAG, "Model not ready at $modelPath - waiting (attempt ${runAttemptCount + 1})")
            val ready = waitForModel(
                isReady    = { modelFilesReady(modelPath) },
                getPercent = { modelDownloadPercent(modelPath) },
                onStage    = { stage ->
                    // ponytail: runBlocking bridge - onStage is non-suspend, setProgress is suspend.
                    // Runs on Dispatchers.Default (doWork thread); setProgress is fast, no deadlock.
                    kotlinx.coroutines.runBlocking { setProgress(stage.toData()) }
                    updateStageNotification(stage)
                },
                isStopped  = { isStopped },
                maxMs      = modelWaitMaxMs,
                pollMs     = modelPollMs
            )
            if (!ready) {
                val pct = modelDownloadPercent(modelPath).takeIf { it > 0 }
                setProgress(IndexStage.WaitingForModel(pct).toData())
                // review-022: this attempt never reaches runIndexing, so it records its own
                // event and prunes. INFO level: waiting for a download is not a problem.
                recordModelWaitRetry()
                return Result.retry()
            }
        }

        // One indexing run per process: the initial, periodic and MediaStore jobs have different
        // unique names and can be released together. Each run loads ~500 MB of models and
        // writes the same DB, so later runs wait here (they are cancelled if stopped meanwhile).
        return INDEX_RUN_LOCK.withLock { withContext(Dispatchers.IO) {
            try {
                runIndexing(modelPath).also {
                    // W29/E19: a Done or WaitingForPermission success before the flag exists is
                    // the user-started first index finishing (deltas are blocked until then).
                    // A Paused success (system stop mid-read) must not set it. Set here, not in
                    // IndexService, so it survives the service process dying. Commit lands
                    // before WorkManager reports SUCCEEDED.
                    markFirstIndexDoneOnSuccess(it)
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                // Worker stopped: rethrow so the coroutine machinery handles it correctly.
                // If caught by the generic Exception handler below, a stop would be
                // misreported as Failed instead of being silently absorbed.
                throw e
            } catch (e: Exception) {
                failureResult(e)
            } catch (e: LinkageError) {
                failureResult(e)
            }
        } }
    }

    private suspend fun failureResult(failure: Throwable): Result {
        Log.e(TAG, "Indexing failed on attempt ${runAttemptCount + 1}", failure)
        val failureCategory = failure.javaClass.simpleName
            .ifEmpty { "Exception" }
            .take(100)
        val failedStage = IndexStage.Failed(failureCategory)
        setProgress(failedStage.toData())
        updateStageNotification(failedStage)
        return if (runAttemptCount >= MAX_RETRIES) {
            Log.e(TAG, "Max retries ($MAX_RETRIES) exceeded - marking as failure")
            Result.failure(
                androidx.work.Data.Builder()
                    .putAll(failedStage.toData())
                    .putLong(KEY_FINISHED_AT, System.currentTimeMillis())
                    .build()
            )
        } else {
            Result.retry()
        }
    }

    /**
     * Production indexing path. Only reached when the model file exists or embedderFactory
     * is injected (test path).
     *
     * Progress is published to WorkManager every 50 files or 1 s, see [ProgressTicker] (and at
     * the end). Because [Indexer.onProgress] is a non-suspend lambda but
     * [setProgress] is a suspend function, we use [kotlinx.coroutines.runBlocking]
     * as a pragmatic bridge inside the lambda. The lambda is always called from
     * a background thread ([Dispatchers.IO]), so runBlocking does not block the
     * main thread.
     */
    private suspend fun runIndexing(modelPath: String): Result {
        promoteToForeground()
        val db = IndexDatabase(applicationContext)
        val repository = IndexRepository(db)
        // M18: unique ID per run, shared across all event writes in this run.
        val runId = java.util.UUID.randomUUID().toString()
        val startedAt = System.currentTimeMillis()
        var embedder: LazyEmbedder? = null
        var visionOutcome = VisionIndexOutcome.COMPLETE
        // Track total files indexed across all passes for Done(count).
        var totalIndexed = 0
        // ponytail: separate counter because ImageIndexer reports CUMULATIVE counts (each
        // tick then final), so += would double-count. Assign here, add at Done.
        var photosIndexed = 0
        // Set to true when isStopped is detected so Paused is not overwritten by Done.
        var wasStopped = false
        // M18: updated by each extractText lambda before calling extractor, so the onWarn
        // closure can include the file name without an additional parameter.
        var currentTarget = ""
        var currentTargetWarned = false

        val rawFilePaths = inputData.getString(KEY_FILE_PATHS)
            ?.split("\n")
            ?.map { it.trim() }
            ?.filter { it.isNotEmpty() }

        val result: Result = try {
            // Branch B only: check access per pass before any work (the model loads lazily).
            // Photos and documents are independent - one missing does not block the other.
            // Only when NEITHER pass has any access do we publish WaitingForPermission.
            if (rawFilePaths.isNullOrEmpty()) {
                val hasPhoto = isPhotoAccessGranted()
                val hasDoc   = isDocumentAccessGranted()
                if (!hasPhoto && !hasDoc) {
                    val whichPerm = when {
                        android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU ->
                            android.Manifest.permission.READ_MEDIA_IMAGES
                        else ->
                            android.Manifest.permission.READ_EXTERNAL_STORAGE
                    }
                    val waitPerm = IndexStage.WaitingForPermission(which = whichPerm)
                    setProgress(waitPerm.toData())
                    updateStageNotification(waitPerm)
                    // Return success - app-open delta re-runs after the user grants access.
                    // E19: carry the stage so the first-index flag rule can tell this from Paused.
                    return Result.success(
                        androidx.work.Data.Builder()
                            .putAll(waitPerm.toData())
                            .putLong(KEY_FINISHED_AT, System.currentTimeMillis())
                            .build()
                    )
                }
            }

            val chunker = Chunker()
            // M18: onWarn fires alongside each Log.w in DocumentExtractor.
            // currentTarget is updated by each extractText lambda before calling extractStream,
            // so the closure always captures the file being processed at that moment.
            val extractor = DocumentExtractor(
                onWarn = { errorClass, msg ->
                    currentTargetWarned = true
                    repository.appendEvent(
                        runId      = runId,
                        stage      = "extraction",
                        level      = "WARN",
                        fileName   = currentTarget,
                        errorClass = errorClass,
                        message    = msg
                    )
                }
            )

            // Built on the first chunk that needs embedding, not here (an unchanged delta never builds it).
            val lazyEmbedder = LazyEmbedder {
                if (embedderFactory != null) {
                    embedderFactory.invoke(modelPath)
                } else {
                    val tokenizerPath = File(modelPath).resolveSibling("tokenizer.json").absolutePath
                    EmbeddingModelFactory.build(modelPath, tokenizerPath)
                }
            }
            embedder = lazyEmbedder

            if (!rawFilePaths.isNullOrEmpty()) {
                // Branch A: explicit file-path list (test path + future explicit callers).
                setProgress(IndexStage.Finding("documents").toData())
                // Publish Reading(0, n) immediately so the chip shows the count before first checkpoint.
                if (rawFilePaths.isNotEmpty()) {
                    val initStage = IndexStage.Reading("documents", 0, rawFilePaths.size)
                    setProgress(initStage.toData())
                    updateStageNotification(initStage)
                }
                val onDocProgress: (Int, Int) -> Unit = { processed, total ->
                    totalIndexed = processed
                    val stage = IndexStage.Reading("documents", processed, total)
                    kotlinx.coroutines.runBlocking { setProgress(stage.toData()) }
                    updateStageNotification(stage)
                }
                val indexer = Indexer(
                    embedder = lazyEmbedder,
                    extractText = { path ->
                        currentTarget = path  // M18: update before extract so onWarn has the name
                        currentTargetWarned = false
                        try {
                            val file = File(path)
                            val mimeType = guessMimeType(file)
                            file.inputStream().use { extractor.extractStream(it, mimeType) }
                        } catch (e: Exception) {
                            Log.w(TAG, "Failed to extract text from $path", e)
                            if (!currentTargetWarned) {
                                repository.appendEvent(runId, "extraction", "WARN", path, e.javaClass.simpleName, "Failed to extract text")
                            }
                            null
                        }
                    },
                    chunkText = { text -> chunker.chunk(text) },
                    repository = repository,
                    checkpointEvery = 50,
                    onProgress = onDocProgress,
                    shouldStop = { isStopped }
                )
                indexer.indexFiles(rawFilePaths)
                if (isStopped) {
                    setProgress(IndexStage.Paused.toData())
                    updateStageNotification(IndexStage.Paused)
                    wasStopped = true
                }
                // Branch A does not run image indexing.

            } else {
                // Branch B: MediaStore pass - documents then photos, each guarded by its own access.
                val hasDoc   = isDocumentAccessGranted()
                val hasPhoto = isPhotoAccessGranted()

                // --- Documents pass ---
                if (hasDoc && !isStopped) {
                    setProgress(IndexStage.Finding("documents").toData())
                    updateStageNotification(IndexStage.Finding("documents"))
                    Log.d(TAG, "Enumerating documents from MediaStore")
                    val enumerator = MediaStoreEnumerator()
                    // Prune guards: access and volumes as they are at listing time.
                    val docAccessAtListing = isDocumentAccessGranted()
                    repository.recordVolumes(presentVolumes())
                    val docTargets = enumerator.enumerateDocuments(applicationContext)
                    Log.d(TAG, "Document enumeration: ${docTargets.size}")

                    // Publish Reading(0, n) immediately so the chip shows before the first 50.
                    if (docTargets.isNotEmpty()) {
                        val initStage = IndexStage.Reading("documents", 0, docTargets.size)
                        setProgress(initStage.toData())
                        updateStageNotification(initStage)
                    }

                    val docDateMap = docTargets.associate { it.uri to it.lastModified }
                    val onDocProgress: (Int, Int) -> Unit = { processed, total ->
                        totalIndexed = processed
                        val stage = IndexStage.Reading("documents", processed, total)
                        kotlinx.coroutines.runBlocking { setProgress(stage.toData()) }
                        updateStageNotification(stage)
                    }
                    val indexer = Indexer(
                        embedder = lazyEmbedder,
                        extractText = { uriString ->
                            currentTarget = uriString  // M18: update before extract
                            currentTargetWarned = false
                            try {
                                extractor.extract(Uri.parse(uriString), applicationContext)
                            } catch (e: Exception) {
                                Log.w(TAG, "Failed to extract text from $uriString", e)
                                if (!currentTargetWarned) {
                                    repository.appendEvent(runId, "extraction", "WARN", uriString, e.javaClass.simpleName, "Failed to extract text")
                                }
                                null
                            }
                        },
                        chunkText = { text -> chunker.chunk(text) },
                        repository = repository,
                        checkpointEvery = 50,
                        onProgress = onDocProgress,
                        lastModifiedOf = { uriString -> docDateMap[uriString] ?: 0L },
                        shouldStop = { isStopped }
                    )
                    val docUris = docTargets.map { it.uri }
                    val docPassFinished = indexer.indexFiles(docUris) && !isStopped
                    // R13: drop rows for documents deleted from the phone. Full enumeration above,
                    // access re-checked, pass finished: see shouldPruneDocuments.
                    if (shouldPruneDocuments(docAccessAtListing, isDocumentAccessGranted(), docPassFinished)) {
                        val removed = repository.pruneDocuments(docUris.toSet(), presentVolumes())
                        Log.d(TAG, "Pruned $removed deleted document(s)")
                    }
                }

                if (isStopped) {
                    setProgress(IndexStage.Paused.toData())
                    updateStageNotification(IndexStage.Paused)
                    wasStopped = true
                } else if (hasPhoto) {
                    // --- Photos pass ---
                    val sharedModel = { (lazyEmbedder.get() as? OnnxEmbedder)?.model }
                    val imgOnProgress: (Int, Int) -> Unit = { processed, total ->
                        photosIndexed = processed  // cumulative from ImageIndexer, not additive
                        val stage = IndexStage.Reading("photos", processed, total)
                        kotlinx.coroutines.runBlocking { setProgress(stage.toData()) }
                        updateStageNotification(stage)
                    }
                    visionOutcome = runImageBranch(
                        repository, imgOnProgress, sharedModel,
                        onFinding = {
                            val s = IndexStage.Finding("photos")
                            kotlinx.coroutines.runBlocking { setProgress(s.toData()) }
                            updateStageNotification(s)
                        },
                        runId = runId
                    )
                }
            }

            // review-021: the terminal publish runs inside the try, so a failure in the final
            // setProgress reaches the catch below (ERROR event) before the finally closes the DB.
            // Paused: already published above, do not overwrite with Done.
            if (wasStopped) {
                Result.success(
                    androidx.work.Data.Builder()
                        .putLong(KEY_STARTED_AT, startedAt)
                        .putLong(KEY_FINISHED_AT, System.currentTimeMillis())
                        .build()
                )
            } else {
                val terminal = terminalOutput(visionOutcome, totalIndexed, photosIndexed)
                if (terminal == null) {
                    Result.retry()
                } else {
                    setProgress(terminal.stage.toData())
                    updateStageNotification(terminal.stage)
                    // WorkManager clears progress once work finishes, so the Done stage also
                    // goes in the output data (the app reads outputData for the celebration).
                    Result.success(
                        androidx.work.Data.Builder()
                            .putAll(terminal.stage.toData())
                            .apply { terminal.status?.let { putString(KEY_STATUS, it) } }
                            .putLong(KEY_STARTED_AT, startedAt)
                            .putLong(KEY_FINISHED_AT, System.currentTimeMillis())
                            .build()
                    )
                }
            }

        } catch (e: Exception) {
            // M18: write run-level failure before finally prunes and closes the DB.
            // CancellationException is rethrown without an event (it is not an indexing failure).
            if (e !is kotlinx.coroutines.CancellationException) {
                repository.appendEvent(
                    runId      = runId,
                    stage      = "worker",
                    level      = "ERROR",
                    fileName   = currentTarget.ifEmpty { null },
                    errorClass = e.javaClass.simpleName,
                    // Never use e.message; a parser exception message can echo file content.
                    message    = "Indexing failed on attempt ${runAttemptCount + 1}"
                )
            }
            throw e
        } catch (e: LinkageError) {
            repository.appendEvent(
                runId      = runId,
                stage      = "worker",
                level      = "ERROR",
                fileName   = currentTarget.ifEmpty { null },
                errorClass = e.javaClass.simpleName,
                message    = "Indexing failed on attempt ${runAttemptCount + 1}"
            )
            throw e
        } finally {
            try { embedder?.close() } catch (e: Exception) {
                Log.w(TAG, "Failed to close embedder", e)
                repository.appendEvent(runId, "worker", "WARN", null, e.javaClass.simpleName, "Failed to close embedder")
            }
            // M18: prune before closing DB so pruneEvents can write.
            repository.pruneEvents()
            db.close()
        }

        return result
    }

    private fun markFirstIndexDoneOnSuccess(result: Result) {
        if (shouldMarkFirstIndexDone(result)) IndexService.setFirstIndexDone(applicationContext, true)
    }

    /**
     * review-022: the model-wait retry returns before [runIndexing] builds its repository.
     * Open a short-lived one so this attempt still records an event and prunes.
     * Never throws: logging must not change the worker result.
     */
    private fun recordModelWaitRetry() {
        try {
            val db = IndexDatabase(applicationContext)
            try {
                val repository = IndexRepository(db)
                repository.appendEvent(
                    runId      = java.util.UUID.randomUUID().toString(),
                    stage      = "worker",
                    level      = "INFO",
                    errorClass = "ModelNotReady",
                    message    = "Model not ready, will retry (attempt ${runAttemptCount + 1})"
                )
                repository.pruneEvents()
            } finally {
                db.close()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to record model-wait event", e)
        }
    }

    /**
     * Returns true if the user has granted any photo read access on this API level
     * (full, or on API 34+ the "selected photos" subset).
     * Mirrors [com.narad.search.settings.PermissionState.isStorageGranted].
     */
    private fun isPhotoAccessGranted(): Boolean = photoAccessLevel() != PhotoAccess.NONE

    private fun photoAccessLevel(): PhotoAccess {
        val ctx = applicationContext
        val pm  = android.content.pm.PackageManager.PERMISSION_GRANTED
        val sdk = android.os.Build.VERSION.SDK_INT
        return photoAccess(
            sdk          = sdk,
            mediaImages  = ctx.checkSelfPermission(android.Manifest.permission.READ_MEDIA_IMAGES) == pm,
            userSelected = sdk >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE &&
                ctx.checkSelfPermission(android.Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED) == pm,
            readExternal = ctx.checkSelfPermission(android.Manifest.permission.READ_EXTERNAL_STORAGE) == pm,
            manager      = sdk >= android.os.Build.VERSION_CODES.R &&
                android.os.Environment.isExternalStorageManager(),
        )
    }

    /**
     * MediaStore volumes mounted right now. Empty on failure, which makes the prune skip.
     * minSdk 29, so getExternalVolumeNames always exists.
     */
    private fun presentVolumes(): Set<String> = try {
        android.provider.MediaStore.getExternalVolumeNames(applicationContext)
    } catch (e: Exception) {
        Log.w(TAG, "Could not list MediaStore volumes, skipping prune", e)
        emptySet()
    }

    /**
     * Returns true if the user has granted document-level storage access.
     * Documents (.pdf, .docx, .txt) require MANAGE_EXTERNAL_STORAGE on API 30+.
     */
    private fun isDocumentAccessGranted(): Boolean {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            android.os.Environment.isExternalStorageManager()
        } else {
            applicationContext.checkSelfPermission(android.Manifest.permission.READ_EXTERNAL_STORAGE) ==
                android.content.pm.PackageManager.PERMISSION_GRANTED
        }
    }


    /**
     * Image indexing branch (Branch B only).
     *
     * Guards:
     *  1. [DeviceTierDetector.detect] must return [imageEmbeddingEnabled]=true (Tier A/B).
     *  2. Both vision model files must have their expected sizes.
     *
     * An absent model is skipped. A partially provisioned model requests a retry.
     *
     * The image encoder (and through [sharedEmbeddingModel] the text model) is built on the first
     * photo that decodes and needs encoding, so a delta with nothing to encode, or only unreadable
     * photos, loads neither the shared model nor the vision encoder. It comes from the injected
     * [imageEncoderFactory] in tests. A text-model build failure propagates as a run failure.
     *
     * Note: the encoder lifecycle is managed within this method (closed in finally).
     * The shared [EmbeddingModel] (from [sharedEmbeddingModel]) is NOT closed here.
     */
    private fun runImageBranch(
        repository: IndexRepository,
        onProgress: (Int, Int) -> Unit,
        sharedEmbeddingModel: () -> EmbeddingModel? = { null },
        /** Called right before enumerateImages so the notification shows Finding("photos"). */
        onFinding: () -> Unit = {},
        /** M18: run ID for event writing. Required; always passed by runIndexing. */
        runId: String
    ): VisionIndexOutcome {
        val visionModelPath = IndexService.getVisionEncoderPath(applicationContext)

        // Capture mutable field to a local val for smart-cast (Kotlin smart-cast requires val).
        val localImageEncoderFactory = imageEncoderFactory

        // Capability and model-file checks run now (cheap). The encoder itself is built by
        // LazyImageEncoder on the first photo that decodes. A build that cannot finish throws
        // PhotoPassAborted with the outcome to return.
        val buildEncoder: () -> ImageEncoder = when {
            localImageEncoderFactory != null -> {
                // Test path: use the injected factory (null skips the branch).
                val buildInjected: () -> ImageEncoder = {
                    localImageEncoderFactory.invoke(visionModelPath) ?: run {
                        Log.d(TAG, "Image encoder factory returned null, skipping image branch")
                        throw PhotoPassAborted(VisionIndexOutcome.COMPLETE)
                    }
                }
                buildInjected
            }
            else -> {
                // Production path: check capability + model presence.
                val flags = try {
                    DeviceTierDetector.detect(applicationContext)
                } catch (e: Exception) {
                    Log.w(TAG, "DeviceTierDetector.detect() failed, skipping image branch", e)
                    repository.appendEvent(runId, "photos", "WARN", null, e.javaClass.simpleName, "DeviceTierDetector failed")
                    return VisionIndexOutcome.COMPLETE
                } catch (e: LinkageError) {
                    Log.w(TAG, "DeviceTierDetector.detect() failed, skipping image branch", e)
                    repository.appendEvent(runId, "photos", "WARN", null, e.javaClass.simpleName, "DeviceTierDetector failed")
                    return VisionIndexOutcome.COMPLETE
                }

                if (!flags.imageEmbeddingEnabled) {
                    Log.d(TAG, "imageEmbeddingEnabled=false (Tier C), skipping image branch")
                    return VisionIndexOutcome.COMPLETE
                }

                when (visionFilesState(visionModelPath)) {
                    VisionFilesState.ABSENT,
                    VisionFilesState.INCOMPLETE -> {
                        val outcome = visionUnavailableOutcome(runAttemptCount)
                        Log.d(
                            TAG,
                            "EG2 vision encoder not ready at $visionModelPath, outcome: $outcome," +
                                " attempt ${runAttemptCount + 1}"
                        )
                        return outcome
                    }
                    VisionFilesState.READY -> Unit
                }

                val buildProduction: () -> ImageEncoder = {
                    // A text-model failure is a run failure (retry), as before the lazy build:
                    // it must not be mistaken for "photos not ready".
                    val sharedModel = try {
                        sharedEmbeddingModel()
                    } catch (e: Exception) {
                        throw TextModelBuildFailed(e)
                    } catch (e: LinkageError) {
                        throw TextModelBuildFailed(e)
                    }
                    if (sharedModel == null) {
                        Log.w(TAG, "No shared EmbeddingModel available, skipping image branch")
                        repository.appendEvent(runId, "photos", "WARN", null, "NoModel", "No shared EmbeddingModel available")
                        throw PhotoPassAborted(VisionIndexOutcome.TEXT_ONLY)
                    }
                    // Memory gate: read after the text model is built, so availMem counts it.
                    // Abort skips the prune below, so photo vectors already indexed stay.
                    val mem = ActivityManager.MemoryInfo()
                    (applicationContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager)
                        .getMemoryInfo(mem)
                    val fits = photoPassFits(mem.availMem, mem.threshold, mem.lowMemory, PHOTO_PASS_HEADROOM_BYTES)
                    Log.i(
                        TAG,
                        "Photo memory gate: availMem=${mem.availMem} threshold=${mem.threshold}" +
                            " lowMemory=${mem.lowMemory} totalMem=${mem.totalMem}" +
                            " decision=${if (fits) "run" else "defer"}"
                    )
                    if (!fits) {
                        repository.appendEvent(runId, "photos", "INFO", null, null, "photos deferred: low memory")
                        throw PhotoPassAborted(VisionIndexOutcome.DEFERRED)
                    }
                    try {
                        EG2BitmapImageEncoder(sharedModel, VisionEncoder(visionModelPath))
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to build EG2BitmapImageEncoder, skipping image branch", e)
                        repository.appendEvent(runId, "photos", "WARN", null, e.javaClass.simpleName, "Failed to build EG2BitmapImageEncoder")
                        throw PhotoPassAborted(VisionIndexOutcome.TEXT_ONLY)
                    } catch (e: LinkageError) {
                        Log.w(TAG, "Failed to build EG2BitmapImageEncoder, skipping image branch", e)
                        repository.appendEvent(runId, "photos", "WARN", null, e.javaClass.simpleName, "Failed to build EG2BitmapImageEncoder")
                        throw PhotoPassAborted(VisionIndexOutcome.TEXT_ONLY)
                    }
                }
                buildProduction
            }
        }

        // Closed in the finally below; closes the inner encoder only if it was ever built.
        val encoder = LazyImageEncoder(buildEncoder)

        try {
            onFinding()
            val enumerator = MediaStoreEnumerator()
            // Prune guards: access and volumes as they are at listing time.
            val accessAtListing = photoAccessLevel()
            repository.recordVolumes(presentVolumes())
            val imageTargets = enumerator.enumerateImages(applicationContext)
            Log.d(TAG, "Image enumeration returned ${imageTargets.size} image(s)")

            if (imageTargets.isEmpty()) {
                Log.d(TAG, "No images found in MediaStore, image branch complete (0 items)")
                return VisionIndexOutcome.COMPLETE
            }

            // Publish Reading(0, n) immediately so the chip shows before the first checkpoint.
            onProgress(0, imageTargets.size)

            val imageDateMap = imageTargets.associate { it.uri to it.lastModified }

            val imageIndexer = ImageIndexer(
                imageEncoder = encoder,
                repository   = repository,
                context      = applicationContext,
                onEvent      = { uri, errorClass, msg ->
                    repository.appendEvent(runId, "photos", "WARN", uri, errorClass, msg)
                }
            )

            val imageUris = imageTargets.map { it.uri }
            val photoPassFinished = imageIndexer.indexImages(
                imageUris  = imageUris,
                dateMapSec = imageDateMap,
                onProgress = { indexed, total ->
                    Log.d(TAG, "Image indexing progress: $indexed / $total")
                    onProgress(indexed, total)
                },
                shouldStop = { isStopped }
            )

            // R13: drop rows for photos deleted from the phone. Needs FULL access: with the
            // user-selected grant the enumeration above is only a subset.
            if (shouldPrunePhotos(accessAtListing, photoAccessLevel(), photoPassFinished && !isStopped)) {
                val removed = repository.prunePhotos(imageUris.toSet(), presentVolumes())
                Log.d(TAG, "Pruned $removed deleted photo(s)")
            }

            Log.d(TAG, "Image branch complete: ${imageTargets.size} image(s) processed")
        } catch (e: PhotoPassAborted) {
            return e.outcome
        } catch (e: TextModelBuildFailed) {
            // Not a photo problem: fail the run (Failed stage + retry), as an eager build did.
            throw e.original
        } catch (e: Exception) {
            Log.e(TAG, "Image branch threw unexpectedly. Documents still indexed.", e)
            repository.appendEvent(runId, "photos", "ERROR", null, e.javaClass.simpleName, "Image branch threw unexpectedly")
            return VisionIndexOutcome.TEXT_ONLY
        } catch (e: LinkageError) {
            Log.e(TAG, "Image branch threw unexpectedly. Documents still indexed.", e)
            repository.appendEvent(runId, "photos", "ERROR", null, e.javaClass.simpleName, "Image branch threw unexpectedly")
            return VisionIndexOutcome.TEXT_ONLY
        } finally {
            try { encoder.close() } catch (e: Exception) {
                Log.w(TAG, "Failed to close image encoder in runImageBranch", e)
                repository.appendEvent(runId, "photos", "WARN", null, e.javaClass.simpleName, "Failed to close image encoder")
            }
        }
        return VisionIndexOutcome.COMPLETE
    }

    /**
     * As a plain job the run was cancelled at 30 minutes (JobScheduler timeout, stress test
     * 2026-10-25), and repeated timeouts get the app throttled. WorkManager ignores the job
     * stop for foreground work. Android refuses a foreground start from the background (an
     * overnight run): the run then goes on as a plain job, as before.
     */
    private suspend fun promoteToForeground() {
        try {
            val notification = IndexNotifications.buildStageNotification(
                applicationContext, IndexStage.Finding("documents")
            )
            setForeground(
                ForegroundInfo(
                    IndexNotifications.NOTIFICATION_ID, notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
                )
            )
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "Could not run in the foreground, running as a plain job", e)
        }
    }

    /**
     * Update the foreground service notification for the given [IndexStage].
     * Delegates to [IndexNotifications.buildStageNotification].
     */
    @SuppressLint("MissingPermission") // Notification denial is handled by the catch below.
    private fun updateStageNotification(stage: IndexStage) {
        try {
            val nm = androidx.core.app.NotificationManagerCompat.from(applicationContext)
            when (stage) {
                // Terminal stages: cancel the ongoing notification instead of posting a new one.
                // WaitingForPermission returns success immediately; cancel so it does not linger.
                is IndexStage.Done, is IndexStage.Failed, IndexStage.Paused,
                is IndexStage.WaitingForPermission ->
                    nm.cancel(IndexNotifications.NOTIFICATION_ID)
                else -> {
                    val notification = IndexNotifications.buildStageNotification(applicationContext, stage)
                    nm.notify(IndexNotifications.NOTIFICATION_ID, notification)
                }
            }
        } catch (e: Exception) {
            // Not wired to appendEvent: notification failure is not an indexing failure,
            // and updateStageNotification can run before the DB is open.
            Log.w(TAG, "Failed to update stage notification", e)
        }
    }

    private fun guessMimeType(file: File): String? {
        return when (file.extension.lowercase()) {
            "pdf"  -> "application/pdf"
            "docx" -> "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            "txt"  -> "text/plain"
            "md"   -> "text/markdown"
            else   -> null
        }
    }
}

/**
 * E19: only a Done or WaitingForPermission success commits the first-index flag. The Paused
 * return is also a success but carries no stage, so a system stop mid-read leaves the flag off.
 */
// Result.Success is flagged RestrictedApi by lint, but it is the only way to read a success.
@SuppressLint("RestrictedApi")
internal fun shouldMarkFirstIndexDone(result: androidx.work.ListenableWorker.Result): Boolean {
    if (result !is androidx.work.ListenableWorker.Result.Success) return false
    val stage = IndexStage.fromData(result.outputData)
    return stage is IndexStage.Done || stage is IndexStage.WaitingForPermission
}
