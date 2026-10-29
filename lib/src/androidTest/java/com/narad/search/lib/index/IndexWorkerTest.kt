package com.narad.search.lib.index

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.work.ExistingWorkPolicy
import androidx.work.ListenableWorker
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.testing.TestListenableWorkerBuilder
import androidx.work.testing.WorkManagerTestInitHelper
import androidx.work.workDataOf
import com.narad.search.lib.index.worker.ImageEncoder
import com.narad.search.lib.index.service.BootReceiver
import com.narad.search.lib.index.service.IndexService
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository
import com.narad.search.lib.index.worker.IndexStage
import com.narad.search.lib.index.worker.IndexWorker
import com.narad.search.lib.query.store.KeywordStore
import com.narad.search.lib.query.store.VectorStore
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Collections
import java.util.UUID
import java.util.concurrent.TimeUnit

/**
 * Test 4: Model-absent → Result.retry()
 * Test 10: End-to-end IndexWorker production path with FakeEmbedder injection (fix item 10).
 * U25 test A: onWorkFinished enqueues exactly one periodic request (NEEDS-DEVICE).
 * U25 test B: foreground-delta enqueue twice keeps exactly one request (NEEDS-DEVICE).
 *
 * Uses a custom WorkerFactory (IndexWorker.Factory) with TestListenableWorkerBuilder so
 * IndexWorker can be tested without the 94 MB ONNX model on the emulator.
 */
@RunWith(AndroidJUnit4::class)
class IndexWorkerTest {

    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    // -------------------------------------------------------------------------
    // Test 4: Model-absent → Result.retry()
    //
    // Uses a null embedderFactory (production path) so the model-file guard fires.
    // /nonexistent/path/model.onnx does not exist → doWork() returns retry().
    // -------------------------------------------------------------------------
    @Test
    fun test4_modelAbsent_returnsRetry() {
        // Build using a factory that passes null embedderFactory so the model-file
        // guard is active (fix: was using plain TestListenableWorkerBuilder which
        // needed a 2-param constructor - now uses Factory for clean injection).
        val worker = TestListenableWorkerBuilder<IndexWorker>(context)
            .setWorkerFactory(object : androidx.work.WorkerFactory() {
                override fun createWorker(
                    appContext: Context,
                    workerClassName: String,
                    workerParameters: WorkerParameters
                ): androidx.work.ListenableWorker? {
                    return if (workerClassName == IndexWorker::class.java.name) {
                        // embedderFactory = null, waitMaxMs short so test does not block
                        IndexWorker(appContext, workerParameters, null).also {
                            it.modelWaitMaxMs = 50L
                            it.modelPollMs    = 5L
                        }
                    } else null
                }
            })
            .setInputData(
                workDataOf(
                    IndexWorker.KEY_MODEL_PATH to "/nonexistent/path/model.onnx"
                )
            )
            .build()

        val result = runBlocking { worker.doWork() }
        assertEquals(
            "IndexWorker should return Result.retry() when model file is absent",
            ListenableWorker.Result.retry(),
            result
        )
    }

    // -------------------------------------------------------------------------
    // U25 regression: "tap Start during model download"
    //
    // Drives IndexWorker with model absent initially, then makes model appear
    // mid-wait and asserts the same run proceeds to Reading/Done without a retry.
    //
    // NEEDS-DEVICE: requires TestListenableWorkerBuilder on an Android runtime.
    //   ./gradlew :lib:connectedDebugAndroidTest \
    //     -Pandroid.testInstrumentationRunnerArguments.class=com.narad.search.lib.index.IndexWorkerTest#u25_regression_tapStartDuringDownload
    // -------------------------------------------------------------------------
    /**
     * U25 regression: "tap Start during model download".
     *
     * Drives IndexWorker with model files absent. Places the files mid-run (on a
     * background thread, after doWork enters waitForModel). Asserts the SAME run
     * succeeds without WorkManager issuing a retry.
     *
     * NEEDS-DEVICE: requires instrumented runtime.
     *   ./gradlew :lib:connectedDebugAndroidTest \
     *     -Pandroid.testInstrumentationRunnerArguments.class=com.narad.search.lib.index.IndexWorkerTest#u25_regression_tapStartDuringDownload
     */
    @Test
    fun u25_regression_tapStartDuringDownload() {
        val modelDir  = context.filesDir.resolve("test-model-u25-${UUID.randomUUID()}")
        modelDir.mkdirs()
        val modelPath = modelDir.resolve("model_quantized.onnx").absolutePath

        // A real temp file for Branch A (KEY_FILE_PATHS) so the permission gate is skipped.
        val inputFile = File(modelDir, "input.txt").also { it.writeText("hello world") }

        val capturedStages = Collections.synchronizedList(mutableListOf<IndexStage?>())

        val worker = TestListenableWorkerBuilder<IndexWorker>(context)
            .setWorkerFactory(object : androidx.work.WorkerFactory() {
                override fun createWorker(
                    appContext: Context,
                    workerClassName: String,
                    workerParameters: WorkerParameters
                ): androidx.work.ListenableWorker? {
                    return if (workerClassName == IndexWorker::class.java.name) {
                        // FakeEmbedder avoids real ONNX; checkModelFiles=true forces the
                        // WaitingForModel guard so the regression path is actually exercised.
                        IndexWorker(appContext, workerParameters) { _ -> FakeEmbedder() }.also {
                            it.checkModelFiles = true
                            it.modelWaitMaxMs  = 5_000L
                            it.modelPollMs     = 50L
                        }
                    } else null
                }
            })
            .setProgressUpdater { _, _, data ->
                capturedStages += IndexStage.fromData(data)
                // ponytail: guava utility classes (Futures/SettableFuture) not on compile CP via
                // work-testing; implement the already-done future inline from the interface only.
                object : com.google.common.util.concurrent.ListenableFuture<Void?> {
                    override fun addListener(r: Runnable, e: java.util.concurrent.Executor) = e.execute(r)
                    override fun cancel(b: Boolean) = false
                    override fun isCancelled() = false
                    override fun isDone() = true
                    override fun get(): Void? = null
                    override fun get(t: Long, u: java.util.concurrent.TimeUnit): Void? = null
                }
            }
            .setInputData(workDataOf(
                IndexWorker.KEY_MODEL_PATH to modelPath,
                // Branch A: explicit file list skips the permission gate entirely.
                IndexWorker.KEY_FILE_PATHS to inputFile.absolutePath
            ))
            .build()

        // Place model files 200 ms into the run so doWork enters waitForModel first.
        val placer = Thread {
            Thread.sleep(200)
            File(modelPath).writeText("fake-model")
            File("${modelPath}_data").writeText("fake-data")
            File(modelDir, "tokenizer.json").writeText("{}")
        }
        placer.start()

        val result = runBlocking { worker.doWork() }
        placer.join()
        modelDir.deleteRecursively()

        // Must succeed in the SAME invocation (no retry).
        assertTrue(
            "Same run must succeed after model appears mid-wait, not retry. result=$result",
            result is ListenableWorker.Result.Success
        )
        // Must have recorded WaitingForModel before Reading/Done.
        val stageTypes = capturedStages.filterNotNull().map { it::class.simpleName }
        assertTrue(
            "Expected WaitingForModel stage before indexing. Got: $stageTypes",
            stageTypes.contains("WaitingForModel")
        )
        val waitIdx = stageTypes.indexOfFirst { it == "WaitingForModel" }
        val readOrDone = stageTypes.drop(waitIdx).any { it == "Reading" || it == "Done" }
        assertTrue(
            "Expected Reading or Done stage after WaitingForModel. Got: $stageTypes",
            readOrDone
        )
    }

    // -------------------------------------------------------------------------
    // Test 10: End-to-end IndexWorker with FakeEmbedder injection (fix item 10)
    //
    // This is the test that would have caught the "worker indexes zero files" defect.
    //
    // Drives IndexWorker.doWork() end-to-end:
    //   - IndexWorker.Factory injects FakeEmbedder (no ONNX model needed)
    //   - KEY_FILE_PATHS (newline-encoded, fix item 8) pointing at 2 real temp files
    // Asserts:
    //   - Result.success()
    //   - IndexRepository.chunkCount() > 0 (chunks were persisted)
    //   - rehydrate into fresh stores finds the chunks
    // -------------------------------------------------------------------------
    @Test
    fun test10_endToEnd_withFakeEmbedder_persistsChunksAndRehydratesSuccessfully() {
        val filesDir = context.filesDir

        // Create 2 real temp text files
        val file1 = File(filesDir, "test_e2e_doc1_${UUID.randomUUID()}.txt").also {
            it.writeText("This is a contract document with a penalty clause for late payment.")
        }
        val file2 = File(filesDir, "test_e2e_doc2_${UUID.randomUUID()}.txt").also {
            it.writeText("Weather report: sunny and warm. No rain expected in the forecast.")
        }

        // Encode file paths with newline separator (fix item 8: no comma corruption)
        val filePathsEncoded = listOf(file1.absolutePath, file2.absolutePath).joinToString("\n")

        // Build the worker with FakeEmbedder injected via Factory.
        // embedderFactory != null → model-file existence check is skipped, so
        // fakeModelPath does not need to exist on disk.
        val worker = TestListenableWorkerBuilder<IndexWorker>(context)
            .setWorkerFactory(IndexWorker.Factory { _ -> FakeEmbedder() })
            .setInputData(
                workDataOf(
                    IndexWorker.KEY_MODEL_PATH to "/fake/model.onnx",
                    IndexWorker.KEY_FILE_PATHS to filePathsEncoded
                )
            )
            .build()

        IndexService.setFirstIndexDone(context, false)
        val result = runBlocking { worker.doWork() }

        assertTrue(
            "IndexWorker should return Result.success() with FakeEmbedder and real files",
            result is ListenableWorker.Result.Success
        )
        assertTrue(
            "a successful run must set first_index_done itself (W29)",
            IndexService.isFirstIndexDone(context)
        )

        // Verify chunks were persisted to the production DB
        // (IndexWorker uses applicationContext → the app's narad_index.db)
        val db = IndexDatabase(context)
        val repository = IndexRepository(db)
        try {
            val count = repository.chunkCount()
            assertTrue(
                "IndexRepository should have at least 2 chunks (one per file) after end-to-end run",
                count > 0
            )

            // Rehydrate into fresh stores and verify search works
            val vs = VectorStore()
            val ks = KeywordStore()
            repository.rehydrate(vs, ks)

            assertTrue("Rehydrated VectorStore should have entries", vs.size > 0)

            val results = ks.search("penalty clause", topK = 5)
            assertFalse("Should find keyword 'penalty clause' after end-to-end indexing", results.isEmpty())
        } finally {
            db.close()
            // Clean up: clear the production DB after this test so we don't pollute other tests
            val cleanupDb = IndexDatabase(context)
            try {
                IndexRepository(cleanupDb).clearAll()
            } finally {
                cleanupDb.close()
            }
            file1.delete()
            file2.delete()
        }
    }

    // -------------------------------------------------------------------------
    // M18 test: one corrupt PDF + one txt file gives exactly 1 event.
    //
    // NEEDS-DEVICE: run on a connected device or emulator with:
    //   ./gradlew :lib:connectedDebugAndroidTest \
    //     -Pandroid.testInstrumentationRunnerArguments.class=com.narad.search.lib.index.IndexWorkerTest#m18_oneCorruptPdf_givesExactlyOneEvent \
    //     --no-daemon
    // -------------------------------------------------------------------------

    /**
     * M18 gate (instrumented): a temp folder with one corrupt PDF and one valid txt file.
     * After doWork():
     *  - Result is success (corrupt file does not fail the run).
     *  - Exactly 1 event is recorded with fileName == "corrupt.pdf".
     *
     * NEEDS-DEVICE: requires a real Android context for IndexDatabase.
     */
    @Test
    fun m18_oneCorruptPdf_givesExactlyOneEvent() {
        val filesDir = context.filesDir

        val corruptPdf = File(filesDir, "corrupt.pdf").also { f ->
            // Corrupt PDF: valid header marker then garbage bytes.
            f.writeBytes("%PDF-1.4\n".toByteArray() + ByteArray(64) { (it * 37).toByte() })
        }
        val validTxt = File(filesDir, "valid_m18_${UUID.randomUUID()}.txt").also { f ->
            f.writeText("This is a valid text document for indexing.")
        }

        val filePathsEncoded = listOf(corruptPdf.absolutePath, validTxt.absolutePath).joinToString("\n")

        val worker = TestListenableWorkerBuilder<IndexWorker>(context)
            .setWorkerFactory(IndexWorker.Factory { _ -> FakeEmbedder() })
            .setInputData(
                workDataOf(
                    IndexWorker.KEY_MODEL_PATH to "/fake/model.onnx",
                    IndexWorker.KEY_FILE_PATHS to filePathsEncoded
                )
            )
            .build()

        val runStart = System.currentTimeMillis()
        val result = runBlocking { worker.doWork() }

        assertTrue(
            "Corrupt PDF must not fail the run",
            result is ListenableWorker.Result.Success
        )

        val db = IndexDatabase(context)
        val repository = IndexRepository(db)
        try {
            val events = repository.listEvents().filter { it.timestampMs >= runStart }
            assertEquals("Exactly 1 event for 1 corrupt PDF", 1, events.size)
            assertEquals(
                "Event fileName must be the bare name of the corrupt file",
                "corrupt.pdf",
                events[0].fileName
            )
        } finally {
            db.close()
            val cleanupDb = IndexDatabase(context)
            try { IndexRepository(cleanupDb).clearAll() } finally { cleanupDb.close() }
            corruptPdf.delete()
            validTxt.delete()
        }
    }

    // -------------------------------------------------------------------------
    // U25 tests: WorkManager state after work finishes (NEEDS-DEVICE).
    //
    // NEEDS-DEVICE: run on a connected device or emulator with:
    //   ./gradlew :lib:connectedDebugAndroidTest \
    //     -Pandroid.testInstrumentationRunnerArguments.class=com.narad.search.lib.index.IndexWorkerTest
    // -------------------------------------------------------------------------

    private val fakeModel by lazy { FakeModelFiles(context) }

    @Before
    fun setUp() {
        WorkManagerTestInitHelper.initializeTestWorkManager(context)
        // The delta enqueue is guarded on the model being on disk (E6).
        fakeModel.create()
        // ...and on the first full index having finished (W29).
        IndexService.setFirstIndexDone(context, true)
    }

    @After
    fun tearDown() {
        fakeModel.delete()
        IndexService.setFirstIndexDone(context, false)
    }

    /**
     * U25-A: [BootReceiver.enqueuePeriodicDeltaWork] called twice (as [IndexService.onWorkFinished]
     * would call it, plus the initial call from [IndexService.enqueueUserInitiatedIndex])
     * leaves exactly ONE ENQUEUED periodic request under [BootReceiver.DELTA_WORK_NAME].
     *
     * Proves KEEP policy is idempotent: double-enqueue does not duplicate the request.
     *
     * NEEDS-DEVICE: requires WorkManagerTestInitHelper and an Android context.
     */
    @Test
    fun u25_onWorkFinished_enqueuesExactlyOnePeriodicRequest() {
        val wm = WorkManager.getInstance(context)
        // Cancel any leftover periodic work from a previous test.
        wm.cancelUniqueWork(BootReceiver.DELTA_WORK_NAME).result.get()

        // Simulate onWorkFinished calling enqueuePeriodicDeltaWork twice.
        BootReceiver.enqueuePeriodicDeltaWork(context)
        BootReceiver.enqueuePeriodicDeltaWork(context)

        val infos = wm.getWorkInfosForUniqueWork(BootReceiver.DELTA_WORK_NAME).get()
        assertEquals(
            "KEEP policy must produce exactly one periodic request even when called twice. Got: " +
            infos.map { "${it.id}:${it.state}" },
            1,
            infos.size
        )
        assertEquals(
            "The single periodic request must be ENQUEUED",
            WorkInfo.State.ENQUEUED,
            infos.single().state
        )
    }

    /**
     * U25-B: [BootReceiver.enqueueForegroundDeltaWork] called twice (simulating two consecutive
     * [Activity.onActivityStarted] events) leaves exactly ONE ENQUEUED one-time request under
     * [BootReceiver.FOREGROUND_DELTA_WORK_NAME].
     *
     * Proves KEEP policy: second enqueue while one is already queued is a no-op.
     *
     * NEEDS-DEVICE: requires WorkManagerTestInitHelper and an Android context.
     */
    @Test
    fun u25_foregroundDelta_enqueueTwice_keepsExactlyOneRequest() {
        val wm = WorkManager.getInstance(context)

        // Guard against a vacuous pass: with the model present the first call must really
        // enqueue a job. If the delta guard silently skipped, the KEEP checks below would
        // still pass because they only look at the request enqueued directly.
        wm.cancelUniqueWork(BootReceiver.FOREGROUND_DELTA_WORK_NAME).result.get()
        wm.pruneWork().result.get()
        BootReceiver.enqueueForegroundDeltaWork(context)
        assertEquals(
            "enqueueForegroundDeltaWork must enqueue a job when the model is present",
            1,
            wm.getWorkInfosForUniqueWork(BootReceiver.FOREGROUND_DELTA_WORK_NAME).get().size
        )
        wm.cancelUniqueWork(BootReceiver.FOREGROUND_DELTA_WORK_NAME).result.get()
        wm.pruneWork().result.get()

        // The test WorkManager runs eligible work at once, so enqueue the first request with a
        // delay to keep it ENQUEUED; otherwise KEEP correctly accepts a new id after it finishes.
        val first = OneTimeWorkRequestBuilder<IndexWorker>()
            .setInitialDelay(1, TimeUnit.HOURS)
            .build()
        wm.enqueueUniqueWork(
            BootReceiver.FOREGROUND_DELTA_WORK_NAME,
            ExistingWorkPolicy.KEEP,
            first
        )
        val idAfterFirst = first.id

        BootReceiver.enqueueForegroundDeltaWork(context)
        val infos = wm.getWorkInfosForUniqueWork(BootReceiver.FOREGROUND_DELTA_WORK_NAME).get()

        assertEquals(
            "KEEP policy must leave exactly one foreground-delta request even when enqueued twice. Got: " +
            infos.map { "${it.id}:${it.state}" },
            1,
            infos.size
        )
        // KEEP policy preserves the original request (same id), not a replacement.
        assertEquals("The original request must still be ENQUEUED", WorkInfo.State.ENQUEUED, infos.single().state)
        assertEquals(
            "KEEP policy must preserve the original request id, not replace it. Got: ${infos.single().id}",
            idAfterFirst,
            infos.single().id
        )
    }

    // -------------------------------------------------------------------------
    // U7: lazy model. A delta builds the text embedder and the vision encoder only when an
    // item needs them.
    //
    // NEEDS-DEVICE: run on a connected device or emulator with:
    //   ./gradlew :lib:connectedDebugAndroidTest \
    //     -Pandroid.testInstrumentationRunnerArguments.class=com.narad.search.lib.index.IndexWorkerTest
    // (connectedAndroidTest uninstalls the app; restore the model afterwards, see AGENTS.md.)
    // -------------------------------------------------------------------------

    private var embedderBuilds = 0
    private var imageEncoderBuilds = 0
    private var seededPhoto: Uri? = null

    private fun grantStorage() {
        val auto = InstrumentationRegistry.getInstrumentation().uiAutomation
        val pkg = context.packageName
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            auto.grantRuntimePermission(pkg, Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            auto.grantRuntimePermission(pkg, Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        android.os.ParcelFileDescriptor.AutoCloseInputStream(
            auto.executeShellCommand("appops set $pkg MANAGE_EXTERNAL_STORAGE allow")
        ).use { it.readBytes() }
    }

    private fun countingWorker(): IndexWorker = TestListenableWorkerBuilder<IndexWorker>(context)
        .setWorkerFactory(
            IndexWorker.Factory.withBothEncoders(
                embedderFactory = { _ -> embedderBuilds++; FakeEmbedder() },
                imageEncoderFactory = { _ ->
                    imageEncoderBuilds++
                    object : ImageEncoder {
                        override fun encode(bitmap: Bitmap) = FloatArray(256).also { it[0] = 1f }
                        override fun close() {}
                    }
                }
            )
        )
        .setInputData(workDataOf(IndexWorker.KEY_MODEL_PATH to "/fake/model.onnx"))
        .build()

    private fun clearIndex() {
        val db = IndexDatabase(context)
        try { IndexRepository(db).clearAll() } finally { db.close() }
    }

    private fun imageVectorCount(): Int {
        val db = IndexDatabase(context)
        return try { IndexRepository(db).imageVectorCount() } finally { db.close() }
    }

    private fun seedPhoto(corrupt: Boolean = false): Uri {
        val bmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888).also { it.eraseColor(Color.rgb(10, 90, 200)) }
        val jpeg = if (corrupt) {
            ByteArray(256) { 7 }  // not an image: the decode fails
        } else {
            ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.JPEG, 90, it) }.toByteArray()
        }
        bmp.recycle()
        val resolver = context.contentResolver
        val cv = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "narad_u7_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/")
        }
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv)
            ?: error("could not insert test photo")
        resolver.openOutputStream(uri)!!.use { it.write(jpeg) }
        resolver.update(uri, ContentValues().apply {
            put(MediaStore.MediaColumns.DATE_MODIFIED, System.currentTimeMillis() / 1000L)
        }, null, null)
        seededPhoto = uri
        return uri
    }

    @After
    fun u7_cleanUp() {
        seededPhoto?.let { runCatching { context.contentResolver.delete(it, null, null) } }
        runCatching { clearIndex() }
    }

    /** A delta with no changed documents or photos builds neither model and still ends complete. */
    @Test
    fun u7_unchangedDelta_buildsNeitherModel_andEndsComplete() {
        grantStorage()
        clearIndex()
        // First run indexes whatever the device holds.
        runBlocking { countingWorker().doWork() }

        embedderBuilds = 0
        imageEncoderBuilds = 0
        val result = runBlocking { countingWorker().doWork() }

        assertEquals("unchanged delta must not build the text embedder", 0, embedderBuilds)
        assertEquals("unchanged delta must not build the image encoder", 0, imageEncoderBuilds)
        val success = result as ListenableWorker.Result.Success
        assertTrue(
            "run must still end Done, got ${IndexStage.fromData(success.outputData)}",
            IndexStage.fromData(success.outputData) is IndexStage.Done
        )
    }

    /** A new photo and no changed documents builds the image encoder once and indexes the photo. */
    @Test
    fun u7_onePhotoNoDocChange_buildsEncoderOnce_andIndexesPhoto() {
        grantStorage()
        clearIndex()
        runBlocking { countingWorker().doWork() }
        val before = imageVectorCount()

        seedPhoto()
        embedderBuilds = 0
        imageEncoderBuilds = 0
        runBlocking { countingWorker().doWork() }

        assertEquals("no document changed, so no text embedder", 0, embedderBuilds)
        assertEquals("the encoder is built once for the new photo", 1, imageEncoderBuilds)
        assertEquals("the new photo is indexed", before + 1, imageVectorCount())
    }

    /** An unreadable pending photo never reaches encode, so it must not build the encoder. */
    @Test
    fun u7_corruptPhoto_buildsNothing_everyDelta() {
        grantStorage()
        clearIndex()
        runBlocking { countingWorker().doWork() }

        seedPhoto(corrupt = true)
        repeat(2) {
            embedderBuilds = 0
            imageEncoderBuilds = 0
            runBlocking { countingWorker().doWork() }
            assertEquals("run ${it + 1}: no text embedder", 0, embedderBuilds)
            assertEquals("run ${it + 1}: the corrupt photo must not build the encoder", 0, imageEncoderBuilds)
        }
    }

    /** A text-model build failure is a failed run (retry), not a success. */
    @Test
    fun u7_embedderBuildFailure_failsTheRun() {
        val file = File(context.filesDir, "u7_fail_${UUID.randomUUID()}.txt").also { it.writeText("some text to embed") }
        try {
            val worker = TestListenableWorkerBuilder<IndexWorker>(context)
                .setWorkerFactory(IndexWorker.Factory { _ -> error("model build failed") })
                .setInputData(
                    workDataOf(
                        IndexWorker.KEY_MODEL_PATH to "/fake/model.onnx",
                        IndexWorker.KEY_FILE_PATHS to file.absolutePath,
                    )
                )
                .build()

            val result = runBlocking { worker.doWork() }

            assertTrue("expected retry, got $result", result is ListenableWorker.Result.Retry)
        } finally {
            file.delete()
        }
    }
}
