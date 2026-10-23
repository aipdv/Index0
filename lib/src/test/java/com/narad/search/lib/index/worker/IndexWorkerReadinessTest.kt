package com.narad.search.lib.index.worker

import android.content.Context
import androidx.work.WorkerParameters
import com.narad.search.lib.index.provision.EmbeddingGemma2Manifest
import com.narad.search.lib.index.provision.EmbeddingGemma2VisionManifest
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile
import java.nio.file.Files

class IndexWorkerReadinessTest {

    /**
     * WorkManager's default factory reflects on (Context, WorkerParameters). The app registers
     * no custom WorkerFactory, so without this constructor every production index job fails
     * with NoSuchMethodException.
     */
    @Test
    fun `IndexWorker has the two-arg constructor WorkManager default factory needs`() {
        IndexWorker::class.java.getConstructor(Context::class.java, WorkerParameters::class.java)
    }

    @Test
    fun `unavailable vision retries three times`() {
        assertEquals(VisionIndexOutcome.RETRY, visionUnavailableOutcome(0))
        assertEquals(VisionIndexOutcome.RETRY, visionUnavailableOutcome(1))
        assertEquals(VisionIndexOutcome.RETRY, visionUnavailableOutcome(2))
    }

    @Test
    fun `unavailable vision falls back to text after retry cap`() {
        assertEquals(VisionIndexOutcome.TEXT_ONLY, visionUnavailableOutcome(3))
        assertEquals("Photos not ready", PHOTOS_NOT_READY_STATUS)
    }

    // ---- review-021: terminal output is computed by one pure function -------------------

    @Test
    fun `complete run reports Done with the summed count`() {
        val t = terminalOutput(VisionIndexOutcome.COMPLETE, totalIndexed = 7, photosIndexed = 5)!!
        assertEquals(IndexStage.Done(count = 12, reason = null), t.stage)
        assertFalse(t.photosNotReady)
    }

    @Test
    fun `complete run with nothing read explains why`() {
        val t = terminalOutput(VisionIndexOutcome.COMPLETE, 0, 0)!!
        assertEquals(IndexStage.Done(count = 0, reason = "Nothing to read yet"), t.stage)
    }

    @Test
    fun `text-only run is Done with photos-not-ready flag and no empty reason`() {
        val t = terminalOutput(VisionIndexOutcome.TEXT_ONLY, 3, 0)!!
        assertEquals(IndexStage.Done(count = 3, reason = null), t.stage)
        assertTrue(t.photosNotReady)
    }

    @Test
    fun `retry outcome publishes no terminal stage`() {
        assertNull(terminalOutput(VisionIndexOutcome.RETRY, 10, 10))
    }

    @Test
    fun `vision files are incomplete until graph and weights have expected sizes`() {
        val dir = Files.createTempDirectory("vision-readiness").toFile()
        try {
            val files = EmbeddingGemma2VisionManifest.manifest.files.associateBy {
                File(it.relativeDest).name
            }
            val graph = File(dir, "vision_encoder_quantized.onnx")
            val weights = File(dir, "vision_encoder_quantized.onnx_data")

            assertEquals(VisionFilesState.ABSENT, visionFilesState(graph.absolutePath))

            graph.writeBytes(ByteArray(files.getValue(graph.name).sizeBytes.toInt()))
            assertEquals(VisionFilesState.INCOMPLETE, visionFilesState(graph.absolutePath))

            weights.writeBytes(ByteArray(1))
            assertEquals(VisionFilesState.INCOMPLETE, visionFilesState(graph.absolutePath))

            weights.delete()
            RandomAccessFile(weights, "rw").use { output ->
                output.setLength(files.getValue(weights.name).sizeBytes)
            }
            assertEquals(VisionFilesState.READY, visionFilesState(graph.absolutePath))
        } finally {
            dir.deleteRecursively()
        }
    }

    // -------------------------------------------------------------------------
    // U25: model-missing guard
    // -------------------------------------------------------------------------

    @Test
    fun `modelFilesReady is false when model file absent`() {
        assertFalse(
            "modelFilesReady must return false for a path that does not exist",
            modelFilesReady("/tmp/definitely-absent-${System.nanoTime()}.onnx")
        )
    }

    @Test
    fun `modelDownloadPercent is 0 when no files on disk`() {
        val percent = modelDownloadPercent("/tmp/absent-dir-${System.nanoTime()}/model.onnx")
        assertEquals("percent must be 0 when no model directory exists", 0, percent)
    }

    @Test
    fun `modelDownloadPercent is 100 when all files are at expected sizes`() {
        val dir = Files.createTempDirectory("model-percent-100").toFile()
        try {
            val manifest = EmbeddingGemma2Manifest.manifest
            val onnxFile = File(dir, "model_quantized.onnx")
            for (mf in manifest.files) {
                RandomAccessFile(File(dir, File(mf.relativeDest).name), "rw").use { raf ->
                    raf.setLength(mf.sizeBytes)
                }
            }
            assertEquals(100, modelDownloadPercent(onnxFile.absolutePath))
        } finally {
            dir.deleteRecursively()
        }
    }

    // -------------------------------------------------------------------------
    // U25: waitForModel logic - JVM-only (extracted pure function)
    //
    // Full regression "tap Start during download" is NEEDS-DEVICE:
    //   ./gradlew :lib:connectedDebugAndroidTest \
    //     -Pandroid.testInstrumentationRunnerArguments.class=com.narad.search.lib.index.IndexWorkerTest#u25_regression_tapStartDuringDownload
    // -------------------------------------------------------------------------

    @Test
    fun `waitForModel returns true immediately when model is already ready`() = runTest {
        val result = waitForModel(
            isReady  = { true },
            getPercent = { 100 },
            onStage  = {},
            isStopped = { false },
            maxMs    = 5_000,
            pollMs   = 10
        )
        assertTrue("waitForModel must return true when model is immediately ready", result)
    }

    @Test
    fun `waitForModel returns false immediately when isStopped is true`() = runTest {
        val result = waitForModel(
            isReady   = { false },
            getPercent = { 0 },
            onStage   = {},
            isStopped = { true },
            maxMs     = 5_000,
            pollMs    = 10
        )
        assertFalse("waitForModel must return false when isStopped on first check", result)
    }

    @Test
    fun `waitForModel returns false after maxMs elapsed without model`() = runTest {
        val result = waitForModel(
            isReady   = { false },
            getPercent = { 0 },
            onStage   = {},
            isStopped = { false },
            maxMs     = 50,
            pollMs    = 1_000  // longer than maxMs so timeout fires first
        )
        assertFalse("waitForModel must return false after timeout", result)
    }

    @Test
    fun `waitForModel returns false when isStopped becomes true mid-wait`() = runTest {
        var stopped = false
        val deferred = async {
            waitForModel(
                isReady    = { false },
                getPercent = { 0 },
                onStage    = {},
                isStopped  = { stopped },
                maxMs      = 10_000,
                pollMs     = 5  // short poll
            )
        }
        delay(20)
        stopped = true
        assertFalse("waitForModel must return false when stopped mid-wait", deferred.await())
    }

    @Test
    fun `waitForModel returns true once model becomes ready during poll`() = runTest {
        var ready = false
        val deferred = async {
            waitForModel(
                isReady   = { ready },
                getPercent = { 50 },
                onStage   = {},
                isStopped = { false },
                maxMs     = 10_000,
                pollMs    = 5  // short poll so the test runs in virtual time
            )
        }
        delay(20)
        ready = true
        assertTrue("waitForModel must return true once model appears", deferred.await())
    }

    @Test
    fun `waitForModel publishes WaitingForModel stage each poll`() = runTest {
        val stages = mutableListOf<IndexStage>()
        waitForModel(
            isReady   = { false },
            getPercent = { 42 },
            onStage   = { stages.add(it) },
            isStopped = { false },
            maxMs     = 30,
            pollMs    = 1_000  // times out before first delay
        )
        // At least one WaitingForModel stage should have been emitted before timeout
        assertTrue(
            "waitForModel must emit at least one WaitingForModel stage",
            stages.any { it is IndexStage.WaitingForModel }
        )
    }

    // -------------------------------------------------------------------------
    // U25: IndexStage sealed class - Data round-trip
    // -------------------------------------------------------------------------

    @Test
    fun `IndexStage WaitingForModel round-trips with percent`() {
        val original = IndexStage.WaitingForModel(percent = 73)
        val recovered = IndexStage.fromData(original.toData())
        assertEquals(original, recovered)
    }

    @Test
    fun `IndexStage WaitingForModel round-trips with null percent`() {
        val original = IndexStage.WaitingForModel(percent = null)
        val recovered = IndexStage.fromData(original.toData())
        assertEquals(original, recovered)
    }

    @Test
    fun `IndexStage WaitingForPermission round-trips`() {
        val original = IndexStage.WaitingForPermission("READ_MEDIA_DOCUMENTS")
        val recovered = IndexStage.fromData(original.toData())
        assertEquals(original, recovered)
    }

    @Test
    fun `IndexStage Finding round-trips`() {
        val original = IndexStage.Finding("documents")
        val recovered = IndexStage.fromData(original.toData())
        assertEquals(original, recovered)
    }

    @Test
    fun `IndexStage Reading round-trips`() {
        val original = IndexStage.Reading(kind = "documents", done = 5, total = 20)
        val recovered = IndexStage.fromData(original.toData())
        assertEquals(original, recovered)
    }

    @Test
    fun `IndexStage Done round-trips with count`() {
        val original = IndexStage.Done(count = 42)
        val recovered = IndexStage.fromData(original.toData())
        assertEquals(original, recovered)
    }

    @Test
    fun `IndexStage Done round-trips with zero count and reason`() {
        val original = IndexStage.Done(count = 0, reason = "Nothing to read yet")
        val recovered = IndexStage.fromData(original.toData())
        assertEquals(original, recovered)
    }

    @Test
    fun `IndexStage Failed round-trips`() {
        val original = IndexStage.Failed(reason = "disk full")
        val recovered = IndexStage.fromData(original.toData())
        assertEquals(original, recovered)
    }

    @Test
    fun `IndexStage Paused round-trips`() {
        val original = IndexStage.Paused
        val recovered = IndexStage.fromData(original.toData())
        assertEquals(original, recovered)
    }

    @Test
    fun `IndexStage fromData returns null for unknown type`() {
        val data = androidx.work.workDataOf(IndexStage.KEY_STAGE to "UnknownType")
        assertNull(IndexStage.fromData(data))
    }

    @Test
    fun `IndexStage fromData returns null when KEY_STAGE absent`() {
        val data = androidx.work.workDataOf("unrelated" to "value")
        assertNull(IndexStage.fromData(data))
    }

    @Test
    fun `IndexStage Reading toData includes backward-compat progress keys`() {
        val stage = IndexStage.Reading(kind = "documents", done = 3, total = 10)
        val data = stage.toData()
        // SearchScreenContent reads these keys until U20 migration
        assertEquals(3, data.getInt("processed", -1))
        assertEquals(10, data.getInt("total", -1))
    }
}
