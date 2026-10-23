package com.narad.search.lib.index.provision

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import okio.Buffer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest

/**
 * Instrumented tests for [ModelProvisioner] using [MockWebServer].
 *
 * Uses SMALL synthetic payloads (a few KB each) - NOT the 94 MB real model.
 *
 * Test coverage:
 *   (a) Both files land at the correct FLAT path; bytes equal what the server served.
 *   (b) StateFlow emits Downloading... then Ready.
 *   (c) Wrong-sha response → Failed AND no final file left on disk.
 *   (d) Idempotency: second provision() returns Ready without new server requests.
 */
@RunWith(AndroidJUnit4::class)
class ModelProvisionerTest {

    private lateinit var mockServer: MockWebServer
    private lateinit var testDestRoot: File

    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    // Synthetic file payloads - small enough to keep the test fast.
    private val fakeOnnxBytes: ByteArray = ByteArray(4096) { it.toByte() }
    private val fakeTokenizerBytes: ByteArray = ByteArray(2048) { (it * 3).toByte() }

    private val fakeOnnxSha256: String by lazy { sha256Hex(fakeOnnxBytes) }
    private val fakeTokenizerSha256: String by lazy { sha256Hex(fakeTokenizerBytes) }

    // Relative destinations (mirrors EG2 production layout, under a test-isolated dir).
    private val onnxRelDest = "models/embeddinggemma-2/model_quantized.onnx"
    private val tokenizerRelDest = "models/embeddinggemma-2/tokenizer.json"

    @Before
    fun setUp() {
        mockServer = MockWebServer()
        mockServer.start()

        // Isolated dest root - will be cleaned in @After.
        testDestRoot = File(context.filesDir, "provisioner_test_${System.currentTimeMillis()}")
        testDestRoot.mkdirs()
    }

    @After
    fun tearDown() {
        try { mockServer.shutdown() } catch (_: Exception) {}
        testDestRoot.deleteRecursively()
    }

    // -------------------------------------------------------------------------
    // Helper: build a test manifest pointing at the MockWebServer
    // -------------------------------------------------------------------------

    private fun buildTestManifest(
        onnxSha: String = fakeOnnxSha256,
        tokenizerSha: String = fakeTokenizerSha256
    ): ModelManifest = ModelManifest(
        modelId = "embeddinggemma-2",
        files = listOf(
            ModelFile(
                url          = "model_quantized.onnx",
                relativeDest = onnxRelDest,
                sha256       = onnxSha,
                sizeBytes    = fakeOnnxBytes.size.toLong()
            ),
            ModelFile(
                url          = "tokenizer.json",
                relativeDest = tokenizerRelDest,
                sha256       = tokenizerSha,
                sizeBytes    = fakeTokenizerBytes.size.toLong()
            )
        )
    )

    private fun buildProvisioner(manifest: ModelManifest): ModelProvisioner =
        ModelProvisioner(
            manifest = manifest,
            destRootProvider = { _ -> testDestRoot }
        )

    private fun mockServerBaseUrl(): String = mockServer.url("/").toString()

    // -------------------------------------------------------------------------
    // Test (a): both files land at the correct FLAT path; bytes match server payload
    // -------------------------------------------------------------------------

    @Test
    fun provision_downloadsAndInstallsBothFiles_exactBytesMatch() = runBlocking {
        // Serve the two synthetic payloads.
        mockServer.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                return when {
                    request.path?.endsWith("model_quantized.onnx") == true ->
                        MockResponse().setBody(Buffer().write(fakeOnnxBytes)).setResponseCode(200)
                    request.path?.endsWith("tokenizer.json") == true ->
                        MockResponse().setBody(Buffer().write(fakeTokenizerBytes)).setResponseCode(200)
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }

        val provisioner = buildProvisioner(buildTestManifest())
        provisioner.provision(context, mockServerBaseUrl())

        // (a) Both files exist at the FLAT dest path under testDestRoot.
        val onnxFile = File(testDestRoot, onnxRelDest)
        val tokenizerFile = File(testDestRoot, tokenizerRelDest)

        assertTrue("ONNX file should exist after provision()", onnxFile.exists())
        assertTrue("Tokenizer file should exist after provision()", tokenizerFile.exists())

        // Mutation-resistant: read back bytes and compare - not just existence.
        assertArrayEquals(
            "ONNX file bytes must exactly equal what the server served",
            fakeOnnxBytes, onnxFile.readBytes()
        )
        assertArrayEquals(
            "Tokenizer file bytes must exactly equal what the server served",
            fakeTokenizerBytes, tokenizerFile.readBytes()
        )
    }

    // -------------------------------------------------------------------------
    // Test (b): StateFlow emits Downloading... then Ready
    // -------------------------------------------------------------------------

    @Test
    fun provision_stateFlow_emitsDownloadingThenReady() = runBlocking {
        mockServer.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                return when {
                    request.path?.endsWith("model_quantized.onnx") == true ->
                        MockResponse().setBody(Buffer().write(fakeOnnxBytes)).setResponseCode(200)
                    request.path?.endsWith("tokenizer.json") == true ->
                        MockResponse().setBody(Buffer().write(fakeTokenizerBytes)).setResponseCode(200)
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }

        val provisioner = buildProvisioner(buildTestManifest())

        // Track states seen during provision (via a side channel list built before provision runs).
        val seenStates = mutableListOf<ProvisionState>()
        val collectJob = this.launch {
            provisioner.stateFlow.collect { seenStates.add(it) }
        }

        provisioner.provision(context, mockServerBaseUrl())
        collectJob.cancel()

        val finalState = provisioner.stateFlow.value

        // The final state must be Ready.
        assertEquals(
            "StateFlow should be Ready after successful provision()",
            ProvisionState.Ready, finalState
        )

        // At least one intermediate state (Downloading or Verifying) must have been seen.
        // Note: with very small test payloads the download may complete before the collector
        // captures a Downloading event (race), but Verifying is always emitted per-file.
        val hadIntermediateState = seenStates.any {
            it is ProvisionState.Downloading || it is ProvisionState.Verifying
        }
        assertTrue(
            "StateFlow should emit at least one Downloading or Verifying state during provision(); " +
            "states seen: $seenStates",
            hadIntermediateState
        )
    }

    // -------------------------------------------------------------------------
    // Test (b) - simple version: provision completes with Ready
    // -------------------------------------------------------------------------

    @Test
    fun provision_returnsReady_onSuccess() = runBlocking {
        mockServer.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                return when {
                    request.path?.endsWith("model_quantized.onnx") == true ->
                        MockResponse().setBody(Buffer().write(fakeOnnxBytes)).setResponseCode(200)
                    request.path?.endsWith("tokenizer.json") == true ->
                        MockResponse().setBody(Buffer().write(fakeTokenizerBytes)).setResponseCode(200)
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }

        val provisioner = buildProvisioner(buildTestManifest())
        provisioner.provision(context, mockServerBaseUrl())

        assertEquals(
            "StateFlow should be Ready after successful provision()",
            ProvisionState.Ready,
            provisioner.stateFlow.value
        )
    }

    // -------------------------------------------------------------------------
    // Test (c): Wrong SHA → Failed AND no final file left on disk
    // -------------------------------------------------------------------------

    @Test
    fun provision_wrongSha_emitsFailed_andNoFinalFileOnDisk() = runBlocking {
        // Serve a different payload so the sha256 won't match.
        val wrongPayload = ByteArray(4096) { 0xFF.toByte() }  // all 0xFF - different sha

        mockServer.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                return when {
                    request.path?.endsWith("model_quantized.onnx") == true ->
                        // Wrong content - same SIZE but different bytes, so sha mismatch
                        // We also need same sizeBytes so the size check passes, leaving sha to fail.
                        MockResponse().setBody(Buffer().write(wrongPayload)).setResponseCode(200)
                    request.path?.endsWith("tokenizer.json") == true ->
                        MockResponse().setBody(Buffer().write(fakeTokenizerBytes)).setResponseCode(200)
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }

        // Build a manifest with the correct sha for fakeOnnxBytes, but the server will
        // send wrongPayload - so sha will mismatch.
        val provisioner = buildProvisioner(buildTestManifest(onnxSha = fakeOnnxSha256))
        provisioner.provision(context, mockServerBaseUrl())

        val state = provisioner.stateFlow.value
        assertTrue(
            "StateFlow should be Failed after sha mismatch, got: $state",
            state is ProvisionState.Failed
        )

        // Critical atomicity check: the final file must NOT exist on disk.
        val onnxFinalFile = File(testDestRoot, onnxRelDest)
        assertFalse(
            "ONNX final file must NOT exist after sha mismatch (atomicity contract)",
            onnxFinalFile.exists()
        )

        // Fix 14: also assert the .tmp scratch file is absent - the provisioner
        // must clean it up on failure so the next provision() starts from scratch.
        val onnxTmpFile = File("${onnxFinalFile.absolutePath}.tmp")
        assertFalse(
            "ONNX .tmp file must NOT remain after sha mismatch (cleanup contract); " +
            "found: ${onnxTmpFile.absolutePath}",
            onnxTmpFile.exists()
        )
    }

    // -------------------------------------------------------------------------
    // U8: free space is checked before any byte downloads. Range resume and .tmp retention are
    // covered on the host by ModelProvisionerResumeTest (MockWebServer).
    // -------------------------------------------------------------------------

    @Test
    fun provision_lowFreeSpace_failsWithClearReason_beforeAnyRequest() = runBlocking {
        val provisioner = ModelProvisioner(
            manifest = buildTestManifest(),
            destRootProvider = { _ -> testDestRoot }
        )
        provisioner.usableSpaceOf = { 1024L }   // smaller than the model files plus margin

        provisioner.provision(context, mockServerBaseUrl())

        val state = provisioner.stateFlow.value
        assertTrue("expected Failed, got $state", state is ProvisionState.Failed)
        assertTrue(
            "reason must name free space, got ${(state as ProvisionState.Failed).reason}",
            state.reason.contains("free space")
        )
        assertEquals("no request may reach the server", 0, mockServer.requestCount)
        assertFalse(File(testDestRoot, onnxRelDest).exists())
        assertFalse(File("${File(testDestRoot, onnxRelDest).absolutePath}.tmp").exists())
    }

    // -------------------------------------------------------------------------
    // Test (d): Idempotency - second provision() returns Ready without new requests
    // -------------------------------------------------------------------------

    @Test
    fun provision_idempotent_secondCallSkipsDownload() = runBlocking {
        var requestCount = 0

        mockServer.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                requestCount++
                return when {
                    request.path?.endsWith("model_quantized.onnx") == true ->
                        MockResponse().setBody(Buffer().write(fakeOnnxBytes)).setResponseCode(200)
                    request.path?.endsWith("tokenizer.json") == true ->
                        MockResponse().setBody(Buffer().write(fakeTokenizerBytes)).setResponseCode(200)
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }

        val provisioner = buildProvisioner(buildTestManifest())

        // First provision - downloads 2 files.
        provisioner.provision(context, mockServerBaseUrl())
        assertEquals("First provision should complete with Ready",
            ProvisionState.Ready, provisioner.stateFlow.value)

        val requestsAfterFirst = requestCount

        // Second provision - already provisioned, must skip download entirely.
        provisioner.provision(context, mockServerBaseUrl())
        assertEquals("Second provision should also be Ready",
            ProvisionState.Ready, provisioner.stateFlow.value)

        val newRequestsOnSecondCall = requestCount - requestsAfterFirst
        assertEquals(
            "Second provision() must not hit the server (idempotency: 0 new requests), " +
            "got $newRequestsOnSecondCall",
            0, newRequestsOnSecondCall
        )
    }

    // -------------------------------------------------------------------------
    // Test: isProvisioned returns false before, true after
    // -------------------------------------------------------------------------

    @Test
    fun isProvisioned_falseBeforeDownload_trueAfter() = runBlocking {
        mockServer.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                return when {
                    request.path?.endsWith("model_quantized.onnx") == true ->
                        MockResponse().setBody(Buffer().write(fakeOnnxBytes)).setResponseCode(200)
                    request.path?.endsWith("tokenizer.json") == true ->
                        MockResponse().setBody(Buffer().write(fakeTokenizerBytes)).setResponseCode(200)
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }

        val provisioner = buildProvisioner(buildTestManifest())

        assertFalse("isProvisioned should be false before any download", provisioner.isProvisioned(context))

        provisioner.provision(context, mockServerBaseUrl())

        assertTrue("isProvisioned should be true after successful provision()", provisioner.isProvisioned(context))
    }

    // -------------------------------------------------------------------------
    // Test: .tmp file is cleaned up after successful install (no leftover temps)
    // -------------------------------------------------------------------------

    @Test
    fun provision_noTmpFilesLeft_afterSuccess() = runBlocking {
        mockServer.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                return when {
                    request.path?.endsWith("model_quantized.onnx") == true ->
                        MockResponse().setBody(Buffer().write(fakeOnnxBytes)).setResponseCode(200)
                    request.path?.endsWith("tokenizer.json") == true ->
                        MockResponse().setBody(Buffer().write(fakeTokenizerBytes)).setResponseCode(200)
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }

        val provisioner = buildProvisioner(buildTestManifest())
        provisioner.provision(context, mockServerBaseUrl())

        // No .tmp files should remain after a successful provision.
        val tmpFiles = testDestRoot.walkTopDown().filter { it.name.endsWith(".tmp") }.toList()
        assertTrue(
            "No .tmp files should remain after successful provision, found: $tmpFiles",
            tmpFiles.isEmpty()
        )
    }

    // -------------------------------------------------------------------------
    // Helper: SHA-256 hex for a ByteArray
    // -------------------------------------------------------------------------

    private fun sha256Hex(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        return digest.digest(bytes).joinToString("") { "%02x".format(it) }
    }
}
