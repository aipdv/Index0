package com.narad.search.lib.index.provision

import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import okhttp3.mockwebserver.SocketPolicy
import okio.Buffer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.file.Files
import java.security.MessageDigest

/**
 * U8: free-space check, Range resume and .tmp retention, against a local MockWebServer.
 * Host JVM only; no device needed.
 */
class ModelProvisionerResumeTest {

    private val dest = "models/test/model.onnx"
    private val payload = ByteArray(300_000) { (it * 31 + 7).toByte() }
    private lateinit var root: File
    private lateinit var server: MockWebServer
    private val tmp get() = File(root, "$dest.tmp")
    private val finalFile get() = File(root, dest)

    @Before
    fun setUp() {
        root = Files.createTempDirectory("provision-resume").toFile()
        server = MockWebServer().also { it.start() }
    }

    @After
    fun tearDown() {
        server.shutdown()
        root.deleteRecursively()
    }

    private fun provisioner(
        sha: String = sha256(payload),
        usableSpace: (File) -> Long = { Long.MAX_VALUE },
    ) = ModelProvisioner(
        ModelManifest(
            modelId = "test",
            files = listOf(ModelFile("model.onnx", dest, sha, payload.size.toLong()))
        )
    ).also { it.usableSpaceOf = usableSpace }

    private fun url() = server.url("/").toString()

    /** Honours Range like a real CDN: 206 with the tail, else 200 with everything. */
    private fun rangeAwareServer() {
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val range = request.getHeader("Range")
                return if (range != null) {
                    val from = range.removePrefix("bytes=").removeSuffix("-").toInt()
                    MockResponse().setResponseCode(206)
                        .setHeader("Content-Range", "bytes $from-${payload.size - 1}/${payload.size}")
                        .setBody(Buffer().write(payload.copyOfRange(from, payload.size)))
                } else {
                    MockResponse().setResponseCode(200).setBody(Buffer().write(payload))
                }
            }
        }
    }

    @Test
    fun `kept tmp is resumed with a Range request and finishes with a matching hash`() = runBlocking {
        rangeAwareServer()
        tmp.parentFile!!.mkdirs()
        tmp.writeBytes(payload.copyOfRange(0, 100_000))

        val p = provisioner()
        p.provision(root, url())

        val req = server.takeRequest()
        assertEquals("bytes=100000-", req.getHeader("Range"))
        assertEquals(ProvisionState.Ready, p.stateFlow.value)
        assertArrayEquals(payload, finalFile.readBytes())
        assertFalse(".tmp is renamed away", tmp.exists())
    }

    @Test
    fun `200 answer to a Range request restarts from zero`() = runBlocking {
        // Server ignores Range. The old prefix is deliberately wrong so a bad append would fail the hash.
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest) =
                MockResponse().setResponseCode(200).setBody(Buffer().write(payload))
        }
        tmp.parentFile!!.mkdirs()
        tmp.writeBytes(ByteArray(50_000) { 0x55 })

        val p = provisioner()
        p.provision(root, url())

        assertNotNull("a Range request was sent", server.takeRequest().getHeader("Range"))
        assertEquals(ProvisionState.Ready, p.stateFlow.value)
        assertArrayEquals(payload, finalFile.readBytes())
        assertFalse(tmp.exists())
    }

    @Test
    fun `416 answer to a Range request deletes the tmp and restarts from zero`() = runBlocking {
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse =
                if (request.getHeader("Range") != null) {
                    MockResponse().setResponseCode(416).setHeader("Content-Range", "bytes */${payload.size}")
                } else {
                    MockResponse().setResponseCode(200).setBody(Buffer().write(payload))
                }
        }
        tmp.parentFile!!.mkdirs()
        tmp.writeBytes(ByteArray(120_000) { 0x33 })

        val p = provisioner()
        p.provision(root, url())

        assertNotNull("first request carried a Range", server.takeRequest().getHeader("Range"))
        assertNull("the retry carries no Range", server.takeRequest().getHeader("Range"))
        assertEquals(2, server.requestCount)
        assertEquals(ProvisionState.Ready, p.stateFlow.value)
        assertArrayEquals(payload, finalFile.readBytes())
        assertFalse(tmp.exists())
    }

    @Test
    fun `connection drop mid-stream keeps the tmp and the next run resumes it`() = runBlocking {
        server.enqueue(
            MockResponse().setResponseCode(200).setBody(Buffer().write(payload))
                .setSocketPolicy(SocketPolicy.DISCONNECT_DURING_RESPONSE_BODY)
        )
        val p = provisioner()
        p.provision(root, url())

        assertTrue("first run fails", p.stateFlow.value is ProvisionState.Failed)
        assertFalse(finalFile.exists())
        assertTrue("partial .tmp is kept", tmp.exists())
        val kept = tmp.length()
        assertTrue("partial, not empty and not complete: $kept", kept > 0 && kept < payload.size)
        assertArrayEquals("kept bytes are a true prefix", payload.copyOfRange(0, kept.toInt()), tmp.readBytes())

        server.takeRequest() // first request, no Range
        rangeAwareServer()
        p.provision(root, url())

        assertEquals("bytes=$kept-", server.takeRequest().getHeader("Range"))
        assertEquals(ProvisionState.Ready, p.stateFlow.value)
        assertArrayEquals(payload, finalFile.readBytes())
        assertFalse(tmp.exists())
    }

    @Test
    fun `sha mismatch deletes the tmp`() = runBlocking {
        rangeAwareServer()
        val p = provisioner(sha = sha256(ByteArray(8)))
        p.provision(root, url())

        assertTrue(p.stateFlow.value is ProvisionState.Failed)
        assertFalse(tmp.exists())
        assertFalse(finalFile.exists())
    }

    @Test
    fun `size mismatch deletes the tmp`() = runBlocking {
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest) =
                MockResponse().setResponseCode(200).setBody(Buffer().write(payload.copyOf(1_000)))
        }
        val p = provisioner()
        p.provision(root, url())

        assertTrue(p.stateFlow.value is ProvisionState.Failed)
        assertFalse(tmp.exists())
        assertFalse(finalFile.exists())
    }

    @Test
    fun `tmp already at full size is verified without a request`() = runBlocking {
        tmp.parentFile!!.mkdirs()
        tmp.writeBytes(payload)

        val p = provisioner()
        p.provision(root, url())

        assertEquals(0, server.requestCount)
        assertEquals(ProvisionState.Ready, p.stateFlow.value)
        assertArrayEquals(payload, finalFile.readBytes())
    }

    @Test
    fun `free space below the model size fails before any request`() = runBlocking {
        rangeAwareServer()
        val p = provisioner(usableSpace = { payload.size.toLong() }) // covers the file, not file + margin
        p.provision(root, url())

        val state = p.stateFlow.value
        assertTrue(state is ProvisionState.Failed)
        assertTrue((state as ProvisionState.Failed).toString(), state.toString().contains("free space"))
        assertEquals("no bytes may be requested", 0, server.requestCount)
        assertNull(tmp.takeIf { it.exists() })
    }

    @Test
    fun `a kept tmp lowers the space needed for the resume`() = runBlocking {
        rangeAwareServer()
        tmp.parentFile!!.mkdirs()
        tmp.writeBytes(payload.copyOfRange(0, 200_000))
        // Only 100_000 bytes remain; free space covers them plus the margin but not the full file.
        val free = 100_000L + ModelProvisioner.FREE_SPACE_MARGIN_BYTES
        val p = provisioner(usableSpace = { free })
        p.provision(root, url())

        assertEquals(ProvisionState.Ready, p.stateFlow.value)
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}
