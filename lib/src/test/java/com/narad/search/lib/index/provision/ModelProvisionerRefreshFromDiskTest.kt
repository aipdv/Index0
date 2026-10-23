package com.narad.search.lib.index.provision

import android.content.Context
import android.content.ContextWrapper
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.nio.file.Files

/** E17: a process restart must not hide a model that is already on disk. */
class ModelProvisionerRefreshFromDiskTest {

    private val payload = ByteArray(2048) { (it * 3).toByte() }
    private val dest = "models/test/model.onnx"
    private val manifest = ModelManifest(
        modelId = "test",
        files = listOf(
            ModelFile(url = "model.onnx", relativeDest = dest, sha256 = "0".repeat(64), sizeBytes = payload.size.toLong())
        )
    )
    private val root = Files.createTempDirectory("refresh-from-disk").toFile()
    private val context: Context = ContextWrapper(null)
    private val provisioner = ModelProvisioner(manifest) { root }

    @Before fun setUp() { root.deleteRecursively(); root.mkdirs() }
    @After fun tearDown() { root.deleteRecursively() }

    private fun writeModel() {
        root.resolve(dest).apply { parentFile!!.mkdirs(); writeBytes(payload) }
    }

    @Test
    fun `files on disk and NotStarted becomes Ready`() {
        writeModel()
        provisioner.refreshFromDisk(context)
        assertEquals(ProvisionState.Ready, provisioner.stateFlow.value)
    }

    @Test
    fun `files missing stays NotStarted`() {
        provisioner.refreshFromDisk(context)
        assertEquals(ProvisionState.NotStarted, provisioner.stateFlow.value)
    }

    @Test
    fun `wrong size on disk stays NotStarted`() {
        root.resolve(dest).apply { parentFile!!.mkdirs(); writeBytes(payload.copyOf(10)) }
        provisioner.refreshFromDisk(context)
        assertEquals(ProvisionState.NotStarted, provisioner.stateFlow.value)
    }

    @Test
    fun `a Failed state is not overwritten even when files are on disk`() = runBlocking {
        val server = MockWebServer()
        server.enqueue(MockResponse().setBody(okio.Buffer().write(payload))) // sha256 will not match
        server.start()
        try {
            provisioner.provision(root, server.url("/").toString())
            assertTrue(provisioner.stateFlow.value is ProvisionState.Failed)
            val failed = provisioner.stateFlow.value

            writeModel()
            provisioner.refreshFromDisk(context)

            assertEquals(failed, provisioner.stateFlow.value)
        } finally {
            server.shutdown()
        }
    }
}
