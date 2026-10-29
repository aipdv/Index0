package com.narad.search.lib.index.provision

import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okio.Buffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.file.Files
import java.security.MessageDigest

class ModelProvisionerConcurrencyTest {

    @Test
    fun `concurrent provision calls download once and install valid file`() = runBlocking {
        val payload = ByteArray(4096) { (it * 7).toByte() }
        val destination = "models/test/model.onnx"
        val root = Files.createTempDirectory("provision-concurrency").toFile()
        val server = MockWebServer()
        server.enqueue(
            MockResponse()
                .setBody(Buffer().write(payload))
                .setBodyDelay(100, java.util.concurrent.TimeUnit.MILLISECONDS)
        )
        server.start()

        try {
            val provisioner = ModelProvisioner(
                ModelManifest(
                    modelId = "test",
                    files = listOf(
                        ModelFile(
                            url = "model.onnx",
                            relativeDest = destination,
                            sha256 = sha256(payload),
                            sizeBytes = payload.size.toLong()
                        )
                    )
                )
            )

            val first = async { provisioner.provision(root, server.url("/").toString()) }
            val second = async { provisioner.provision(root, server.url("/").toString()) }
            first.await()
            second.await()

            assertEquals(1, server.requestCount)
            assertArrayEquals(payload, root.resolve(destination).readBytes())
            assertEquals(ProvisionState.Ready, provisioner.stateFlow.value)
        } finally {
            server.shutdown()
            root.deleteRecursively()
        }
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { "%02x".format(it) }
}
