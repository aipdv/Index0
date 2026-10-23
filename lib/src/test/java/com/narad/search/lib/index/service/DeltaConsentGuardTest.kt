package com.narad.search.lib.index.service

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import java.nio.file.Files

/**
 * E6: the app-open and periodic delta jobs only refresh an existing index, so they
 * must not be enqueued before the model is on disk (before the user accepts Gemma Terms).
 * W29: nor before the first full index has finished (that run is the user-started
 * foreground run; deltas rely on job quota).
 */
class DeltaConsentGuardTest {

    @Test
    fun `no model files means nothing is enqueued`() {
        val dir = Files.createTempDirectory("delta-guard-absent").toFile()
        try {
            val enqueued = mutableListOf<String>()
            val modelPath = File(dir, "model_quantized.onnx").absolutePath
            BootReceiver.enqueueIfModelReady(modelPath, firstIndexDone = true) { enqueued += it }
            assertEquals(emptyList<String>(), enqueued)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun `partial model files means nothing is enqueued`() {
        val dir = Files.createTempDirectory("delta-guard-partial").toFile()
        try {
            val graph = File(dir, "model_quantized.onnx").apply { writeBytes(ByteArray(1)) }
            val enqueued = mutableListOf<String>()
            BootReceiver.enqueueIfModelReady(graph.absolutePath, firstIndexDone = true) { enqueued += it }
            assertEquals(emptyList<String>(), enqueued)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun `all model artefacts present means the delta is enqueued with the model path`() {
        val dir = Files.createTempDirectory("delta-guard-ready").toFile()
        try {
            val graph = File(dir, "model_quantized.onnx").apply { writeBytes(ByteArray(1)) }
            File(dir, "model_quantized.onnx_data").writeBytes(ByteArray(1))
            File(dir, "tokenizer.json").writeBytes(ByteArray(1))
            val enqueued = mutableListOf<String>()
            BootReceiver.enqueueIfModelReady(graph.absolutePath, firstIndexDone = true) { enqueued += it }
            assertEquals(listOf(graph.absolutePath), enqueued)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun `model present but first index not done means nothing is enqueued`() {
        val dir = Files.createTempDirectory("delta-guard-first-index").toFile()
        try {
            val graph = File(dir, "model_quantized.onnx").apply { writeBytes(ByteArray(1)) }
            File(dir, "model_quantized.onnx_data").writeBytes(ByteArray(1))
            File(dir, "tokenizer.json").writeBytes(ByteArray(1))
            val enqueued = mutableListOf<String>()
            BootReceiver.enqueueIfModelReady(graph.absolutePath, firstIndexDone = false) { enqueued += it }
            assertEquals(emptyList<String>(), enqueued)
        } finally {
            dir.deleteRecursively()
        }
    }

    // ---- Index0: overnight toggle -----------------------------------------------

    private fun readyModel(dir: File): File {
        val graph = File(dir, "model_quantized.onnx").apply { writeBytes(ByteArray(1)) }
        File(dir, "model_quantized.onnx_data").writeBytes(ByteArray(1))
        File(dir, "tokenizer.json").writeBytes(ByteArray(1))
        return graph
    }

    @Test
    fun `overnight off means the periodic job is not enqueued even when ready`() {
        val dir = Files.createTempDirectory("delta-guard-overnight-off").toFile()
        try {
            val graph = readyModel(dir)
            val enqueued = mutableListOf<String>()
            BootReceiver.enqueuePeriodicIfAllowed(false, graph.absolutePath, firstIndexDone = true) { enqueued += it }
            assertEquals(emptyList<String>(), enqueued)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun `overnight on with model ready and first index done enqueues the periodic job`() {
        val dir = Files.createTempDirectory("delta-guard-overnight-on").toFile()
        try {
            val graph = readyModel(dir)
            val enqueued = mutableListOf<String>()
            BootReceiver.enqueuePeriodicIfAllowed(true, graph.absolutePath, firstIndexDone = true) { enqueued += it }
            assertEquals(listOf(graph.absolutePath), enqueued)
            enqueued.clear()
            BootReceiver.enqueuePeriodicIfAllowed(true, graph.absolutePath, firstIndexDone = false) { enqueued += it }
            assertEquals(emptyList<String>(), enqueued)
        } finally {
            dir.deleteRecursively()
        }
    }
}
