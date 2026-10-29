package com.narad.search.lib.index

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.narad.search.lib.index.provision.EmbeddingGemma2Manifest
import com.narad.search.lib.index.provision.EmbeddingGemma2VisionManifest
import com.narad.search.lib.index.service.IndexService
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Every production entry point (Start Indexing, boot, MediaStore) passes
 * IndexService.getDefaultModelPath to the worker. It must be exactly where the provisioner
 * installs the graph, or indexing retries forever on a real install.
 */
@RunWith(AndroidJUnit4::class)
class ModelPathTest {

    @Test
    fun defaultPaths_matchWhereProvisionerInstalls() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals(
            File(context.filesDir, "models/embeddinggemma-2/model_quantized.onnx").absolutePath,
            IndexService.getDefaultModelPath(context)
        )
        assertEquals(
            File(context.filesDir, "models/embeddinggemma-2/vision_encoder_quantized.onnx").absolutePath,
            IndexService.getVisionEncoderPath(context)
        )
        // The graph's weights must sit next to it under the name baked into the graph.
        val textFiles = EmbeddingGemma2Manifest.manifest.files.map { it.relativeDest }
        val visionFiles = EmbeddingGemma2VisionManifest.manifest.files.map { it.relativeDest }
        assertEquals(true, "models/embeddinggemma-2/model_quantized.onnx_data" in textFiles)
        assertEquals(true, "models/embeddinggemma-2/vision_encoder_quantized.onnx_data" in visionFiles)
    }
}
