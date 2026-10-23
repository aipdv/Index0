package com.narad.search.lib.query.model.image

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.narad.search.lib.query.model.ortEnvironment
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.min

/**
 * EmbeddingGemma 2 vision encoder - wraps `vision_encoder_quantized.onnx`.
 *
 * Inputs (spec / ONNX introspection):
 *   `pixel_values`      float [1, 630, 768]  - padded patches (zeros for unused)
 *   `pixel_position_ids` int64 [1, 630, 2]   - (col, row) per patch; (−1, −1) for padding
 *
 * Output:
 *   `image_features`    float [n_soft, 512]  - pooled per-soft-token features
 *
 * @param modelPath  Absolute path to `vision_encoder_quantized.onnx`.
 *   The `vision_encoder_quantized.onnx_data` sibling must exist in the same directory.
 */
class VisionEncoder(modelPath: String) : AutoCloseable {

    companion object {
        private const val MAX_PATCHES = EG2ImagePreprocessor.MAX_PATCHES
        private const val PATCH_FEAT  = EG2ImagePreprocessor.PATCH_FEAT   // 768
        const val FEAT_DIM            = 512
    }

    // Shared process-wide env - NEVER close (mirrors EmbeddingModel pattern)
    private val env: OrtEnvironment = ortEnvironment()

    private val session: OrtSession = env.createSession(
        modelPath,
        OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(min(4, Runtime.getRuntime().availableProcessors()))
            addConfigEntry("session.intra_op.allow_spinning", "0")
        }
    )

    /**
     * Run the vision encoder on preprocessed [pixelValues] and [positionIds].
     *
     * @param pixelValues  Flattened [MAX_PATCHES × 768] float array.
     * @param positionIds  Flattened [MAX_PATCHES × 2]  long array.
     * @return Flat [n_soft × 512] float array.
     */
    fun encode(pixelValues: FloatArray, positionIds: LongArray): FloatArray {
        check(pixelValues.size == MAX_PATCHES * PATCH_FEAT) {
            "pixelValues must be ${MAX_PATCHES * PATCH_FEAT} floats, got ${pixelValues.size}"
        }
        check(positionIds.size == MAX_PATCHES * 2) {
            "positionIds must be ${MAX_PATCHES * 2} longs, got ${positionIds.size}"
        }

        val pvTensor  = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(pixelValues),
            longArrayOf(1L, MAX_PATCHES.toLong(), PATCH_FEAT.toLong())
        )
        val pidTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(positionIds),
            longArrayOf(1L, MAX_PATCHES.toLong(), 2L)
        )

        try {
            session.run(mapOf("pixel_values" to pvTensor, "pixel_position_ids" to pidTensor)).use { out ->
                @Suppress("UNCHECKED_CAST")
                val feats = out["image_features"].get().value as Array<FloatArray>
                // feats: [n_soft][512] - flatten
                val flat = FloatArray(feats.size * FEAT_DIM)
                for (i in feats.indices) {
                    feats[i].copyInto(flat, i * FEAT_DIM)
                }
                return flat
            }
        } finally {
            pvTensor.close()
            pidTensor.close()
        }
    }

    override fun close() {
        // Close ONLY the session - NOT the process-wide OrtEnvironment singleton
        session.close()
    }
}
