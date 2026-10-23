package com.narad.search.lib.query.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.narad.search.lib.query.engine.QueryEmbedder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.sqrt
import kotlin.math.min

/**
 * Loads EmbeddingGemma 2 (OUR tuned ONNX, MatMulNBits accuracy_level=4) and
 * produces 256-dimensional L2-normalised text embeddings.
 *
 * Pooling strategy: **sentence_embedding** output - the model's own pooled vector
 * (no CLS slicing; Gemma 2 handles pooling internally). The 768-dim output is
 * Matryoshka-sliced to [DIM]=256 and then L2-normalised.
 *
 * Inputs: `input_ids`, `attention_mask` (int64 [1,n]) plus three empty float32
 * tensors for unused modalities: `image_features`, `video_features`, `audio_features`
 * (shape [0,512]). All five must be present or the graph aborts.
 *
 * Session options: [OrtSession.SessionOptions.OptLevel.ALL_OPT] and
 * intraOpNumThreads = min(4, availableProcessors).
 *
 * @param modelPath  Absolute filesystem path to `model_quantized.onnx`.
 *                   The data file `model_quantized.onnx_data` must exist
 *                   in the SAME directory (graph-internal name; do not rename).
 * @param tokenizer  Any [Tokenizer] implementation. The EG2 tokenizer adds
 *                   `<bos>` automatically; callers do NOT add special tokens.
 */
class EmbeddingModel(
    private val modelPath: String,
    private val tokenizer: Tokenizer
) : QueryEmbedder, AutoCloseable {

    companion object {
        /** Matryoshka slice dimension stored and compared at search time. */
        const val DIM = 256

        // EG2 requires the full 768-dim head; we slice down after inference.
        private const val RAW_DIM = 768

        /** Prompt prefix for search queries (asymmetric dual-embed). */
        private const val QUERY_PREFIX = "task: search result | query: "

        /** Prompt prefix for indexed document chunks. */
        private const val DOC_PREFIX = "title: none | text: "
    }

    // Process-wide OrtEnvironment singleton - NEVER close it (U4 fix).
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
     * Embed a search query with the asymmetric query prompt.
     *
     * The prompt `"task: search result | query: "` is prepended automatically.
     *
     * @return L2-normalised 256-dim embedding.
     */
    override fun embedQuery(text: String): FloatArray = embed("$QUERY_PREFIX$text")

    /**
     * Produce a 256-dim image embedding from pre-computed vision-encoder features.
     *
     * The LM input sequence is:
     *   [BOS=2, BOI=255999, IMG_TOKEN=258880 × nSoft, EOI=258882, EOS=1]
     *
     * [imageFeatures] must be a flat [nSoft × 512] float array from [VisionEncoder.encode].
     * The count of IMG_TOKEN (258880) entries MUST equal [nSoft] - a mismatch causes
     * the ONNX graph to return garbage.
     *
     * Uses the SAME session as [embedQuery] (shared LM - never load a second 300 MB session).
     *
     * @return L2-normalised 256-dim embedding (same space as text queries).
     */
    fun embedImageFeatures(imageFeatures: FloatArray, nSoft: Int): FloatArray {
        require(imageFeatures.size == nSoft * 512) {
            "imageFeatures must have nSoft × 512 = ${nSoft * 512} elements, got ${imageFeatures.size}"
        }
        // Sequence: BOS, BOI, IMG×nSoft, EOI, EOS
        val inputIds = LongArray(nSoft + 4).also { ids ->
            ids[0] = 2L          // BOS
            ids[1] = 255999L     // BOI
            for (i in 0 until nSoft) ids[2 + i] = 258880L  // image soft token
            ids[2 + nSoft] = 258882L  // EOI
            ids[3 + nSoft] = 1L       // EOS
        }
        val seqLen = inputIds.size.toLong()
        val mask   = LongArray(inputIds.size) { 1L }

        val inputs = mapOf(
            "input_ids" to OnnxTensor.createTensor(
                env, LongBuffer.wrap(inputIds), longArrayOf(1L, seqLen)
            ),
            "attention_mask" to OnnxTensor.createTensor(
                env, LongBuffer.wrap(mask), longArrayOf(1L, seqLen)
            ),
            "image_features" to OnnxTensor.createTensor(
                env, FloatBuffer.wrap(imageFeatures), longArrayOf(nSoft.toLong(), 512L)
            ),
            "video_features" to OnnxTensor.createTensor(
                env, FloatBuffer.allocate(0), longArrayOf(0L, 512L)
            ),
            "audio_features" to OnnxTensor.createTensor(
                env, FloatBuffer.allocate(0), longArrayOf(0L, 512L)
            ),
        )
        try {
            session.run(inputs, setOf("sentence_embedding")).use { out ->
                @Suppress("UNCHECKED_CAST")
                val raw = (out[0].value as Array<FloatArray>)[0]
                val sliced = raw.copyOfRange(0, DIM)
                val result = l2Normalize(sliced)
                check(result.none { it.isNaN() }) { "Image embedding contains NaN (nSoft=$nSoft)" }
                return result
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    /**
     * Embed a document chunk with the document prompt.
     *
     * The prompt `"title: none | text: "` is prepended automatically.
     *
     * @return L2-normalised 256-dim embedding.
     */
    fun embedDocument(text: String): FloatArray = embed("$DOC_PREFIX$text")

    /** Runs the ONNX graph on [promptedText] and returns a 256-dim L2-normalised vector. */
    private fun embed(promptedText: String): FloatArray {
        val tokenized = tokenizer.encode(promptedText)
        val n = tokenized.inputIds.size.toLong()

        // Bug watch: each modality tensor must be a SEPARATE FloatBuffer instance.
        val inputs = mapOf(
            "input_ids" to OnnxTensor.createTensor(
                env, LongBuffer.wrap(tokenized.inputIds), longArrayOf(1L, n)
            ),
            "attention_mask" to OnnxTensor.createTensor(
                env, LongBuffer.wrap(tokenized.attentionMask), longArrayOf(1L, n)
            ),
            // Three empty [0,512] float32 tensors for unused modalities (required by graph).
            "image_features" to OnnxTensor.createTensor(
                env, FloatBuffer.allocate(0), longArrayOf(0L, 512L)
            ),
            "video_features" to OnnxTensor.createTensor(
                env, FloatBuffer.allocate(0), longArrayOf(0L, 512L)
            ),
            "audio_features" to OnnxTensor.createTensor(
                env, FloatBuffer.allocate(0), longArrayOf(0L, 512L)
            ),
        )

        try {
            session.run(inputs, setOf("sentence_embedding")).use { out ->
                @Suppress("UNCHECKED_CAST")
                val raw = (out[0].value as Array<FloatArray>)[0]

                check(raw.size == RAW_DIM) {
                    "Expected ${RAW_DIM}-dim sentence_embedding, got ${raw.size}"
                }

                // Matryoshka: slice first DIM values, then L2-normalise the slice.
                val sliced = raw.copyOfRange(0, DIM)
                val result = l2Normalize(sliced)

                check(result.none { it.isNaN() }) {
                    "Embedding contains NaN values for prompted text (len=$n tokens)"
                }

                return result
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    override fun close() {
        // Close ONLY the session, NOT the process-wide OrtEnvironment singleton (U4 fix).
        session.close()
        (tokenizer as? AutoCloseable)?.close()
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var norm = 0f
        for (x in v) norm += x * x
        norm = sqrt(norm.toDouble()).toFloat()
        if (norm < 1e-12f) return v.copyOf()
        return FloatArray(v.size) { v[it] / norm }
    }
}

/**
 * The process-wide [OrtEnvironment] with ONNX Runtime telemetry switched off. Narad is on-device
 * only; ORT telemetry would send the Android ID over the network. Best effort: the setter can
 * throw on a platform build without telemetry, which is fine because there is nothing to turn off.
 * The manifest also removes the provider that starts it (lib AndroidManifest.xml).
 */
internal fun ortEnvironment(): OrtEnvironment =
    OrtEnvironment.getEnvironment().also { env -> runCatching { env.setTelemetry(false) } }
