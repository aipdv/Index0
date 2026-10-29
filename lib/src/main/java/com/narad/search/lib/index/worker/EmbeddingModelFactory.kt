package com.narad.search.lib.index.worker

import com.narad.search.lib.query.model.DjlTokenizer
import com.narad.search.lib.query.model.EmbeddingModel

/**
 * Factory that constructs a production [OnnxEmbedder] from paths to the ONNX
 * model file and the HuggingFace tokenizer.json.
 *
 * The on-device tokenizer is [DjlTokenizer], backed by the Rust tokenizer engine
 * bundled in the `ai.djl.android:tokenizer-native` AAR (`libdjl_tokenizer.so`,
 * arm64-v8a + x86_64). DJL auto-detects Android and loads the library via
 * `System.loadLibrary` - no network access is required (offline-first, B4 resolved).
 *
 * EmbeddingGemma 2 uses SentencePiece BPE (Gemma 2 vocabulary). [DjlTokenizer]
 * wraps the same Rust tokenizer engine used by the JVM host tests, guaranteeing
 * token-ID parity. Truncation is set explicitly to 512 tokens in [DjlTokenizer].
 *
 * Production caller: [IndexWorker.runIndexing] calls
 * `EmbeddingModelFactory.build(modelPath, tokenizerPath)` where
 * `tokenizerPath = File(modelPath).resolveSibling("tokenizer.json").absolutePath`.
 * Model layout (FLAT): `filesDir/models/embeddinggemma-2/{model_quantized.onnx,
 * model_quantized.onnx_data, tokenizer.json}`.
 */
object EmbeddingModelFactory {

    /**
     * Build a production [OnnxEmbedder] for the given model and tokenizer paths.
     *
     * @param modelPath         Absolute path to `model_quantized.onnx`.
     *                          `model_quantized.onnx_data` must be in the same directory.
     * @param tokenizerJsonPath Absolute path to the companion `tokenizer.json`.
     * @param tokenBudget       Padded tokens per batched model run (by device tier).
     * @return A fully-wired [OnnxEmbedder] ready to produce 256-dim embeddings.
     * @throws java.io.IOException if tokenizerJsonPath is not a readable tokenizer.json.
     * @throws ai.onnxruntime.OrtException if modelPath does not point to a valid ONNX model.
     */
    fun build(
        modelPath: String,
        tokenizerJsonPath: String,
        tokenBudget: Int = EmbeddingModel.DEFAULT_TOKEN_BUDGET,
    ): OnnxEmbedder {
        val tokenizer = DjlTokenizer(tokenizerJsonPath)
        return try {
            OnnxEmbedder(EmbeddingModel(modelPath, tokenizer), tokenBudget)
        } catch (e: Exception) {
            tokenizer.close()
            throw e
        } catch (e: LinkageError) {
            tokenizer.close()
            throw e
        }
    }
}
