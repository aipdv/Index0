package com.narad.search.lib.query.model

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer as DjlHuggingFaceTokenizer

/**
 * Production [Tokenizer] backed by DJL HuggingFace Tokenizers on Android.
 *
 * On Android, the Rust tokenizer engine is provided by the bundled native
 * library `libdjl_tokenizer.so` (shipped via the `ai.djl.android:tokenizer-native`
 * AAR). DJL auto-detects Android and calls `System.loadLibrary("djl_tokenizer")`
 * rather than attempting a network download - the app remains fully offline.
 *
 * Supports the EmbeddingGemma 2 tokenizer format (SentencePiece BPE, Gemma 2
 * vocab, `model_max_length` = 8192 in the file config). Truncation is set
 * EXPLICITLY to 512 tokens here so on-device inference stays within the
 * budget that the ONNX graph was calibrated for. The EG2 tokenizer prepends
 * `<bos>` (token id=2) automatically via its `BOS` added-token; callers do
 * NOT add special tokens manually.
 *
 * Named [DjlTokenizer] (not `HuggingFaceTokenizer`) because the simple name
 * `HuggingFaceTokenizer` is already taken by the test sourceset wrapper;
 * a duplicate simple name in the same package would cause a class-name clash
 * when both sourcesets are visible (e.g. in androidTest).
 *
 * @param tokenizerJsonPath Absolute filesystem path to the `tokenizer.json`
 *   that accompanies the model (e.g. `filesDir/models/embeddinggemma-2/tokenizer.json`).
 * @throws java.io.IOException if tokenizerJsonPath is not a readable tokenizer.json.
 */
class DjlTokenizer(tokenizerJsonPath: String) : Tokenizer, AutoCloseable {

    companion object {
        init {
            // Privacy: every DJL tokenizer build otherwise phones home (djl-telemetry on AWS and
            // the EC2 metadata IP). Set before the first newInstance; a class static initialiser
            // runs once, ahead of any constructor.
            System.setProperty("OPT_OUT_TRACKING", "true")
            System.setProperty("ai.djl.offline", "true")
        }
    }

    private val delegate: DjlHuggingFaceTokenizer =
        DjlHuggingFaceTokenizer.newInstance(
            java.nio.file.Paths.get(tokenizerJsonPath),
            // Explicitly truncate to 512 tokens (EG2 graph calibration budget).
            // The EG2 tokenizer.json sets model_max_length=8192; we must override it.
            mapOf("maxLength" to "512", "truncation" to "true")
        )

    override fun close() {
        delegate.close()
    }

    override fun encode(text: String): Tokenized {
        val encoding = delegate.encode(text)
        return Tokenized(
            inputIds = encoding.ids,
            attentionMask = encoding.attentionMask
        )
    }
}
