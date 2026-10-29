package com.narad.search.lib.query.model

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer as DjlTokenizer

/**
 * [Tokenizer] implementation backed by DJL HuggingFace Tokenizers (native Rust
 * library, macOS arm64-compatible).
 *
 * Lives in the **test** sourceset so the DJL native dependency is never
 * packaged into the Android APK.
 *
 * @param tokenizerJsonPath  Absolute path to the HuggingFace `tokenizer.json`
 *                           file that accompanies the model.
 */
class HuggingFaceTokenizer(tokenizerJsonPath: String) : Tokenizer {

    private val delegate: DjlTokenizer =
        DjlTokenizer.newInstance(java.nio.file.Paths.get(tokenizerJsonPath))

    override fun encode(text: String): Tokenized {
        val encoding = delegate.encode(text)
        val inputIds     = encoding.ids          // LongArray
        val attentionMask = encoding.attentionMask // LongArray
        return Tokenized(inputIds = inputIds, attentionMask = attentionMask)
    }
}
