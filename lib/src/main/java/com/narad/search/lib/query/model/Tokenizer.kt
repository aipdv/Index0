package com.narad.search.lib.query.model

/**
 * Data class holding the output of a tokenizer encode call.
 *
 * @param inputIds  Token IDs, shape [sequenceLength].
 * @param attentionMask  Attention mask (1 = real token, 0 = padding), same shape.
 */
data class Tokenized(
    val inputIds: LongArray,
    val attentionMask: LongArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Tokenized) return false
        return inputIds.contentEquals(other.inputIds) &&
            attentionMask.contentEquals(other.attentionMask)
    }

    override fun hashCode(): Int {
        var result = inputIds.contentHashCode()
        result = 31 * result + attentionMask.contentHashCode()
        return result
    }
}

/**
 * Minimal tokenizer interface.  The main source-set declares this as a pure-Kotlin
 * interface so that [EmbeddingModel] compiles without any native dependency.
 *
 * Production implementation: [DjlTokenizer] in the main sourceset, backed by
 * `libdjl_tokenizer.so` bundled in the `ai.djl.android:tokenizer-native` AAR.
 * Test-only implementation: [HuggingFaceTokenizer] in the test sourceset, which
 * uses the macOS-native DJL binary and is never included in the Android APK.
 */
interface Tokenizer {
    /**
     * Encode [text] into token IDs and a matching attention mask.
     * CLS / BOS token should already be prepended and EOS / SEP appended by
     * the implementation so callers don't have to handle it.
     */
    fun encode(text: String): Tokenized
}
