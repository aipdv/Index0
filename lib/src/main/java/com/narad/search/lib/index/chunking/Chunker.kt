package com.narad.search.lib.index.chunking

/**
 * Splits a text string into overlapping chunks suitable for embedding.
 *
 * **Algorithm:**
 * - Target chunk size: 512 characters (character-level approximation of token count;
 *   exact token counting is expensive and unnecessary for chunking purposes - the plan
 *   spec endorses `text.length / 4` as the fast approximation, but we use raw char
 *   windows to guarantee the 550-char test gate assertion).
 * - Overlap: 64 characters between consecutive chunks, so context is not lost at
 *   boundaries (Bug Watch: "keep the overlap so context isn't lost").
 * - Returns an empty list for blank/empty input.
 * - A text shorter than or equal to [CHUNK_SIZE] chars is returned as a single chunk.
 *
 * **No Android imports.** This class is pure Kotlin and runs on the host JVM.
 */
class Chunker(
    /** Maximum number of characters per chunk. */
    private val chunkSize: Int = CHUNK_SIZE,
    /** Number of characters shared between consecutive chunks. */
    private val overlapSize: Int = OVERLAP_SIZE
) {

    companion object {
        /** Default chunk window (character count approximating 512 tokens). */
        const val CHUNK_SIZE = 512
        /** Default overlap (character count approximating 64 tokens). */
        const val OVERLAP_SIZE = 64
    }

    init {
        require(chunkSize > 0) { "chunkSize must be positive, got $chunkSize" }
        require(overlapSize >= 0) { "overlapSize must be non-negative, got $overlapSize" }
        require(overlapSize < chunkSize) { "overlapSize ($overlapSize) must be < chunkSize ($chunkSize)" }
    }

    /**
     * Split [text] into overlapping chunks.
     *
     * @param text  The source text. Blank or empty input returns an empty list.
     * @return      A list of non-empty string chunks.  Each chunk is at most
     *              [chunkSize] characters long.  Consecutive chunks share a
     *              [overlapSize]-character overlap at their boundary.
     */
    fun chunk(text: String): List<String> {
        if (text.isBlank()) return emptyList()

        // Trim trailing whitespace once to avoid a spurious short trailing chunk.
        val trimmed = text.trimEnd()
        if (trimmed.isEmpty()) return emptyList()

        if (trimmed.length <= chunkSize) return listOf(trimmed)

        val result = mutableListOf<String>()
        val step = chunkSize - overlapSize   // advance per chunk (= 448 with defaults)

        var start = 0
        while (start < trimmed.length) {
            val end = minOf(start + chunkSize, trimmed.length)
            result.add(trimmed.substring(start, end))
            if (end == trimmed.length) break   // final chunk consumed
            start += step
        }

        return result
    }
}
