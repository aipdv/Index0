package com.narad.search.lib.index.chunking

import org.junit.Assert.*
import org.junit.Test

class ChunkerTest {

    private val chunker = Chunker()

    // -------------------------------------------------------------------------
    // DoD: chunk("") → empty list
    // -------------------------------------------------------------------------
    @Test
    fun `chunk empty string returns empty list`() {
        val result = chunker.chunk("")
        assertTrue("Expected empty list for empty input", result.isEmpty())
    }

    // -------------------------------------------------------------------------
    // DoD: chunk("short") → exactly one chunk
    // -------------------------------------------------------------------------
    @Test
    fun `chunk short text returns exactly one chunk`() {
        val result = chunker.chunk("short")
        assertEquals("Expected exactly one chunk for short text", 1, result.size)
        assertEquals("Chunk content should match input", "short", result[0])
    }

    // -------------------------------------------------------------------------
    // DoD: chunk(longText): every chunk ≤ 550 chars
    // -------------------------------------------------------------------------
    @Test
    fun `chunk long text produces chunks each at most 550 chars`() {
        val longText = "a".repeat(3000)
        val chunks = chunker.chunk(longText)
        assertTrue("Expected at least 4 chunks for 3000-char input", chunks.size >= 4)
        for ((index, chunk) in chunks.withIndex()) {
            assertTrue(
                "Chunk $index has ${chunk.length} chars, expected ≤ 550",
                chunk.length <= 550
            )
        }
    }

    // -------------------------------------------------------------------------
    // DoD: chunk(longText): consecutive chunks overlap by ~64 chars
    // The overlap region from the end of chunk[i] should appear at the start
    // of chunk[i+1].
    // -------------------------------------------------------------------------
    @Test
    fun `chunk long text has ~64 char overlap between consecutive chunks`() {
        // Use a realistic text (not all-same chars) so overlap content is meaningful.
        val words = (1..600).map { "word$it" }.joinToString(" ")
        val chunks = chunker.chunk(words)

        assertTrue("Expected at least 2 chunks for overlap test", chunks.size >= 2)

        for (i in 0 until chunks.size - 1) {
            val current = chunks[i]
            val next = chunks[i + 1]

            // The last N chars of current should appear at the start of next,
            // where N is the overlap window (≈64 chars).
            // We test with a 32-char tolerance: overlap should be between 32 and 128 chars.
            val overlapFound = findOverlapLength(current, next)
            assertTrue(
                "Chunk $i → ${i + 1}: expected overlap ≥ 32 chars, found $overlapFound chars",
                overlapFound >= 32
            )
            assertTrue(
                "Chunk $i → ${i + 1}: expected overlap ≤ 128 chars, found $overlapFound chars",
                overlapFound <= 128
            )
        }
    }

    // -------------------------------------------------------------------------
    // Verify exact overlap region matches between consecutive chunks.
    // Use a text where we can predict the boundary.
    // -------------------------------------------------------------------------
    @Test
    fun `overlap region of consecutive chunks matches exactly`() {
        // Create text whose chunk boundaries we can reason about.
        // Chunker uses 512-char window with 64-char overlap.
        // Build a text of exactly 800 chars so we get at least 2 chunks.
        val text = (1..160).joinToString("") { i -> ('A' + (i % 26)).toString().repeat(5) }
        assertEquals("Test setup: text should be 800 chars", 800, text.length)

        val chunks = chunker.chunk(text)
        assertTrue("Expected at least 2 chunks from 800-char text", chunks.size >= 2)

        // The suffix of chunks[0] that equals the prefix of chunks[1] is the overlap.
        val overlapLen = findOverlapLength(chunks[0], chunks[1])
        assertTrue("Overlap must be at least 32 chars", overlapLen >= 32)

        val tailOfFirst = chunks[0].takeLast(overlapLen)
        val headOfSecond = chunks[1].take(overlapLen)
        assertEquals(
            "The overlap region from the end of chunk[0] must match the start of chunk[1]",
            tailOfFirst,
            headOfSecond
        )
    }

    // -------------------------------------------------------------------------
    // Whitespace/blank text: should behave like empty
    // -------------------------------------------------------------------------
    @Test
    fun `chunk blank whitespace returns empty list`() {
        val result = chunker.chunk("   \n\t  ")
        assertTrue("Expected empty list for blank text", result.isEmpty())
    }

    // -------------------------------------------------------------------------
    // Single long word exactly at the chunk boundary
    // -------------------------------------------------------------------------
    @Test
    fun `chunk text at exactly 512 chars returns one chunk`() {
        val text = "x".repeat(512)
        val chunks = chunker.chunk(text)
        assertEquals("512 chars should fit in exactly one chunk", 1, chunks.size)
        assertEquals("Chunk content should be preserved", text, chunks[0])
    }

    @Test
    fun `chunk text at 513 chars returns two chunks`() {
        val text = "x".repeat(513)
        val chunks = chunker.chunk(text)
        assertEquals("513 chars should require two chunks", 2, chunks.size)
    }

    // -------------------------------------------------------------------------
    // Helper: find the longest suffix of `a` that is a prefix of `b`.
    // -------------------------------------------------------------------------
    private fun findOverlapLength(a: String, b: String): Int {
        val maxCheck = minOf(a.length, b.length, 256) // check up to 256 chars
        for (len in maxCheck downTo 1) {
            if (a.endsWith(b.substring(0, len))) return len
        }
        return 0
    }
}
