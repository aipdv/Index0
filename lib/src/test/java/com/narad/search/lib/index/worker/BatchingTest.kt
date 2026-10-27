package com.narad.search.lib.index.worker

import com.narad.search.lib.device.Tier
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

private fun vec(text: String) = floatArrayOf(text.hashCode().toFloat())

class BatchingTest {

    /** Counts calls. Each text gets its own vector: [text.hashCode()]. */
    private class CountingEmbedder : Embedder {
        var embedCalls = 0
        val batches = mutableListOf<List<String>>()
        override fun embed(text: String): FloatArray { embedCalls++; return vec(text) }
        override fun embedBatch(texts: List<String>): List<FloatArray> { batches += texts; return texts.map(::vec) }
        override fun close() {}
    }

    private val chunks = (0 until 19).map { "chunk $it" }

    @Test
    fun embedChunks_callsEmbedBatchPerBatch_neverEmbed_andKeepsOrder() {
        val fake = CountingEmbedder()

        val out = embedChunks(chunks, 8, fake) { false }!!

        assertEquals(0, fake.embedCalls)
        assertEquals(listOf(8, 8, 3), fake.batches.map { it.size }) // ceil(19 / 8) = 3 calls
        assertEquals(chunks.map { vec(it)[0] }, out.map { it[0] })
    }

    @Test
    fun embedChunks_batchOne_isOneCallPerChunk() {
        val fake = CountingEmbedder()
        embedChunks(chunks, 1, fake) { false }
        assertEquals(chunks.size, fake.batches.size)
    }

    @Test
    fun embedChunks_stopAfterFirstBatch_returnsNull() {
        val fake = CountingEmbedder()
        var checks = 0

        val out = embedChunks(chunks, 8, fake) { checks++ > 0 }

        assertNull("stopped: the caller must leave the file unmarked", out)
        assertEquals(1, fake.batches.size)
    }

    @Test
    fun lazyEmbedder_forwardsEmbedBatch_once() {
        val fake = CountingEmbedder()
        val lazy = LazyEmbedder { fake }

        lazy.embedBatch(listOf("a", "b", "c"))

        assertEquals(listOf(listOf("a", "b", "c")), fake.batches)
        assertEquals("the interface default (one embed per text) must not run", 0, fake.embedCalls)
    }

    @Test
    fun tokenBudget_byTier() {
        assertEquals(1024, tokenBudget(Tier.C))
        assertEquals(2048, tokenBudget(Tier.B))
        assertEquals(2048, tokenBudget(Tier.A))
    }
}
