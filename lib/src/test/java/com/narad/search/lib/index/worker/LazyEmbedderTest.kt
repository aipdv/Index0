package com.narad.search.lib.index.worker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class LazyEmbedderTest {

    private class CountingEmbedder : Embedder {
        var closed = 0
        override fun embed(text: String) = floatArrayOf(text.length.toFloat())
        override fun close() { closed++ }
    }

    @Test
    fun neverUsed_neverBuilds_andCloseIsSafe() {
        var builds = 0
        val lazy = LazyEmbedder { builds++; CountingEmbedder() }

        lazy.close()

        assertEquals(0, builds)
        assertFalse(lazy.isBuilt)
    }

    @Test
    fun firstEmbedBuilds_once_andSharedWithGet() {
        var builds = 0
        val inner = CountingEmbedder()
        val lazy = LazyEmbedder { builds++; inner }

        assertEquals(3f, lazy.embed("abc")[0], 0f)
        lazy.embed("de")
        assertSame("the photo branch must see the same instance", inner, lazy.get())

        assertEquals(1, builds)
        assertTrue(lazy.isBuilt)
    }

    @Test
    fun close_closesBuiltEmbedderOnce() {
        val inner = CountingEmbedder()
        val lazy = LazyEmbedder { inner }
        lazy.embed("x")

        lazy.close()

        assertEquals(1, inner.closed)
    }

    @Test
    fun failedBuild_isRetriedNotCached() {
        var builds = 0
        val lazy = LazyEmbedder { if (builds++ == 0) error("boom") else CountingEmbedder() }

        val failure = runCatching { lazy.embed("x") }.exceptionOrNull()
        // The build error reaches the caller (IndexWorker turns it into a failed run, not a skipped branch).
        assertEquals("boom", failure?.message)
        assertFalse(lazy.isBuilt)
        lazy.embed("x")

        assertEquals(2, builds)
    }
}
