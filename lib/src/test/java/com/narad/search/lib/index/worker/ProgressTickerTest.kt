package com.narad.search.lib.index.worker

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ProgressTickerTest {

    private var now = 0L
    private fun ticker() = ProgressTicker(everyFiles = 50, everyMs = 1_000, nowMs = { now })

    @Test
    fun `ticks at every 50th file even when files are fast`() {
        val t = ticker()
        assertFalse(t.due(1))
        assertFalse(t.due(49))
        assertTrue(t.due(50))
        assertTrue(t.due(100))
    }

    @Test
    fun `ticks after one second even before 50 files`() {
        val t = ticker()
        now = 999
        assertFalse(t.due(3))
        now = 1_000
        assertTrue(t.due(4))
        now = 1_500
        assertFalse(t.due(5))
        now = 2_000
        assertTrue(t.due(6))
    }

    @Test
    fun `no tick at zero and final is pending only for an unpublished count`() {
        val t = ticker()
        now = 5_000
        assertFalse(t.due(0))
        assertFalse(t.pending(0))
        assertTrue(t.due(7))
        assertFalse(t.pending(7))
        assertTrue(t.pending(8))
    }
}
