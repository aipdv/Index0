package me.zhanghai.android.files.index

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SemanticSearchTest {

    private val docs = "/storage/emulated/0/Documents"

    @Test
    fun `hits keep their order and a file shows once`() {
        val hits = listOf("$docs/b.pdf", "$docs/a.pdf", "$docs/b.pdf")
        assertEquals(listOf("$docs/b.pdf", "$docs/a.pdf"), scopeHits(hits, docs, emptyList()))
    }

    @Test
    fun `scope matches on a slash boundary`() {
        val hits = listOf("$docs/rent.pdf", "$docs/sub/trip.txt")
        assertEquals(emptyList<String>(), scopeHits(hits, "/storage/emulated/0/Doc", emptyList()))
        assertEquals(hits, scopeHits(hits, docs, emptyList()))
        assertEquals(hits, scopeHits(hits, "$docs/", emptyList()))
    }

    @Test
    fun `unresolved and excluded hits are dropped`() {
        val hits = listOf(null, "$docs/Work/c.txt", "$docs/Home/d.txt", "$docs/.secret/b.txt")
        assertEquals(listOf("$docs/Home/d.txt"), scopeHits(hits, docs, listOf("$docs/Work")))
    }

    @Test
    fun `non local or non storage folders get no semantic rows`() {
        val hits = listOf("$docs/rent.pdf")
        assertEquals(emptyList<String>(), scopeHits(hits, null, emptyList()))
        assertEquals(emptyList<String>(), scopeHits(hits, "/", emptyList()))
        assertNull(semanticScope("/data/local/tmp"))
        assertEquals("/storage/emulated/0", semanticScope("/storage/emulated/0/"))
    }

    @Test
    fun `app open delta runs at most every two minutes`() {
        val now = 10 * 60 * 1000L
        assertEquals(true, shouldRunForegroundDelta(0L, now))
        assertEquals(false, shouldRunForegroundDelta(now - 60_000L, now))
        assertEquals(true, shouldRunForegroundDelta(now - FOREGROUND_DELTA_MIN_GAP_MS, now))
        assertEquals(true, shouldRunForegroundDelta(now + 1, now))
    }
}
