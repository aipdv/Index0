package com.narad.search.lib.index

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository
import com.narad.search.lib.index.worker.Embedder
import com.narad.search.lib.index.worker.Indexer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

/**
 * One file whose embedding throws must not fail the run: the other files are indexed, the bad
 * file is marked (not retried until it changes), and a broken model still ends the run.
 */
@RunWith(AndroidJUnit4::class)
class IndexerEmbedFailTest {

    /** Throws for any text that contains "bad". Counts calls. */
    private class ThrowingEmbedder : Embedder {
        val seen = mutableListOf<String>()
        private val inner = FakeEmbedder()
        override fun embed(text: String): FloatArray {
            seen += text
            if ("bad" in text) throw IllegalStateException("model rejected the input")
            return inner.embed(text)
        }
        override fun close() {}
    }

    private fun withRepository(block: (IndexRepository) -> Unit) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dbName = "indexer_embed_fail_test_${System.currentTimeMillis()}.db"
        try { block(IndexRepository(IndexDatabase(context, dbName))) } finally { context.deleteDatabase(dbName) }
    }

    @Test
    fun badFile_isSkippedAndMarked_runCompletes_noRetry() = withRepository { repository ->
        val files = listOf("/ef/a.txt", "/ef/bad.txt", "/ef/c.txt", "/ef/d.txt")
        val embedder = ThrowingEmbedder()
        val failed = mutableListOf<String>()
        val done = mutableMapOf<String, Int>()
        fun indexer() = Indexer(
            embedder = embedder,
            extractText = { path -> "text of $path" },
            chunkText = { listOf(it) },
            repository = repository,
            lastModifiedOf = { 1000L },
            onFileDone = { path, _, chunks, _, _ -> done[path] = chunks },
            onEmbedFail = { path, _ -> failed += path },
        )

        assertTrue("the run completes", indexer().indexFiles(files))

        assertEquals(listOf("/ef/bad.txt"), failed)
        assertEquals(mapOf("/ef/a.txt" to 1, "/ef/bad.txt" to 0, "/ef/c.txt" to 1, "/ef/d.txt" to 1), done)
        assertEquals("the other 3 files are indexed", 3, repository.chunkCount())
        assertFalse("the bad file counts as handled", repository.fileNeedsIndex("/ef/bad.txt", 1000L))
        assertTrue("it is read again once it changes", repository.fileNeedsIndex("/ef/bad.txt", 2000L))

        embedder.seen.clear()
        indexer().indexFiles(files)
        assertTrue("no retry loop: the second run embeds nothing", embedder.seen.isEmpty())
    }

    @Test
    fun brokenModel_stillFailsTheRun() = withRepository { repository ->
        val files = (1..5).map { "/ef/bad$it.txt" }
        val indexer = Indexer(
            embedder = ThrowingEmbedder(),
            extractText = { path -> "text of $path" },
            chunkText = { listOf(it) },
            repository = repository,
            lastModifiedOf = { 1000L },
        )
        try {
            indexer.indexFiles(files)
            fail("3 failures in a row must end the run")
        } catch (e: IllegalStateException) {
            assertTrue("files after the limit stay unmarked", repository.fileNeedsIndex("/ef/bad4.txt", 1000L))
        }
    }
}
