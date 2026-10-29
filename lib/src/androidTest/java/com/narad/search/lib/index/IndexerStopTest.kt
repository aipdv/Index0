package com.narad.search.lib.index

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository
import com.narad.search.lib.index.worker.Indexer
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * When WorkManager stops the worker (idle window ends, ~10 min limit), the Indexer must stop,
 * keep the files it finished, not save a half-embedded file, and the next run must resume
 * with the interrupted file and the rest.
 */
@RunWith(AndroidJUnit4::class)
class IndexerStopTest {

    @Test
    fun stopsBetweenFiles_andNextRunResumes() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dbName = "indexer_stop_test_${System.currentTimeMillis()}.db"
        val repository = IndexRepository(IndexDatabase(context, dbName))
        try {
            val files = (1..5).map { "/stop-test/doc$it.txt" }
            val extracted = mutableListOf<String>()
            fun indexer(stopAfter: Int) = Indexer(
                embedder = FakeEmbedder(),
                extractText = { path -> extracted.add(path); "text of $path" },
                chunkText = { listOf(it) },
                repository = repository,
                lastModifiedOf = { 1000L },
                // Becomes true once the (stopAfter+1)-th file is extracted, i.e. mid-file.
                shouldStop = { extracted.size > stopAfter }
            )

            indexer(stopAfter = 2).indexFiles(files)
            assertEquals("first run finishes 2 files and stops inside the 3rd", files.take(3), extracted)

            extracted.clear()
            indexer(stopAfter = Int.MAX_VALUE).indexFiles(files)
            assertEquals(
                "second run skips the 2 saved files and redoes the interrupted 3rd",
                files.drop(2), extracted
            )
        } finally {
            context.deleteDatabase(dbName)
        }
    }
}
