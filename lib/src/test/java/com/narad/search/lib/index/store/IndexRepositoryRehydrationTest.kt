package com.narad.search.lib.index.store

import com.narad.search.lib.query.store.KeywordStore
import com.narad.search.lib.query.store.VectorStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class IndexRepositoryRehydrationTest {

    @Test
    fun `keyword-only rehydration does not retain vectors`() {
        val vectorStore = VectorStore()
        val keywordStore = KeywordStore()

        rehydrateChunk(
            id = "doc#0",
            filePath = "/docs/contract.txt",
            chunkOffset = 0,
            modelId = "",
            vectorBlob = null,
            content = "late payment penalty",
            vectorStore = vectorStore,
            keywordStore = keywordStore
        )

        assertEquals(0, vectorStore.size)
        assertEquals(1, keywordStore.size)
        assertTrue(keywordStore.search("penalty", 10).isNotEmpty())
    }
}
