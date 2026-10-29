package com.narad.search.lib.query.engine

import com.narad.search.lib.query.TestModel
import com.narad.search.lib.query.model.EmbeddingModel
import com.narad.search.lib.query.model.HuggingFaceTokenizer
import com.narad.search.lib.query.store.KeywordStore
import com.narad.search.lib.query.store.VectorStore
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class QueryEngineTest {

    private lateinit var tokenizer: HuggingFaceTokenizer
    private lateinit var embeddingModel: EmbeddingModel
    private lateinit var vectorStore: VectorStore
    private lateinit var keywordStore: KeywordStore
    private lateinit var engine: QueryEngine

    @Before
    fun setUp() {
        TestModel.skipIfMissing()
        tokenizer      = HuggingFaceTokenizer(TestModel.tokenizerJsonPath)
        embeddingModel = EmbeddingModel(TestModel.onnxPath, tokenizer)
        vectorStore    = VectorStore()
        keywordStore   = KeywordStore()
        engine         = QueryEngine(embeddingModel, vectorStore, keywordStore)
    }

    @After
    fun tearDown() {
        if (::embeddingModel.isInitialized) embeddingModel.close()
    }

    // -------------------------------------------------------------------------
    // DoD: Hybrid rerank - semantically-close doc beats exact-keyword-only doc
    //
    // Setup:
    //   "semClose"  - content is semantically similar to "penalty clause" but
    //                 does NOT contain those exact words.  Only in vector store.
    //   "kwOnly"    - content contains the literal word "penalty" but is about a
    //                 completely different topic (sports).  Only in keyword store.
    //
    // The hybrid engine must rank "semClose" above "kwOnly" because α=0.6 gives
    // more weight to semantic similarity.
    // -------------------------------------------------------------------------
    @Test
    fun `hybrid semantically close doc ranks above exact keyword only doc`() {
        // embedDocument applies the doc prompt; embedQuery (via engine.search) applies query prompt.
        val semCloseText = "The contractual fine for breach of agreement is ten percent of total value."
        val semCloseVec  = embeddingModel.embedDocument(semCloseText)

        val kwOnlyText = "The soccer team was awarded a penalty kick after the foul."

        // Index semClose in VectorStore only (no keyword entry)
        vectorStore.add("semClose", "/contracts/breach.txt", 0, "embeddinggemma-2", semCloseVec)

        // Index kwOnly in KeywordStore only (no vector entry)
        keywordStore.add("kwOnly", "/sports/soccer.txt", 0, kwOnlyText)

        val results = engine.search("penalty clause", topK = 10)

        val semCloseResult = results.find { it.filePath == "/contracts/breach.txt" }
        val kwOnlyResult   = results.find { it.filePath == "/sports/soccer.txt" }

        assertNotNull("semClose doc must appear in results", semCloseResult)
        assertNotNull("kwOnly doc must appear in results",   kwOnlyResult)

        println("semClose score = ${semCloseResult!!.score}")
        println("kwOnly   score = ${kwOnlyResult!!.score}")

        assertTrue(
            "Semantically-close doc (${semCloseResult.score}) must rank above " +
                "exact-keyword-only doc (${kwOnlyResult.score})",
            semCloseResult.score > kwOnlyResult.score
        )
    }

    // -------------------------------------------------------------------------
    // DoD: Dedup - two chunks from the same file collapse to one FileResult
    // -------------------------------------------------------------------------
    @Test
    fun `dedup two chunks from same file collapse to one FileResult`() {
        val textA = "The penalty clause states ten percent fine for breach."
        val textB = "Late delivery triggers the penalty as stipulated in section 5."

        val vecA = embeddingModel.embedDocument(textA)
        val vecB = embeddingModel.embedDocument(textB)

        vectorStore.add("chunk0", "/contracts/lease.txt", 0,   "embeddinggemma-2", vecA)
        vectorStore.add("chunk1", "/contracts/lease.txt", 512, "embeddinggemma-2", vecB)

        keywordStore.add("chunk0", "/contracts/lease.txt", 0,   textA)
        keywordStore.add("chunk1", "/contracts/lease.txt", 512, textB)

        val results = engine.search("penalty clause", topK = 10)

        val leaseResults = results.filter { it.filePath == "/contracts/lease.txt" }
        assertEquals(
            "Two chunks from the same file must collapse to exactly one FileResult",
            1,
            leaseResults.size
        )
    }

    // -------------------------------------------------------------------------
    // Basic sanity: search on empty stores returns empty list
    // -------------------------------------------------------------------------
    @Test
    fun `search on empty stores returns empty list`() {
        val results = engine.search("penalty clause", topK = 10)
        assertTrue(results.isEmpty())
    }

    @Test
    fun `fast search uses keywords without embedding`() {
        var embedCalls = 0
        keywordStore.add("doc", "/contract.txt", 0, "late payment penalty")
        val fastEngine = QueryEngine(
            embeddingModel = QueryEmbedder {
                embedCalls++
                floatArrayOf(1f, 0f)
            },
            vectorStore = vectorStore,
            keywordStore = keywordStore,
            fastSearchOnly = true
        )

        val results = fastEngine.searchAll("penalty", topK = 10)

        assertEquals(0, embedCalls)
        assertEquals(listOf("/contract.txt"), results.documents.map { it.filePath })
        assertTrue(results.images.isEmpty())
    }

    @Test
    fun `combined search embeds once and skips empty photo store`() {
        var embedCalls = 0
        vectorStore.add("doc", "/contract.txt", 0, "embeddinggemma-2", floatArrayOf(1f, 0f))
        val sharedEngine = QueryEngine(
            embeddingModel = QueryEmbedder {
                embedCalls++
                floatArrayOf(1f, 0f)
            },
            vectorStore = vectorStore,
            keywordStore = keywordStore
        )

        val results = sharedEngine.searchAll("contract", topK = 10)

        assertEquals(1, embedCalls)
        assertEquals(listOf("/contract.txt"), results.documents.map { it.filePath })
        assertTrue(results.images.isEmpty())
    }

    @Test
    fun `combined document and photo search shares one embedding`() {
        var embedCalls = 0
        val imageStore = VectorStore().apply {
            add("photo", "content://photos/1", 0, "embeddinggemma-2", floatArrayOf(1f, 0f))
        }
        val sharedEngine = QueryEngine(
            embeddingModel = QueryEmbedder {
                embedCalls++
                floatArrayOf(1f, 0f)
            },
            vectorStore = vectorStore,
            keywordStore = keywordStore,
            imageVectorStore = imageStore
        )

        val results = sharedEngine.searchAll("photo", topK = 10)

        assertEquals(1, embedCalls)
        assertEquals(listOf("content://photos/1"), results.images.map { it.filePath })
    }

    // -------------------------------------------------------------------------
    // FileResult.chunkOffset equals the best-scoring chunk's index (fix 3)
    //
    // Two chunks from the same file; chunkOffset=2 is the query-relevant one.
    // QueryEngine must collapse to one FileResult whose chunkOffset is 2.
    // -------------------------------------------------------------------------
    @Test
    fun `FileResult chunkOffset is the best-scoring chunk index`() {
        val relevantText  = "The penalty clause states ten percent fine for breach of contract."
        val irrelevantText = "Unrelated content about weather, sunshine, and seasonal forecasts."

        val relevantVec   = embeddingModel.embedDocument(relevantText)
        val irrelevantVec = embeddingModel.embedDocument(irrelevantText)

        // chunkOffset 2 is relevant; chunkOffset 0 is irrelevant.
        // Irrelevant chunk inserted FIRST so "keep first chunk" logic would return chunkOffset=0.
        vectorStore.add("doc#0", "/contracts/lease.txt", 0, "embeddinggemma-2", irrelevantVec)
        vectorStore.add("doc#2", "/contracts/lease.txt", 2, "embeddinggemma-2", relevantVec)

        keywordStore.add("doc#0", "/contracts/lease.txt", 0, irrelevantText)
        keywordStore.add("doc#2", "/contracts/lease.txt", 2, relevantText)

        val results = engine.search("penalty clause", topK = 10)
        val leaseResult = results.find { it.filePath == "/contracts/lease.txt" }

        assertNotNull("lease.txt must appear in results", leaseResult)
        assertEquals(
            "FileResult.chunkOffset must be 2 (the best-scoring chunk index); " +
            "got ${leaseResult!!.chunkOffset}",
            2,
            leaseResult.chunkOffset
        )
    }

    // -------------------------------------------------------------------------
    // Basic ranking: the best-matching doc should rank first
    // -------------------------------------------------------------------------
    @Test
    fun `best matching doc ranks first`() {
        val contractText = "penalty clause breach of contract fine ten percent"
        val weatherText  = "sunny weather forecast rain clouds wind"

        val contractVec = embeddingModel.embedDocument(contractText)
        val weatherVec  = embeddingModel.embedDocument(weatherText)

        vectorStore.add("c1", "/c.txt", 0, "embeddinggemma-2", contractVec)
        vectorStore.add("w1", "/w.txt", 0, "embeddinggemma-2", weatherVec)

        keywordStore.add("c1", "/c.txt", 0, contractText)
        keywordStore.add("w1", "/w.txt", 0, weatherText)

        val results = engine.search("penalty clause", topK = 5)
        assertTrue("Should return at least 2 results", results.size >= 2)
        assertEquals("Contract doc should rank first", "/c.txt", results[0].filePath)
    }

    // -------------------------------------------------------------------------
    // DoD "Success Looks Like": Hindi query "पेनल्टी क्लॉज़" → contract ranks #1.
    //
    // Seeds 3 documents with REAL embeddings from EmbeddingGemma 2 using
    // embedDocument() (with doc prompt). QueryEngine uses embedQuery() (with query
    // prompt) internally - asymmetric dual-embed as designed.
    //
    // EG2 is proven multilingual (16/16 on trial). This assertion must not be weakened.
    // -------------------------------------------------------------------------
    @Test
    fun `hindi query penalty clause ranks contract document first`() {
        val contractText = "The rental agreement includes a penalty clause: ten percent of rent " +
            "due as a fine for breach of contract or late payment as stipulated in section 7."
        val weatherText  = "Sunny weather forecast for the week with light rain expected on Thursday " +
            "and strong winds along the coastal areas."
        val invoiceText  = "Quarterly invoice total for Q3 amounts to fifty thousand rupees including " +
            "GST. Payment due within thirty days of invoice date."

        vectorStore.add("c1", "/contracts/rental.txt",  0, "embeddinggemma-2", embeddingModel.embedDocument(contractText))
        vectorStore.add("w1", "/weather/forecast.txt",  0, "embeddinggemma-2", embeddingModel.embedDocument(weatherText))
        vectorStore.add("i1", "/invoices/q3.txt",       0, "embeddinggemma-2", embeddingModel.embedDocument(invoiceText))

        keywordStore.add("c1", "/contracts/rental.txt",  0, contractText)
        keywordStore.add("w1", "/weather/forecast.txt",  0, weatherText)
        keywordStore.add("i1", "/invoices/q3.txt",       0, invoiceText)

        // QueryEngine.search() calls embedQuery("पेनल्टी क्लॉज़") internally.
        val results = engine.search("पेनल्टी क्लॉज़", topK = 10)

        assertTrue("Should return at least 3 results", results.size >= 3)

        results.forEachIndexed { idx, r ->
            println("Hindi query rank ${idx + 1}: ${r.filePath}  score=${r.score}")
        }

        assertEquals(
            "Hindi query 'पेनल्टी क्लॉज़' must rank the contract document #1 " +
                "(actual ranking: ${results.map { it.filePath }})",
            "/contracts/rental.txt",
            results[0].filePath
        )
    }
}

/** Matched-term union needs no embedding model, so it runs in the model-free JVM gate. */
class QueryEngineMatchedTermsTest {

    private fun fastEngine(store: KeywordStore) = QueryEngine(
        embeddingModel = null,
        vectorStore = VectorStore(),
        keywordStore = store,
        fastSearchOnly = true
    )

    @Test
    fun `matched terms are the union across all chunks of a file`() {
        val kw = KeywordStore()
        kw.add("c0", "/ghar.txt", 0, "bijli connection number 4521")
        kw.add("c1", "/ghar.txt", 1, "monthly bill and rent")
        kw.add("other", "/other.txt", 0, "bill of lading")

        val results = fastEngine(kw).search("bijli bill", topK = 10).associateBy { it.filePath }

        assertEquals(setOf("bijli", "bill"), results.getValue("/ghar.txt").matchedTerms)
        assertEquals(setOf("bill"), results.getValue("/other.txt").matchedTerms)
    }

    @Test
    fun `stopword only query finds nothing`() {
        val kw = KeywordStore()
        kw.add("c0", "/a.txt", 0, "the quick brown fox")
        assertTrue(fastEngine(kw).search("the", topK = 10).isEmpty())
    }
}

/**
 * Query vector cache. Uses a counting fake embedder, so it needs no model files and runs
 * even with NARAD_SKIP_MODEL_TESTS=1.
 */
class QueryEngineQueryCacheTest {

    private val embedded = mutableListOf<String>()

    private fun engine(): QueryEngine {
        val keywords = KeywordStore().apply { add("d", "/a.txt", 0, "penalty clause") }
        return QueryEngine(
            embeddingModel = QueryEmbedder { text -> embedded += text; floatArrayOf(1f, 0f) },
            vectorStore = VectorStore(),
            keywordStore = keywords,
        )
    }

    @Test
    fun `same query twice embeds once`() {
        val engine = engine()

        val first = engine.searchAll("penalty", topK = 5)
        val second = engine.searchAll("penalty", topK = 5)

        assertEquals(listOf("penalty"), embedded)
        assertEquals(first.documents, second.documents)
    }

    @Test
    fun `search and searchAll share the cache`() {
        val engine = engine()

        engine.search("penalty", topK = 5)
        engine.searchAll("penalty", topK = 5)

        assertEquals(1, embedded.size)
    }

    @Test
    fun `cache keeps the last eight queries and drops the least recently used`() {
        val engine = engine()
        (1..8).forEach { engine.search("q$it", topK = 1) }
        engine.search("q1", topK = 1)      // q1 is now the most recent
        engine.search("q9", topK = 1)      // evicts q2, the least recent
        embedded.clear()

        engine.search("q1", topK = 1)      // still cached
        assertEquals(emptyList<String>(), embedded)
        engine.search("q2", topK = 1)      // evicted, embeds again
        assertEquals(listOf("q2"), embedded)
    }

    @Test
    fun `warmUp embeds once and does not fill the cache`() {
        val engine = engine()

        engine.warmUp()
        engine.search("penalty", topK = 5)

        assertEquals(listOf("warm up", "penalty"), embedded)
    }
}
