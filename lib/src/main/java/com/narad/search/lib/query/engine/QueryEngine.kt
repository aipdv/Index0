package com.narad.search.lib.query.engine

import com.narad.search.lib.query.store.KeywordStore
import com.narad.search.lib.query.store.VectorStore

fun interface QueryEmbedder {
    fun embedQuery(text: String): FloatArray
}

data class QueryResults(
    val documents: List<FileResult>,
    val images: List<ImageResult>
)

/**
 * Composes [QueryEmbedder], [VectorStore], and [KeywordStore] into a hybrid
 * semantic + keyword search engine.
 *
 * **Algorithm (per query):**
 * 1. Embed the query with [EmbeddingModel].
 * 2. Retrieve top-[topK] hits from [VectorStore] (cosine similarity) and
 *    top-[topK] hits from [KeywordStore] (BM25).
 * 3. Min-max normalise each list's scores to [0, 1] independently.
 * 4. Merge both lists and compute a hybrid score:
 *    `hybridScore = α × cosNorm + (1 − α) × bm25Norm`
 *    where α = 0.6 (semantic weight) by default.
 * 5. Deduplicate at the chunk level (same id from both stores → take max score
 *    for each component).
 * 6. Collapse to file level: group by filePath, keep the best hybrid score.
 * 7. Return sorted descending by hybrid score.
 *
 * @param embeddingModel    Shared query embedder (not owned; caller closes it).
 * @param vectorStore       In-memory vector store for text chunks (256-dim).
 * @param keywordStore      In-memory keyword store.
 * @param imageVectorStore  In-memory vector store for image vectors (256-dim, same space).
 *                          Pass an empty [VectorStore] if image indexing is disabled (Tier C).
 * @param alpha             Semantic weight in [0, 1]; default 0.6.
 */
class QueryEngine(
    private val embeddingModel: QueryEmbedder?,
    private val vectorStore: VectorStore,
    private val keywordStore: KeywordStore,
    private val imageVectorStore: VectorStore = VectorStore(),
    private val alpha: Float = 0.6f,
    private val fastSearchOnly: Boolean = false
) {
    private companion object {
        const val QUERY_CACHE_SIZE = 8
    }

    init {
        require(fastSearchOnly || embeddingModel != null) {
            "A query embedder is required for semantic search"
        }
    }

    // Last QUERY_CACHE_SIZE query vectors. Typing, deleting and retyping the same text, or
    // re-running a query after a refresh, then costs no model call. Access-ordered, so the
    // least recently used entry leaves first. Guarded by itself: searches run on IO threads.
    private val queryCache = object : LinkedHashMap<String, FloatArray>(QUERY_CACHE_SIZE, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, FloatArray>?) =
            size > QUERY_CACHE_SIZE
    }

    private fun embedQuery(query: String): FloatArray {
        synchronized(queryCache) { queryCache[query] }?.let { return it }
        val vector = embeddingModel!!.embedQuery(query)
        synchronized(queryCache) { queryCache[query] = vector }
        return vector
    }

    /**
     * Run one throwaway embed so the first real search does not pay the model's first-call cost.
     * Bypasses the cache. No-op in fast-search mode.
     */
    fun warmUp() {
        if (!fastSearchOnly) embeddingModel!!.embedQuery("warm up")
    }

    /**
     * Search for [query] and return deduplicated, file-level results sorted by
     * hybrid score descending.
     *
     * @param query  Natural-language query (any language supported by the model).
     * @param topK   Number of candidates to pull from each store before merging.
     *               The returned list may be shorter if fewer documents are indexed.
     */
    fun search(query: String, topK: Int = 10): List<FileResult> {
        val queryVector = if (fastSearchOnly) null else embedQuery(query)
        return searchDocuments(query, topK, queryVector)
    }

    fun searchAll(query: String, topK: Int = 10): QueryResults {
        if (fastSearchOnly) {
            return QueryResults(
                documents = searchDocuments(query, topK, null),
                images = emptyList()
            )
        }

        val queryVector = embedQuery(query)
        return QueryResults(
            documents = searchDocuments(query, topK, queryVector),
            images = if (imageVectorStore.size == 0) {
                emptyList()
            } else {
                searchImages(queryVector, topK)
            }
        )
    }

    private fun searchDocuments(
        query: String,
        topK: Int,
        queryVector: FloatArray?
    ): List<FileResult> {
        val vectorHits = queryVector?.let { vectorStore.search(it, topK) }.orEmpty()
        val keywordHits = keywordStore.search(query, topK)

        // 3. Min-max normalise each list
        val normVector  = minMaxNormVector(vectorHits)
        val normKeyword = minMaxNormKeyword(keywordHits)

        // 4 & 5. Merge by chunk id, deduplicating
        // Build a map: id → (filePath, chunkOffset, cosNorm, bm25Norm)
        data class ChunkScore(
            val filePath: String,
            val chunkOffset: Int,
            var cosNorm: Float = 0f,
            var bm25Norm: Float = 0f,
            var cosRaw: Float = 0f,
            var kwHit: Boolean = false,
            var matched: Set<String> = emptySet()
        )

        val merged = mutableMapOf<String, ChunkScore>()

        for ((i, n) in normVector.withIndex()) {
            val cs = merged.getOrPut(n.id) { ChunkScore(n.filePath, n.chunkOffset) }
            cs.cosNorm = n.score
            cs.cosRaw = vectorHits[i].score
        }
        for ((i, n) in normKeyword.withIndex()) {
            val cs = merged.getOrPut(n.id) { ChunkScore(n.filePath, n.chunkOffset) }
            cs.bm25Norm = n.score
            cs.kwHit = true
            cs.matched = keywordHits[i].matchedTerms
        }

        // Hybrid score
        // 6. Deduplicate to file level: keep best score + its chunkOffset per filePath, and
        // carry the file's best raw cosine and whether any chunk had a keyword hit.
        data class FileBest(val score: Float, val chunkOffset: Int, val cosine: Float, val keywordHit: Boolean, val matched: Set<String>)
        val fileBest = mutableMapOf<String, FileBest>()
        for (cs in merged.values) {
            val hybridScore = alpha * cs.cosNorm + (1f - alpha) * cs.bm25Norm
            val prev = fileBest[cs.filePath]
            val cosine = maxOf(prev?.cosine ?: 0f, cs.cosRaw)
            val kw = (prev?.keywordHit ?: false) || cs.kwHit
            // ponytail: matched terms only come from the top-K keyword chunks, so on a big corpus
            // a file's other term can fall out of topK and the file misses full coverage.
            // Upgrade: per-file coverage computed in KeywordStore over all chunks of a file.
            val matched = (prev?.matched ?: emptySet()) + cs.matched
            fileBest[cs.filePath] = if (prev == null || hybridScore > prev.score) {
                FileBest(hybridScore, cs.chunkOffset, cosine, kw, matched)
            } else {
                prev.copy(cosine = cosine, keywordHit = kw, matched = matched)
            }
        }

        // 7. Sort descending
        return fileBest.entries
            .sortedByDescending { it.value.score }
            .map { (path, best) ->
                FileResult(path, best.score, best.chunkOffset, best.cosine, best.keywordHit, best.matched)
            }
    }

    /**
     * Search indexed images by text query using plain cosine similarity.
     *
     * Text queries and image embeddings share the EmbeddingGemma 2 vector space,
     * so [EmbeddingModel.embedQuery] produces a compatible query vector.
     *
     * Images are NOT mixed into [search] results - they live in a separate ranked list
     * because cross-modal cosines run lower than text-text, and images have no keyword
     * score. Callers display the two lists separately in the UI.
     *
     * @param query  Natural-language query (same languages as [search]).
     * @param topK   Maximum results to return.
     * @return List of [ImageResult] sorted by cosine score descending.
     */
    fun searchImages(query: String, topK: Int = 10): List<ImageResult> {
        if (fastSearchOnly || imageVectorStore.size == 0) return emptyList()
        val queryVector = embedQuery(query)
        return searchImages(queryVector, topK)
    }

    private fun searchImages(queryVector: FloatArray, topK: Int): List<ImageResult> {
        return imageVectorStore.search(queryVector, topK).map { hit ->
            ImageResult(filePath = hit.filePath, score = hit.score)
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /** Normalised VectorResult reuses the same data class but score is in [0,1]. */
    private data class NormVectorResult(
        val id: String,
        val filePath: String,
        val chunkOffset: Int,
        val modelId: String,
        val score: Float
    )

    private data class NormKeywordResult(
        val id: String,
        val filePath: String,
        val chunkOffset: Int,
        val score: Float
    )

    private fun minMaxNormVector(hits: List<VectorResult>): List<NormVectorResult> {
        if (hits.isEmpty()) return emptyList()
        val min = hits.minOf { it.score }
        val max = hits.maxOf { it.score }
        val range = max - min
        return hits.map { r ->
            NormVectorResult(
                id = r.id,
                filePath = r.filePath,
                chunkOffset = r.chunkOffset,
                modelId = r.modelId,
                score = if (range < 1e-9f) 1f else (r.score - min) / range
            )
        }
    }

    private fun minMaxNormKeyword(hits: List<KeywordResult>): List<NormKeywordResult> {
        if (hits.isEmpty()) return emptyList()
        val min = hits.minOf { it.score }
        val max = hits.maxOf { it.score }
        val range = max - min
        return hits.map { r ->
            NormKeywordResult(
                id = r.id,
                filePath = r.filePath,
                chunkOffset = r.chunkOffset,
                score = if (range < 1e-9f) 1f else (r.score - min) / range
            )
        }
    }
}
