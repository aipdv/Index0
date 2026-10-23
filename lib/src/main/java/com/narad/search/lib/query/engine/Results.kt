package com.narad.search.lib.query.engine

/** A single hit from the vector store. */
data class VectorResult(
    val id: String,
    val filePath: String,
    val chunkOffset: Int,
    val modelId: String,
    /** Cosine similarity score in [0, 1] (vectors are L2-normalised). */
    val score: Float
)

/** A single hit from the keyword store. */
data class KeywordResult(
    val id: String,
    val filePath: String,
    val chunkOffset: Int,
    /** Raw BM25 score (non-negative). */
    val score: Float,
    /** Query terms (see queryTerms) that this chunk contains. */
    val matchedTerms: Set<String> = emptySet()
)

/**
 * A deduplicated, file-level search result combining vector and keyword signals.
 *
 * @param filePath    Absolute (or content-URI) path to the file.
 * @param score       Hybrid score in [0, 1] after min-max normalisation and merging.
 * @param chunkOffset Index (0-based) of the best-scoring chunk within the file.
 *                    Callers use this to load a snippet via [IndexRepository.chunkContent].
 *                    Defaults to 0 so existing callers compile without change.
 * @param cosine      Best raw cosine of any chunk in the file (0 when no vector hit).
 *                    Unlike [score] it is not rescaled per query, so callers can use it
 *                    as an absolute relevance floor.
 * @param keywordHit  True when any chunk of the file matched the query in the keyword store.
 * @param matchedTerms Union of the query terms matched by any chunk of the file.
 */
data class FileResult(
    val filePath: String,
    val score: Float,
    val chunkOffset: Int = 0,
    val cosine: Float = 0f,
    val keywordHit: Boolean = false,
    val matchedTerms: Set<String> = emptySet()
)

/**
 * A ranked image result from [QueryEngine.searchImages].
 *
 * @param filePath  Absolute path or content-URI string of the image.
 * @param score     Raw cosine similarity in [0, 1] (L2-normalised vectors).
 */
data class ImageResult(
    val filePath: String,
    val score: Float
)
