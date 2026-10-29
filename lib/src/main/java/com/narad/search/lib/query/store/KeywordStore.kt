package com.narad.search.lib.query.store

import com.narad.search.lib.query.engine.KeywordResult
import kotlin.math.ln

/**
 * Words that carry no meaning for search: English, Hinglish and Devanagari. The single shared list: the
 * keyword ranking, the snippet window and the app's highlight terms all drop these from the
 * QUERY. The index still stores them, so no reindex is needed.
 */
val QUERY_STOPWORDS: Set<String> = setOf(
    // English
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "is", "are", "was", "were", "be", "it", "this", "that", "as", "if", "so", "my", "me", "i",
    // Hinglish
    "hai", "hain", "ho", "tha", "thi", "aur", "ya", "kya", "ka", "ki", "ke", "ko", "se", "mein",
    "par", "pe", "wali", "wala", "wale", "ek", "yeh", "ye", "woh", "wo", "na", "bhi",
    // Devanagari, escaped: hai, ka, ki, ke, aur, mein, se, ko, par
    "\u0939\u0948", "\u0915\u093E", "\u0915\u0940", "\u0915\u0947", "\u0914\u0930",
    "\u092E\u0947\u0902", "\u0938\u0947", "\u0915\u094B", "\u092A\u0930"
)

private val TOKEN_SPLIT = Regex("[^\\p{L}\\p{M}\\p{N}]+")

/**
 * Lowercase and split on characters that are not letters, combining marks or digits. Marks
 * stay inside a token so Devanagari matras and the virama do not cut a word apart. KeywordStore
 * is in memory and rebuilt from the stored chunk text on every start, so no reindex is needed.
 */
internal fun tokenizeText(text: String): List<String> =
    text.lowercase().split(TOKEN_SPLIT).filter { it.isNotEmpty() }

/**
 * The meaningful words of a search [query]: lowercased, split like the index, stopwords
 * removed, duplicates dropped, query order kept.
 */
fun queryTerms(query: String): List<String> =
    tokenizeText(query).filter { it !in QUERY_STOPWORDS }.distinct()

/**
 * In-memory keyword store backed by an inverted index with BM25 scoring.
 *
 * Tokenisation: lowercase + split on runs of characters that are not a letter,
 * combining mark or digit (see [tokenizeText]). Keeping marks keeps Devanagari
 * words whole, and CJK and other Unicode letters pass through.
 *
 * BM25 parameters follow the standard Okapi BM25 defaults:
 *   - k1 = 1.5
 *   - b  = 0.75
 *
 * On-device persistence is deferred to U4 (Decision B2).
 */
class KeywordStore {

    // BM25 hyper-parameters
    private val k1 = 1.5f
    private val b  = 0.75f

    private data class Document(
        val id: String,
        val filePath: String,
        val chunkOffset: Int,
        /** Term-frequency map for this document (lower-cased tokens). */
        val tf: Map<String, Int>,
        val length: Int            // total token count
    )

    /** docId → Document */
    private val docs = mutableMapOf<String, Document>()

    /** term → set of docIds that contain it */
    private val invertedIndex = mutableMapOf<String, MutableSet<String>>()

    /** Running sum of all document lengths - used to compute average. */
    private var totalLength = 0L

    /**
     * Add (or update) a document.
     *
     * Re-inserting a document with the same [id] replaces the old entry,
     * keeping the index idempotent.
     *
     * @param id          Unique chunk identifier.
     * @param filePath    Absolute path to the source file.
     * @param chunkOffset Byte or token offset inside the file.
     * @param content     Raw text content of this chunk.
     */
    fun add(
        id: String,
        filePath: String,
        chunkOffset: Int,
        content: String
    ) {
        // Remove the old entry if it exists, so re-indexing is idempotent.
        remove(id)

        val tokens = tokenize(content)
        val tf = tokens.groupingBy { it }.eachCount()
        val doc = Document(id, filePath, chunkOffset, tf, tokens.size)

        docs[id] = doc
        totalLength += tokens.size
        for (term in tf.keys) {
            invertedIndex.getOrPut(term) { mutableSetOf() }.add(id)
        }
    }

    /**
     * Return the [topK] documents most relevant to [query] according to BM25,
     * sorted descending.
     *
     * Returns an empty list when the store contains no documents.
     */
    fun search(query: String, topK: Int): List<KeywordResult> {
        if (docs.isEmpty()) return emptyList()

        val queryTerms = queryTerms(query)
        if (queryTerms.isEmpty()) return emptyList()

        val N = docs.size.toDouble()
        val avgDl = if (docs.isEmpty()) 1.0 else totalLength.toDouble() / docs.size

        // Accumulate BM25 scores per docId
        val scores = mutableMapOf<String, Double>()
        val matched = mutableMapOf<String, MutableSet<String>>()

        for (term in queryTerms) {
            val matchingIds = invertedIndex[term] ?: continue
            val df = matchingIds.size.toDouble()
            // IDF - smooth BM25 variant: ln((N - df + 0.5) / (df + 0.5) + 1)
            val idf = ln((N - df + 0.5) / (df + 0.5) + 1.0)

            for (docId in matchingIds) {
                val doc = docs[docId] ?: continue
                val freq = (doc.tf[term] ?: 0).toDouble()
                val dl = doc.length.toDouble()
                val tf = (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * dl / avgDl))
                scores[docId] = (scores[docId] ?: 0.0) + idf * tf
                matched.getOrPut(docId) { linkedSetOf() }.add(term)
            }
        }

        return scores.entries
            .sortedByDescending { it.value }
            .take(topK)
            .mapNotNull { (docId, score) ->
                val doc = docs[docId] ?: return@mapNotNull null
                KeywordResult(
                    id = docId,
                    filePath = doc.filePath,
                    chunkOffset = doc.chunkOffset,
                    score = score.toFloat(),
                    matchedTerms = matched[docId].orEmpty()
                )
            }
    }

    /** Number of documents currently stored. */
    val size: Int get() = docs.size

    /** Remove all documents. */
    fun clear() {
        docs.clear()
        invertedIndex.clear()
        totalLength = 0L
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /** Lowercase and split on characters that are not letters, marks or digits. */
    internal fun tokenize(text: String): List<String> = tokenizeText(text)

    /** Remove a document from the index, updating all data structures. */
    private fun remove(id: String) {
        val doc = docs.remove(id) ?: return
        totalLength -= doc.length
        for (term in doc.tf.keys) {
            val set = invertedIndex[term] ?: continue
            set.remove(id)
            if (set.isEmpty()) invertedIndex.remove(term)
        }
    }
}
