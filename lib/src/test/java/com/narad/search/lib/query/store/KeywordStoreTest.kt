package com.narad.search.lib.query.store

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class KeywordStoreTest {

    private lateinit var store: KeywordStore

    @Before
    fun setUp() {
        store = KeywordStore()
    }

    // -------------------------------------------------------------------------
    // DoD: KeywordStore.search("penalty") finds doc containing "penalty clause"
    // -------------------------------------------------------------------------
    @Test
    fun `search finds doc containing queried term`() {
        store.add("doc1", "/contracts/lease.txt", 0, "The penalty clause states that late payment incurs a 10% fee.")
        store.add("doc2", "/weather/report.txt",  0, "Sunny skies with a chance of afternoon showers.")

        val results = store.search("penalty", topK = 5)

        assertTrue("Should find at least one result", results.isNotEmpty())
        assertEquals("Top result must be doc1 (contains 'penalty')", "doc1", results[0].id)
    }

    @Test
    fun `search on empty store returns empty list`() {
        val results = store.search("penalty", topK = 5)
        assertTrue(results.isEmpty())
    }

    @Test
    fun `search with no matching term returns empty list`() {
        store.add("doc1", "/f.txt", 0, "The quick brown fox jumps over the lazy dog.")
        val results = store.search("penalty", topK = 5)
        assertTrue(results.isEmpty())
    }

    @Test
    fun `add with duplicate id replaces old entry`() {
        store.add("doc1", "/f.txt", 0, "penalty clause contract")
        store.add("doc1", "/f.txt", 0, "weather forecast rain")  // replace

        val results = store.search("penalty", topK = 5)
        assertTrue("After replacement, 'penalty' should not be found", results.isEmpty())

        val weatherResults = store.search("weather", topK = 5)
        assertEquals("doc1", weatherResults[0].id)
    }

    @Test
    fun `results are sorted descending by BM25 score`() {
        // doc1 mentions "penalty" 3 times; doc2 mentions it once
        store.add("doc1", "/a.txt", 0, "penalty penalty penalty clause in contract")
        store.add("doc2", "/b.txt", 0, "the penalty for lateness is small")

        val results = store.search("penalty", topK = 2)
        assertEquals(2, results.size)
        assertEquals("doc1 should rank first (higher TF)", "doc1", results[0].id)
        assertTrue(results[0].score >= results[1].score)
    }

    @Test
    fun `Devanagari text can be indexed and found`() {
        store.add("hindi1", "/hi/doc.txt", 0, "पेनल्टी क्लॉज़ अनुबंध में है")
        val results = store.search("पेनल्टी", topK = 5)
        assertTrue("Hindi term should be found", results.isNotEmpty())
        assertEquals("hindi1", results[0].id)
    }

    @Test
    fun `tokenize splits on non-letter-digit boundaries preserving Devanagari`() {
        val tokens = store.tokenize("penalty-clause, contract!")
        assertTrue("'penalty' should be a token", "penalty" in tokens)
        assertTrue("'clause' should be a token", "clause" in tokens)
        assertTrue("'contract' should be a token", "contract" in tokens)
    }

    @Test
    fun `tokenize preserves Devanagari codepoints`() {
        val tokens = store.tokenize("पेनल्टी क्लॉज़")
        assertTrue("Devanagari token should survive", tokens.any { it.contains('प') })
    }

    @Test
    fun `search multi-term query ranks correctly`() {
        store.add("exact", "/e.txt", 0, "the penalty clause contract dispute resolution")
        store.add("partial", "/p.txt", 0, "the penalty for speeding on the highway")

        val results = store.search("penalty clause contract", topK = 5)
        assertTrue("Should return results", results.isNotEmpty())
        assertEquals("exact match doc should rank first", "exact", results[0].id)
    }

    // -------------------------------------------------------------------------
    // U3: stopwords never drive ranking; matched terms come back with each hit
    // -------------------------------------------------------------------------
    @Test
    fun `search of a stopword alone returns no hits`() {
        store.add("doc1", "/a.txt", 0, "the train reaches the station by noon")
        assertTrue(store.search("the", topK = 5).isEmpty())
        assertTrue(store.search("by the", topK = 5).isEmpty())
    }

    @Test
    fun `stopwords are dropped from a mixed query so only the real word ranks`() {
        // "the" is in every doc, so it must not pull the unrelated doc in.
        store.add("bill", "/bill.txt", 0, "electricity bill due this month")
        store.add("noise", "/noise.txt", 0, "the cat sat by the window")
        val results = store.search("the bill", topK = 5)
        assertEquals(listOf("bill"), results.map { it.id })
    }

    @Test
    fun `hinglish filler words are dropped`() {
        store.add("a", "/a.txt", 0, "ghar ka kharcha hai aur kya")
        assertTrue(store.search("ka hai aur kya ki", topK = 5).isEmpty())
        assertEquals(listOf("a"), store.search("ghar ka kharcha", topK = 5).map { it.id })
    }

    @Test
    fun `hit carries the query terms it matched`() {
        store.add("c1", "/f.txt", 0, "bijli connection details")
        store.add("c2", "/g.txt", 0, "water bill and bijli")
        val results = store.search("bijli bill", topK = 5).associateBy { it.id }
        assertEquals(setOf("bijli"), results.getValue("c1").matchedTerms)
        assertEquals(setOf("bijli", "bill"), results.getValue("c2").matchedTerms)
    }

    @Test
    fun `queryTerms lowercases splits on punctuation drops stopwords and dedupes`() {
        assertEquals(listOf("temple", "river"), queryTerms(" Temple, by the  river! temple"))
        assertEquals(emptyList<String>(), queryTerms("a the"))
        assertEquals(listOf("tv"), queryTerms("TV"))
    }

    @Test
    fun `queryTerms keeps Devanagari words whole and drops Devanagari stopwords`() {
        assertEquals(listOf("बिजली", "बिल"), queryTerms("बिजली बिल"))
        assertEquals(listOf("बिल"), queryTerms("बिल का है"))
    }

    @Test
    fun `unrelated Hindi document does not cover a Hindi query`() {
        // The old split cut "बिल" into "ब" and "ल"; this text holds both letters-as-words.
        store.add("weather", "/hi/w.txt", 0, "आज मौसम बहुत अच्छा है, बात करने के लिए समय नहीं")
        store.add("bill", "/hi/b.txt", 0, "बिजली बिल जमा करना है")
        val results = store.search("बिल", topK = 5)
        assertEquals(listOf("bill"), results.map { it.id })
        assertEquals(setOf("बिल"), results[0].matchedTerms)
    }
}
