package com.narad.search.lib.query.store

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.sqrt

class VectorStoreTest {

    private lateinit var store: VectorStore

    @Before
    fun setUp() {
        store = VectorStore()
    }

    // -------------------------------------------------------------------------
    // DoD: VectorStore.search returns closest vector as rank 1 across 5 entries
    // -------------------------------------------------------------------------
    @Test
    fun `search returns closest vector as rank 1 across 5 seeded entries`() {
        // Seed 5 distinct unit vectors in 384-dim space (most dims are 0)
        val dim = 384

        // Vector[0]: points mostly in direction of feature-0
        val v0 = unitVector(dim, 0)
        // Vector[1]: points mostly in direction of feature-1
        val v1 = unitVector(dim, 1)
        // Vector[2]: points mostly in direction of feature-2
        val v2 = unitVector(dim, 2)
        // Vector[3]: blend of features 0 and 1 (diagonal)
        val v3 = l2Normalize(FloatArray(dim).also { it[0] = 1f; it[1] = 1f })
        // Vector[4]: blend of features 2 and 3
        val v4 = l2Normalize(FloatArray(dim).also { it[2] = 1f; it[3] = 1f })

        store.add("id0", "/file/a.txt", 0, "m", v0)
        store.add("id1", "/file/b.txt", 0, "m", v1)
        store.add("id2", "/file/c.txt", 0, "m", v2)
        store.add("id3", "/file/d.txt", 0, "m", v3)
        store.add("id4", "/file/e.txt", 0, "m", v4)

        // Query vector = v2 (exactly); closest should be id2 at cosine = 1.0
        val results = store.search(v2, topK = 5)

        assertEquals("Should return all 5 results", 5, results.size)
        assertEquals("Top result must be id2 (exact match)", "id2", results[0].id)
        assertEquals(1.0f, results[0].score, 1e-5f)
    }

    @Test
    fun `search on empty store returns empty list`() {
        val q = unitVector(384, 0)
        val results = store.search(q, topK = 5)
        assertTrue(results.isEmpty())
    }

    @Test
    fun `add with duplicate id replaces existing entry`() {
        val v1 = unitVector(384, 0)
        val v2 = unitVector(384, 1)
        store.add("dup", "/f.txt", 0, "m", v1)
        store.add("dup", "/f.txt", 0, "m", v2)  // replace with v2

        assertEquals("Should have exactly 1 entry after replace", 1, store.size)

        // Query with v2 - the stored vector should now be v2
        val results = store.search(v2, topK = 1)
        assertEquals(1.0f, results[0].score, 1e-5f)
    }

    @Test
    fun `search results are sorted descending by score`() {
        val dim = 384
        // Two vectors at different angles from query
        val vClose  = l2Normalize(FloatArray(dim).also { it[0] = 0.99f; it[1] = 0.01f })
        val vFar    = l2Normalize(FloatArray(dim).also { it[0] = 0.1f;  it[1] = 0.9f  })
        val query   = unitVector(dim, 0)  // points along feature-0

        store.add("close", "/close.txt", 0, "m", vClose)
        store.add("far",   "/far.txt",   0, "m", vFar)

        val results = store.search(query, topK = 2)
        assertEquals("close", results[0].id)
        assertEquals("far",   results[1].id)
        assertTrue(results[0].score > results[1].score)
    }

    // -------------------------------------------------------------------------
    // U1 characterization: 50k vectors at DIM=256 (EmbeddingModel.DIM)
    // -------------------------------------------------------------------------

    @Test
    fun `load 50k 256-dim vectors completes under 1500 ms and search is correct`() {
        val dim = 256 // EmbeddingModel.DIM
        val count = 50_000
        val rng = java.util.Random(42L)

        val startMs = System.currentTimeMillis()
        repeat(count) { i ->
            store.add(
                "id$i", "/f/$i.txt", i, "m",
                l2Normalize(FloatArray(dim) { rng.nextFloat() - 0.5f })
            )
        }
        val elapsedMs = System.currentTimeMillis() - startMs
        println("VectorStore: 50k 256-dim inserts took ${elapsedMs}ms")

        assertTrue(
            "50k inserts must complete in < 1.5 s (took ${elapsedMs}ms; the old O(n^2) load took about 2.7 s)",
            elapsedMs < 1_500L
        )
        assertEquals("Size must be 50k after 50k unique adds", count, store.size)

        // Insert a known vector as the needle, then verify it ranks #1.
        // ponytail: reuse unitVector(dim, 0), maximally orthogonal to random cloud
        val needle = unitVector(dim, 0)
        store.add("needle", "/needle.txt", 0, "m", needle)
        val top = store.search(needle, topK = 1)
        assertEquals("needle must rank #1 for exact-match query", "needle", top[0].id)
        assertEquals(1.0f, top[0].score, 1e-4f)
    }

    @Test
    fun `duplicate id replace is idempotent at 50k scale`() {
        val dim = 256
        val rng = java.util.Random(7L)
        repeat(50_000) { i ->
            store.add("id$i", "/f/$i.txt", i, "m",
                l2Normalize(FloatArray(dim) { rng.nextFloat() - 0.5f }))
        }
        val v2 = unitVector(dim, 0)
        store.add("id0", "/f/0.txt", 0, "m", v2)  // replace first entry
        assertEquals("Replace must not grow size", 50_000, store.size)
        val top = store.search(v2, topK = 1)
        assertEquals("id0", top[0].id)
        assertEquals(1.0f, top[0].score, 1e-4f)
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private fun unitVector(dim: Int, hotIdx: Int): FloatArray =
        FloatArray(dim).also { it[hotIdx] = 1.0f }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var norm = 0.0f
        for (x in v) norm += x * x
        norm = sqrt(norm.toDouble()).toFloat()
        return if (norm < 1e-12f) v.copyOf() else FloatArray(v.size) { i -> v[i] / norm }
    }
}
