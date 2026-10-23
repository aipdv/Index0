package com.narad.search.lib.query.model

import com.narad.search.lib.query.TestModel
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.sqrt

class EmbeddingModelTest {

    private lateinit var tokenizer: HuggingFaceTokenizer
    private lateinit var model: EmbeddingModel

    @Before
    fun setUp() {
        TestModel.skipIfMissing()
        tokenizer = HuggingFaceTokenizer(TestModel.tokenizerJsonPath)
        model = EmbeddingModel(
            modelPath = TestModel.onnxPath,
            tokenizer = tokenizer
        )
    }

    @After
    fun tearDown() {
        if (::model.isInitialized) model.close()
    }

    // -------------------------------------------------------------------------
    // DoD: embedQuery/embedDocument return 256-element FloatArray, no NaN
    // -------------------------------------------------------------------------
    @Test
    fun `embedQuery hello returns 256-element array with no NaN`() {
        val v = model.embedQuery("hello")
        assertEquals("Expected 256-dim embedding", EmbeddingModel.DIM, v.size)
        assertFalse("Embedding must not contain NaN", v.any { it.isNaN() })
    }

    @Test
    fun `embedDocument returns 256-element array for English`() {
        val v = model.embedDocument("The rental agreement includes a penalty clause.")
        assertEquals("Expected 256-dim embedding", EmbeddingModel.DIM, v.size)
        assertFalse("Embedding must not contain NaN", v.any { it.isNaN() })
    }

    @Test
    fun `embedQuery returns valid 256-element non-zero vector for Hindi`() {
        val v = model.embedQuery("नमस्ते")
        assertEquals("Expected 256-dim Hindi embedding", EmbeddingModel.DIM, v.size)
        assertFalse("Hindi embedding must not contain NaN", v.any { it.isNaN() })
        val norm = sqrt(v.sumOf { (it * it).toDouble() }).toFloat()
        assertTrue("Hindi embedding must have non-zero norm, got $norm", norm > 1e-4f)
    }

    // -------------------------------------------------------------------------
    // DoD: semantically similar docs score higher than dissimilar ones
    // EG2 is a much better model than Granite: large cosine margin expected.
    // -------------------------------------------------------------------------
    @Test
    fun `penalty clause doc vs penalty clause query has high cosine`() {
        val q = model.embedQuery("penalty clause contract")
        val d = model.embedDocument("The contractual fine for breach of agreement is ten percent of total value.")
        val cos = dot(q, d)
        println("cosine(query:penalty clause contract, doc:contractual fine) = $cos")
        assertTrue(
            "Expected cosine > 0.7 for semantically close query/doc pair, got $cos",
            cos > 0.7f
        )
    }

    @Test
    fun `penalty clause vs weather forecast scores materially below similar pair`() {
        val q = model.embedQuery("penalty clause")
        val dClose = model.embedDocument("The contractual fine for breach is ten percent.")
        val dFar = model.embedDocument("Sunny weather forecast with light rain on Thursday.")
        val cosSimilar = dot(q, dClose)
        val cosDistant = dot(q, dFar)
        println("cosine(query:penalty, doc:breach) = $cosSimilar")
        println("cosine(query:penalty, doc:weather) = $cosDistant")
        // EG2 at 256-dims has a much better spread than Granite.
        assertTrue(
            "Similar pair ($cosSimilar) must beat distant pair ($cosDistant) by >= 0.15",
            cosSimilar - cosDistant >= 0.15f
        )
    }

    // -------------------------------------------------------------------------
    // Verify L2-normalisation
    // -------------------------------------------------------------------------
    @Test
    fun `returned vector is L2-normalised`() {
        val v = model.embedDocument("contract law")
        val norm = sqrt(v.sumOf { (it * it).toDouble() }).toFloat()
        assertEquals("L2 norm should be ~1.0", 1.0f, norm, 1e-4f)
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }
}
