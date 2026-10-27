package com.narad.search.lib.query.model

import com.narad.search.lib.query.TestModel
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Batched document embedding must equal the single path (research spec section 7.6).
 * Model-backed: skipped by NARAD_SKIP_MODEL_TESTS=1 or when no model folder is found.
 */
class EmbeddingModelBatchTest {

    private lateinit var model: EmbeddingModel
    private lateinit var single: List<FloatArray>

    // Hindi text as escapes: no Devanagari in repo files.
    private val hindiRent = "नमस्ते, यह किराये का अनुबंध है। " +
        "मकान मालिक हर महीने की पाँच तारीख तक किराया लेगा। "
    private val hindiBank = "बैंक खाते का विवरण: जमा राशि, " +
        "निकासी और शेष राशि नीचे दी गई है। "

    private fun chunk(s: String) = s.repeat(512 / s.length + 1).take(512)

    private val texts = listOf(
        "hi",
        chunk("The tenant pays the rent on the fifth day of each month. A late payment adds a penalty of two percent. "),
        chunk("Photosynthesis turns light, water and carbon dioxide into sugar and oxygen inside the chloroplast. "),
        chunk("The meeting moved to Thursday. Please bring the quarterly budget and the hiring plan for review. "),
        "Signed in Pune on the twelfth of March. Both parties keep one copy of this agreement.",
        chunk("12/03/2024 UPI-4471823390 HDFC0001234 DR 1,250.00 CR 0.00 BAL 48,312.75 "),
        chunk("0045 7781 2290 3341 INR 99,120.00 05-11-2023 NEFT/N309231234567/ "),
        chunk(hindiRent),
        chunk(hindiBank),
        "Kal subah 10 baje doctor ka appointment hai, report saath le jana mat bhoolna.",
        "INVOICE\nBill to: Asha Traders\n\nItem\tQty\tPrice\nPaper A4\t10\t2,500\n\nTotal due: 2,500",
        "lorem ".repeat(60).trim(),
        " \n\n   \n ".repeat(40), // blank chunk: a window inside a PDF's whitespace run
    )

    @Before
    fun setUp() {
        TestModel.skipIfMissing()
        model = EmbeddingModel(TestModel.onnxPath, DjlTokenizer(TestModel.tokenizerJsonPath))
        single = texts.map(model::embedDocument)
    }

    @After
    fun tearDown() {
        if (::model.isInitialized) model.close()
    }

    @Test
    fun oneRun_matchesSingle() = assertParity(model.embedDocuments(texts, tokenBudget = 1 shl 20))

    @Test
    fun defaultBudget_matchesSingle() = assertParity(model.embedDocuments(texts))

    @Test
    fun shuffled_matchesSingle() {
        val order = texts.indices.shuffled(Random(7))
        val out = model.embedDocuments(order.map { texts[it] })
        val back = arrayOfNulls<FloatArray>(texts.size)
        order.forEachIndexed { pos, i -> back[i] = out[pos] }
        assertParity(back.map { it!! })
    }

    @Test
    fun smallBudget_splits_matchesSingle() = assertParity(model.embedDocuments(texts, tokenBudget = 300))

    private fun assertParity(batched: List<FloatArray>) {
        assertEquals(texts.size, batched.size)
        for (i in texts.indices) {
            val v = batched[i]
            assertEquals(EmbeddingModel.DIM, v.size)
            assertFalse("row $i has NaN", v.any { it.isNaN() })
            assertEquals("row $i norm", 1.0, sqrt(v.sumOf { (it * it).toDouble() }), 1e-4)
            val c = cos(single[i], v)
            // ponytail: 0.9995, not 0.9999. Unpadded rows are bit-exact; a padded row drifts up to
            // 1.4e-4 (0.999858, Apple M3, 4 ORT threads): float order in the longer softmax, amplified
            // by the int8 activation step. The drift does not grow with pad length.
            assertTrue("row $i: cos(single, batched) = $c", c >= 0.9995)
        }
        // Row mix-up: a batched row must not look like another text's single vector.
        for (i in texts.indices) for (j in texts.indices) {
            if (i != j && cos(single[i], single[j]) < 0.95) {
                assertTrue("row $i looks like text $j", cos(batched[i], single[j]) < 0.99)
            }
        }
    }

    private fun cos(a: FloatArray, b: FloatArray): Double {
        var dot = 0.0; var na = 0.0; var nb = 0.0
        for (k in a.indices) { dot += a[k] * b[k]; na += a[k] * a[k]; nb += b[k] * b[k] }
        return dot / sqrt(na * nb)
    }
}
