package me.zhanghai.android.files.index

import org.junit.Assert.assertEquals
import org.junit.Test

class RelevanceTest {

    private val none = emptySet<String>()

    private fun docs(cosines: List<Float>, terms: List<String>, matched: List<Set<String>> = cosines.map { none }) =
        strongDocs(cosines, matched, terms)

    // Real raw cosines from narad36, "train ticket": IRCTC ticket, Varanasi plan, Diwali list, rest.
    @Test fun `one clear winner is the only strong document`() {
        val mask = docs(listOf(0.803f, 0.669f, 0.660f, 0.616f), listOf("train", "ticket"))
        assertEquals(listOf(true, false, false, false), mask)
    }

    // "doctor ki parchi": two doctor files tie, the next ones sit just under the band.
    @Test fun `close hits inside the band are all strong`() {
        val mask = docs(listOf(0.730f, 0.730f, 0.693f, 0.690f), listOf("doctor", "parchi"))
        assertEquals(listOf(true, true, false, false), mask)
    }

    // "qwerty zxcv plonk": nothing clears the floor.
    @Test fun `nonsense query has no strong hits`() {
        assertEquals(listOf(false, false), docs(listOf(0.645f, 0.640f), listOf("qwerty", "zxcv", "plonk")))
    }

    // "bijli bill": the file holds bijli in one chunk and bill in another, cosine under the floor.
    @Test fun `file matching every meaningful term is strong without a cosine pass`() {
        val mask = docs(
            cosines = listOf(0.640f, 0.630f),
            terms = listOf("bijli", "bill"),
            matched = listOf(setOf("bijli", "bill"), setOf("bill")),
        )
        assertEquals(listOf(true, false), mask)
    }

    @Test fun `keyword only search needs every term covered`() {
        val mask = docs(
            cosines = listOf(0f, 0f),
            terms = listOf("train", "ticket"),
            matched = listOf(setOf("train", "ticket"), setOf("train")),
        )
        assertEquals(listOf(true, false), mask)
    }

    // Query "a" or "the": no meaningful word, so nothing is strong, even with a high cosine.
    @Test fun `query with no meaningful terms marks nothing strong`() {
        val terms = meaningfulTerms("the")
        assertEquals(emptyList<String>(), terms)
        assertEquals(listOf(false, false), docs(listOf(0.90f, 0.80f), terms, listOf(setOf("x"), none)))
        assertEquals(listOf(false, false), docs(listOf(0f, 0f), terms)) // Tier C
        assertEquals(listOf(false), strongPhotos(listOf(0.90f), terms))
    }

    // "temple by the river" photos: Varanasi ghats, then the Taj Mahal.
    @Test fun `photo band keeps only the clear winner`() {
        assertEquals(listOf(true, false), strongPhotos(listOf(0.758f, 0.666f), listOf("temple", "river")))
    }

    @Test fun `empty input gives empty mask`() {
        assertEquals(emptyList<Boolean>(), strongMask(emptyList(), PHOTO_FLOOR, PHOTO_GAP))
        assertEquals(emptyList<Boolean>(), strongPhotos(emptyList(), listOf("x")))
    }

    // ---- weak floor and cap --------------------------------------------------------------

    private fun hit(cosine: Float, kw: Boolean = false, strong: Boolean = false) = Hit(strong, cosine, kw)

    @Test fun `cosine under the weak floor with no keyword hit is dropped`() {
        val (d, _) = bucketHits(listOf(hit(0.58f), hit(0.62f), hit(0.58f, kw = true)), emptyList())
        assertEquals(listOf(Bucket.DROP, Bucket.WEAK, Bucket.WEAK), d)
    }

    @Test fun `strong hits stay strong and are never capped`() {
        val strong = List(12) { hit(0.8f, strong = true) }
        val (d, _) = bucketHits(strong, emptyList())
        assertEquals(List(12) { Bucket.STRONG }, d)
    }

    // Gibberish: best cosine 0.645, no keyword hit. 0 strong, at most 10 less related.
    @Test fun `gibberish gives no strong rows and at most ten less related`() {
        val cos = List(15) { 0.645f - it * 0.001f }
        val strong = docs(cos, listOf("qwerty", "zxcv"))
        assertEquals(List(15) { false }, strong)
        val (d, p) = bucketHits(cos.mapIndexed { i, c -> hit(c, strong = strong[i]) }, listOf(hit(0.64f)))
        val weak = (d + p).count { it == Bucket.WEAK }
        assertEquals(LESS_RELATED_MAX, weak)
        // The best-ranked rows are the ones kept: the 10 best of 16 (docs 0.645..0.636, photo 0.64).
        assertEquals(Bucket.WEAK, p[0])
        assertEquals(Bucket.DROP, d.last())
    }

    @Test fun `cap counts documents and photos together`() {
        val (d, p) = bucketHits(List(8) { hit(0.80f - it * 0.01f) }, List(8) { hit(0.65f - it * 0.001f) })
        assertEquals(LESS_RELATED_MAX, (d + p).count { it == Bucket.WEAK })
        assertEquals(8, d.count { it == Bucket.WEAK }) // all docs outrank the photos
        assertEquals(2, p.count { it == Bucket.WEAK })
    }

    // ---- chip counts ------------------------------------------------------------------

    @Test fun `chip counts come from strong rows only`() {
        fun doc(p: String, strong: Boolean) = DocRow(p, 0f, 0, p, "", 0L, FileKind.TXT, "", strong)
        val split = splitRows(
            docs = listOf(doc("a", true), doc("b", false), doc("c", false)),
            photos = listOf(PhotoRow("p1", 0.65f, false)),
        )
        assertEquals(1, split.mainDocs.size)
        assertEquals(0, split.mainPhotos.size) // "Photos 0": the chip is disabled
        assertEquals(2, split.lessDocs.size)
        assertEquals(1, split.lessPhotos.size)
        assertEquals(false, split.noStrong)
    }

    @Test fun `nothing strong gives an empty main list and everything under less related`() {
        val split = splitRows(
            docs = listOf(DocRow("a", 0f, 0, "a", "", 0L, FileKind.TXT, "", false)),
            photos = emptyList(),
        )
        assertEquals(true, split.noStrong)
        assertEquals(0, split.mainDocs.size)
        assertEquals(1, split.lessDocs.size)
    }
}
