package me.zhanghai.android.files.index

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM tests for snippet centering and FileKind mapping.
 *
 * Bug Watch coverage:
 * - Devanagari snippet centering (multi-byte text, char offsets)
 * - FileKind mapping correctness
 * - Snippet stays within maxLen
 */
class SnippetTest {

    // ---- buildSnippet -------------------------------------------------------

    @Test
    fun snippet_centersOnFirstTermHit() {
        val prefix  = "a".repeat(100) + " "
        val content = prefix + "TARGET" + " " + "a".repeat(100)
        val snippet = buildSnippet(content, listOf("target"), maxLen = 50)
        assertTrue(
            "Snippet must contain the matched term; got: $snippet",
            snippet.contains("TARGET", ignoreCase = true)
        )
        assertTrue(
            "Snippet must not exceed maxLen; got ${snippet.length}",
            snippet.length <= 50
        )
    }

    @Test
    fun snippet_neverExceedsMaxLen() {
        val content = "x".repeat(1000)
        val snippet = buildSnippet(content, listOf("x"), maxLen = 160)
        assertTrue(
            "Snippet length ${snippet.length} must be <= 160",
            snippet.length <= 160
        )
    }

    @Test
    fun snippet_noMatchReturnsPrefixWindow() {
        val content = "The quick brown fox jumps over the lazy dog."
        val snippet = buildSnippet(content, listOf("nonexistent_term"), maxLen = 160)
        assertTrue(
            "Snippet must be non-empty for non-blank content",
            snippet.isNotEmpty()
        )
        assertTrue(
            "Snippet must not exceed maxLen",
            snippet.length <= 160
        )
    }

    @Test
    fun snippet_emptyContent_returnsEmpty() {
        val snippet = buildSnippet("", listOf("query"), maxLen = 160)
        assertEquals("Empty content → empty snippet", "", snippet)
    }

    /**
     * Bug Watch: Devanagari chunk + Hinglish query must not throw, must contain the
     * matched term, and must produce a snippet of at most 160 chars (char count, not
     * byte count).
     *
     * This tests that the implementation uses char-based indexing (Kotlin String is
     * UTF-16, NOT byte-indexed), so Devanagari multi-byte code points are handled
     * correctly.
     *
     * The hit term "दंड" is placed after char 200 so this test can fail if centering
     * is broken or the snippet is truncated before reaching the hit.
     */
    @Test
    fun snippet_devanagariChunk_hinglishQuery_noCrash() {
        // Pad with 210 Devanagari chars before the hit so the term is well past char 200.
        val prefix  = "क".repeat(210) + " "     // 211 chars - all before the hit (a word start needs a gap)
        val hit     = "दंड"                       // the matched term
        val suffix  = "क".repeat(60)            // 60 chars after the hit
        val content = prefix + hit + suffix      // total > 270 chars
        assertTrue("Content must be longer than 160 chars", content.length > 160)

        // Hinglish/transliterated query terms - "dand" won't match by substring,
        // but "दंड" (Devanagari) is the exact hit we expect to find.
        val queryTerms = listOf("kiraya", "दंड")
        val snippet = buildSnippet(content, queryTerms, maxLen = 160)

        assertTrue(
            "Devanagari snippet must not exceed 160 chars; got ${snippet.length}",
            snippet.length <= 160
        )
        assertTrue(
            "Snippet must contain the matched Devanagari term 'दंड' (hit at char 210); " +
            "got: '$snippet'",
            snippet.contains("दंड")
        )
        assertTrue("Snippet must not be empty for non-empty content", snippet.isNotEmpty())
    }

    @Test
    fun snippet_devanagariTermFound_centeredCorrectly() {
        // Place "दंड" at char position 80
        val before  = "क".repeat(40) + " " // 40 Devanagari chars plus a gap
        val content = before + "दंड" + " " + "क".repeat(40)
        val snippet = buildSnippet(content, listOf("दंड"), maxLen = 30)
        assertTrue(
            "Snippet should contain the matched Devanagari term; got: $snippet",
            snippet.contains("दंड")
        )
        assertTrue(snippet.length <= 30)
    }

    // U3: the window follows the first meaningful word start, not a stopword or a mid-word match.
    private val itinerary =
        "Day 1: reach by train at 6 am, then ghat walk and a long breakfast of poha and tea " +
            "near the station before the tour begins. Evening: temple at dusk, aarti by the river. " + "Day 2: free morning, shopping and an early night. ".repeat(4)

    @Test
    fun snippet_showsFirstMeaningfulMatchNotAStopword() {
        val terms = meaningfulTerms("temple by the")
        assertEquals(listOf("temple"), terms)
        val snippet = buildSnippet(itinerary, terms, maxLen = 160)
        assertTrue("temple must be in the first 80 chars; got: $snippet", snippet.take(80).contains("temple"))
        val marked = highlightRanges(snippet, terms).map { snippet.substring(it.first, it.last + 1) }
        assertEquals(listOf("temple"), marked) // "then" and "the" are not marked
    }

    @Test
    fun snippet_startsAboutTwentyFiveCharsBeforeTheHit() {
        val content = "x ".repeat(100) + "temple " + "y ".repeat(100)
        val snippet = buildSnippet(content, listOf("temple"), maxLen = 160)
        // 25 characters of lead; trim() may eat one leading space.
        assertTrue("got ${snippet.indexOf("temple")}", snippet.indexOf("temple") in 24..25)
    }

    @Test
    fun snippet_midWordMatchDoesNotCentre() {
        // "restrain" contains "train" but is not a word start, so the window stays at the start.
        val content = "restrain the dog. " + "filler ".repeat(60) + "train arrives"
        val snippet = buildSnippet(content, listOf("train"), maxLen = 80)
        assertTrue(snippet.contains("train arrives"))
    }

    // The window never starts inside a word: back to the word start, with a leading "..." when cut.
    private val slip = "Electronic reservation slip. Train 12321 Howrah Mumbai Mail, CSMT to Varanasi. PNR 4521 " +
        "lorem ipsum dolor sit amet ".repeat(8)

    @Test
    fun snippet_startAtTextStartHasNoEllipsis() {
        // The 25-char lead lands inside "Electronic"; the word start is the text start.
        val snippet = buildSnippet(slip, listOf("train"), maxLen = 160)
        assertTrue("got: $snippet", snippet.startsWith("Electronic reservation"))
    }

    @Test
    fun snippet_cutStartMovesBackToAWordBoundaryAndAddsEllipsis() {
        val content = "lorem ipsum dolor sit amet ".repeat(8) + slip
        assertTrue(content.length > 400)
        val snippet = buildSnippet(content, listOf("train"), maxLen = 160)
        assertTrue("got: $snippet", snippet.startsWith("…"))
        val firstWord = snippet.drop(1).substringBefore(' ')
        assertTrue("'$firstWord' is not a whole word of the text; got: $snippet", content.split(' ').contains(firstWord))
        assertTrue(snippet.contains("Train 12321"))
        assertTrue("got ${snippet.length}", snippet.length <= 160)
        assertEquals(listOf("Train"), highlightRanges(snippet, listOf("train")).map { snippet.substring(it.first, it.last + 1) })
    }

    @Test
    fun snippet_hitNearTheEndOfAChunk_staysInTheFirstVisibleLine() {
        // A 448-char chunk with the hit 10 chars from the end: a shorter snippet, hit up front.
        val words = "alpha bravo charlie delta echo foxtrot "
        val content = words.repeat(12).take(436) + " Train 12321"
        assertEquals(448, content.length)
        val snippet = buildSnippet(content, listOf("train"), maxLen = 160)
        val at = snippet.indexOf("Train")
        assertTrue("hit at $at must be within the first 40 visible chars; got: $snippet", at in 0..40)
        assertTrue(snippet.endsWith("12321"))
        val firstWord = snippet.removePrefix("\u2026").substringBefore(' ')
        assertTrue("'$firstWord' cut mid-word; got: $snippet", content.split(' ').contains(firstWord))
    }

    // Chunks after the first begin at a raw cut (the Chunker cuts every 448 chars).
    private val chunkText = "ure from the main terminal and then a little more text to read. " + "lorem ipsum dolor sit amet ".repeat(8)

    @Test
    fun snippet_firstChunk_startsAtTextStartWithNoEllipsis() {
        val snippet = buildSnippet(chunkText, listOf("terminal"), maxLen = 160, startsMidText = false)
        assertTrue("got: $snippet", snippet.startsWith("ure from"))
    }

    @Test
    fun snippet_laterChunk_dropsThePartialWordAndAddsEllipsis() {
        val snippet = buildSnippet(chunkText, listOf("terminal"), maxLen = 160, startsMidText = true)
        assertTrue("got: $snippet", snippet.startsWith("\u2026from the main terminal"))
        assertTrue(snippet.length <= 160)
    }

    @Test
    fun snippet_laterShortChunk_alsoGetsTheEllipsis() {
        val snippet = buildSnippet("ure from the main terminal", listOf("terminal"), startsMidText = true)
        assertEquals("\u2026from the main terminal", snippet)
    }

    @Test
    fun snippet_laterChunk_neverDropsTheHitWord() {
        // The hit is the first word of the chunk: keep it, mark the cut.
        val snippet = buildSnippet("terminal map of the station. " + "x ".repeat(100), listOf("terminal"), startsMidText = true)
        assertTrue("got: $snippet", snippet.startsWith("\u2026terminal"))
    }

    @Test
    fun snippet_laterChunkWithNoHit_stillDropsThePartialWord() {
        // Meaning-only match or a stopword query: no term hits, the cut must still not open mid-word.
        for (terms in listOf(listOf("railway", "ticket"), emptyList())) {
            val snippet = buildSnippet(chunkText, terms, maxLen = 160, startsMidText = true)
            assertTrue("terms=$terms got: $snippet", snippet.startsWith("\u2026from the main terminal"))
        }
    }

    @Test
    fun snippet_laterChunk_skipsPunctuationAfterThePartialWord() {
        val snippet = buildSnippet("ure, from the main terminal", listOf("terminal"), startsMidText = true)
        assertEquals("\u2026from the main terminal", snippet)
    }

    @Test
    fun snippet_noMatchPrefixWindowHasNoEllipsis() {
        assertTrue(buildSnippet(slip, listOf("nothing"), maxLen = 160).startsWith("Electronic"))
    }

    // Devanagari words with matras, found through meaningfulTerms (not a hand-passed term).
    @Test
    fun snippet_devanagariQueryWithMatras_marksTheWholeWord() {
        val terms = meaningfulTerms("बिजली बिल")
        assertEquals(listOf("बिजली", "बिल"), terms)
        val content = "आज " + "मौसम बहुत अच्छा है ".repeat(15) + "इस महीने का बिजली बिल जमा करना है " + "समय ".repeat(40)
        val snippet = buildSnippet(content, terms, maxLen = 80)
        val marked = highlightRanges(snippet, terms).map { snippet.substring(it.first, it.last + 1) }
        assertEquals(listOf("बिजली", "बिल"), marked)
        assertTrue("snippet must reach the hit; got: $snippet", snippet.contains("बिजली"))
    }

    // ---- fileKindFromMime ---------------------------------------------------

    @Test
    fun fileKind_pdf() {
        assertEquals(FileKind.PDF, fileKindFromMime("application/pdf"))
    }

    @Test
    fun fileKind_doc() {
        assertEquals(FileKind.DOC, fileKindFromMime("application/msword"))
        assertEquals(FileKind.DOC, fileKindFromMime("application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
    }

    @Test
    fun fileKind_txt() {
        assertEquals(FileKind.TXT, fileKindFromMime("text/plain"))
        assertEquals(FileKind.TXT, fileKindFromMime("text/markdown"))
    }

    @Test
    fun fileKind_image() {
        assertEquals(FileKind.IMAGE, fileKindFromMime("image/jpeg"))
        assertEquals(FileKind.IMAGE, fileKindFromMime("image/png"))
    }

    @Test
    fun fileKind_unknown_fallsBackToTxt() {
        assertEquals(FileKind.TXT, fileKindFromMime(null))
        assertEquals(FileKind.TXT, fileKindFromMime("application/octet-stream"))
    }
}