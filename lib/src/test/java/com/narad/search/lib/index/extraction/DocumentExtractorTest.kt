package com.narad.search.lib.index.extraction

import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.pdmodel.PDPage
import com.tom_roush.pdfbox.pdmodel.PDPageContentStream
import com.tom_roush.pdfbox.pdmodel.encryption.AccessPermission
import com.tom_roush.pdfbox.pdmodel.encryption.StandardProtectionPolicy
import com.tom_roush.pdfbox.pdmodel.font.PDType1Font
import org.apache.poi.xwpf.usermodel.XWPFDocument
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Host-JVM tests for [DocumentExtractor].
 *
 * PDFs are built in-test with PdfBox (no committed binary fixtures for English/encrypted).
 * GlyphList and AFM resources are provided by the `extractPdfboxTestResources`
 * Gradle task that unzips the pdfbox-android AAR assets and adds them as a test resource dir.
 * `testOptions.unitTests.isReturnDefaultValues = true` silences `android.util.Log` stubs.
 *
 * DOCX parity: [poiReference] is the oracle (old POI behavior). SAX-based extractor must
 * match it for paragraph content. POI stays on testImplementation after the Gradle split.
 */
class DocumentExtractorTest {

    private val extractor = DocumentExtractor()

    // ─────────────────────────────────────────────────────────────────────────
    // PDF: English Helvetica
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `extractStream from PdfBox-built English PDF contains known sentence`() {
        val knownText = "The quick brown fox jumps over the lazy dog."
        val pdfBytes = buildHelveticaPdf(knownText)

        val result = extractor.extractStream(pdfBytes.inputStream(), "application/pdf")

        assertNotNull("extractStream must return non-null for a valid PDF", result)
        assertTrue(
            "Extracted text must contain the known sentence.\nExtracted: $result",
            result!!.contains("quick brown fox")
        )
    }

    @Test
    fun `extractStream for user-password PDF returns null without throwing`() {
        val pdfBytes = buildPasswordProtectedPdf(userPassword = "s3cr3t")

        val result = extractor.extractStream(pdfBytes.inputStream(), "application/pdf")

        assertNull("User-password PDF must return null (not throw)", result)
    }

    @Test
    fun `extractStream for PDF-header garbage bytes returns null without throwing`() {
        val garbage = "%PDF-1.4\n".toByteArray() + ByteArray(64) { (it * 37).toByte() }

        val result = extractor.extractStream(garbage.inputStream(), "application/pdf")

        assertNull("Corrupt PDF must return null (not throw)", result)
    }

    @Test
    fun `extractStream for truncated PDF returns null or partial text without throwing`() {
        val fullBytes = buildHelveticaPdf("Truncation test line one. Truncation test line two.")
        val full = extractor.extractStream(fullBytes.inputStream(), "application/pdf")
        assertNotNull("Full PDF must extract", full)
        val half = fullBytes.copyOfRange(0, fullBytes.size / 2)

        val result = extractor.extractStream(half.inputStream(), "application/pdf")
        assertTrue(
            "Truncated PDF must return null or a substring of the full text. Got: $result",
            result == null || full!!.contains(result.trim())
        )
    }

    @Test
    fun `extractStream for two-page PDF returns every page exactly once`() {
        val pdfBytes = buildHelveticaPdf(
            listOf("Alpha deposit clause.", "Alpha second line."),
            listOf("Bravo eviction notice.", "Bravo second line.")
        )

        val result = extractor.extractStream(pdfBytes.inputStream(), "application/pdf")

        assertNotNull(result)
        for (word in listOf("Alpha deposit", "Alpha second", "Bravo eviction", "Bravo second")) {
            assertEquals("'$word' must appear exactly once in: $result", 1, result!!.split(word).size - 1)
        }
        assertTrue("Page order must be kept", result!!.indexOf("Alpha") < result.indexOf("Bravo"))
    }

    @Test
    fun `extractStream for empty-page PDF returns null`() {
        val pdfBytes = buildEmptyPagePdf()

        val result = extractor.extractStream(pdfBytes.inputStream(), "application/pdf")

        assertNull("Empty-page PDF should return null (no text)", result)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DOCX: known content round-trips (characterization and parity tests)
    // [poiReference] captures the old POI behavior as oracle.
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `extractStream from test DOCX returns non-empty string`() {
        val docxBytes = buildTestDocx("Contract clause: the penalty fee is ten percent of total value.")

        val result = extractor.extractStream(
            docxBytes.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNotNull("extractStream should return non-null for DOCX", result)
        assertTrue("Extracted DOCX text must be non-empty", result!!.isNotBlank())
    }

    @Test
    fun `extractStream from DOCX contains known paragraph text`() {
        val knownText = "Rental agreement penalty clause section five."
        val docxBytes = buildTestDocx(knownText)

        val result = extractor.extractStream(
            docxBytes.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNotNull(result)
        assertTrue(
            "Extracted DOCX text should contain known content.\nExtracted: $result",
            result!!.contains("penalty clause")
        )
    }

    /** Parity: SAX output exactly matches POI reference for a single-paragraph DOCX. */
    @Test
    fun `extractStream DOCX SAX output matches POI reference for single paragraph`() {
        val text = "Contract clause: rent is due on the first of each month."
        val docxBytes = buildTestDocx(text)

        val poiResult = poiReference(docxBytes)
        val saxResult = extractor.extractStream(
            docxBytes.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNotNull("SAX extractor must return non-null for valid DOCX", saxResult)
        assertNotNull("POI reference must return non-null", poiResult)
        assertEquals(
            "SAX output must exactly match POI reference for single-paragraph DOCX",
            poiResult, saxResult
        )
    }

    /**
     * Parity: one paragraph with multiple runs (multiple w:r elements).
     * POI's para.text concatenates runs; SAX accumulates across multiple w:t elements.
     */
    @Test
    fun `extractStream DOCX multi-run paragraph matches POI reference`() {
        val docxBytes = buildMultiRunDocx(listOf("First part ", "second part ", "third."))

        val poiResult = poiReference(docxBytes)
        val saxResult = extractor.extractStream(
            docxBytes.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNotNull(saxResult)
        assertNotNull(poiResult)
        assertEquals("Multi-run paragraph SAX must match POI reference", poiResult, saxResult)
    }

    /**
     * Parity: tab and line break within a run.
     * w:tab must emit '\t' and w:br must emit '\n' to match POI para.text output.
     */
    @Test
    fun `extractStream DOCX tab and line break match POI reference`() {
        val docxBytes = buildTabBreakDocx()

        val poiResult = poiReference(docxBytes)
        val saxResult = extractor.extractStream(
            docxBytes.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNotNull("SAX must return non-null for tab/break DOCX", saxResult)
        assertNotNull("POI reference must return non-null for tab/break DOCX", poiResult)
        assertEquals("Tab and line break must match POI reference", poiResult, saxResult)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ISO/IEC 29500 Strict namespace (http://purl.oclc.org/ooxml/wordprocessingml/main)
    // Files saved by Word in Strict mode use this URI instead of the Transitional one.
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `extractStream DOCX with Strict namespace extracts same text as Transitional`() {
        val text = "Strict mode document text for parity check."
        val transitionalDocx = buildSingleEntryZip(
            "word/document.xml",
            minimalDocumentXml(text).toByteArray(Charsets.UTF_8)
        )
        val strictXml =
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://purl.oclc.org/ooxml/wordprocessingml/main"><w:body><w:p><w:r><w:t>$text</w:t></w:r></w:p></w:body></w:document>"""
        val strictDocx = buildSingleEntryZip(
            "word/document.xml",
            strictXml.toByteArray(Charsets.UTF_8)
        )

        val transitionalResult = extractor.extractStream(
            transitionalDocx.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        val strictResult = extractor.extractStream(
            strictDocx.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNotNull("Transitional DOCX must extract text", transitionalResult)
        assertEquals("Strict namespace must produce the same text as Transitional", transitionalResult, strictResult)
    }

    // ─────────────────────────────────────────────────────────────────────────
    @Test
    fun `extractStream DOCX accepts an alternate WordprocessingML prefix`() {
        val text = "Prefix-independent document text."
        val standardDocx = buildSingleEntryZip(
            "word/document.xml",
            minimalDocumentXml(text).toByteArray(Charsets.UTF_8)
        )
        val alternateXml =
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><x:document xmlns:x="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><x:body><x:p><x:r><x:t>$text</x:t></x:r></x:p></x:body></x:document>"""
        val alternateDocx = buildSingleEntryZip(
            "word/document.xml",
            alternateXml.toByteArray(Charsets.UTF_8)
        )

        val standardResult = extractor.extractStream(
            standardDocx.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        val alternateResult = extractor.extractStream(
            alternateDocx.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNotNull("Standard-prefix DOCX must extract text", standardResult)
        assertEquals("Alternate prefix must produce the same text", standardResult, alternateResult)
    }

    // DOCX security: containsDoctype (M6, KTD6)
    // Tested directly against the internal function so the test does not depend
    // on the JVM SAX parser's own DOCTYPE handling (which Xerces also blocks,
    // but Android's libcore SAX factory does not support disallow-doctype-decl).
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `containsDoctype detects DOCTYPE in UTF-8`() {
        val xml = """<?xml version="1.0"?><!DOCTYPE foo []><root/>""".toByteArray(Charsets.UTF_8)
        assertTrue("UTF-8 DOCTYPE must be detected", extractor.containsDoctype(xml))
    }

    @Test
    fun `containsDoctype accepts clean UTF-8 without DOCTYPE`() {
        val xml = """<?xml version="1.0"?><root/>""".toByteArray(Charsets.UTF_8)
        assertFalse("Clean UTF-8 XML must not be flagged", extractor.containsDoctype(xml))
    }

    @Test
    fun `containsDoctype refuses UTF-16 BOM encoded bytes`() {
        // Charsets.UTF_16 adds a BOM (0xFE 0xFF); subsequent chars contain 0x00 bytes.
        val xml = """<?xml version="1.0"?><root/>""".toByteArray(Charsets.UTF_16)
        assertTrue("UTF-16 BOM bytes contain null bytes and must be refused", extractor.containsDoctype(xml))
    }

    @Test
    fun `containsDoctype refuses UTF-16LE encoded bytes`() {
        val xml = """<?xml version="1.0"?><root/>""".toByteArray(Charsets.UTF_16LE)
        assertTrue("UTF-16LE bytes contain null bytes and must be refused", extractor.containsDoctype(xml))
    }

    @Test
    fun `containsDoctype refuses UTF-16BE encoded bytes`() {
        val xml = """<?xml version="1.0"?><root/>""".toByteArray(Charsets.UTF_16BE)
        assertTrue("UTF-16BE bytes contain null bytes and must be refused", extractor.containsDoctype(xml))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DOCX security: DOCTYPE guard in the extraction path (M6, KTD6)
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `extractStream refuses DOCX whose document xml contains DOCTYPE`() {
        val xmlWithDoctype = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body><w:p><w:r><w:t>safe text</w:t></w:r></w:p></w:body>
</w:document>""".toByteArray(Charsets.UTF_8)

        val result = extractor.extractStream(
            buildSingleEntryZip("word/document.xml", xmlWithDoctype).inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNull("DOCX whose word/document.xml contains DOCTYPE must be refused", result)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DOCX security: zip-bomb / cap guard (M6)
    //
    // Two tests:
    // (A) Honest header: entry size is correct; content genuinely inflates past cap.
    //     This test FAILS if the cap is removed (extractor would return non-null text).
    //     This is the primary test for the readCapped() guard.
    //
    // (B) Lying header: ZipEntry.size = 10 (lying) while inflated content is large.
    //     ZipInputStream may throw ZipException on the size mismatch; either way the
    //     extractor returns null. Proves the code handles malformed-header archives.
    // ─────────────────────────────────────────────────────────────────────────

    /** (A) Honest header: cap genuinely rejects an oversized entry. */
    @Test
    fun `extractStream refuses DOCX whose document entry genuinely inflates past cap`() {
        // Any minimal XML is well over 100 bytes; assert > 100 to make the test self-documenting.
        val xml = minimalDocumentXml("Direct cap test.").toByteArray(Charsets.UTF_8)
        assertTrue("XML fixture must be larger than the test cap", xml.size > 100)

        val zipBytes = buildSingleEntryZip("word/document.xml", xml)

        val result = DocumentExtractor(maxDocxBytes = 100L).extractStream(
            zipBytes.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNull("Entry inflating past cap must be refused (honest header)", result)
    }

    /** (B) Lying header: local header declares size = 10; inflated content is much larger. */
    @Test
    fun `extractStream refuses DOCX whose entry declares small size but inflates past cap`() {
        val xml = minimalDocumentXml("Security test paragraph. ".repeat(20)).toByteArray(Charsets.UTF_8)
        val streamingZip = buildSingleEntryZip("word/document.xml", xml)
        val patchedZip = patchFirstEntryUncompressedSize(streamingZip, fakeUncompressedSize = 10)

        // Cap = 100: declared size (10) < cap, actual inflated (~700+) > cap -> refuse.
        val result = DocumentExtractor(maxDocxBytes = 100L).extractStream(
            patchedZip.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNull(
            "DOCX entry that declares small size but inflates past cap must be refused",
            result
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DOCX: zip without word/document.xml
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `extractStream for DOCX zip without word document xml returns null`() {
        val zipBytes = buildSingleEntryZip("other/file.xml", "<root/>".toByteArray())

        val result = extractor.extractStream(
            zipBytes.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertNull("Zip with no word/document.xml must return null", result)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Unknown / null MIME
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `extractStream for unknown MIME returns null without throwing`() {
        val result = extractor.extractStream(
            "dummy content".byteInputStream(),
            "application/x-unknown-format"
        )
        assertNull("Unknown MIME should return null, not throw", result)
    }

    @Test
    fun `extractStream for null MIME returns null`() {
        val result = extractor.extractStream("data".byteInputStream(), null)
        assertNull("null MIME should return null", result)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Plain text / Markdown
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `extractStream for text_plain returns raw text`() {
        val content = "Hello world. This is a plain text document about penalty clauses."
        val result = extractor.extractStream(content.byteInputStream(Charsets.UTF_8), "text/plain")

        assertNotNull("text/plain must return non-null", result)
        assertEquals("text/plain content should be returned verbatim", content, result)
    }

    @Test
    fun `extractStream for text_markdown returns raw text`() {
        val content = "# Heading\n\nSome **markdown** content about contracts."
        val result = extractor.extractStream(content.byteInputStream(Charsets.UTF_8), "text/markdown")

        assertNotNull("text/markdown must return non-null", result)
        assertTrue("Markdown content should be returned", result!!.contains("markdown"))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // U8: 1 MB text cap during extraction
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `extractStream for 5 MB text file keeps only the first 1 MB and still indexes`() {
        val oneMb = 1024 * 1024
        // Distinct head and tail so truncation is observable, not just the length.
        val head = "HEADMARK "
        val tail = " TAILMARK"
        val body = StringBuilder(head)
        while (body.length < 5 * oneMb) body.append("lorem ipsum dolor sit amet ")
        body.append(tail)

        val result = extractor.extractStream(body.toString().byteInputStream(Charsets.UTF_8), "text/plain")

        assertNotNull("an oversize text file must still yield text", result)
        assertEquals("exactly the first 1 MB is kept", oneMb, result!!.length)
        assertTrue(result.startsWith(head))
        assertFalse("the tail past the cap must be dropped", result.contains("TAILMARK"))
    }

    @Test
    fun `extractStream for text exactly at the cap keeps every byte`() {
        val small = DocumentExtractor(maxTextBytes = 100)
        val exact = "a".repeat(100)
        assertEquals(exact, small.extractStream(exact.byteInputStream(), "text/plain"))
        assertEquals(exact, small.extractStream(("$exact" + "b".repeat(50)).byteInputStream(), "text/plain"))
    }

    @Test
    fun `extractStream for 2000-page PDF stops at the page limit`() {
        val pages = (1..2000).map { listOf("PAGEMARK${"%04d".format(it)} body text for the page") }
        val pdf = buildHelveticaPdf(*pages.toTypedArray())

        val result = extractor.extractStream(pdf.inputStream(), "application/pdf")

        assertNotNull(result)
        assertTrue("first page is read", result!!.contains("PAGEMARK0001"))
        assertTrue("last page inside the limit is read", result.contains("PAGEMARK0500"))
        assertFalse("pages past the limit are not read", result.contains("PAGEMARK0501"))
        assertFalse(result.contains("PAGEMARK2000"))
    }

    @Test
    fun `extractStream for PDF stops once extracted text reaches the cap`() {
        val capped = DocumentExtractor(maxTextBytes = 300)
        val pages = (1..40).map { listOf("PAGEMARK${"%04d".format(it)} " + "word ".repeat(10)) }
        val pdf = buildHelveticaPdf(*pages.toTypedArray())

        val result = capped.extractStream(pdf.inputStream(), "application/pdf")

        assertNotNull(result)
        assertTrue("text is capped, got ${result!!.length}", result.length <= 300)
        assertTrue(result.contains("PAGEMARK0001"))
        assertFalse("later pages are cut off", result.contains("PAGEMARK0040"))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // review-035: entry-count limit
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Entry-count guard: with limit=3, the 4th entry is word/document.xml with valid text.
     * Without the guard the extractor would reach and parse that entry and return text.
     * With the guard it aborts after entry 3 and returns null with exactly one
     * EntryLimitExceeded event.
     */
    @Test
    fun `extractStream refuses DOCX with entry count above limit`() {
        val events = mutableListOf<Pair<String, String>>()
        val docXml = minimalDocumentXml("should not be read").toByteArray(Charsets.UTF_8)
        val zip = buildMultiEntryZip(entries = listOf(
            "other1.xml" to "<a/>".toByteArray(),
            "other2.xml" to "<b/>".toByteArray(),
            "other3.xml" to "<c/>".toByteArray(),
            "word/document.xml" to docXml          // 4th entry; guard fires before this is read
        ))
        val result = DocumentExtractor(
            maxDocxEntries = 3,
            onWarn = { errorClass, msg -> events.add(errorClass to msg) }
        ).extractStream(
            zip.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assertNull("Entry limit exceeded must return null", result)
        assertEquals("Exactly one EntryLimitExceeded event", 1, events.size)
        assertEquals("EntryLimitExceeded", events[0].first)
    }

    /** A normal DOCX (few entries) must still extract correctly under the default limit. */
    @Test
    fun `extractStream normal DOCX succeeds under entry limit`() {
        val docxBytes = buildTestDocx("Entry limit does not affect normal files.")
        val result = extractor.extractStream(
            docxBytes.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assertNotNull("Normal DOCX must still extract under the entry count limit", result)
        assertTrue(result!!.contains("Entry limit"))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DOCX: corrupted bytes
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `extractStream for corrupted DOCX bytes returns null without throwing`() {
        val garbledBytes = byteArrayOf(
            0x50, 0x4B, 0x03, 0x04,
            0x00, 0x00, 0xFF.toByte(), 0xFF.toByte(),
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        )
        val result = extractor.extractStream(
            garbledBytes.inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assertNull("Corrupted DOCX bytes must return null", result)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers: build in-memory documents with PdfBox / POI
    // ─────────────────────────────────────────────────────────────────────────

    private fun buildHelveticaPdf(text: String): ByteArray = buildHelveticaPdf(listOf(text))

    private fun buildHelveticaPdf(vararg pages: List<String>): ByteArray {
        val baos = ByteArrayOutputStream()
        PDDocument().use { doc ->
            for (lines in pages) {
                val page = PDPage()
                doc.addPage(page)
                PDPageContentStream(doc, page).use { cs -> writeLines(cs, lines) }
            }
            doc.save(baos)
        }
        return baos.toByteArray()
    }

    private fun writeLines(cs: PDPageContentStream, lines: List<String>) {
        cs.beginText()
        cs.setFont(PDType1Font.HELVETICA, 12f)
        cs.setLeading(14f)
        cs.newLineAtOffset(72f, 720f)
        for (line in lines) {
            cs.showText(line)
            cs.newLine()
        }
        cs.endText()
    }

    private fun buildPasswordProtectedPdf(userPassword: String): ByteArray {
        val baos = ByteArrayOutputStream()
        PDDocument().use { doc ->
            val page = PDPage()
            doc.addPage(page)
            PDPageContentStream(doc, page).use { cs -> writeLines(cs, listOf("Secret deposit clause.")) }
            val policy = StandardProtectionPolicy("ownerPass", userPassword, AccessPermission())
            policy.encryptionKeyLength = 40
            doc.protect(policy)
            doc.save(baos)
        }
        return baos.toByteArray()
    }

    private fun buildEmptyPagePdf(): ByteArray {
        val baos = ByteArrayOutputStream()
        PDDocument().use { doc ->
            doc.addPage(PDPage())
            doc.save(baos)
        }
        return baos.toByteArray()
    }

    /** Build a single-paragraph DOCX using POI (used as both fixture and oracle). */
    private fun buildTestDocx(text: String): ByteArray {
        val baos = ByteArrayOutputStream()
        XWPFDocument().use { xwpf ->
            xwpf.createParagraph().createRun().setText(text)
            xwpf.write(baos)
        }
        return baos.toByteArray()
    }

    /** Build a DOCX with one paragraph containing multiple runs (multiple w:r elements). */
    private fun buildMultiRunDocx(runs: List<String>): ByteArray {
        val baos = ByteArrayOutputStream()
        XWPFDocument().use { xwpf ->
            val para = xwpf.createParagraph()
            for (run in runs) para.createRun().setText(run)
            xwpf.write(baos)
        }
        return baos.toByteArray()
    }

    /**
     * Build a DOCX with one run containing text, a tab, more text, a line break, and more text.
     * Used to verify that w:tab and w:br are emitted as '\t' and '\n' by the SAX extractor.
     */
    private fun buildTabBreakDocx(): ByteArray {
        val baos = ByteArrayOutputStream()
        XWPFDocument().use { xwpf ->
            val run = xwpf.createParagraph().createRun()
            run.setText("Name:")
            run.addTab()
            run.setText("John")
            run.addBreak()
            run.setText("Springfield")
            xwpf.write(baos)
        }
        return baos.toByteArray()
    }

    /**
     * POI-based oracle capturing the old extraction behavior.
     * Body paragraphs first, then tables (cells space-joined). Used for parity assertions.
     * Note: table order differs from SAX (document order), so only use for paragraph-only fixtures.
     */
    private fun poiReference(docxBytes: ByteArray): String? {
        return try {
            XWPFDocument(docxBytes.inputStream()).use { xwpf ->
                val sb = StringBuilder()
                for (para in xwpf.paragraphs) {
                    val t = para.text
                    if (t.isNotBlank()) { sb.append(t); sb.append('\n') }
                }
                for (table in xwpf.tables) {
                    for (row in table.rows) {
                        for (cell in row.tableCells) {
                            val ct = cell.text.trim()
                            if (ct.isNotBlank()) { sb.append(ct); sb.append(' ') }
                        }
                        sb.append('\n')
                    }
                }
                sb.toString().trim().ifEmpty { null }
            }
        } catch (e: Exception) {
            null
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Security test fixtures
    // ─────────────────────────────────────────────────────────────────────────

    /** Build a zip with multiple entries, each a (name, content) pair. */
    private fun buildMultiEntryZip(entries: List<Pair<String, ByteArray>>): ByteArray {
        val baos = ByteArrayOutputStream()
        ZipOutputStream(baos).use { zip ->
            for ((name, content) in entries) {
                zip.putNextEntry(ZipEntry(name))
                zip.write(content)
                zip.closeEntry()
            }
        }
        return baos.toByteArray()
    }

    /**
     * Build a zip with a single DEFLATED entry named [entryName] containing [content].
     * In streaming mode (bit 3 set), ZipEntry.size == -1 on read-back.
     */
    private fun buildSingleEntryZip(entryName: String, content: ByteArray): ByteArray {
        val baos = ByteArrayOutputStream()
        ZipOutputStream(baos).use { zip ->
            zip.putNextEntry(ZipEntry(entryName))
            zip.write(content)
            zip.closeEntry()
        }
        return baos.toByteArray()
    }

    /**
     * Patches the first entry's local file header so ZipEntry.size == [fakeUncompressedSize]
     * while the actual DEFLATE stream inflates to the original content size.
     *
     * Technique:
     * 1. Read the real CRC-32 and compressed size from the central directory (authoritative).
     * 2. Patch the local file header:
     *    a. Clear bit 3 of general-purpose flags (removes data-descriptor; ZipInputStream
     *       then reads sizes from the local header instead of the trailing descriptor).
     *    b. Write the real CRC-32 and compressed size (so the inflater gets the right data).
     *    c. Write [fakeUncompressedSize] as the uncompressed size (the lie).
     *
     * After patching, ZipInputStream.nextEntry().getSize() == [fakeUncompressedSize],
     * yet reading the entry inflates the full original content. A cap that counts actual
     * inflated bytes (not declared size) will catch this and abort.
     */
    private fun patchFirstEntryUncompressedSize(zipBytes: ByteArray, fakeUncompressedSize: Int): ByteArray {
        val bytes = zipBytes.copyOf()

        // Find End of Central Directory: scan backwards for signature 0x06054b50.
        var eocdOffset = -1
        for (i in bytes.size - 22 downTo maxOf(0, bytes.size - 22 - 65535)) {
            if (bytes.readInt32LE(i) == 0x06054b50.toInt()) { eocdOffset = i; break }
        }
        check(eocdOffset >= 0) { "EOCD not found in zip" }

        val cdOffset = bytes.readInt32LE(eocdOffset + 16)
        check(bytes.readInt32LE(cdOffset) == 0x02014b50.toInt()) { "Central directory signature not found" }

        val crc32 = bytes.readInt32LE(cdOffset + 16)
        val compressedSize = bytes.readInt32LE(cdOffset + 20)
        val localHeaderOffset = bytes.readInt32LE(cdOffset + 42)

        // Local header layout (little-endian):
        //   +0 signature (4)    +4 version needed (2)    +6 flags (2)
        //   +8 method (2)       +10 mod time (2)          +12 mod date (2)
        //   +14 CRC-32 (4)      +18 compressed size (4)   +22 uncompressed size (4)
        val lh = localHeaderOffset
        bytes[lh + 6] = (bytes[lh + 6].toInt() and 0xF7).toByte() // clear bit 3 (data descriptor)
        bytes.writeInt32LE(lh + 14, crc32)
        bytes.writeInt32LE(lh + 18, compressedSize)
        bytes.writeInt32LE(lh + 22, fakeUncompressedSize)

        return bytes
    }

    /** Minimal word/document.xml string for crafting test zip entries. */
    private fun minimalDocumentXml(text: String): String =
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>$text</w:t></w:r></w:p></w:body></w:document>"""

    // ─────────────────────────────────────────────────────────────────────────
    // Zip byte helpers (little-endian int32 read/write)
    // ─────────────────────────────────────────────────────────────────────────

    private fun ByteArray.readInt32LE(offset: Int): Int =
        (this[offset].toInt() and 0xFF) or
        ((this[offset + 1].toInt() and 0xFF) shl 8) or
        ((this[offset + 2].toInt() and 0xFF) shl 16) or
        ((this[offset + 3].toInt() and 0xFF) shl 24)

    private fun ByteArray.writeInt32LE(offset: Int, value: Int) {
        this[offset] = (value and 0xFF).toByte()
        this[offset + 1] = ((value ushr 8) and 0xFF).toByte()
        this[offset + 2] = ((value ushr 16) and 0xFF).toByte()
        this[offset + 3] = ((value ushr 24) and 0xFF).toByte()
    }
}
