package com.narad.search.lib.index.extraction

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.pdmodel.PDPage
import com.tom_roush.pdfbox.pdmodel.PDPageContentStream
import com.tom_roush.pdfbox.pdmodel.encryption.AccessPermission
import com.tom_roush.pdfbox.pdmodel.encryption.StandardProtectionPolicy
import com.tom_roush.pdfbox.pdmodel.font.PDType1Font
import org.junit.Assert.*
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream

/**
 * Device (instrumented) tests for [DocumentExtractor].
 *
 * Fixture coverage:
 *   - English Helvetica PDF built in-test: must contain "quick brown fox"
 *   - Hindi Unicode PDF from assets (CoreText-shaped, N8): must contain समझौता, मकान, जमा
 *   - User-password PDF: must return null without throwing
 *   - Class-init poisoning order: encrypted first, then English - English must still extract
 *
 * Hindi fixture (`lib/src/androidTest/assets/pdf/hindi_unicode.pdf`) generated with the CoreText
 * Swift snippet (CoreText, Kohinoor Devanagari) from text:
 *   "किराया समझौता / मकान मालिक को जमा राशि लौटानी होगी।"
 * with font "Kohinoor Devanagari", size 18. Words with the ि matra extract in visual order
 * (`िकराया` → broken); assert only the three words without ि: समझौता, मकान, जमा.
 *
 * [PDFBoxResourceLoader.init] is called in [setUpClass] using the instrumentation target context.
 * The lib test APK has no [NaradApplication], so we call the lib helper directly here.
 */
@RunWith(AndroidJUnit4::class)
class PdfExtractionDeviceTest {

    private val extractor = DocumentExtractor()

    companion object {
        @BeforeClass
        @JvmStatic
        fun setUpClass() {
            // Initialise PdfBox-Android glyph lists and AFM data once per test process.
            // On device this must happen before any PDDocument or PDType1Font access.
            val ctx = InstrumentationRegistry.getInstrumentation().targetContext
            DocumentExtractor.init(ctx)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // G5 case 1: English Helvetica PDF built in-test - T1
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun extractEnglishPdfReturnsKnownWords() {
        val knownText = "The quick brown fox jumps over the lazy dog."
        val pdfBytes = buildHelveticaPdf(knownText)

        val result = extractor.extractStream(pdfBytes.inputStream(), "application/pdf")

        assertNotNull("English PDF extraction must not return null", result)
        assertTrue(
            "Extracted text must contain 'quick brown fox'.\nExtracted: $result",
            result!!.contains("quick brown fox")
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // G5 case 2: Hindi Unicode fixture - T2
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun extractHindiPdfReturnsDevanagariWords() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val pdfBytes = ctx.assets.open("pdf/hindi_unicode.pdf").readBytes()

        val result = extractor.extractStream(pdfBytes.inputStream(), "application/pdf")

        // Print full extracted string so G5 logcat grep can verify.
        println("F1-HINDI: $result")

        assertNotNull("Hindi PDF extraction must not return null", result)
        // Words without the ि matra: CoreText outputs them correctly.
        // Words with ि (किराया, मालिक, राशि) extract in visual order - do not assert those.
        assertTrue("Must contain समझौता", result!!.contains("समझौता"))
        assertTrue("Must contain मकान", result.contains("मकान"))
        assertTrue("Must contain जमा", result.contains("जमा"))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // G5 case 3: User-password PDF - T3
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun extractEncryptedPdfReturnsNull() {
        val pdfBytes = buildPasswordProtectedPdf(userPassword = "s3cr3t")

        val result = extractor.extractStream(pdfBytes.inputStream(), "application/pdf")

        assertNull("User-password PDF must return null (not throw)", result)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // G5 case 4: Class-init poisoning order check
    // Extract encrypted first, then English - English must still work.
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun classInitOrderEncryptedThenEnglishStillExtracts() {
        // Encrypted first
        val encBytes = buildPasswordProtectedPdf(userPassword = "poison")
        val encResult = extractor.extractStream(encBytes.inputStream(), "application/pdf")
        assertNull("Encrypted PDF must return null", encResult)

        // English second - must not be poisoned by the earlier null-returning extraction
        val engBytes = buildHelveticaPdf("Class init order check passes if this extracts.")
        val engResult = extractor.extractStream(engBytes.inputStream(), "application/pdf")
        assertNotNull("English PDF after encrypted must still extract", engResult)
        assertTrue(engResult!!.contains("order check"))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private fun buildHelveticaPdf(text: String): ByteArray {
        val baos = ByteArrayOutputStream()
        PDDocument().use { doc ->
            val page = PDPage()
            doc.addPage(page)
            PDPageContentStream(doc, page).use { cs ->
                cs.beginText()
                cs.setFont(PDType1Font.HELVETICA, 12f)
                cs.newLineAtOffset(72f, 720f)
                cs.showText(text)
                cs.endText()
            }
            doc.save(baos)
        }
        return baos.toByteArray()
    }

    /** Page has real text, so a null result proves the password check, not an empty page. */
    private fun buildPasswordProtectedPdf(userPassword: String): ByteArray {
        val baos = ByteArrayOutputStream()
        PDDocument().use { doc ->
            val page = PDPage()
            doc.addPage(page)
            PDPageContentStream(doc, page).use { cs ->
                cs.beginText()
                cs.setFont(PDType1Font.HELVETICA, 12f)
                cs.newLineAtOffset(72f, 720f)
                cs.showText("Secret deposit clause.")
                cs.endText()
            }
            val policy = StandardProtectionPolicy(
                "ownerPass",
                userPassword,
                AccessPermission()
            )
            policy.encryptionKeyLength = 40  // RC4-40 - no BouncyCastle dependency
            doc.protect(policy)
            doc.save(baos)
        }
        return baos.toByteArray()
    }
}
