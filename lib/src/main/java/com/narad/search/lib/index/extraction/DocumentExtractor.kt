package com.narad.search.lib.index.extraction

import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.io.MemoryUsageSetting
import com.tom_roush.pdfbox.io.ScratchFile
import com.tom_roush.pdfbox.pdfparser.PDFParser
import com.tom_roush.pdfbox.text.PDFTextStripper
import java.io.ByteArrayOutputStream
import java.io.FilterInputStream
import java.io.InputStream
import java.util.zip.ZipInputStream
import javax.xml.parsers.SAXParserFactory
import org.xml.sax.Attributes
import org.xml.sax.helpers.DefaultHandler

/**
 * Extracts plain text from document files.
 *
 * Split design for host-JVM testability:
 *
 * 1. [extractStream] - runs on host JVM (only android.util.Log, stubbed in tests). Dispatches
 *    by MIME type:
 *    - "application/pdf" -> PdfBox-Android text extraction (Apache-2.0)
 *    - "application/vnd.openxmlformats-officedocument.wordprocessingml.document" (DOCX) ->
 *      ZipInputStream + SAX over word/document.xml. No Apache POI in the production classpath.
 *    - "text/plain", "text/markdown" -> raw InputStream.readText
 *    - anything else (including null) -> returns null (no exception thrown)
 *
 * 2. [extract] - Android wrapper (Android imports allowed). Resolves a
 *    android.net.Uri to an InputStream + MIME type via ContentResolver,
 *    then delegates to [extractStream]. This path is NEEDS-DEVICE and is not
 *    exercised in the host gate tests.
 *
 * Security (KTD6, M6):
 * - Actual inflated bytes are counted as they are read; ZipEntry.size is never consulted.
 *   Entries that expand past [maxDocxBytes] are refused (zip-bomb guard).
 * - word/document.xml is refused if it contains a null byte (UTF-16/32 encoding) OR if its
 *   ISO-8859-1 view contains "<!DOCTYPE". This catches both UTF-8 and UTF-16/32 DOCTYPE
 *   injections. See [containsDoctype].
 * - The SAX parser has external-general-entities and external-parameter-entities disabled as
 *   defence-in-depth. The byte pre-check is the portable authoritative guard because Android's
 *   SAX factory (libcore) does not support "disallow-doctype-decl".
 * - Entries before word/document.xml are drained through a counted reader (not closeEntry),
 *   capped at a separate 100 MB skip budget, so a junk entry cannot inflate unbounded.
 *
 * Bug Watch addressed:
 * - PdfBox-Android: org.bouncycastle excluded in lib/build.gradle.kts (F1).
 *   Encrypted PDFs may throw Error subclasses (NoClassDefFoundError); caught via Throwable.
 * - Encrypted DOCX: the file is an OLE container, not a standard ZIP. ZipInputStream finds
 *   no word/document.xml entry and returns null. No POI dependency needed for this path.
 * - Image URIs are NOT handled here; image files are indexed via the embedding pipeline (F1).
 *
 * M18 event wiring: [onWarn] is called alongside each existing Log.w failure. The caller
 * (IndexWorker) injects a closure that writes the event to [IndexRepository.appendEvent].
 * Messages passed to [onWarn] are short static strings and exception class names only;
 * they never contain extracted file content (KTD18).
 *
 * @param maxDocxBytes Maximum inflated bytes read from word/document.xml. Entries that
 *   expand past this limit are refused. Default is 10 MB (ample for realistic office docs).
 *   Inject a smaller value in tests to keep fixtures small and fast.
 * @param maxDocxEntries Maximum number of ZIP entries visited before aborting. A ZIP with
 *   millions of empty entries can tie up the worker without inflating many bytes; this limit
 *   stops iteration early. Default is 10_000 (real DOCX files have tens to a few hundred
 *   entries). Inject a smaller value in tests.
 * @param onWarn Called alongside each Log.w failure. Receives the exception class name (or a
 *   short token) and a brief static message. Never contains file content. Null = no-op.
 */
class DocumentExtractor(
    // ponytail: injectable cap keeps bomb tests fast; 10 MB covers real DOCX files
    internal val maxDocxBytes: Long = 10L * 1024 * 1024,
    // ponytail: 10_000 entries covers all real DOCX files (tens to hundreds of entries);
    // a ZIP bomb via millions of empty entries is refused early. Raise only if a legitimate
    // DOCX exceeds this (unlikely: OOXML spec defines a small fixed set of parts).
    internal val maxDocxEntries: Int = 10_000,
    // ponytail: callback fires alongside Log.w; null default keeps existing callers unchanged
    internal val onWarn: ((errorClass: String, message: String) -> Unit)? = null,
    // U8: at most this much text per file reaches the embedder. A longer file keeps its first
    // maxTextBytes (bytes for plain text, chars for PDF) and still indexes. 1 MB is far past
    // anything a chunker needs and keeps a huge log or PDF from stalling or OOM-ing a run.
    internal val maxTextBytes: Int = 1024 * 1024,
    // U8: PDFs read at most this many pages. Pages beyond it are not indexed.
    internal val maxPdfPages: Int = 500
) {

    // -------------------------------------------------------------------------
    // Host-JVM-testable core: only android.util.Log below this line
    // -------------------------------------------------------------------------

    /**
     * Extract plain text from [input] given its [mimeType].
     *
     * @param input     Byte stream of the document. The caller owns the stream;
     *                  this function reads but does not close it.
     * @param mimeType  MIME type string (e.g. "application/pdf"). null or
     *                  an unrecognised type returns null.
     * @return          Extracted text, or null if the MIME type is unsupported
     *                  or extraction fails. Never throws.
     */
    fun extractStream(input: InputStream, mimeType: String?): String? {
        if (mimeType == null) return null

        // Normalise: strip parameters (e.g. "text/plain; charset=utf-8" -> "text/plain")
        val baseMime = mimeType.substringBefore(';').trim().lowercase()

        return when (baseMime) {
            "application/pdf" -> extractPdf(input)
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" -> extractDocx(input)
            "text/plain", "text/markdown" -> extractPlainText(input)
            else -> null
        }
    }

    // -------------------------------------------------------------------------
    // Private extraction helpers (no Android imports)
    // -------------------------------------------------------------------------

    private fun extractPdf(input: InputStream): String? {
        return try {
            // Temp-file-only memory setting streams large PDFs through java.io.tmpdir
            // (app cache dir on Android) instead of the heap. We own the ScratchFile so
            // its temp file is deleted even when parsing throws a non-IOException;
            // PDDocument.load only cleans up on IOException.
            ScratchFile(MemoryUsageSetting.setupTempFileOnly()).use { scratch ->
                val parser = PDFParser(scratch.createBuffer(input), "", scratch)
                parser.parse()
                parser.pdDocument.use { doc ->
                    val stripper = PDFTextStripper().apply { endPage = maxPdfPages }
                    val out = CappedWriter(maxTextBytes)
                    try {
                        stripper.writeText(doc, out)
                    } catch (_: TextCapReached) {
                        // Cap hit: keep what was written so far.
                    }
                    out.toString().trim().ifEmpty { null }
                }
            }
        } catch (t: Throwable) {
            // ponytail: catches Throwable not Exception. PdfBox raises Error subclasses:
            // ExceptionInInitializerError (class-init without PDFBoxResourceLoader),
            // NoClassDefFoundError (SecurityProvider / PublicKeySecurityHandler reference BC,
            // which is excluded), OOM, StackOverflow on malformed object graphs.
            // A bare `catch (Exception)` would let these escape, fail the indexing run, and
            // hit the same file on every retry. null is safe: Indexer marks the file indexed
            // and moves on. Do NOT widen catches in IndexWorker (would hide model/DB failures).
            warn(t.javaClass.simpleName, "PDF extraction failed")
            null
        }
    }

    /**
     * Extract text from a DOCX file using ZipInputStream + SAX over word/document.xml.
     *
     * No Apache POI is used here. POI remains on the test classpath only as a fixture builder
     * and parity oracle. After the Gradle split (second commit, post W1 merge), poi-ooxml moves
     * from implementation to testImplementation; this method has no production POI dependency.
     *
     * Security:
     * - Counts actual inflated bytes; ZipEntry.size is never read or trusted.
     * - Entries inflating past [maxDocxBytes] are refused.
     * - Entries before word/document.xml are drained through a 100 MB total skip budget;
     *   closeEntry() is not used for skipped entries.
     * - word/document.xml containing null bytes or "<!DOCTYPE" is refused before SAX parsing.
     * - SAX external entities are disabled.
     */
    private fun extractDocx(input: InputStream): String? {
        val zip = ZipInputStream(object : FilterInputStream(input) {
            override fun close() = Unit
        })
        // ponytail: 100 MB total budget for all entries before word/document.xml;
        // drainCounted enforces this without closeEntry() inflating them unchecked.
        var skipBudget = 100L * 1024 * 1024
        var entryCount = 0
        return try {
            var entry = zip.nextEntry
            while (entry != null) {
                entryCount++
                if (entryCount > maxDocxEntries) {
                    warn("EntryLimitExceeded", "DOCX entry count exceeded limit")
                    return null
                }
                if (entry.name == "word/document.xml") {
                    // readCapped counts actual inflated bytes; never consults entry.size.
                    val bytes = readCapped(zip, maxDocxBytes)
                    if (bytes == null) {
                        warn("CapExceeded", "DOCX word/document.xml exceeded cap")
                        return null // abort: do not call closeEntry on a partially-read bomb
                    }
                    // Found the target; stop iterating. Refuse DOCTYPE before SAX (KTD6).
                    if (containsDoctype(bytes)) {
                        warn("DoctypeRefused", "DOCX DOCTYPE in word/document.xml")
                        return null
                    }
                    return parseSaxDocumentXml(bytes)
                }
                // Skip this entry. Drain through a counted reader (not closeEntry) so that a
                // large junk entry before word/document.xml cannot inflate unbounded.
                val consumed = drainCounted(zip, skipBudget)
                if (consumed == null) {
                    warn("SkipBudgetExceeded", "DOCX skipped entries exceeded 100 MB budget")
                    return null
                }
                skipBudget -= consumed
                entry = zip.nextEntry
            }
            warn(
                java.io.FileNotFoundException::class.java.simpleName,
                "DOCX word/document.xml missing"
            )
            null // no word/document.xml (encrypted DOCX, corrupt zip, or empty archive)
        } catch (t: Throwable) {
            // Catches Throwable for the same reason as extractPdf: OOM or Error subclasses
            // must not escape extractStream and crash the indexing worker on every retry.
            warn(t.javaClass.simpleName, "DOCX extraction failed")
            null
        } finally {
            runCatching { zip.close() }
        }
    }

    /**
     * Returns true if [bytes] should be refused before SAX parsing.
     *
     * Two refusal conditions:
     * 1. Any 0x00 byte: valid UTF-8 XML never contains a null byte; UTF-16/32 always does.
     *    Refusing null bytes blocks UTF-16/32 DOCTYPE injections that bypass the ISO-8859-1 check,
     *    because Android's SAX factory (libcore) does not support "disallow-doctype-decl" and
     *    safeFeature() silently ignores that gap.
     * 2. ISO-8859-1 view contains "<!DOCTYPE" (case-insensitive): catches DOCTYPE in UTF-8 XML.
     *
     * Internal visibility allows tests to exercise this function directly, independent of
     * the SAX parser's own DOCTYPE handling (which also blocks DOCTYPE on some runtimes, but
     * not on Android's libcore SAX factory).
     */
    internal fun containsDoctype(bytes: ByteArray): Boolean {
        // Any null byte means UTF-16/32 encoding (or a deliberately injected null).
        // Refuse all encodings other than UTF-8. OOXML mandates UTF-8 for word/document.xml.
        if (bytes.any { it == 0.toByte() }) return true
        // UTF-8: check for DOCTYPE in ISO-8859-1 view. ISO-8859-1 never throws on decode.
        return bytes.toString(Charsets.ISO_8859_1).contains("<!DOCTYPE", ignoreCase = true)
    }

    /**
     * Reads bytes from [stream] up to [cap], counting actual bytes returned by read().
     * Returns the accumulated bytes, or null if [cap] is exceeded before the stream ends.
     * With [truncate] the first [cap] bytes are returned instead of null, and reading stops there.
     */
    private fun readCapped(stream: InputStream, cap: Long, truncate: Boolean = false): ByteArray? {
        val baos = ByteArrayOutputStream()
        val buf = ByteArray(8192)
        var total = 0L
        var n: Int
        while (stream.read(buf).also { n = it } != -1) {
            if (total + n > cap) {
                if (!truncate) return null
                // Truncate mode: keep the bytes up to the cap and stop reading.
                baos.write(buf, 0, (cap - total).toInt())
                return baos.toByteArray()
            }
            total += n
            baos.write(buf, 0, n)
        }
        return baos.toByteArray()
    }

    /**
     * Reads and discards bytes from [stream] up to [budget] bytes, counting actual bytes.
     * Returns the bytes consumed, or null if [budget] is exceeded.
     * Used to drain skipped zip entries without allocating them in memory.
     */
    private fun drainCounted(stream: InputStream, budget: Long): Long? {
        val buf = ByteArray(8192)
        var total = 0L
        var n: Int
        while (stream.read(buf).also { n = it } != -1) {
            total += n
            if (total > budget) return null
        }
        return total
    }

    /**
     * Parse [bytes] as DOCX word/document.xml and join w:t text runs.
     *
     * SAX is used instead of Android XmlPullParser: on the host JVM XmlPullParser is a stub
     * that returns empty values without an error, making host tests unreliable (KTD6).
     * javax.xml.parsers.SAXParserFactory is available on both Android and JDK 8+.
     *
     * Inline separators:
     * - w:tab   appends '\t' (tab stop between run segments)
     * - w:br    appends '\n' (line break; includes page and column breaks)
     * - w:cr    appends '\n' (carriage return; treated as line break)
     *
     * Paragraph ends (</w:p>) append a newline so multi-paragraph documents read naturally.
     * Tables are extracted in document order (cells interleaved with paragraphs), which differs
     * slightly from the old POI code (paragraphs first, then tables). For plain-text semantic
     * search this is equivalent: we only need the words, not the layout.
     */
    private fun parseSaxDocumentXml(bytes: ByteArray): String? {
        val factory = SAXParserFactory.newInstance()
        factory.isNamespaceAware = true
        // Defence-in-depth: disable external entities. DOCTYPE is already refused by containsDoctype.
        safeFeature(factory, "http://xml.org/sax/features/external-general-entities", false)
        safeFeature(factory, "http://xml.org/sax/features/external-parameter-entities", false)
        // Note: "disallow-doctype-decl" is not supported on Android's libcore SAX factory.
        // containsDoctype() is the authoritative guard for that gap.
        safeFeature(factory, "http://apache.org/xml/features/disallow-doctype-decl", true)

        val sb = StringBuilder()
        var inText = false
        val handler = object : DefaultHandler() {
            override fun startElement(
                uri: String?, localName: String?, qName: String?, attrs: Attributes?
            ) {
                if (uri != WORDPROCESSINGML_NAMESPACE && uri != WORDPROCESSINGML_NAMESPACE_STRICT) return
                when (localName) {
                    "t" -> inText = true
                    "tab" -> sb.append('\t')
                    "br", "cr" -> sb.append('\n')
                }
            }

            override fun endElement(uri: String?, localName: String?, qName: String?) {
                if (uri != WORDPROCESSINGML_NAMESPACE && uri != WORDPROCESSINGML_NAMESPACE_STRICT) return
                when (localName) {
                    "t" -> inText = false
                    // Append newline at paragraph end if there is buffered text.
                    "p" -> if (sb.isNotEmpty() && sb.last() != '\n') sb.append('\n')
                }
            }

            // SAX may split a single text node across multiple characters() calls.
            // Accumulate all of them while inside a w:t element.
            override fun characters(ch: CharArray?, start: Int, length: Int) {
                if (inText && ch != null) sb.append(ch, start, length)
            }
        }

        return try {
            factory.newSAXParser().parse(bytes.inputStream(), handler)
            val result = sb.toString().trim()
            if (result.isEmpty()) null else result
        } catch (e: Exception) {
            warn(e.javaClass.simpleName, "SAX parse failed")
            null
        }
    }

    private fun safeFeature(factory: SAXParserFactory, feature: String, enabled: Boolean) {
        try {
            factory.setFeature(feature, enabled)
        } catch (_: Exception) {
            // Not all runtimes support every feature name. The containsDoctype() pre-check on
            // raw bytes is the portable authoritative guard for DOCTYPE.
        }
    }

    private fun extractPlainText(input: InputStream): String? {
        return try {
            // U8: bounded read. A 5 MB log keeps its first 1 MB (a cut multi-byte character at
            // the end decodes to U+FFFD, harmless for search).
            val text = readCapped(input, maxTextBytes.toLong(), truncate = true)!!
                .toString(Charsets.UTF_8)
            // Return the raw text verbatim (including leading/trailing whitespace),
            // matching the test assertion assertEquals(content, result).
            text
        } catch (t: Throwable) {
            warn(t.javaClass.simpleName, "Plain text extraction failed")
            null
        }
    }

    // -------------------------------------------------------------------------
    // Android wrapper: Android imports are confined to this section only.
    // This function is NOT compiled or executed on the host JVM test gate.
    // -------------------------------------------------------------------------

    /**
     * Android-only convenience wrapper. Resolves [uri] to an InputStream and
     * MIME type via [context]'s ContentResolver, then delegates to [extractStream].
     *
     * Image MIME types (image/jpeg, image/png, etc.) are not handled here; they
     * are indexed separately via the image embedding pipeline (F1: text recognition removed).
     *
     * This path is NEEDS-DEVICE: it cannot be exercised on the host JVM
     * because android.net.Uri and ContentResolver require a real Android runtime.
     *
     * @param uri      Content URI (MediaStore, FileProvider, etc.)
     * @param context  Android context providing access to ContentResolver.
     * @return         Extracted text, or null on unsupported MIME or error.
     */
    fun extract(
        uri: android.net.Uri,
        context: android.content.Context
    ): String? {
        val contentResolver = context.contentResolver
        val mimeType = contentResolver.getType(uri)
        val stream = contentResolver.openInputStream(uri)
        if (stream == null) {
            warn(
                java.io.FileNotFoundException::class.java.simpleName,
                "Document input stream unavailable"
            )
            return null
        }
        return stream.use { extractStream(it, mimeType) }
    }

    private fun warn(errorClass: String, message: String) {
        android.util.Log.w("DocumentExtractor", "$message: $errorClass")
        runCatching { onWarn?.invoke(errorClass, message) }
    }

    /** Thrown by [CappedWriter] to stop PDFTextStripper once the text cap is reached. */
    private class TextCapReached : RuntimeException() {
        // No stack trace: this is control flow, not an error.
        override fun fillInStackTrace(): Throwable = this
    }

    /** Collects text up to [cap] chars, then throws [TextCapReached] so the stripper stops. */
    private class CappedWriter(private val cap: Int) : java.io.Writer() {
        private val sb = StringBuilder()
        override fun write(cbuf: CharArray, off: Int, len: Int) {
            val room = cap - sb.length
            if (len > room) {
                sb.append(cbuf, off, room)
                throw TextCapReached()
            }
            sb.append(cbuf, off, len)
        }
        override fun flush() = Unit
        override fun close() = Unit
        override fun toString(): String = sb.toString()
    }

    companion object {
        // Transitional (OOXML ECMA-376 / ISO 29500 Part 4)
        private const val WORDPROCESSINGML_NAMESPACE =
            "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        // ponytail: Strict (ISO/IEC 29500 Part 1). Same element names, different URI.
        private const val WORDPROCESSINGML_NAMESPACE_STRICT =
            "http://purl.oclc.org/ooxml/wordprocessingml/main"

        /**
         * Initialise PdfBox-Android resources (glyphlist, AFM, CMaps) for the process.
         *
         * Must be called once in `Application.onCreate` before any PDF extraction.
         * Without this call, GlyphList class-init fails with ExceptionInInitializerError
         * and every PDF returns null (see docs/plans/2026-10-09-003-f1-fdroid-license-plan.md).
         *
         * Kept in the lib module so the app module never needs a direct `pdfbox-android`
         * dependency.
         *
         * @param context Any context; PdfBox keeps only the application context's AssetManager.
         */
        fun init(context: android.content.Context) {
            PDFBoxResourceLoader.init(context)
        }
    }
}
