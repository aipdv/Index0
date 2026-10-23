package com.narad.search.lib.index.worker

import com.narad.search.lib.index.extraction.DocumentExtractor
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * M18 gate: event wiring for DocumentExtractor.
 *
 * Tests that onWarn fires exactly once for one corrupt PDF and that
 * event messages never contain extracted file content.
 *
 * IndexWorker wires onWarn -> repository.appendEvent. That path requires
 * Android context (instrumented). Here we verify the callback contract on
 * the host JVM, which is the entire observable surface of M18 at this level.
 */
class IndexEventWiringTest {

    // Corrupt PDF bytes: valid header marker, garbage body.
    // PdfBox throws on parse, extractPdf catches Throwable, onWarn fires once.
    private val corruptPdfBytes =
        "%PDF-1.4\n".toByteArray() + ByteArray(64) { (it * 37).toByte() }

    @Test
    fun `one corrupt PDF extraction triggers exactly one event`() {
        val events = mutableListOf<Pair<String, String>>()
        val extractor = DocumentExtractor(
            onWarn = { errorClass, msg -> events.add(errorClass to msg) }
        )

        extractor.extractStream(corruptPdfBytes.inputStream(), "application/pdf")

        assertEquals("Exactly one event for one corrupt PDF", 1, events.size)
    }

    @Test
    fun `successful plain text extraction triggers no events`() {
        val events = mutableListOf<Pair<String, String>>()
        val extractor = DocumentExtractor(
            onWarn = { errorClass, msg -> events.add(errorClass to msg) }
        )

        extractor.extractStream("Valid content for search.".byteInputStream(), "text/plain")

        assertTrue("No events for successful extraction", events.isEmpty())
    }

    @Test
    fun `event content does not contain extracted file content`() {
        // ponytail: known unique string injected into corrupt bytes.
        // If the error message or errorClass leaked file content this would fail.
        val knownContent = "UNIQUE_SECRET_9423_NOT_IN_EVENTS"
        val events = mutableListOf<String>()
        val corruptBytes = knownContent.toByteArray() + ByteArray(80) { it.toByte() }
        val extractor = DocumentExtractor(
            onWarn = { errorClass, msg -> events.add("$errorClass|$msg") }
        )

        extractor.extractStream(corruptBytes.inputStream(), "application/pdf")

        assertEquals("Corrupt extraction must emit one event", 1, events.size)
        events.forEach { event ->
            assertFalse(
                "Event must not contain extracted file content. Got: $event",
                event.contains(knownContent)
            )
        }
    }

    @Test
    fun `DOCX without document XML triggers exactly one event`() {
        val events = mutableListOf<Pair<String, String>>()
        val extractor = DocumentExtractor(
            onWarn = { errorClass, msg -> events.add(errorClass to msg) }
        )
        val output = ByteArrayOutputStream()
        ZipOutputStream(output).use { zip ->
            zip.putNextEntry(ZipEntry("other/file.xml"))
            zip.write("<root/>".toByteArray())
            zip.closeEntry()
        }

        extractor.extractStream(
            output.toByteArray().inputStream(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assertEquals(
            listOf("FileNotFoundException" to "DOCX word/document.xml missing"),
            events
        )
    }

    @Test
    fun `plain text read failure triggers exactly one event`() {
        val events = mutableListOf<Pair<String, String>>()
        val extractor = DocumentExtractor(
            onWarn = { errorClass, msg -> events.add(errorClass to msg) }
        )
        val failingInput = object : InputStream() {
            override fun read(): Int = throw IOException("private file detail")
        }

        extractor.extractStream(failingInput, "text/plain")

        assertEquals(
            listOf("IOException" to "Plain text extraction failed"),
            events
        )
    }
}
