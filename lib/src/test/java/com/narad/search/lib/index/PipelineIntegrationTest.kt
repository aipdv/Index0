package com.narad.search.lib.index

import com.narad.search.lib.index.chunking.Chunker
import com.narad.search.lib.index.extraction.DocumentExtractor
import com.narad.search.lib.query.TestModel
import com.narad.search.lib.query.engine.QueryEngine
import com.narad.search.lib.query.model.EmbeddingModel
import com.narad.search.lib.query.model.HuggingFaceTokenizer
import com.narad.search.lib.query.store.KeywordStore
import com.narad.search.lib.query.store.VectorStore
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.pdmodel.PDPage
import com.tom_roush.pdfbox.pdmodel.PDPageContentStream
import com.tom_roush.pdfbox.pdmodel.font.PDType1Font
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.io.ByteArrayOutputStream

/**
 * Integration test: build a PDF in-test with PdfBox → [DocumentExtractor.extractStream] →
 * [Chunker.chunk] → embed each chunk with real [EmbeddingModel] → add to [VectorStore] and
 * [KeywordStore] → [QueryEngine.search] for a phrase from the passage returns that content.
 *
 * Proves the pipeline end-to-end on the host JVM.
 */
class PipelineIntegrationTest {

    private lateinit var tokenizer: HuggingFaceTokenizer
    private lateinit var embeddingModel: EmbeddingModel
    private lateinit var vectorStore: VectorStore
    private lateinit var keywordStore: KeywordStore
    private lateinit var queryEngine: QueryEngine
    private val extractor = DocumentExtractor()
    private val chunker = Chunker()

    @Before
    fun setUp() {
        TestModel.skipIfMissing()
        tokenizer = HuggingFaceTokenizer(TestModel.tokenizerJsonPath)
        embeddingModel = EmbeddingModel(TestModel.onnxPath, tokenizer)
        vectorStore = VectorStore()
        keywordStore = KeywordStore()
        queryEngine = QueryEngine(embeddingModel, vectorStore, keywordStore)
    }

    @After
    fun tearDown() {
        if (::embeddingModel.isInitialized) embeddingModel.close()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Main integration: PdfBox-built PDF → extract → chunk → embed → index → query
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `pipeline PDF extract chunk embed index and search returns matching result`() {
        val filePath = "/fake/docs/rental-agreement.pdf"

        // Build a PDF with a passage that contains the query keyword "penalty".
        val passage = "This rental agreement includes a penalty clause. The tenant agrees to pay " +
            "a fine equivalent to ten percent of the monthly rent for any breach of contract."
        val pdfBytes = buildHelveticaPdf(passage)

        // Step 1: Extract text from the PDF
        val extracted = extractor.extractStream(pdfBytes.inputStream(), "application/pdf")
        assertNotNull("PDF extraction must return non-null text", extracted)
        assertTrue("Extracted text must contain known content", extracted!!.contains("penalty"))

        // Step 2: Chunk the extracted text
        val chunks = chunker.chunk(extracted)
        assertTrue("Chunking must produce at least one chunk", chunks.isNotEmpty())

        // Step 3: Embed each chunk and add to stores.
        chunks.forEachIndexed { offset, chunkText ->
            val chunkId = "$filePath#$offset"
            val embedding = embeddingModel.embedDocument(chunkText)
            vectorStore.add(chunkId, filePath, offset, "embeddinggemma-2", embedding)
            keywordStore.add(chunkId, filePath, offset, chunkText)
        }

        // Also index an unrelated document (so search must discriminate)
        val unrelatedPath = "/fake/docs/weather.txt"
        val unrelatedPassage = "The weather forecast predicts sunny skies with light clouds. " +
            "Temperature will remain around 25 degrees Celsius."
        val unrelatedChunks = chunker.chunk(unrelatedPassage)
        unrelatedChunks.forEachIndexed { offset, chunkText ->
            val chunkId = "$unrelatedPath#$offset"
            val embedding = embeddingModel.embedDocument(chunkText)
            vectorStore.add(chunkId, unrelatedPath, offset, "embeddinggemma-2", embedding)
            keywordStore.add(chunkId, unrelatedPath, offset, chunkText)
        }

        // Step 4: Search
        val results = queryEngine.search("penalty clause rental agreement", topK = 10)

        assertTrue("Search must return at least one result", results.isNotEmpty())

        val pdfResult = results.find { it.filePath == filePath }
        assertNotNull(
            "PDF document must appear in search results.\nAll results: ${results.map { it.filePath }}",
            pdfResult
        )

        val weatherResult = results.find { it.filePath == unrelatedPath }
        if (weatherResult != null && pdfResult != null) {
            assertTrue(
                "PDF (score=${pdfResult.score}) must rank above weather doc (score=${weatherResult.score})",
                pdfResult.score >= weatherResult.score
            )
        }

        println("Pipeline integration results:")
        results.forEachIndexed { i, r -> println("  ${i + 1}. ${r.filePath} score=${r.score}") }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Verify the chunker produces non-empty chunks from PdfBox-extracted text
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    fun `pipeline chunks from extracted PDF are all non-empty`() {
        // 30 lines, ~1350 chars: forces several 512-char chunks, as at 22ae886.
        val pdfBytes = buildHelveticaPdf(*Array(30) { "The penalty clause is enforceable under law." })
        val extracted = extractor.extractStream(pdfBytes.inputStream(), "application/pdf")

        assertNotNull(extracted)
        val chunks = chunker.chunk(extracted!!)
        assertTrue("Should produce several chunks, got ${chunks.size}", chunks.size >= 2)
        for (chunk in chunks) {
            assertTrue("Each chunk must be non-empty", chunk.isNotBlank())
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helper
    // ─────────────────────────────────────────────────────────────────────────

    /** One page, one Helvetica text line per entry in [lines]. */
    private fun buildHelveticaPdf(vararg lines: String): ByteArray {
        val baos = ByteArrayOutputStream()
        PDDocument().use { doc ->
            val page = PDPage()
            doc.addPage(page)
            PDPageContentStream(doc, page).use { cs ->
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
            doc.save(baos)
        }
        return baos.toByteArray()
    }
}
