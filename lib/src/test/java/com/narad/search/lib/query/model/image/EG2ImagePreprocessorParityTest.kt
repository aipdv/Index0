package com.narad.search.lib.query.model.image

import com.narad.search.lib.query.TestModel
import com.narad.search.lib.query.model.DjlTokenizer
import com.narad.search.lib.query.model.EmbeddingModel
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * JVM parity test for [EG2ImagePreprocessor] and the image embedding pipeline.
 *
 * For 2 synthetic test images (linear-gradient, generated with the same formula
 * used to create the Python fixtures - no copyright):
 *   (a) Resized dimensions and n_soft match Python reference.
 *   (b) pixel_values max-abs-diff <= 0.02 vs Python bicubic reference.
 *   (c) position_ids match exactly.
 *   (d) Full image embedding cosine >= 0.98 vs Python reference embedding.
 *
 * Fixture files (in test/resources):
 *   landscape_pixel_values.bin / portrait_pixel_values.bin  - Python reference (float32 LE, 630x768)
 *   landscape_position_ids.bin / portrait_position_ids.bin  - Python reference (int64 LE, 630x2)
 *   landscape_embedding.bin / portrait_embedding.bin        - Python reference (float32 LE, 256)
 *
 * ARGB pixels are generated programmatically (same formula as the PIL PNG generation)
 * to avoid java.awt dependency. On a linear gradient, bilinear == bicubic exactly.
 */
class EG2ImagePreprocessorParityTest {

    private lateinit var embeddingModel: EmbeddingModel
    private lateinit var visionEncoder: VisionEncoder
    private val preprocessor = EG2ImagePreprocessor(BilinearResizeFn)

    @Before
    fun setUp() {
        TestModel.skipIfMissing()
        val tokenizer = DjlTokenizer(TestModel.tokenizerJsonPath)
        embeddingModel = EmbeddingModel(TestModel.onnxPath, tokenizer)
        visionEncoder  = VisionEncoder(TestModel.visionEncoderPath)
    }

    @After
    fun tearDown() {
        if (::visionEncoder.isInitialized) visionEncoder.close()
        if (::embeddingModel.isInitialized) embeddingModel.close()
    }

    // -------------------------------------------------------------------------
    // Landscape 640x427 (R=left-to-right, G=top-to-bottom, B=128)
    // -------------------------------------------------------------------------

    @Test
    fun landscape_resize_dims_and_nsoft_match_python_reference() {
        val (tw, th) = preprocessor.computeTargetSize(640, 427)
        assertEquals("landscape tw", 480, tw)
        assertEquals("landscape th", 288, th)
        val nSoft = (tw / EG2ImagePreprocessor.SIDE_MULT) * (th / EG2ImagePreprocessor.SIDE_MULT)
        assertEquals("landscape n_soft", 60, nSoft)
    }

    @Test
    fun landscape_pixel_values_max_abs_diff_le_0_02_vs_python_bicubic_reference() {
        val result = preprocessLandscape()
        val ref    = loadFloatBin("landscape_pixel_values.bin")
        val diff   = maxAbsDiff(result.pixelValues, ref)
        println("Landscape pixel_values max-abs-diff: $diff")
        assertTrue("pixel_values max-abs-diff $diff must be <= 0.02", diff <= 0.02f)
    }

    @Test
    fun landscape_position_ids_match_python_reference_exactly() {
        val result = preprocessLandscape()
        val ref    = loadLongBin("landscape_position_ids.bin")
        assertArrayEquals("landscape position_ids", ref, result.positionIds)
    }

    @Test
    fun landscape_full_image_embedding_cosine_ge_0_98_vs_python_reference() {
        val result        = preprocessLandscape()
        val imageFeatures = visionEncoder.encode(result.pixelValues, result.positionIds)
        val nSoft         = imageFeatures.size / VisionEncoder.FEAT_DIM
        val kotlinEmb     = embeddingModel.embedImageFeatures(imageFeatures, nSoft)
        val refEmb        = loadFloatBin("landscape_embedding.bin")
        val cosine        = cosine(kotlinEmb, refEmb)
        println("Landscape image embedding cosine vs Python: $cosine")
        assertTrue("embedding cosine $cosine must be >= 0.98", cosine >= 0.98f)
    }

    // -------------------------------------------------------------------------
    // Portrait 300x900 (diagonal gradient)
    // -------------------------------------------------------------------------

    @Test
    fun portrait_resize_dims_and_nsoft_match_python_reference() {
        val (tw, th) = preprocessor.computeTargetSize(300, 900)
        assertEquals("portrait tw", 192, tw)
        assertEquals("portrait th", 672, th)
        val nSoft = (tw / EG2ImagePreprocessor.SIDE_MULT) * (th / EG2ImagePreprocessor.SIDE_MULT)
        assertEquals("portrait n_soft", 56, nSoft)
    }

    @Test
    fun portrait_pixel_values_max_abs_diff_le_0_02_vs_python_bicubic_reference() {
        val result = preprocessPortrait()
        val ref    = loadFloatBin("portrait_pixel_values.bin")
        val diff   = maxAbsDiff(result.pixelValues, ref)
        println("Portrait pixel_values max-abs-diff: $diff")
        assertTrue("pixel_values max-abs-diff $diff must be <= 0.02", diff <= 0.02f)
    }

    @Test
    fun portrait_position_ids_match_python_reference_exactly() {
        val result = preprocessPortrait()
        val ref    = loadLongBin("portrait_position_ids.bin")
        assertArrayEquals("portrait position_ids", ref, result.positionIds)
    }

    @Test
    fun portrait_full_image_embedding_cosine_ge_0_98_vs_python_reference() {
        val result        = preprocessPortrait()
        val imageFeatures = visionEncoder.encode(result.pixelValues, result.positionIds)
        val nSoft         = imageFeatures.size / VisionEncoder.FEAT_DIM
        val kotlinEmb     = embeddingModel.embedImageFeatures(imageFeatures, nSoft)
        val refEmb        = loadFloatBin("portrait_embedding.bin")
        val cosine        = cosine(kotlinEmb, refEmb)
        println("Portrait image embedding cosine vs Python: $cosine")
        assertTrue("embedding cosine $cosine must be >= 0.98", cosine >= 0.98f)
    }

    // -------------------------------------------------------------------------
    // Pixel generation (same formula as the PIL synthetic image generation)
    // -------------------------------------------------------------------------

    /** Generate ARGB IntArray for landscape_640x427.
     *  R = x*255/639, G = y*255/426, B = 128. Linear gradients. */
    private fun makeLandscapePixels(): Triple<IntArray, Int, Int> {
        val w = 640; val h = 427
        val pixels = IntArray(w * h) { idx ->
            val x = idx % w; val y = idx / w
            val r = (x * 255 / (w - 1))
            val g = (y * 255 / (h - 1))
            val b = 128
            (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Triple(pixels, w, h)
    }

    /** Generate ARGB IntArray for portrait_300x900.
     *  R = (x + y*300/900)*255/(300+300), G=100, B=y*255/899. */
    private fun makePortraitPixels(): Triple<IntArray, Int, Int> {
        val w = 300; val h = 900
        val pixels = IntArray(w * h) { idx ->
            val x = idx % w; val y = idx / w
            val rRaw = ((x + y * w.toDouble() / h) * 255 / (w + w)).toInt().coerceIn(0, 255)
            val g = 100
            val b = (y * 255 / (h - 1))
            (0xFF shl 24) or (rRaw shl 16) or (g shl 8) or b
        }
        return Triple(pixels, w, h)
    }

    private fun preprocessLandscape(): EG2ImagePreprocessor.PreprocessResult {
        val (pix, w, h) = makeLandscapePixels()
        return preprocessor.preprocess(pix, w, h)
    }

    private fun preprocessPortrait(): EG2ImagePreprocessor.PreprocessResult {
        val (pix, w, h) = makePortraitPixels()
        return preprocessor.preprocess(pix, w, h)
    }

    // -------------------------------------------------------------------------
    // Fixture IO helpers
    // -------------------------------------------------------------------------

    private fun loadFloatBin(resource: String): FloatArray {
        val bytes = javaClass.classLoader!!.getResourceAsStream(resource)!!.readBytes()
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buf.remaining()).also { buf.get(it) }
    }

    private fun loadLongBin(resource: String): LongArray {
        val bytes = javaClass.classLoader!!.getResourceAsStream(resource)!!.readBytes()
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer()
        return LongArray(buf.remaining()).also { buf.get(it) }
    }

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
        var max = 0f
        for (i in a.indices) { val d = abs(a[i] - b[i]); if (d > max) max = d }
        return max
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0.0; var na = 0.0; var nb = 0.0
        for (i in a.indices) {
            dot += a[i].toDouble() * b[i].toDouble()
            na  += a[i].toDouble() * a[i].toDouble()
            nb  += b[i].toDouble() * b[i].toDouble()
        }
        val denom = sqrt(na * nb)
        return if (denom < 1e-12) 0f else (dot / denom).toFloat()
    }
}
