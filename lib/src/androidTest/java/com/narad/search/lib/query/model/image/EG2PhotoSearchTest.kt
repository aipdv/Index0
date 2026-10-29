package com.narad.search.lib.query.model.image

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository
import com.narad.search.lib.query.engine.QueryEngine
import com.narad.search.lib.query.model.DjlTokenizer
import com.narad.search.lib.query.model.EmbeddingModel
import com.narad.search.lib.query.store.VectorStore
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.max
import kotlin.system.measureTimeMillis

// Device test (M2 DoD #2): EG2 photo search on 15 Wikipedia images.
// Push images first: adb push /tmp/narad-eg2/images /data/local/tmp/eg2/images
// Push models to: /data/local/tmp/eg2/models/embeddinggemma-2/ (model_quantized.onnx,
//   model_quantized.onnx_data, vision_encoder_quantized.onnx,
//   vision_encoder_quantized.onnx_data, tokenizer.json)
// Hard asserts: >=14/15 English, >=14/15 Hindi top-1. Hinglish >=12/15 (softer).
@RunWith(AndroidJUnit4::class)
class EG2PhotoSearchTest {

    companion object {
        private const val IMAGE_DIR = "/data/local/tmp/eg2/images"
        private const val MODEL_DIR = "/data/local/tmp/eg2"  // same as M1 trial setup

        // Key -> (English, Hindi, Hinglish) - mirrors eval_images.py
        private val QUERIES = mapOf(
            "taj"      to Triple("white marble monument with domes",
                                 "सफेद संगमरमर का गुंबद वाला स्मारक",
                                 "safed gumbad wala makbara"),
            "cricket"  to Triple("batsman playing a cricket match",
                                 "क्रिकेट मैच खेलता बल्लेबाज़",
                                 "cricket match ki photo"),
            "ganga"    to Triple("ghats on the river in Varanasi",
                                 "बनारस में नदी के घाट",
                                 "banaras ke ghat"),
            "dog"      to Triple("dogs resting in the snow",
                                 "बर्फ में आराम करते कुत्ते",
                                 "kutte baraf me"),
            "cat"      to Triple("a cat with blue eyes",
                                 "नीली आंखों वाली बिल्ली",
                                 "billi ki photo"),
            "biryani"  to Triple("plate of rice biryani",
                                 "बिरयानी की प्लेट",
                                 "biryani wala khana"),
            "rickshaw" to Triple("three wheeled auto rickshaw",
                                 "तीन पहिये वाला ऑटो रिक्शा",
                                 "auto rickshaw"),
            "elephant" to Triple("elephant in the forest",
                                 "जंगल में हाथी",
                                 "haathi jungle me"),
            "mango"    to Triple("ripe mango cut in half",
                                 "आधा कटा हुआ पका आम",
                                 "kata hua aam"),
            "diwali"   to Triple("oil lamps on rangoli",
                                 "रंगोली पर जलते दीये",
                                 "diye aur rangoli"),
            "everest"  to Triple("snowy mountain peak",
                                 "बर्फ से ढकी पहाड़ की चोटी",
                                 "barfeela pahad"),
            "bicycle"  to Triple("old black bicycle",
                                 "पुरानी काली साइकिल",
                                 "purani cycle"),
            "laptop"   to Triple("laptop computers",
                                 "लैपटॉप कंप्यूटर",
                                 "laptop ki photo"),
            "train"    to Triple("railway company logo",
                                 "रेलवे का लोगो",
                                 "railway ka logo"),
            "banana"   to Triple("bunches of bananas",
                                 "केले के गुच्छे",
                                 "kele ka guchha"),
        )
    }

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private lateinit var embeddingModel: EmbeddingModel
    private lateinit var visionEncoder: VisionEncoder

    @Before
    fun setUp() {
        val modelBase = "$MODEL_DIR/models/embeddinggemma-2"
        val modelPath = "$modelBase/model_quantized.onnx"
        val tokenizerPath = "$modelBase/tokenizer.json"
        val visionPath = "$modelBase/vision_encoder_quantized.onnx"

        require(File(modelPath).exists()) { "LM graph not found at $modelPath" }
        require(File(visionPath).exists()) { "Vision graph not found at $visionPath" }
        require(File(tokenizerPath).exists()) { "tokenizer.json not found at $tokenizerPath" }

        val tokenizer = DjlTokenizer(tokenizerPath)
        embeddingModel = EmbeddingModel(modelPath, tokenizer)
        visionEncoder  = VisionEncoder(visionPath)
    }

    @After
    fun tearDown() {
        visionEncoder.close()
        embeddingModel.close()
    }

    @Test
    fun photo_search_production_path_scores_ge14_15_english_and_hindi() {
        // ------------------------------------------------------------------
        // 1. Discover image files
        // ------------------------------------------------------------------
        val imageDir = File(IMAGE_DIR)
        require(imageDir.isDirectory) { "Push images to $IMAGE_DIR first" }
        val imageFiles = imageDir.listFiles { f -> f.extension in setOf("jpg", "jpeg", "png") }
            ?.sortedBy { it.nameWithoutExtension }
            ?: error("No image files found in $IMAGE_DIR")

        println("Found ${imageFiles.size} images in $IMAGE_DIR")
        require(imageFiles.size == 15) { "Expected 15 images, got ${imageFiles.size}" }

        // ------------------------------------------------------------------
        // 2. Embed each photo → in-memory 256-dim VectorStore
        // ------------------------------------------------------------------
        val preprocessor = EG2ImagePreprocessor { srcPixels, srcW, srcH, dstW, dstH ->
            // Android bilinear via Bitmap API
            val src = Bitmap.createBitmap(srcW, srcH, Bitmap.Config.ARGB_8888)
            src.setPixels(srcPixels, 0, srcW, 0, 0, srcW, srcH)
            val scaled = Bitmap.createScaledBitmap(src, dstW, dstH, true)
            src.recycle()
            val dst = IntArray(dstW * dstH)
            scaled.getPixels(dst, 0, dstW, 0, 0, dstW, dstH)
            if (scaled !== src) scaled.recycle()
            dst
        }

        // Persist through the production repository, then rehydrate for search (round trip).
        val db = IndexDatabase(context, "eg2_photo_test_${System.currentTimeMillis()}.db")
        val repo = IndexRepository(db)
        val keyToFilePath    = mutableMapOf<String, String>()
        var totalEmbedMs     = 0L

        for (imgFile in imageFiles) {
            val key = imgFile.nameWithoutExtension  // e.g. "taj"
            var bitmap: Bitmap? = null
            val embedMs = measureTimeMillis {
                // Two-pass decode with inSampleSize (OOM guard for large photos)
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(imgFile.absolutePath, bounds)
                val maxDim = max(bounds.outWidth, bounds.outHeight)
                val sampleSize = if (maxDim > 1080) {
                    val ceil = (maxDim + 1079) / 1080
                    Integer.highestOneBit(ceil).let { if (it < ceil) it shl 1 else it }
                } else 1
                bitmap = BitmapFactory.decodeFile(imgFile.absolutePath,
                    BitmapFactory.Options().apply { inSampleSize = sampleSize })

                val bmp = bitmap!!
                val w = bmp.width; val h = bmp.height
                val pixels = IntArray(w * h)
                bmp.getPixels(pixels, 0, w, 0, 0, w, h)

                val pre           = preprocessor.preprocess(pixels, w, h)
                val imageFeatures = visionEncoder.encode(pre.pixelValues, pre.positionIds)
                val nSoft         = imageFeatures.size / VisionEncoder.FEAT_DIM
                val embedding     = embeddingModel.embedImageFeatures(imageFeatures, nSoft)

                repo.upsertImageVector(id = key, filePath = imgFile.absolutePath, vector = embedding)
                keyToFilePath[key] = imgFile.absolutePath
            }
            bitmap?.recycle()
            totalEmbedMs += embedMs
            println("  $key embedded in ${embedMs} ms")
        }

        val msPerPhoto = totalEmbedMs / imageFiles.size
        println("Average embed: $msPerPhoto ms/photo")

        // ------------------------------------------------------------------
        // 3. Build a QueryEngine with the image VectorStore
        // ------------------------------------------------------------------
        val imageVectorStore = VectorStore()
        repo.rehydrateImages(imageVectorStore)
        assertTrue("all photos must persist and rehydrate", imageVectorStore.size == imageFiles.size)
        val queryEngine = QueryEngine(
            embeddingModel    = embeddingModel,
            vectorStore       = VectorStore(),  // empty text store (not tested here)
            keywordStore      = com.narad.search.lib.query.store.KeywordStore(),
            imageVectorStore  = imageVectorStore
        )

        // ------------------------------------------------------------------
        // 4. Run queries for all 3 languages
        // ------------------------------------------------------------------
        val results = mutableMapOf<String, Triple<Int, Int, List<String>>>()  // lang -> (top1, total, misses)

        for ((langIdx, langName) in listOf("English", "Hindi", "Hinglish").withIndex()) {
            var top1 = 0
            val misses = mutableListOf<String>()

            for ((key, queryTriple) in QUERIES) {
                val query = when (langIdx) {
                    0 -> queryTriple.first
                    1 -> queryTriple.second
                    else -> queryTriple.third
                }

                val hits = queryEngine.searchImages(query, topK = 15)
                val topKey = hits.firstOrNull()?.let { hit ->
                    File(hit.filePath).nameWithoutExtension
                } ?: ""

                if (topKey == key) {
                    top1++
                } else {
                    misses.add("$query -> $topKey (expected $key)")
                }
            }
            results[langName] = Triple(top1, QUERIES.size, misses)
        }

        // ------------------------------------------------------------------
        // 5. Report and assert
        // ------------------------------------------------------------------
        println("\n=== EG2 Photo Search Results ===")
        println("ms/photo: $msPerPhoto")
        for ((lang, r) in results) {
            val (top1, total, misses) = r
            println("$lang: $top1/$total top-1")
            misses.forEach { println("  MISS: $it") }
        }

        db.close()

        val (enTop1, enTotal, _) = results["English"]!!
        val (hiTop1, hiTotal, _) = results["Hindi"]!!
        val (hlTop1, hlTotal, _) = results["Hinglish"]!!

        assertTrue("English top-1 $enTop1/$enTotal must be >= 14/15", enTop1 >= 14)
        assertTrue("Hindi top-1 $hiTop1/$hiTotal must be >= 14/15", hiTop1 >= 14)
        // Hinglish: report but softer assert
        assertTrue("Hinglish top-1 $hlTop1/$hlTotal must be >= 12/15", hlTop1 >= 12)
    }
}
