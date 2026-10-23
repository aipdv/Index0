package com.narad.search.lib.query.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.junit.AfterClass
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * On-device trial of EmbeddingGemma 2 (int8 ONNX, onnx-community build) for Hindi -> English
 * retrieval, using the app's DjlTokenizer and ONNX Runtime 1.30. Logs speed and native memory.
 *
 * Setup (corpus = Wikipedia intros, built by the host trial script):
 *   adb shell mkdir -p /data/local/tmp/eg2
 *   adb push model_quantized.onnx model_quantized.onnx_data tokenizer.json docs.json /data/local/tmp/eg2/
 */
@RunWith(AndroidJUnit4::class)
class EmbeddingGemma2TrialTest {

    companion object {
        private const val TAG = "EG2Trial"
        private const val DIR = "/data/local/tmp/eg2"

        private lateinit var env: OrtEnvironment
        private lateinit var session: OrtSession
        private lateinit var tokenizer: DjlTokenizer
        private var sessionNativeMb = 0L
        private var loadMs = 0L

        @BeforeClass @JvmStatic
        fun setUp() {
            env = OrtEnvironment.getEnvironment()
            val before = Debug.getNativeHeapAllocatedSize()
            val t = System.nanoTime()
            // Optional instrumentation arg `eg2Model` selects a variant (e.g. the accuracy_level=4 graph).
            val model = InstrumentationRegistry.getArguments().getString("eg2Model") ?: "$DIR/model_quantized.onnx"
            Log.i(TAG, "model=$model")
            val opts = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(4)
            }
            session = env.createSession(model, opts)
            loadMs = (System.nanoTime() - t) / 1_000_000
            sessionNativeMb = (Debug.getNativeHeapAllocatedSize() - before) / (1024 * 1024)
            tokenizer = DjlTokenizer("$DIR/tokenizer.json")
        }

        @AfterClass @JvmStatic
        fun tearDown() {
            tokenizer.close()
            session.close()
        }
    }

    private fun embed(text: String): FloatArray {
        val tok = tokenizer.encode(text)
        val n = tok.inputIds.size.toLong()
        val empty = FloatBuffer.allocate(0)
        val inputs = mapOf(
            "input_ids" to OnnxTensor.createTensor(env, LongBuffer.wrap(tok.inputIds), longArrayOf(1, n)),
            "attention_mask" to OnnxTensor.createTensor(env, LongBuffer.wrap(tok.attentionMask), longArrayOf(1, n)),
            "image_features" to OnnxTensor.createTensor(env, empty, longArrayOf(0, 512)),
            "video_features" to OnnxTensor.createTensor(env, empty, longArrayOf(0, 512)),
            "audio_features" to OnnxTensor.createTensor(env, empty, longArrayOf(0, 512)),
        )
        try {
            session.run(inputs, setOf("sentence_embedding")).use { out ->
                @Suppress("UNCHECKED_CAST")
                val v = (out[0].value as Array<FloatArray>)[0]
                val norm = sqrt(v.fold(0f) { acc, x -> acc + x * x })
                return FloatArray(v.size) { v[it] / norm }
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    private fun dot(a: FloatArray, b: FloatArray) = a.indices.fold(0f) { acc, i -> acc + a[i] * b[i] }

    @Test
    fun hindiAndHinglishQueriesFindEnglishDocs() {
        val json = JSONArray(File("$DIR/docs.json").readText())
        val docs = (0 until json.length()).map { json.getJSONObject(it) }.filter { it.getString("lang") == "en" }

        var t = System.nanoTime()
        val docVecs = docs.map { it.getString("topic") to embed("title: none | text: " + it.getString("text").take(2000)) }
        val docMs = (System.nanoTime() - t) / 1_000_000 / docs.size

        val queries = listOf(
            "बारिश का मौसम कब और क्यों आता है" to "monsoon",
            "फेफड़ों की बीमारी जो खांसी से फैलती है" to "tb",
            "पौधे सूरज की रोशनी से अपना भोजन कैसे बनाते हैं" to "photo",
            "आगरा में सफेद संगमरमर का मकबरा" to "taj",
            "सामान और सेवाओं पर लगने वाला अप्रत्यक्ष कर" to "gst",
            "सालाना कमाई पर सरकार को दिया जाने वाला टैक्स" to "incometax",
            "किरायेदार को तय समय के लिए ज़मीन का हक" to "lease",
            "बल्ले और गेंद से ग्यारह खिलाड़ियों का खेल" to "cricket",
            "देश का सर्वोच्च कानून और नागरिकों के मौलिक अधिकार" to "constitution",
            "खून में शुगर का स्तर बढ़ जाने की बीमारी" to "diabetes",
            "मोबाइल से तुरंत बैंक खाते में पैसे भेजना" to "upi",
            "हिमालय से निकलकर बंगाल की खाड़ी में गिरने वाली पवित्र नदी" to "ganga",
            "salary pe kitna tax lagta hai" to "incometax",
            "baarish ka mausam kab aata hai" to "monsoon",
            "phone se paise transfer karna" to "upi",
            "blood sugar badhne ki bimari" to "diabetes",
        )
        var hits = 0
        t = System.nanoTime()
        for ((q, topic) in queries) {
            val qv = embed("task: search result | query: $q")
            val best = docVecs.maxBy { dot(it.second, qv) }.first
            if (best == topic) hits++ else Log.w(TAG, "MISS: $q -> $best (want $topic)")
        }
        val queryMs = (System.nanoTime() - t) / 1_000_000 / queries.size

        val report = "RESULT hits=$hits/${queries.size} loadMs=$loadMs docMs=$docMs queryMs=$queryMs " +
            "sessionNativeMb=$sessionNativeMb totalNativeMb=${Debug.getNativeHeapAllocatedSize() / (1024 * 1024)}"
        Log.i(TAG, report)
        println(report)
        assertTrue(report, hits >= queries.size - 1)
    }
}
