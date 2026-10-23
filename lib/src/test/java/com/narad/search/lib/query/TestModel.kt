package com.narad.search.lib.query

import java.io.File
import org.junit.Assume

/**
 * Resolves the `models/embeddinggemma-2` directory robustly.
 *
 * Gradle's test working directory may be the module root (`lib/`) or the repo
 * root depending on the Gradle version and configuration.  This helper walks
 * upward from `user.dir` until it finds the marker directory, then falls back
 * to an absolute path.
 *
 * Model: EmbeddingGemma 2 (OUR tuned ONNX, MatMulNBits accuracy_level=4).
 * Device layout (FLAT): `models/embeddinggemma-2/{model_quantized.onnx,
 * model_quantized.onnx_data, tokenizer.json}`.
 *
 * CI gate: set NARAD_SKIP_MODEL_TESTS=1 to skip the model-backed suites.
 * Call [skipIfMissing] at the top of every @Before or @BeforeClass that
 * accesses model files.
 */
object TestModel {

    private const val FALLBACK_ROOT = "/Users/prada/Coding/narad-eg2"
    private const val MODEL_DIR_NAME = "models/embeddinggemma-2"

    /**
     * Returns an absolute [File] pointing to the `models/embeddinggemma-2`
     * directory.  Throws [IllegalStateException] if the directory cannot be
     * found or does not exist.
     */
    val modelDir: File by lazy {
        findModelDir() ?: File(FALLBACK_ROOT, MODEL_DIR_NAME).also {
            check(it.exists()) {
                "Could not locate embeddinggemma-2 model directory.  " +
                    "Expected at $FALLBACK_ROOT/$MODEL_DIR_NAME or a parent of ${System.getProperty("user.dir")}"
            }
        }
    }

    /** Absolute path to the tuned ONNX graph file (498 KB; data file in same dir). */
    val onnxPath: String by lazy {
        val f = File(modelDir, "model_quantized.onnx")
        check(f.exists()) { "ONNX model not found at ${f.absolutePath}" }
        f.absolutePath
    }

    /** Absolute path to the `tokenizer.json` file. */
    val tokenizerJsonPath: String by lazy {
        val f = File(modelDir, "tokenizer.json")
        check(f.exists()) { "tokenizer.json not found at ${f.absolutePath}" }
        f.absolutePath
    }

    /** Absolute path to the vision encoder ONNX graph (165 KB). */
    val visionEncoderPath: String by lazy {
        val f = File(modelDir, "vision_encoder_quantized.onnx")
        check(f.exists()) { "vision_encoder_quantized.onnx not found at ${f.absolutePath}" }
        f.absolutePath
    }

    // -------------------------------------------------------------------------
    // CI skip helpers
    // -------------------------------------------------------------------------

    /**
     * Returns true only when model tests should run.
     * False when NARAD_SKIP_MODEL_TESTS=1 or the model folder is absent.
     * Never throws.
     */
    fun shouldRunModelTests(): Boolean {
        if (System.getenv("NARAD_SKIP_MODEL_TESTS") == "1") return false
        return findModelDir() != null || File(FALLBACK_ROOT, MODEL_DIR_NAME).isDirectory
    }

    /**
     * Skips the current JUnit test (via Assume) when model files are absent
     * or NARAD_SKIP_MODEL_TESTS=1 is set.  Call this at the top of every
     * @Before or guarded @BeforeClass before accessing any TestModel property.
     */
    fun skipIfMissing() {
        val skipEnv = System.getenv("NARAD_SKIP_MODEL_TESTS") == "1"
        val folderPresent = !skipEnv && (findModelDir() != null || File(FALLBACK_ROOT, MODEL_DIR_NAME).isDirectory)
        val reason = when {
            skipEnv -> "NARAD_SKIP_MODEL_TESTS=1"
            !folderPresent -> "model folder absent (looked in parents of ${System.getProperty("user.dir")} and $FALLBACK_ROOT/$MODEL_DIR_NAME)"
            else -> null
        }
        if (reason != null) {
            Assume.assumeTrue("Skipping model-backed test: $reason", false)
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private fun findModelDir(): File? {
        var dir = File(System.getProperty("user.dir") ?: return null).canonicalFile
        repeat(8) { // walk up at most 8 levels
            val candidate = File(dir, MODEL_DIR_NAME)
            if (candidate.isDirectory) return candidate
            dir = dir.parentFile ?: return null
        }
        return null
    }
}
