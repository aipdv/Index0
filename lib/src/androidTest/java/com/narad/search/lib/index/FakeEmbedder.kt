package com.narad.search.lib.index

import com.narad.search.lib.index.worker.Embedder
import kotlin.math.sqrt

/**
 * Deterministic fake embedder for orchestration / persistence / dedup tests.
 *
 * Produces a stable, L2-normalised 384-float vector from a text hash so that
 * identical input always yields the same output. Different texts yield
 * different vectors (the hash spreads across the dimension axes), satisfying
 * ranking assertions without pushing the 94 MB ONNX model to the emulator.
 *
 * L10 compliance: tests using this embedder assert RANKING (result #1 for the
 * matching file), not absolute cosine thresholds.
 */
class FakeEmbedder : Embedder {

    override fun close() { /* no-op: no native resources to free */ }

    override fun embed(text: String): FloatArray {
        val seed = text.fold(0L) { acc, c -> acc * 31L + c.code }
        val raw = FloatArray(256) { i ->
            // Deterministic but dimension-varying: each dim gets a unique value
            val h = (seed xor (i.toLong() * 2654435761L)).toInt()
            // Use a soft-sine-like spread via bit manipulation
            val f = (h and 0xFFFF).toFloat() / 0xFFFF.toFloat() - 0.5f
            f
        }
        // L2-normalise
        var norm = 0.0f
        for (x in raw) norm += x * x
        norm = sqrt(norm.toDouble()).toFloat()
        if (norm < 1e-12f) {
            raw[0] = 1.0f
            return raw
        }
        return FloatArray(raw.size) { i -> raw[i] / norm }
    }
}
