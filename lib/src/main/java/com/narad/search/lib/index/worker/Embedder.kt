package com.narad.search.lib.index.worker

/**
 * Seam interface for text embedding.
 *
 * Production implementation: [OnnxEmbedder] (wraps [com.narad.search.lib.query.model.EmbeddingModel]).
 * Test implementation: FakeEmbedder (in androidTest sourceset) - deterministic hash-based vector,
 * no ONNX model required on the emulator.
 *
 * The interface exists solely to allow injection in [Indexer] and [IndexWorker] tests
 * without loading the 94 MB ONNX model. It does NOT modify [EmbeddingModel]'s signature.
 *
 * Extends [AutoCloseable] so [IndexWorker] can always call close() in a finally block
 * without a cast - both production ([OnnxEmbedder]) and test ([FakeEmbedder]) implementations
 * must implement it (FakeEmbedder's close() is a no-op).
 */
interface Embedder : AutoCloseable {
    /**
     * Embed [text] and return an L2-normalised [FloatArray] of length 256.
     */
    fun embed(text: String): FloatArray
}
