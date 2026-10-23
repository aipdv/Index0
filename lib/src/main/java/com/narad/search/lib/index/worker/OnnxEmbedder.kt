package com.narad.search.lib.index.worker

import com.narad.search.lib.query.model.EmbeddingModel

/**
 * Production [Embedder] adapter that delegates to [EmbeddingModel].
 *
 * This wrapper does NOT modify [EmbeddingModel]'s public API (locked decision §3).
 * It exists only to satisfy the [Embedder] interface so [Indexer] and [IndexWorker]
 * can be tested with a [FakeEmbedder] without loading the ONNX model.
 *
 * Implements [AutoCloseable] so the [EmbeddingModel] (which holds an OrtSession)
 * can be closed when the [IndexWorker] finishes.
 */
class OnnxEmbedder(val model: EmbeddingModel) : Embedder, AutoCloseable {
    // Route through embedDocument so the document prompt prefix is applied.
    override fun embed(text: String): FloatArray = model.embedDocument(text)
    override fun close() = model.close()
}
