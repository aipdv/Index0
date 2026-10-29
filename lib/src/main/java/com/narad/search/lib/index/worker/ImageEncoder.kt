package com.narad.search.lib.index.worker

import android.graphics.Bitmap

/**
 * Seam interface for image embedding.
 *
 * Production implementation: [EG2BitmapImageEncoder] (EmbeddingGemma 2 shared vector space).
 * Test implementation: [FakeImageEncoder] (in androidTest sourceset) - deterministic 256-dim vector,
 * no ONNX model required on the emulator.
 *
 * Extends [AutoCloseable] so [IndexWorker] can always call close() in a finally block without a cast.
 * Production implementation closes the underlying VisionEncoder session; test implementation is a no-op.
 *
 * Mirrors the [Embedder] pattern for text embedding.
 */
interface ImageEncoder : AutoCloseable {
    /**
     * Encode [bitmap] and return a 256-dimensional L2-normalised embedding in EG2 vector space.
     *
     * The bitmap lifecycle is owned by the caller - this method does NOT recycle it.
     */
    fun encode(bitmap: Bitmap): FloatArray
}
