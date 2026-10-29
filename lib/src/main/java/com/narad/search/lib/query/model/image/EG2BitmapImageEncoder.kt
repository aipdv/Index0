package com.narad.search.lib.query.model.image

import android.graphics.Bitmap
import android.graphics.Canvas
import com.narad.search.lib.index.worker.ImageEncoder
import com.narad.search.lib.query.model.EmbeddingModel

/**
 * Production [ImageEncoder] implementation for EmbeddingGemma 2 image search.
 *
 * Pipeline per bitmap:
 *   1. Coerce to software ARGB_8888 (handles HARDWARE bitmaps from ImageDecoder/Glide).
 *   2. Extract ARGB pixel array.
 *   3. Run [EG2ImagePreprocessor] (resize + patchify).
 *   4. Run [VisionEncoder] → image_features [nSoft × 512].
 *   5. Run shared [EmbeddingModel.embedImageFeatures] → 256-dim L2-normalised vector.
 *
 * The [embeddingModel] is SHARED with the text indexing branch - do NOT close it here.
 * The [visionEncoder] is owned by this class and closed in [close].
 *
 * @param embeddingModel  Shared EG2 LM session (caller closes it after the indexing run).
 * @param visionEncoder   Vision encoder session (owned here; closed in [close]).
 */
class EG2BitmapImageEncoder(
    private val embeddingModel: EmbeddingModel,
    private val visionEncoder: VisionEncoder
) : ImageEncoder {

    // Same pure-Kotlin resizer the JVM parity test checks against the Python reference.
    private val preprocessor = EG2ImagePreprocessor(BilinearResizeFn)

    override fun encode(bitmap: Bitmap): FloatArray {
        val soft = toSoftwareArgb8888(bitmap)
        try {
            val w = soft.width
            val h = soft.height
            val pixels = IntArray(w * h)
            soft.getPixels(pixels, 0, w, 0, 0, w, h)

            val pre = preprocessor.preprocess(pixels, w, h)

            val imageFeatures = visionEncoder.encode(pre.pixelValues, pre.positionIds)
            val nSoft = imageFeatures.size / VisionEncoder.FEAT_DIM

            return embeddingModel.embedImageFeatures(imageFeatures, nSoft)
        } finally {
            if (soft !== bitmap) soft.recycle()
        }
    }

    /** Closes ONLY the [visionEncoder] - the shared [embeddingModel] is NOT closed here. */
    override fun close() {
        visionEncoder.close()
    }

    // -------------------------------------------------------------------------

    private fun toSoftwareArgb8888(src: Bitmap): Bitmap {
        if (src.config == Bitmap.Config.ARGB_8888) return src
        val out = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(src, 0f, 0f, null)
        return out
    }
}
