package com.narad.search.lib.query.model.image

import kotlin.math.floor
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Pure-Kotlin EmbeddingGemma 2 image preprocessor.
 *
 * Converts an ARGB [IntArray] (with [srcWidth] × [srcHeight]) into the padded
 * `pixel_values` and `pixel_position_ids` tensors the vision encoder expects.
 *
 * This class has NO Android imports so it is JVM-testable on the host machine.
 * A thin Android adapter ([EG2ImagePreprocessorAndroid]) wraps it for use with
 * [android.graphics.Bitmap].
 *
 * Constants (from HF Gemma4ImageProcessor / processor_config.json):
 *   patch_size = 16
 *   pooling_kernel_size = 3
 *   side_multiple = 48   (= patch_size × pooling_kernel_size)
 *   max_soft_tokens = 70
 *   max_patches = 630    (= max_soft_tokens × pooling_kernel_size²)
 *   patch_features = 768 (= patch_size² × 3 channels)
 *
 * Pipeline per image:
 *   1. Compute target size (tw, th): keep aspect ratio, quantise to multiples of 48.
 *   2. Resize to (tw, th) via the caller-supplied [ResizeFn].
 *   3. Rescale pixel values ÷ 255 (no mean/std normalisation).
 *   4. Patchify: row-major, each patch flattened as [py][px][channel] (channel-last).
 *   5. Pad to [MAX_PATCHES] patches with zeros; pad position IDs with (−1, −1).
 *
 * @param resizeFn  Caller-supplied resizer. Receives source ARGB pixels + dimensions,
 *   must return ARGB pixels at the requested (dstW, dstH). JVM tests supply a bilinear
 *   AWT implementation; Android supplies a Bitmap-based adapter.
 */
class EG2ImagePreprocessor(
    private val resizeFn: ResizeFn
) {

    // ponytail: constants match processor_config.json exactly - no config object needed
    companion object {
        const val PATCH_SIZE   = 16
        const val POOL_K       = 3
        const val SIDE_MULT    = PATCH_SIZE * POOL_K   // 48
        const val MAX_SOFT     = 70
        const val MAX_PATCHES  = MAX_SOFT * POOL_K * POOL_K  // 630
        const val PATCH_FEAT   = PATCH_SIZE * PATCH_SIZE * 3  // 768
    }

    /** Function type alias for the injectable resize implementation. */
    fun interface ResizeFn {
        fun resize(srcPixels: IntArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int): IntArray
    }

    data class PreprocessResult(
        /** Flattened [MAX_PATCHES × PATCH_FEAT] float32 array (ready for [1,MAX_PATCHES,PATCH_FEAT] tensor). */
        val pixelValues: FloatArray,
        /** Flattened [MAX_PATCHES × 2] int64 array, (x=col, y=row), padded with (−1,−1). */
        val positionIds: LongArray,
        val resizedWidth: Int,
        val resizedHeight: Int,
        val nSoft: Int,   // (tw/SIDE_MULT) × (th/SIDE_MULT)
        val nPatches: Int // (tw/PATCH_SIZE) × (th/PATCH_SIZE)
    )

    /**
     * Preprocess [srcPixels] (ARGB packed ints, row-major) of size [srcW]×[srcH].
     */
    fun preprocess(srcPixels: IntArray, srcW: Int, srcH: Int): PreprocessResult {
        val (tw, th) = computeTargetSize(srcW, srcH)
        val resized  = resizeFn.resize(srcPixels, srcW, srcH, tw, th)

        val nPw      = tw / PATCH_SIZE    // patches wide
        val nPh      = th / PATCH_SIZE    // patches tall
        val nPatches = nPw * nPh
        val nSoft    = (tw / SIDE_MULT) * (th / SIDE_MULT)

        val pixelValues = FloatArray(MAX_PATCHES * PATCH_FEAT)
        val positionIds = LongArray(MAX_PATCHES * 2) { -1L }

        var patchIdx = 0
        for (row in 0 until nPh) {
            for (col in 0 until nPw) {
                val pvBase = patchIdx * PATCH_FEAT
                var pvOff  = pvBase
                // Flatten patch in channel-last order: [py][px][R,G,B]
                for (py in 0 until PATCH_SIZE) {
                    for (px in 0 until PATCH_SIZE) {
                        val argb = resized[(row * PATCH_SIZE + py) * tw + (col * PATCH_SIZE + px)]
                        pixelValues[pvOff++] = ((argb shr 16) and 0xFF) / 255.0f  // R
                        pixelValues[pvOff++] = ((argb shr  8) and 0xFF) / 255.0f  // G
                        pixelValues[pvOff++] = ( argb         and 0xFF) / 255.0f  // B
                    }
                }
                // Position: (x=col, y=row)
                positionIds[patchIdx * 2    ] = col.toLong()
                positionIds[patchIdx * 2 + 1] = row.toLong()
                patchIdx++
            }
        }
        // Remaining slots stay zeros / -1 (already initialized above)

        return PreprocessResult(
            pixelValues   = pixelValues,
            positionIds   = positionIds,
            resizedWidth  = tw,
            resizedHeight = th,
            nSoft         = nSoft,
            nPatches      = nPatches
        )
    }

    // -------------------------------------------------------------------------
    // Target-size formula (from HF Gemma4ImageProcessor)
    // -------------------------------------------------------------------------

    /** Compute (tw, th) as the largest multiples of [SIDE_MULT] that fit ≤ [MAX_PATCHES] patches. */
    internal fun computeTargetSize(w: Int, h: Int): Pair<Int, Int> {
        val maxPatches = (MAX_SOFT * POOL_K * POOL_K).toDouble()
        val patchArea  = (PATCH_SIZE * PATCH_SIZE).toDouble()
        val factor     = sqrt(maxPatches * patchArea / (h.toDouble() * w.toDouble()))
        var tw         = (floor(w * factor / SIDE_MULT) * SIDE_MULT).toInt()
        var th         = (floor(h * factor / SIDE_MULT) * SIDE_MULT).toInt()

        if (tw == 0 && th == 0) {
            tw = SIDE_MULT; th = SIDE_MULT
        } else if (tw == 0) {
            tw = SIDE_MULT
            val maxOther = (MAX_PATCHES / (POOL_K * POOL_K)) * SIDE_MULT
            th = min((h / w) * SIDE_MULT, maxOther)
        } else if (th == 0) {
            th = SIDE_MULT
            val maxOther = (MAX_PATCHES / (POOL_K * POOL_K)) * SIDE_MULT
            tw = min((w / h) * SIDE_MULT, maxOther)
        }

        return tw to th
    }
}

/**
 * Pure-Kotlin bilinear resize for use in JVM (host) tests.
 *
 * For smooth gradient images this is pixel-accurate vs bicubic (linear functions
 * interpolate exactly under any affine-interpolation scheme), keeping the
 * max-abs-diff of pixel_values below the 0.02 parity bound.
 */
object BilinearResizeFn : EG2ImagePreprocessor.ResizeFn {
    override fun resize(srcPixels: IntArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int): IntArray {
        val dst = IntArray(dstW * dstH)
        for (dy in 0 until dstH) {
            // Centre-aligned: src coordinate of dst pixel centre
            val sy = ((dy + 0.5) * srcH / dstH - 0.5).coerceAtLeast(0.0)  // no extrapolation at the edge
            val y0 = sy.toInt().coerceIn(0, srcH - 1)
            val y1 = (y0 + 1).coerceIn(0, srcH - 1)
            val vy = sy - y0

            for (dx in 0 until dstW) {
                val sx = ((dx + 0.5) * srcW / dstW - 0.5).coerceAtLeast(0.0)
                val x0 = sx.toInt().coerceIn(0, srcW - 1)
                val x1 = (x0 + 1).coerceIn(0, srcW - 1)
                val vx = sx - x0

                // Bilinear over 4 neighbours
                val p00 = srcPixels[y0 * srcW + x0]
                val p10 = srcPixels[y0 * srcW + x1]
                val p01 = srcPixels[y1 * srcW + x0]
                val p11 = srcPixels[y1 * srcW + x1]

                // Interpolate each channel separately
                fun chan(shift: Int): Int {
                    val c00 = (p00 shr shift) and 0xFF
                    val c10 = (p10 shr shift) and 0xFF
                    val c01 = (p01 shr shift) and 0xFF
                    val c11 = (p11 shr shift) and 0xFF
                    val top    = c00 + (c10 - c00) * vx
                    val bottom = c01 + (c11 - c01) * vx
                    return (top + (bottom - top) * vy).toInt().coerceIn(0, 255)
                }

                val r = chan(16); val g = chan(8); val b = chan(0)
                dst[dy * dstW + dx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return dst
    }
}
