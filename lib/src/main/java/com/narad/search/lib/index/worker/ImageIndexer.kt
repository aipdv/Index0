package com.narad.search.lib.index.worker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository

/**
 * Plain, injectable image-indexing pipeline.
 *
 * Mirrors [Indexer] (text side) but for image embeddings via EmbeddingGemma 2.
 *
 * Algorithm (per image URI):
 *  1. Check [repository.fileNeedsIndex] - skip if unchanged (dedup/idempotency).
 *  2. Decode the bitmap from the content URI via two-pass bounds-decode + inSampleSize
 *     to avoid OOM on 48 MP camera photos.
 *  3. Encode via [imageEncoder.encode(bitmap)] → 256-dim L2-normalised vector.
 *  4. Persist via [repository.upsertImageVector] (id = uri string, filePath = uri string).
 *  5. Record dedup row via [repository.markFileIndexed] (chunkCount=1, modelId=IMAGE_MODEL_ID).
 *  6. Recycle intermediate bitmaps to avoid native heap pressure.
 *
 * Dependencies are constructor-injected so tests can substitute a fake [imageEncoder].
 *
 * M18 event wiring: [onEvent] is called alongside each existing Log.w failure. The uri
 * is already known here, so the caller (IndexWorker) need not track current-target for images.
 * Messages are short static strings and exception class names; no file content (KTD18).
 *
 * @param imageEncoder  Image encoder (production: [EG2BitmapImageEncoder]; test: [FakeImageEncoder]).
 *                      Caller owns lifecycle (close in finally).
 * @param repository    Durable index storage.
 * @param context       Android context for ContentResolver bitmap decoding.
 * @param onEvent       Called alongside each Log.w failure. Receives uri, errorClass, message.
 *                      Null = no-op (default; preserves existing callers).
 */
class ImageIndexer(
    private val imageEncoder: ImageEncoder,
    private val repository: IndexRepository,
    private val context: Context,
    // ponytail: callback fires alongside Log.w; null default keeps existing callers unchanged
    private val onEvent: ((uri: String, errorClass: String, message: String) -> Unit)? = null,
    // Called after each photo that was read (not for unchanged photos): uri, ms, result.
    private val onPhotoDone: (uri: String, ms: Long, result: String) -> Unit = { _, _, _ -> }
) {

    companion object {
        private const val TAG = "ImageIndexer"

        /** Maximum dimension (px) before downsampling. */
        private const val MAX_IMAGE_DIMENSION = 1080
    }

    /**
     * Index all images in [imageUris].
     *
     * For each URI: dedup → decode → encode → persist.
     * Skips URIs that fail to decode (logs a warning; does not throw).
     *
     * @param imageUris     List of content-URI strings for images to index.
     * @param dateMapSec    Map of URI → DATE_MODIFIED (seconds) for dedup.
     *                      If a URI is absent, lastModified=0 is used (always re-indexed).
     * @param onProgress    Optional progress callback (indexed, total), every 50 photos or 1 s and at the end.
     * @return true when every URI was visited; false when [shouldStop] ended the pass early.
     *         The prune (U8) must only run after a true.
     */
    fun indexImages(
        imageUris: List<String>,
        dateMapSec: Map<String, Long>,
        onProgress: (indexed: Int, total: Int) -> Unit = { _, _ -> },
        shouldStop: () -> Boolean = { false }
    ): Boolean {
        val total = imageUris.size
        var indexed = 0
        var finished = true
        // Skipped and unreadable photos tick too, so a mostly unchanged library still moves.
        val ticker = ProgressTicker()
        fun step() {
            indexed++
            if (total > 0 && ticker.due(indexed)) onProgress(indexed, total)
        }

        for (uriString in imageUris) {
            if (shouldStop()) { finished = false; break }
            val lastModified = dateMapSec[uriString] ?: 0L

            // Dedup: skip if not changed since last indexed.
            if (!repository.fileNeedsIndex(uriString, lastModified)) {
                step()
                continue
            }

            val photoStart = System.nanoTime()
            fun photoDone(result: String) = onPhotoDone(uriString, (System.nanoTime() - photoStart) / 1_000_000, result)

            // Decode the bitmap with bounds-decode + inSampleSize (OOM guard for 48 MP photos).
            val bitmap: Bitmap? = decodeBitmapFromUri(uriString)
            if (bitmap == null) {
                Log.w(TAG, "Failed to decode bitmap for $uriString - skipping")
                // onEvent is NOT invoked here: decodeBitmapFromUri already fired it
                // at the specific failure point (UnknownDimensions or Exception).
                // Invoking again here would double-count for one corrupt image.
                photoDone(RESULT_DECODE_FAIL)
                step()
                continue
            }

            // Not reported from the finally: a PhotoPassAborted (gate) photo was never encoded.
            var result = RESULT_INDEXED
            try {
                // Encode via EG2 vision encoder + shared LM session.
                val vector: FloatArray = imageEncoder.encode(bitmap)

                // Persist image vector.
                repository.upsertImageVector(
                    id        = uriString,
                    filePath  = uriString,
                    vector    = vector,
                    modelId   = IndexDatabase.IMAGE_MODEL_ID,
                    dim       = IndexDatabase.IMAGE_DIM_VALUE
                )

                // Mark file indexed for dedup (chunkCount=1 for images - one vector per file).
                repository.markFileIndexed(
                    filePath     = uriString,
                    mimeType     = null,   // MIME type is not tracked here; URI encodes it via MediaStore
                    lastModified = lastModified,
                    chunkCount   = 1,
                    modelId      = IndexDatabase.IMAGE_MODEL_ID,
                    dim          = IndexDatabase.IMAGE_DIM_VALUE
                )

                Log.d(TAG, "Indexed image: $uriString (dim=${vector.size})")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to encode or persist image $uriString", e)
                onEvent?.invoke(uriString, e.javaClass.simpleName, "Failed to encode or persist image")
                result = RESULT_ENCODE_FAIL
            } finally {
                bitmap.recycle()
            }
            photoDone(result)

            step()
        }

        // Always emit the terminal progress update (covers a stop between ticks too).
        if (total > 0) onProgress(indexed, total)
        return finished
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Two-pass bitmap decode with inSampleSize downsampling.
     *
     * Pass 1: BitmapFactory.Options.inJustDecodeBounds - reads dimensions only, no allocation.
     * Pass 2: Decode with inSampleSize = next power of two ≥ ceil(maxDim / MAX_IMAGE_DIMENSION).
     *
     * **Power-of-two rounding:** Android rounds `inSampleSize` DOWN to the nearest power of
     * two at decode time, so a raw ceiling value (e.g. 3 for a 3240px photo) would become 2,
     * decoding at 1620px and still risking OOM. By rounding UP to the next power of two (here 4),
     * the decoded dimension is guaranteed to be ≤ MAX_IMAGE_DIMENSION before the encoder's
     * `createScaledBitmap` step (Bug Watch U5 item: "Bitmap decoding OOM on large photos").
     *
     * @return Decoded [Bitmap], or null if the URI cannot be opened or decoded.
     */
    private fun decodeBitmapFromUri(uriString: String): Bitmap? {
        return try {
            val uri = Uri.parse(uriString)
            val resolver = context.contentResolver

            // Pass 1: bounds only.
            val boundsOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            resolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, boundsOpts)
            }

            val maxDim = maxOf(boundsOpts.outWidth, boundsOpts.outHeight)
            if (maxDim <= 0) {
                Log.w(TAG, "Could not determine image dimensions for $uriString")
                onEvent?.invoke(uriString, "UnknownDimensions", "Could not determine image dimensions")
                return null
            }

            val sampleSize = if (maxDim > MAX_IMAGE_DIMENSION) {
                // Ceiling integer division, then round UP to the next power of two so that
                // Android's internal downward power-of-two clamp still yields ≤ MAX_IMAGE_DIMENSION.
                val ceil = (maxDim + MAX_IMAGE_DIMENSION - 1) / MAX_IMAGE_DIMENSION
                Integer.highestOneBit(ceil).let { if (it < ceil) it shl 1 else it }
            } else {
                1
            }

            // Pass 2: full decode with downsampling.
            val decodeOpts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
            val bmp = resolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, decodeOpts)
            }
            if (bmp == null) onEvent?.invoke(uriString, "BitmapDecodeNull", "Failed to decode bitmap")
            bmp
        } catch (e: Exception) {
            Log.w(TAG, "Exception decoding bitmap for $uriString", e)
            onEvent?.invoke(uriString, e.javaClass.simpleName, "Exception decoding bitmap")
            null
        }
    }
}
