package com.narad.search.lib.index.enumeration

import android.content.ContentUris
import android.content.Context
import android.provider.MediaStore

/**
 * Stable identifier for a MediaStore document ready to be indexed.
 *
 * @param uri          Content-URI string (e.g. `content://media/external/file/42`).
 *                     Used as the stable primary key through the entire indexing pipeline -
 *                     it flows into `file_index.file_path` and chunk IDs (`<uri>#<offset>`).
 * @param lastModified `DATE_MODIFIED` value from MediaStore, in **seconds** (NOT millis).
 *                     Dedup is per-key consistent: URI keys always compare against their own
 *                     stored seconds value; File-path keys always compare against millis.
 *                     Never convert - store and compare as-is.
 * @param mimeType     MIME type as reported by MediaStore (e.g. `"text/plain"`), or null.
 */
data class IndexTarget(
    val uri: String,
    val lastModified: Long,
    val mimeType: String?
)

/**
 * Queries MediaStore for indexable document and image files.
 *
 * Collection used: `MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL)` - single
 * collection that covers Downloads, DCIM, Pictures, and all MediaStore-accessible locations,
 * avoiding duplicate-collection URI collisions.  Both [enumerateDocuments] and [enumerateImages]
 * use this collection so that dedup keys (file_index.file_path, image_vectors.id) are uniform
 * across the indexing pipeline - all URIs have the form `content://media/external/file/<id>`.
 *
 * The app's own inserted files are always readable; broad storage permission
 * (`READ_EXTERNAL_STORAGE` / `READ_MEDIA_IMAGES`) is needed to read other apps' files.
 * The enumerator does **not** gate on permission - it queries and returns whatever the runtime
 * allows. An empty result on a fresh device is therefore valid.
 *
 * **Capability gating** is intentionally NOT done here.  The caller ([IndexWorker]) is
 * responsible for checking [DeviceTierDetector.detect(context).imageEmbeddingEnabled] before
 * calling [enumerateImages].
 */
class MediaStoreEnumerator {

    companion object {
        /**
         * MIME types for documents this enumerator returns.
         * Exposed so instrumented tests can assert the exact set without duplicating the string.
         */
        val DOCUMENT_MIME_TYPES: Set<String> = setOf(
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/markdown"
        )

        /**
         * MIME types for images this enumerator returns.
         *
         * Covers JPEG, PNG, and WebP - the three formats with widest Android MediaStore coverage.
         * GIF and HEIC are intentionally excluded: GIF is often animated (not semantic images);
         * HEIC requires API 30+ BitmapFactory support (conditional handling adds complexity).
         *
         * Exposed so instrumented tests can assert the exact set without duplicating the constant.
         */
        val IMAGE_MIME_TYPES: Set<String> = setOf(
            "image/jpeg",
            "image/png",
            "image/webp"
        )
    }

    /**
     * Enumerate all indexable documents visible to the app.
     *
     * @param context   Android context used to obtain the ContentResolver.
     * @param sinceSec  If non-null, only files with `DATE_MODIFIED > sinceSec` are returned
     *                  (delta enumeration for incremental re-index runs).
     *                  If null, all matching documents are returned (full scan).
     * @return          List of [IndexTarget] entries, one per file.
     *                  Empty list if nothing matches or the app has no storage permission.
     */
    fun enumerateDocuments(context: Context, sinceSec: Long? = null): List<IndexTarget> {
        require(sinceSec == null || sinceSec < 10_000_000_000L) {
            "sinceSec must be in SECONDS (MediaStore DATE_MODIFIED), not millis"
        }

        val collection = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL)

        // DISPLAY_NAME is intentionally omitted: IndexTarget has no displayName field,
        // so fetching it would waste cursor memory without being consumed.
        val projection = arrayOf(
            MediaStore.Files.FileColumns._ID,
            MediaStore.Files.FileColumns.MIME_TYPE,
            MediaStore.Files.FileColumns.DATE_MODIFIED,
            // Index0: file path for the excluded-folders rule.
            MediaStore.Files.FileColumns.DATA
        )

        // Build the MIME-type IN clause.
        // Placeholders: one ? per MIME type, plus an optional DATE_MODIFIED clause.
        val mimeList = DOCUMENT_MIME_TYPES.toList()
        val inPlaceholders = mimeList.joinToString(",") { "?" }

        val selectionArgs: Array<String>
        val selection: String

        if (sinceSec != null) {
            selection = "${MediaStore.Files.FileColumns.MIME_TYPE} IN ($inPlaceholders)" +
                " AND ${MediaStore.Files.FileColumns.DATE_MODIFIED} > ?"
            selectionArgs = (mimeList + sinceSec.toString()).toTypedArray()
        } else {
            selection = "${MediaStore.Files.FileColumns.MIME_TYPE} IN ($inPlaceholders)"
            selectionArgs = mimeList.toTypedArray()
        }

        val results = mutableListOf<IndexTarget>()
        val userExcluded = ExcludedFolders.userFolders(context)

        context.contentResolver.query(
            collection,
            projection,
            selection,
            selectionArgs,
            // Newest first: a long first index makes recent files searchable early.
            // Nothing depends on the order (dedup per file, prune uses sets).
            "${MediaStore.Files.FileColumns.DATE_MODIFIED} DESC"
        )?.use { cursor ->
            val idCol       = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID)
            val mimeCol     = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.MIME_TYPE)
            val modifiedCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DATE_MODIFIED)
            val dataCol     = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DATA)

            while (cursor.moveToNext()) {
                val path = cursor.getString(dataCol)
                if (path != null && ExcludedFolders.isExcluded(path, userExcluded)) continue
                val id           = cursor.getLong(idCol)
                val mimeType     = cursor.getString(mimeCol)
                val lastModified = cursor.getLong(modifiedCol) // seconds, not millis - do NOT convert

                // Build stable content URI: content://media/external/file/<id>
                val uri = ContentUris.withAppendedId(collection, id).toString()

                results.add(IndexTarget(uri = uri, lastModified = lastModified, mimeType = mimeType))
            }
        }

        return results
    }

    /**
     * Enumerate all indexable images visible to the app.
     *
     * Returns [IndexTarget] entries for JPEG, PNG, and WebP images (see [IMAGE_MIME_TYPES]).
     * The URI form is `content://media/external/file/<id>` (Files collection) - identical to
     * [enumerateDocuments], so the dedup key space is uniform across the indexing pipeline.
     *
     * **Capability gating:** this method does NOT check `imageEmbeddingEnabled`.
     * The caller (e.g. [com.narad.search.lib.index.worker.IndexWorker]) must guard.
     *
     * @param context   Android context used to obtain the ContentResolver.
     * @param sinceSec  If non-null, only files with `DATE_MODIFIED > sinceSec` are returned
     *                  (delta enumeration for incremental re-index runs).
     *                  If null, all matching images are returned (full scan).
     * @return          List of [IndexTarget] entries, one per image file.
     *                  Empty list if nothing matches or the app has no storage permission.
     */
    fun enumerateImages(context: Context, sinceSec: Long? = null): List<IndexTarget> {
        require(sinceSec == null || sinceSec < 10_000_000_000L) {
            "sinceSec must be in SECONDS (MediaStore DATE_MODIFIED), not millis"
        }

        val collection = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL)

        val projection = arrayOf(
            MediaStore.Files.FileColumns._ID,
            MediaStore.Files.FileColumns.MIME_TYPE,
            MediaStore.Files.FileColumns.DATE_MODIFIED,
            // Index0: file path for the excluded-folders rule.
            MediaStore.Files.FileColumns.DATA
        )

        val mimeList = IMAGE_MIME_TYPES.toList()
        val inPlaceholders = mimeList.joinToString(",") { "?" }

        val selectionArgs: Array<String>
        val selection: String

        if (sinceSec != null) {
            selection = "${MediaStore.Files.FileColumns.MIME_TYPE} IN ($inPlaceholders)" +
                " AND ${MediaStore.Files.FileColumns.DATE_MODIFIED} > ?"
            selectionArgs = (mimeList + sinceSec.toString()).toTypedArray()
        } else {
            selection = "${MediaStore.Files.FileColumns.MIME_TYPE} IN ($inPlaceholders)"
            selectionArgs = mimeList.toTypedArray()
        }

        val results = mutableListOf<IndexTarget>()
        val userExcluded = ExcludedFolders.userFolders(context)

        context.contentResolver.query(
            collection,
            projection,
            selection,
            selectionArgs,
            // Newest first: a long first index makes recent files searchable early.
            // Nothing depends on the order (dedup per file, prune uses sets).
            "${MediaStore.Files.FileColumns.DATE_MODIFIED} DESC"
        )?.use { cursor ->
            val idCol       = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID)
            val mimeCol     = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.MIME_TYPE)
            val modifiedCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DATE_MODIFIED)
            val dataCol     = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DATA)

            while (cursor.moveToNext()) {
                val path = cursor.getString(dataCol)
                if (path != null && ExcludedFolders.isExcluded(path, userExcluded)) continue
                val id           = cursor.getLong(idCol)
                val mimeType     = cursor.getString(mimeCol)
                val lastModified = cursor.getLong(modifiedCol) // seconds, not millis - do NOT convert

                // Build stable content URI: content://media/external/file/<id>
                val uri = ContentUris.withAppendedId(collection, id).toString()

                results.add(IndexTarget(uri = uri, lastModified = lastModified, mimeType = mimeType))
            }
        }

        return results
    }
}
