package com.narad.search.lib.index.store

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import com.narad.search.lib.BuildConfig
import com.narad.search.lib.index.worker.eventCap
import com.narad.search.lib.query.store.KeywordStore
import com.narad.search.lib.query.store.VectorStore
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "IndexRepository"
private const val MEDIA_URI_PREFIX = "content://media/"
internal const val CONFIG_PRUNE_VOLUMES = "prune_volumes"

/** One row from the index_events table (KTD18). Never contains file content. */
data class IndexEvent(
    val id: Long,
    val timestampMs: Long,
    val runId: String,
    val stage: String,
    val level: String,
    val fileName: String?,
    val errorClass: String?,
    val message: String?
)

internal fun rehydrateChunk(
    id: String,
    filePath: String,
    chunkOffset: Int,
    modelId: String,
    vectorBlob: ByteArray?,
    content: String,
    vectorStore: VectorStore?,
    keywordStore: KeywordStore
) {
    if (vectorBlob != null && vectorStore != null) {
        val buffer = ByteBuffer.wrap(vectorBlob).order(ByteOrder.LITTLE_ENDIAN)
        val vector = FloatArray(vectorBlob.size / 4) { buffer.float }
        vectorStore.add(id, filePath, chunkOffset, modelId, vector)
    }
    keywordStore.add(id, filePath, chunkOffset, content)
}

/**
 * Durable index operations built on top of [IndexDatabase].
 *
 * Key responsibilities:
 *  - [upsertChunk]        -- persist a chunk row (idempotent via INSERT OR REPLACE)
 *  - [fileNeedsIndex]     -- check whether a file has changed since last indexed
 *  - [markFileIndexed]    -- record that a file has been fully indexed
 *  - [rehydrate]          -- repopulate in-memory [VectorStore] and [KeywordStore] from DB rows
 *  - [getConfig]          -- read a config key from index_config
 *  - [setConfig]          -- write a config key to index_config
 *  - [chunkCount]         -- total number of chunk rows
 *  - [indexedFileCount]   -- total number of file_index rows
 *  - [pruneDocuments] / [prunePhotos] -- delete rows for files gone from MediaStore (U8)
 *  - [clearAll]           -- wipe embedding tables (favorites and events are kept; see [clearAll])
 *  - [addFavorite]        -- bookmark a content URI (idempotent; adding twice keeps one row)
 *  - [removeFavorite]     -- remove a bookmarked URI
 *  - [listFavorites]      -- newest-first list of bookmarked URIs
 *  - [appendEvent]        -- write one index_events row; errors are caught and logged only
 *  - [pruneEvents]        -- delete rows older than 7 days, keep at most 500
 *  - [listEvents]         -- newest-first list of index events
 *
 * Float <-> BLOB helpers are dimension-agnostic (text and image vectors are both 256 -> 1024 bytes).
 * Values are encoded as little-endian float32 (4 bytes per element).
 */
class IndexRepository(private val db: IndexDatabase) {

    // -------------------------------------------------------------------------
    // Chunk persistence
    // -------------------------------------------------------------------------

    /**
     * Write all chunks for a single file inside a single SQLite transaction.
     *
     * Fix item 11: per-chunk inserts without a transaction are O(N) fsync cost.
     * Wrapping all chunk writes for one file in a transaction amortises the cost
     * to O(1) fsync per file, which is critical for 5000-file index builds.
     *
     * Also handles orphan-chunk cleanup (fix item 6): calls [deleteChunksForFile]
     * before writing so stale rows from a previous run with more chunks are removed.
     *
     * @param filePath   Source file path (used for orphan cleanup and all chunk rows).
     * @param chunks     List of (chunkOffset, vector, content) tuples to write.
     * @param modelId    Embedding model identifier.
     * @param dim        Embedding dimension (256).
     */
    fun upsertChunksForFile(
        filePath: String,
        chunks: List<Triple<Int, FloatArray, String>>,
        modelId: String,
        dim: Int
    ) {
        val w = db.writableDatabase
        w.beginTransaction()
        try {
            // Fix item 6: delete stale chunks BEFORE writing new ones
            deleteChunksForFile(filePath)

            for ((chunkOffset, vector, content) in chunks) {
                val id = "$filePath#$chunkOffset"
                val cv = android.content.ContentValues().apply {
                    put(IndexDatabase.COL_ID, id)
                    put(IndexDatabase.COL_FILE_PATH, filePath)
                    put(IndexDatabase.COL_CHUNK_OFFSET, chunkOffset)
                    put(IndexDatabase.COL_MODEL_ID, modelId)
                    put(IndexDatabase.COL_DIM, dim)
                    put(IndexDatabase.COL_VECTOR, floatArrayToBlob(vector))
                    put(IndexDatabase.COL_CONTENT, content)
                }
                w.insertWithOnConflict(
                    IndexDatabase.TABLE_CHUNKS,
                    null,
                    cv,
                    android.database.sqlite.SQLiteDatabase.CONFLICT_REPLACE
                )
                try {
                    w.delete(IndexDatabase.TABLE_CHUNKS_FTS, "chunk_id = ?", arrayOf(id))
                    val ftsValues = android.content.ContentValues().apply {
                        put("content", content)
                        put("chunk_id", id)
                    }
                    w.insert(IndexDatabase.TABLE_CHUNKS_FTS, null, ftsValues)
                } catch (_: Exception) { /* FTS not available */ }
            }
            w.setTransactionSuccessful()
        } finally {
            w.endTransaction()
        }
    }

    /**
     * Insert or replace a chunk record.
     *
     * [id] must be `"<filePath>#<chunkOffset>"` so it is globally unique and
     * matches the idempotency contract from VectorStore/KeywordStore.
     */
    fun upsertChunk(
        id: String,
        filePath: String,
        chunkOffset: Int,
        modelId: String,
        dim: Int,
        vector: FloatArray,
        content: String
    ) {
        val w = db.writableDatabase
        val cv = ContentValues().apply {
            put(IndexDatabase.COL_ID, id)
            put(IndexDatabase.COL_FILE_PATH, filePath)
            put(IndexDatabase.COL_CHUNK_OFFSET, chunkOffset)
            put(IndexDatabase.COL_MODEL_ID, modelId)
            put(IndexDatabase.COL_DIM, dim)
            put(IndexDatabase.COL_VECTOR, floatArrayToBlob(vector))
            put(IndexDatabase.COL_CONTENT, content)
        }
        w.insertWithOnConflict(
            IndexDatabase.TABLE_CHUNKS,
            null,
            cv,
            SQLiteDatabase.CONFLICT_REPLACE
        )

        // Mirror into FTS table (delete old row first to keep FTS in sync).
        // Wrapped in try-catch: FTS5 may not be available on all builds;
        // FTS4 fallback may also be absent in extreme cases. The FTS table
        // is a secondary durability mirror -- its absence does not break the
        // core pipeline which rehydrates from chunks.content.
        try {
            w.delete(
                IndexDatabase.TABLE_CHUNKS_FTS,
                "chunk_id = ?",
                arrayOf(id)
            )
            val ftsValues = ContentValues().apply {
                put("content", content)
                put("chunk_id", id)
            }
            w.insert(IndexDatabase.TABLE_CHUNKS_FTS, null, ftsValues)
        } catch (_: Exception) {
            // FTS not available; skip silently
        }
    }

    // -------------------------------------------------------------------------
    // Chunk deletion
    // -------------------------------------------------------------------------

    /**
     * Delete ALL chunk rows (and their FTS mirrors) for [filePath].
     *
     * Must be called BEFORE writing a (re)indexed file's chunks so that stale
     * rows from a previous run with more chunks are not left as orphans.
     *
     * Fix item 6: without this, re-indexing a file that now yields fewer chunks
     * leaves orphan rows (e.g. old #3, #4 remain when new content yields only 2 chunks).
     */
    fun deleteChunksForFile(filePath: String) {
        val w = db.writableDatabase
        // Collect chunk IDs for this file so we can delete matching FTS rows too.
        val ids = mutableListOf<String>()
        val cursor = w.query(
            IndexDatabase.TABLE_CHUNKS,
            arrayOf(IndexDatabase.COL_ID),
            "${IndexDatabase.COL_FILE_PATH} = ?",
            arrayOf(filePath),
            null, null, null
        )
        cursor.use { c ->
            while (c.moveToNext()) ids.add(c.getString(0))
        }

        // Delete from primary chunks table
        w.delete(
            IndexDatabase.TABLE_CHUNKS,
            "${IndexDatabase.COL_FILE_PATH} = ?",
            arrayOf(filePath)
        )

        // Delete matching FTS rows (by chunk_id)
        if (ids.isNotEmpty()) {
            try {
                for (id in ids) {
                    w.delete(
                        IndexDatabase.TABLE_CHUNKS_FTS,
                        "chunk_id = ?",
                        arrayOf(id)
                    )
                }
            } catch (_: Exception) {
                // FTS not available; skip
            }
        }
    }

    // -------------------------------------------------------------------------
    // File index tracking
    // -------------------------------------------------------------------------

    /**
     * Returns true if the file at [filePath] has not been indexed yet, or if
     * its last-modified timestamp has changed since the last index run.
     *
     * Delegates dedup responsibility: if false, the [Indexer] skips the file.
     */
    fun fileNeedsIndex(filePath: String, lastModified: Long): Boolean {
        val cursor = db.readableDatabase.query(
            IndexDatabase.TABLE_FILE_INDEX,
            arrayOf(IndexDatabase.COL_FI_LAST_MODIFIED),
            "${IndexDatabase.COL_FI_FILE_PATH} = ?",
            arrayOf(filePath),
            null, null, null
        )
        return cursor.use { c ->
            if (!c.moveToFirst()) {
                true  // never indexed
            } else {
                val storedMod = c.getLong(0)
                storedMod != lastModified
            }
        }
    }

    /**
     * Record that [filePath] has been fully indexed.
     *
     * [chunkCount] is the number of chunks written for this file during this run.
     */
    fun markFileIndexed(
        filePath: String,
        mimeType: String?,
        lastModified: Long,
        chunkCount: Int,
        modelId: String = IndexDatabase.MODEL_ID,
        dim: Int = IndexDatabase.DIM_VALUE
    ) {
        val cv = ContentValues().apply {
            put(IndexDatabase.COL_FI_FILE_PATH, filePath)
            put(IndexDatabase.COL_FI_MIME_TYPE, mimeType)
            put(IndexDatabase.COL_FI_LAST_MODIFIED, lastModified)
            put(IndexDatabase.COL_FI_LAST_INDEXED, System.currentTimeMillis())
            put(IndexDatabase.COL_FI_CHUNK_COUNT, chunkCount)
            put(IndexDatabase.COL_FI_MODEL_ID, modelId)
            put(IndexDatabase.COL_FI_DIM, dim)
        }
        db.writableDatabase.insertWithOnConflict(
            IndexDatabase.TABLE_FILE_INDEX,
            null,
            cv,
            SQLiteDatabase.CONFLICT_REPLACE
        )
    }

    // -------------------------------------------------------------------------
    // Rehydration
    // -------------------------------------------------------------------------

    /**
     * Read all chunk rows from the database and load them into the given
     * in-memory stores.
     *
     * This is the "kill/restart preserves the index" mechanism: on every cold
     * start, call this before serving any queries.
     */
    fun rehydrate(vectorStore: VectorStore, keywordStore: KeywordStore) {
        val cursor = db.readableDatabase.query(
            IndexDatabase.TABLE_CHUNKS,
            arrayOf(
                IndexDatabase.COL_ID,
                IndexDatabase.COL_FILE_PATH,
                IndexDatabase.COL_CHUNK_OFFSET,
                IndexDatabase.COL_MODEL_ID,
                IndexDatabase.COL_VECTOR,
                IndexDatabase.COL_CONTENT
            ),
            null, null, null, null, null
        )
        cursor.use { c ->
            while (c.moveToNext()) {
                val id = c.getString(0)
                val filePath = c.getString(1)
                val chunkOffset = c.getInt(2)
                val modelId = c.getString(3)
                val blob = c.getBlob(4)
                val content = c.getString(5)

                rehydrateChunk(
                    id = id,
                    filePath = filePath,
                    chunkOffset = chunkOffset,
                    modelId = modelId,
                    vectorBlob = blob,
                    content = content,
                    vectorStore = vectorStore,
                    keywordStore = keywordStore
                )
            }
        }
    }

    fun rehydrateKeywords(keywordStore: KeywordStore) {
        val cursor = db.readableDatabase.query(
            IndexDatabase.TABLE_CHUNKS,
            arrayOf(
                IndexDatabase.COL_ID,
                IndexDatabase.COL_FILE_PATH,
                IndexDatabase.COL_CHUNK_OFFSET,
                IndexDatabase.COL_CONTENT
            ),
            null, null, null, null, null
        )
        cursor.use { c ->
            while (c.moveToNext()) {
                rehydrateChunk(
                    id = c.getString(0),
                    filePath = c.getString(1),
                    chunkOffset = c.getInt(2),
                    modelId = "",
                    vectorBlob = null,
                    content = c.getString(3),
                    vectorStore = null,
                    keywordStore = keywordStore
                )
            }
        }
    }

    // -------------------------------------------------------------------------
    // Prune (U8, R13). DATA-LOSS PATH: callers must have verified access and a finished pass.
    // -------------------------------------------------------------------------

    /**
     * Delete document rows (file_index + chunks + FTS) whose MediaStore URI is absent from [seen].
     *
     * Caller contract: [seen] is the result of a FULL enumeration (not a delta) made in this run,
     * with full document access, and the document pass ran to the end without a stop.
     * Only `content://media/` keys are touched, so File-path rows from the explicit-path branch
     * are never pruned. Photo rows (those with an image_vectors row) are left to [prunePhotos].
     *
     * Skips (returns 0) when [seen] is empty or [volumesAllowPrune] says no.
     * ponytail: an empty enumeration is treated as a failed query, not "all deleted". The last
     * deleted document therefore lingers until the next non-empty run. Cheap, and never loses data.
     *
     * @return number of documents removed.
     */
    fun pruneDocuments(seen: Set<String>, presentVolumes: Set<String>): Int {
        if (seen.isEmpty() || !volumesAllowPrune(presentVolumes)) return 0
        val w = db.writableDatabase
        val stale = linkedSetOf<String>()
        w.rawQuery(
            "SELECT ${IndexDatabase.COL_FI_FILE_PATH} FROM ${IndexDatabase.TABLE_FILE_INDEX} " +
                "WHERE ${IndexDatabase.COL_FI_FILE_PATH} LIKE '$MEDIA_URI_PREFIX%' " +
                "AND ${IndexDatabase.COL_FI_FILE_PATH} NOT IN " +
                "(SELECT ${IndexDatabase.COL_IV_FILE_PATH} FROM ${IndexDatabase.TABLE_IMAGE_VECTORS})",
            null
        ).use { c -> while (c.moveToNext()) c.getString(0).let { if (it !in seen) stale.add(it) } }
        // Chunk rows can outlive a file_index row (marked indexed with no text writes none, but a
        // crash between the two writes could); sweep chunk-only orphans too.
        w.rawQuery(
            "SELECT DISTINCT ${IndexDatabase.COL_FILE_PATH} FROM ${IndexDatabase.TABLE_CHUNKS} " +
                "WHERE ${IndexDatabase.COL_FILE_PATH} LIKE '$MEDIA_URI_PREFIX%'",
            null
        ).use { c ->
            while (c.moveToNext()) c.getString(0).let { if (it !in seen) stale.add(it) }
        }
        w.beginTransaction()
        try {
            for (path in stale) {
                deleteChunksForFile(path)
                w.delete(IndexDatabase.TABLE_FILE_INDEX, "${IndexDatabase.COL_FI_FILE_PATH} = ?", arrayOf(path))
            }
            w.setTransactionSuccessful()
        } finally {
            w.endTransaction()
        }
        return stale.size
    }

    /**
     * Delete photo rows (image_vectors + file_index + favorites) whose URI is absent from [seen].
     *
     * Caller contract: [seen] is a FULL image enumeration made with FULL photo access (never the
     * user-selected-only grant, which lists a subset), and the photo pass finished without a stop.
     * Same empty-set and volume guards as [pruneDocuments].
     *
     * @return number of photos removed.
     */
    fun prunePhotos(seen: Set<String>, presentVolumes: Set<String>): Int {
        if (seen.isEmpty() || !volumesAllowPrune(presentVolumes)) return 0
        val w = db.writableDatabase
        val stale = mutableListOf<String>()
        w.rawQuery(
            "SELECT ${IndexDatabase.COL_IV_ID}, ${IndexDatabase.COL_IV_FILE_PATH} " +
                "FROM ${IndexDatabase.TABLE_IMAGE_VECTORS} " +
                "WHERE ${IndexDatabase.COL_IV_FILE_PATH} LIKE '$MEDIA_URI_PREFIX%'",
            null
        ).use { c ->
            while (c.moveToNext()) if (c.getString(1) !in seen) stale.add(c.getString(0))
        }
        w.beginTransaction()
        try {
            for (id in stale) {
                // Image rows use id == file_path == uri (see ImageIndexer).
                w.delete(IndexDatabase.TABLE_IMAGE_VECTORS, "${IndexDatabase.COL_IV_ID} = ?", arrayOf(id))
                w.delete(IndexDatabase.TABLE_FILE_INDEX, "${IndexDatabase.COL_FI_FILE_PATH} = ?", arrayOf(id))
                w.delete(IndexDatabase.TABLE_FAVORITES, "${IndexDatabase.COL_FAV_URI} = ?", arrayOf(id))
            }
            w.setTransactionSuccessful()
        } finally {
            w.endTransaction()
        }
        return stale.size
    }

    /**
     * True when a prune may touch rows: every volume recorded by an earlier run is present now.
     * On true, records the present set. A recorded volume that is absent (SD card removed or not
     * mounted) means some rows may live there, so nothing is deleted.
     *
     * ponytail: rows carry no volume (keys are content://media/external/file/<id>), so this is
     * all-or-nothing. A card removed for good blocks pruning until a reindex (clearAll resets the
     * record). Upgrade path: store the VOLUME_NAME per row (IndexTarget + schema bump) and prune
     * per volume.
     */
    internal fun volumesAllowPrune(present: Set<String>): Boolean {
        if (present.isEmpty()) return false
        val recorded = getConfig(CONFIG_PRUNE_VOLUMES)?.split('\n')?.filter { it.isNotEmpty() }.orEmpty()
        if (!present.containsAll(recorded)) return false
        setConfig(CONFIG_PRUNE_VOLUMES, present.sorted().joinToString("\n"))
        return true
    }

    /**
     * Adds [present] to the recorded volume set (union, never shrinks). No-op when empty.
     * Called before each listing so a volume that was indexed in a run that never reached a prune
     * is still on record when it later goes missing. See [volumesAllowPrune].
     */
    fun recordVolumes(present: Set<String>) {
        if (present.isEmpty()) return
        val recorded = getConfig(CONFIG_PRUNE_VOLUMES)?.split('\n')?.filter { it.isNotEmpty() }.orEmpty()
        setConfig(CONFIG_PRUNE_VOLUMES, (recorded.toSet() + present).sorted().joinToString("\n"))
    }

    // -------------------------------------------------------------------------
    // Config
    // -------------------------------------------------------------------------

    fun getConfig(key: String): String? {
        val cursor = db.readableDatabase.query(
            IndexDatabase.TABLE_INDEX_CONFIG,
            arrayOf(IndexDatabase.COL_IC_VALUE),
            "${IndexDatabase.COL_IC_KEY} = ?",
            arrayOf(key),
            null, null, null
        )
        return cursor.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
    }

    fun setConfig(key: String, value: String) {
        val cv = ContentValues().apply {
            put(IndexDatabase.COL_IC_KEY, key)
            put(IndexDatabase.COL_IC_VALUE, value)
        }
        db.writableDatabase.insertWithOnConflict(
            IndexDatabase.TABLE_INDEX_CONFIG,
            null,
            cv,
            SQLiteDatabase.CONFLICT_REPLACE
        )
    }

    // -------------------------------------------------------------------------
    // Snippet fetch (lazy -- only for returned results)
    // -------------------------------------------------------------------------

    /**
     * Load the text content of a single chunk from the database.
     *
     * This is intentionally a per-result fetch (not bulk-loaded for the whole index)
     * so that snippet text is never held in RAM for every indexed chunk simultaneously.
     * Bug Watch: "Snippet text loaded for every chunk -> OOM on 4 GB phones."
     *
     * @param filePath    The file path key stored in [IndexDatabase.TABLE_CHUNKS].
     * @param chunkOffset The chunk index (0-based) identifying the chunk within the file.
     * @return The text content of the chunk, or `null` if no matching row is found.
     */
    fun chunkContent(filePath: String, chunkOffset: Int): String? {
        // Query by primary key id = "$filePath#$chunkOffset" -- same format used on insert.
        val id = "$filePath#$chunkOffset"
        val cursor = db.readableDatabase.query(
            IndexDatabase.TABLE_CHUNKS,
            arrayOf(IndexDatabase.COL_CONTENT),
            "${IndexDatabase.COL_ID} = ?",
            arrayOf(id),
            null, null, null,
            "1"  // LIMIT 1 -- primary key is unique, but belt-and-suspenders
        )
        return cursor.use { c -> if (c.moveToFirst()) c.getString(0) else null }
    }

    // -------------------------------------------------------------------------
    // Counts
    // -------------------------------------------------------------------------

    /** Total number of chunk rows persisted. */
    fun chunkCount(): Int {
        val cursor = db.readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM ${IndexDatabase.TABLE_CHUNKS}", null
        )
        return cursor.use { c -> if (c.moveToFirst()) c.getInt(0) else 0 }
    }

    /** Total number of files recorded in file_index. */
    fun indexedFileCount(): Int {
        val cursor = db.readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM ${IndexDatabase.TABLE_FILE_INDEX}", null
        )
        return cursor.use { c -> if (c.moveToFirst()) c.getInt(0) else 0 }
    }

    /**
     * Returns the [IndexDatabase.COL_FI_LAST_INDEXED] timestamp (epoch millis) of the
     * most recently indexed file, or `null` if no files have been indexed yet.
     *
     * Used by the Settings tab's IndexStatusCard to display "last run N minutes ago".
     * Runs on a background thread (caller's responsibility) to avoid blocking the main thread.
     */
    fun lastIndexedAt(): Long? {
        val cursor = db.readableDatabase.rawQuery(
            "SELECT MAX(${IndexDatabase.COL_FI_LAST_INDEXED}) FROM ${IndexDatabase.TABLE_FILE_INDEX}",
            null
        )
        return cursor.use { c ->
            if (c.moveToFirst() && !c.isNull(0)) c.getLong(0) else null
        }
    }

    // -------------------------------------------------------------------------
    // Image vector persistence (U5)
    // -------------------------------------------------------------------------

    /**
     * Insert or replace an image embedding row.
     *
     * [id] should be a unique string for the image (e.g. the file path or content URI).
     * [vector] must be a 256-element float32 L2-normalised array from EmbeddingGemma 2.
     *
     * Idempotent: calling again with the same [id] replaces the previous row (INSERT OR REPLACE).
     *
     * @param id        Unique image identifier.
     * @param filePath  Absolute file path or content URI string for the image.
     * @param vector    256-element L2-normalised EG2 image embedding.
     * @param modelId   Model identifier (use [IndexDatabase.IMAGE_MODEL_ID] = "embeddinggemma-2").
     * @param dim       Embedding dimension (use [IndexDatabase.IMAGE_DIM_VALUE] = 256).
     */
    fun upsertImageVector(
        id: String,
        filePath: String,
        vector: FloatArray,
        modelId: String = IndexDatabase.IMAGE_MODEL_ID,
        dim: Int = IndexDatabase.IMAGE_DIM_VALUE
    ) {
        val cv = ContentValues().apply {
            put(IndexDatabase.COL_IV_ID, id)
            put(IndexDatabase.COL_IV_FILE_PATH, filePath)
            put(IndexDatabase.COL_IV_VECTOR, floatArrayToBlob(vector))
            put(IndexDatabase.COL_IV_MODEL_ID, modelId)
            put(IndexDatabase.COL_IV_DIM, dim)
        }
        db.writableDatabase.insertWithOnConflict(
            IndexDatabase.TABLE_IMAGE_VECTORS,
            null,
            cv,
            SQLiteDatabase.CONFLICT_REPLACE
        )
    }

    /**
     * Total number of image vector rows persisted.
     */
    fun imageVectorCount(): Int {
        val cursor = db.readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM ${IndexDatabase.TABLE_IMAGE_VECTORS}", null
        )
        return cursor.use { c -> if (c.moveToFirst()) c.getInt(0) else 0 }
    }

    /**
     * Read all image_vectors rows and load them into the given in-memory [VectorStore].
     *
     * This is the image-side equivalent of [rehydrate] -- call on cold start to rebuild
     * the 256-dim in-memory image vector store from the persisted SQLite rows.
     *
     * The [VectorStore] is dimension-agnostic, so a separate 256-dim instance is used for
     * image vectors (do NOT mix with the 256-dim text vector store -- separate VectorStore
     * instances prevent accidental cross-modal dot products).
     *
     * @param imageVectorStore A [VectorStore] instance dedicated to image vectors (256-dim).
     */
    fun rehydrateImages(imageVectorStore: VectorStore) {
        val cursor = db.readableDatabase.query(
            IndexDatabase.TABLE_IMAGE_VECTORS,
            arrayOf(
                IndexDatabase.COL_IV_ID,
                IndexDatabase.COL_IV_FILE_PATH,
                IndexDatabase.COL_IV_MODEL_ID,
                IndexDatabase.COL_IV_VECTOR
            ),
            null, null, null, null, null
        )
        cursor.use { c ->
            while (c.moveToNext()) {
                val id = c.getString(0)
                val filePath = c.getString(1)
                val modelId = c.getString(2)
                val blob = c.getBlob(3)
                val vector = blobToFloatArray(blob)

                imageVectorStore.add(
                    id = id,
                    filePath = filePath,
                    chunkOffset = 0,  // images have no chunk offset; use 0
                    modelId = modelId,
                    vector = vector
                )
            }
        }
    }

    // -------------------------------------------------------------------------
    // Favorites (v4, KTD16)
    // -------------------------------------------------------------------------

    /**
     * Add [uri] to favorites. Idempotent: if the URI already exists, the original
     * added_at timestamp and list position are preserved (CONFLICT_IGNORE).
     */
    fun addFavorite(uri: String) {
        val cv = ContentValues().apply {
            put(IndexDatabase.COL_FAV_URI, uri)
            put(IndexDatabase.COL_FAV_ADDED_AT, System.currentTimeMillis())
        }
        db.writableDatabase.insertWithOnConflict(
            IndexDatabase.TABLE_FAVORITES,
            null,
            cv,
            SQLiteDatabase.CONFLICT_IGNORE
        )
    }

    /**
     * Remove [uri] from favorites. No-op if not present.
     * Used by the "open failed" path to clean up a stale favorite.
     */
    fun removeFavorite(uri: String) {
        db.writableDatabase.delete(
            IndexDatabase.TABLE_FAVORITES,
            "${IndexDatabase.COL_FAV_URI} = ?",
            arrayOf(uri)
        )
    }

    /**
     * List all favorited URIs, newest first (ORDER BY added_at DESC).
     */
    fun listFavorites(): List<String> {
        val cursor = db.readableDatabase.query(
            IndexDatabase.TABLE_FAVORITES,
            arrayOf(IndexDatabase.COL_FAV_URI),
            null, null, null, null,
            "${IndexDatabase.COL_FAV_ADDED_AT} DESC"
        )
        return cursor.use { c ->
            buildList {
                while (c.moveToNext()) add(c.getString(0))
            }
        }
    }

    /** Total number of favorite rows. Useful for testing. */
    fun favoriteCount(): Int {
        val cursor = db.readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM ${IndexDatabase.TABLE_FAVORITES}", null
        )
        return cursor.use { c -> if (c.moveToFirst()) c.getInt(0) else 0 }
    }

    // -------------------------------------------------------------------------
    // Index events (v4, KTD18)
    // -------------------------------------------------------------------------

    /**
     * Append one event to the index_events log.
     *
     * KTD18 enforcement: [fileName] is stripped to the bare file name (substringAfterLast('/'))
     * so no directory paths are stored. [message] is capped at 200 characters. Neither field
     * ever contains extracted file content.
     *
     * Write errors are caught and logged via Logcat only -- a logging failure never
     * fails an index run.
     */
    fun appendEvent(
        runId: String,
        stage: String,
        level: String,
        fileName: String? = null,
        errorClass: String? = null,
        message: String? = null
    ) {
        try {
            val cv = ContentValues().apply {
                put(IndexDatabase.COL_IE_TIMESTAMP_MS, System.currentTimeMillis())
                put(IndexDatabase.COL_IE_RUN_ID, runId)
                put(IndexDatabase.COL_IE_STAGE, stage)
                put(IndexDatabase.COL_IE_LEVEL, level)
                // KTD18: strip path prefix; never store directory parts
                if (fileName != null) put(IndexDatabase.COL_IE_FILE_NAME, fileName.substringAfterLast('/'))
                if (errorClass != null) put(IndexDatabase.COL_IE_ERROR_CLASS, errorClass)
                // KTD18: cap message at 200 chars so no large extracts can sneak in
                if (message != null) put(IndexDatabase.COL_IE_MESSAGE, message.take(200))
            }
            db.writableDatabase.insert(IndexDatabase.TABLE_INDEX_EVENTS, null, cv)
        } catch (e: Exception) {
            Log.w(TAG, "appendEvent failed: ${e.message}")
        }
    }

    /**
     * Prune index_events: delete rows older than 7 days, then keep only the newest 500
     * (5,000 in a NARAD_METRICS build, so 3 or 4 nights of run lines survive).
     * Call at the end of each index run (from W18 wiring).
     *
     * Errors are caught and logged -- a SQLiteFullException or locked DB must not
     * escape and fail a run.
     */
    fun pruneEvents() {
        try {
            val w = db.writableDatabase
            val sevenDaysAgo = System.currentTimeMillis() - 7L * 24 * 3600 * 1000
            // Delete by age first.
            w.delete(
                IndexDatabase.TABLE_INDEX_EVENTS,
                "${IndexDatabase.COL_IE_TIMESTAMP_MS} < ?",
                arrayOf(sevenDaysAgo.toString())
            )
            // Keep only the newest N: delete everything not in the top N by id desc.
            w.execSQL(
                "DELETE FROM ${IndexDatabase.TABLE_INDEX_EVENTS} " +
                "WHERE ${IndexDatabase.COL_IE_ID} NOT IN (" +
                "  SELECT ${IndexDatabase.COL_IE_ID} FROM ${IndexDatabase.TABLE_INDEX_EVENTS} " +
                "  ORDER BY ${IndexDatabase.COL_IE_ID} DESC LIMIT ${eventCap(BuildConfig.METRICS)}" +
                ")"
            )
        } catch (e: Exception) {
            Log.w(TAG, "pruneEvents failed: ${e.message}")
        }
    }

    /**
     * Return the newest [limit] index events, most-recent first.
     */
    fun listEvents(limit: Int = 500): List<IndexEvent> {
        val cursor = db.readableDatabase.query(
            IndexDatabase.TABLE_INDEX_EVENTS,
            arrayOf(
                IndexDatabase.COL_IE_ID,
                IndexDatabase.COL_IE_TIMESTAMP_MS,
                IndexDatabase.COL_IE_RUN_ID,
                IndexDatabase.COL_IE_STAGE,
                IndexDatabase.COL_IE_LEVEL,
                IndexDatabase.COL_IE_FILE_NAME,
                IndexDatabase.COL_IE_ERROR_CLASS,
                IndexDatabase.COL_IE_MESSAGE
            ),
            null, null, null, null,
            "${IndexDatabase.COL_IE_ID} DESC",
            limit.toString()
        )
        return cursor.use { c ->
            buildList {
                while (c.moveToNext()) {
                    add(
                        IndexEvent(
                            id = c.getLong(0),
                            timestampMs = c.getLong(1),
                            runId = c.getString(2),
                            stage = c.getString(3),
                            level = c.getString(4),
                            fileName = if (c.isNull(5)) null else c.getString(5),
                            errorClass = if (c.isNull(6)) null else c.getString(6),
                            message = if (c.isNull(7)) null else c.getString(7)
                        )
                    )
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Wipe
    // -------------------------------------------------------------------------

    /**
     * Wipe embedding tables (chunks, file_index, image_vectors) and re-seed config.
     *
     * Favorites and index_events are intentionally kept:
     *   - Favorites are user data and must survive a reindex.
     *   - Events are an audit log; [pruneEvents] handles capacity.
     *
     * Stale favorites are cleaned individually via [removeFavorite] on the "open failed" path.
     */
    fun clearAll() {
        val w = db.writableDatabase
        w.delete(IndexDatabase.TABLE_CHUNKS, null, null)
        try { w.delete(IndexDatabase.TABLE_CHUNKS_FTS, null, null) } catch (_: Exception) {}
        w.delete(IndexDatabase.TABLE_FILE_INDEX, null, null)
        try { w.delete(IndexDatabase.TABLE_IMAGE_VECTORS, null, null) } catch (_: Exception) {}
        // A reindex starts a fresh volume record (see volumesAllowPrune).
        w.delete(IndexDatabase.TABLE_INDEX_CONFIG, "${IndexDatabase.COL_IC_KEY} = ?", arrayOf(CONFIG_PRUNE_VOLUMES))
        // Re-seed config defaults
        w.execSQL(
            "INSERT OR REPLACE INTO ${IndexDatabase.TABLE_INDEX_CONFIG} VALUES ('text_model_id', '${IndexDatabase.MODEL_ID}')"
        )
        w.execSQL(
            "INSERT OR REPLACE INTO ${IndexDatabase.TABLE_INDEX_CONFIG} VALUES ('text_dim', '${IndexDatabase.DIM_VALUE}')"
        )
        w.execSQL(
            "INSERT OR REPLACE INTO ${IndexDatabase.TABLE_INDEX_CONFIG} VALUES ('image_model_id', '${IndexDatabase.IMAGE_MODEL_ID}')"
        )
        w.execSQL(
            "INSERT OR REPLACE INTO ${IndexDatabase.TABLE_INDEX_CONFIG} VALUES ('image_dim', '${IndexDatabase.IMAGE_DIM_VALUE}')"
        )
    }

    // -------------------------------------------------------------------------
    // Float <-> BLOB helpers
    // -------------------------------------------------------------------------

    /**
     * Encode a float32 array to a little-endian byte array (4 bytes per float).
     * 256 floats -> 1 024 bytes.
     */
    fun floatArrayToBlob(floats: FloatArray): ByteArray {
        val buf = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (f in floats) buf.putFloat(f)
        return buf.array()
    }

    /**
     * Decode a little-endian byte array back to a float32 array.
     */
    fun blobToFloatArray(bytes: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(bytes.size / 4)
        for (i in floats.indices) floats[i] = buf.getFloat()
        return floats
    }
}
