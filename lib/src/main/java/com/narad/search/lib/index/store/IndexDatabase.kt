package com.narad.search.lib.index.store

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

/**
 * SQLiteOpenHelper that owns the Narad index schema.
 *
 * Schema (no sqlite-vec -- uses built-in Android SQLite + FTS):
 *
 *   chunks         -- one row per text chunk; embedding stored as float32 BLOB (256 x 4 bytes LE)
 *   chunks_fts     -- FTS virtual table (FTS5 if available, FTS4 fallback) for durable keyword search
 *   file_index     -- one row per indexed file (dedup / resume)
 *   index_config   -- key/value settings (model_id, dim)
 *   image_vectors  -- 256-dim EG2 image embeddings (M2)
 *   favorites      -- content URIs bookmarked by the user (v4, KTD16)
 *   index_events   -- capped audit log of indexing events (v4, KTD18)
 *
 * The U4 query path rehydrates into in-memory VectorStore/KeywordStore, so the
 * FTS virtual table is a secondary durability mirror (not the query path).
 *
 * The database file name is passed by the caller so tests can use an
 * isolated file name without polluting the production DB.
 */
class IndexDatabase(
    context: Context,
    dbName: String = DB_NAME
) : SQLiteOpenHelper(context, dbName, null, DB_VERSION) {

    companion object {
        const val DB_NAME = "narad_index.db"
        /**
         * DB_VERSION history:
         *   v3: swapped text embedding model from Granite-97m-r2 (384-dim) to EmbeddingGemma 2
         *       (256-dim Matryoshka slice). onUpgrade clears all text tables and image_vectors.
         *   v4: additive -- adds favorites (KTD16) and index_events (KTD18). Existing data is kept.
         */
        const val DB_VERSION = 4

        // Table / column names
        const val TABLE_CHUNKS = "chunks"
        const val TABLE_CHUNKS_FTS = "chunks_fts"
        const val TABLE_FILE_INDEX = "file_index"
        const val TABLE_INDEX_CONFIG = "index_config"

        /** Image embedding store -- 256-dim EG2 vectors (1024 bytes per row). */
        const val TABLE_IMAGE_VECTORS = "image_vectors"

        /** Favorites table (v4, KTD16): one row per bookmarked content URI. */
        const val TABLE_FAVORITES = "favorites"

        /** Index event log (v4, KTD18): capped at 500 rows or 7 days. */
        const val TABLE_INDEX_EVENTS = "index_events"

        const val COL_ID = "id"                          // TEXT PRIMARY KEY "<filePath>#<offset>"
        const val COL_FILE_PATH = "file_path"
        const val COL_CHUNK_OFFSET = "chunk_offset"
        const val COL_MODEL_ID = "model_id"
        const val COL_DIM = "dim"
        const val COL_VECTOR = "vector"                  // BLOB -- 256 float32 LE = 1024 bytes
        const val COL_CONTENT = "content"

        const val COL_FI_FILE_PATH = "file_path"
        const val COL_FI_MIME_TYPE = "mime_type"
        const val COL_FI_LAST_MODIFIED = "last_modified"
        const val COL_FI_LAST_INDEXED = "last_indexed"
        const val COL_FI_CHUNK_COUNT = "chunk_count"
        const val COL_FI_MODEL_ID = "model_id"
        const val COL_FI_DIM = "dim"

        const val COL_IC_KEY = "key"
        const val COL_IC_VALUE = "value"

        // image_vectors column names (mirrors chunks pattern)
        /** Primary key for image_vectors: unique image identifier (e.g. file path or content URI). */
        const val COL_IV_ID = "id"
        const val COL_IV_FILE_PATH = "file_path"
        /** BLOB: 256 float32 values in little-endian byte order = 1024 bytes. */
        const val COL_IV_VECTOR = "vector"
        const val COL_IV_MODEL_ID = "model_id"
        const val COL_IV_DIM = "dim"

        // favorites column names (v4, KTD16)
        /** Content URI or file path; primary key -- ensures one row per favorite. */
        const val COL_FAV_URI = "uri"
        /** Epoch millis when the favorite was added; used for ORDER BY added_at DESC. */
        const val COL_FAV_ADDED_AT = "added_at"

        // index_events column names (v4, KTD18)
        /** Auto-increment id; used for count-capped pruning (newest 500). */
        const val COL_IE_ID = "id"
        const val COL_IE_TIMESTAMP_MS = "timestamp_ms"
        const val COL_IE_RUN_ID = "run_id"
        const val COL_IE_STAGE = "stage"
        const val COL_IE_LEVEL = "level"
        /** File name only -- never full path or file content. */
        const val COL_IE_FILE_NAME = "file_name"
        const val COL_IE_ERROR_CLASS = "error_class"
        const val COL_IE_MESSAGE = "message"

        // Seeded config values (M1: EmbeddingGemma 2 with 256-dim Matryoshka slice)
        const val MODEL_ID = "embeddinggemma-2"
        const val DIM_VALUE = 256

        /** EG2 image embedding dimension (same Matryoshka 256-dim space as text). */
        const val IMAGE_DIM_VALUE = 256

        /** Model identifier for image vectors (shared EG2 embedding space). */
        const val IMAGE_MODEL_ID = "embeddinggemma-2"
    }

    /** Whether this database instance successfully created an FTS table. */
    @Volatile
    var hasFts: Boolean = false
        private set

    override fun onCreate(db: SQLiteDatabase) {
        // 1. chunks -- primary embedding store
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS $TABLE_CHUNKS (
                $COL_ID           TEXT PRIMARY KEY,
                $COL_FILE_PATH    TEXT NOT NULL,
                $COL_CHUNK_OFFSET INTEGER NOT NULL,
                $COL_MODEL_ID     TEXT NOT NULL,
                $COL_DIM          INTEGER NOT NULL,
                $COL_VECTOR       BLOB NOT NULL,
                $COL_CONTENT      TEXT NOT NULL
            )
            """.trimIndent()
        )

        // 2. chunks_fts -- FTS virtual table for durable keyword search.
        // Prefer FTS5; fall back to FTS4 if FTS5 is not compiled into
        // the device's SQLite (AOSP emulators often omit FTS5).
        // The U4 query path rehydrates from chunks.content (not from FTS),
        // so FTS absence does not break core functionality.
        hasFts = tryCreateFts5(db)
        if (!hasFts) hasFts = tryCreateFts4(db)

        // 3. file_index -- per-file dedup / resume table
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS $TABLE_FILE_INDEX (
                $COL_FI_FILE_PATH     TEXT PRIMARY KEY,
                $COL_FI_MIME_TYPE     TEXT,
                $COL_FI_LAST_MODIFIED INTEGER,
                $COL_FI_LAST_INDEXED  INTEGER,
                $COL_FI_CHUNK_COUNT   INTEGER,
                $COL_FI_MODEL_ID      TEXT,
                $COL_FI_DIM           INTEGER
            )
            """.trimIndent()
        )

        // 4. index_config -- key/value settings
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS $TABLE_INDEX_CONFIG (
                $COL_IC_KEY   TEXT PRIMARY KEY,
                $COL_IC_VALUE TEXT
            )
            """.trimIndent()
        )

        // 5. image_vectors -- 256-dim EG2 image embedding store (M2)
        //    Each row holds a single image embedding (1024 bytes = 256 float32 LE).
        //    Mirrors the chunks table pattern; uses INSERT OR REPLACE for idempotent upserts.
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS $TABLE_IMAGE_VECTORS (
                $COL_IV_ID        TEXT PRIMARY KEY,
                $COL_IV_FILE_PATH TEXT NOT NULL,
                $COL_IV_VECTOR    BLOB NOT NULL,
                $COL_IV_MODEL_ID  TEXT NOT NULL,
                $COL_IV_DIM       INTEGER NOT NULL
            )
            """.trimIndent()
        )

        // 6. favorites -- user-bookmarked content URIs (v4, KTD16)
        createFavoritesTable(db)

        // 7. index_events -- capped audit log (v4, KTD18)
        createIndexEventsTable(db)

        // Seed default config
        db.execSQL(
            "INSERT OR REPLACE INTO $TABLE_INDEX_CONFIG VALUES ('text_model_id', '$MODEL_ID')"
        )
        db.execSQL(
            "INSERT OR REPLACE INTO $TABLE_INDEX_CONFIG VALUES ('text_dim', '$DIM_VALUE')"
        )
        db.execSQL(
            "INSERT OR REPLACE INTO $TABLE_INDEX_CONFIG VALUES ('image_model_id', '${IMAGE_MODEL_ID}')"
        )
        db.execSQL(
            "INSERT OR REPLACE INTO $TABLE_INDEX_CONFIG VALUES ('image_dim', '${IMAGE_DIM_VALUE}')"
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 3) {
            // DB_VERSION 2 -> 3: EmbeddingGemma 2 text swap (384->256-dim) AND image swap
            // (EG2 replaces old 512-dim image model). Drop ALL tables so both indices are
            // rebuilt with the correct model. onCreate uses CREATE TABLE IF NOT EXISTS.
            try { db.execSQL("DROP TABLE IF EXISTS $TABLE_CHUNKS") } catch (_: Exception) {}
            try { db.execSQL("DROP TABLE IF EXISTS $TABLE_CHUNKS_FTS") } catch (_: Exception) {}
            try { db.execSQL("DROP TABLE IF EXISTS $TABLE_FILE_INDEX") } catch (_: Exception) {}
            try { db.execSQL("DROP TABLE IF EXISTS $TABLE_INDEX_CONFIG") } catch (_: Exception) {}
            try { db.execSQL("DROP TABLE IF EXISTS $TABLE_IMAGE_VECTORS") } catch (_: Exception) {}
            // favorites and index_events did not exist yet; DROP is a no-op here.
            try { db.execSQL("DROP TABLE IF EXISTS $TABLE_FAVORITES") } catch (_: Exception) {}
            try { db.execSQL("DROP TABLE IF EXISTS $TABLE_INDEX_EVENTS") } catch (_: Exception) {}
            // Recreate everything (including v4 tables) and return.
            onCreate(db)
            return
        }
        if (oldVersion < 4) {
            // DB_VERSION 3 -> 4: additive -- just add the two new tables.
            // Existing chunks, file_index, image_vectors data is preserved (no data loss).
            createFavoritesTable(db)
            createIndexEventsTable(db)
        }
    }

    // -------------------------------------------------------------------------
    // Private schema creation helpers
    // -------------------------------------------------------------------------

    internal fun createFavoritesTable(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS $TABLE_FAVORITES (
                $COL_FAV_URI      TEXT PRIMARY KEY,
                $COL_FAV_ADDED_AT INTEGER NOT NULL
            )
            """.trimIndent()
        )
    }

    internal fun createIndexEventsTable(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS $TABLE_INDEX_EVENTS (
                $COL_IE_ID           INTEGER PRIMARY KEY AUTOINCREMENT,
                $COL_IE_TIMESTAMP_MS INTEGER NOT NULL,
                $COL_IE_RUN_ID       TEXT NOT NULL,
                $COL_IE_STAGE        TEXT NOT NULL,
                $COL_IE_LEVEL        TEXT NOT NULL,
                $COL_IE_FILE_NAME    TEXT,
                $COL_IE_ERROR_CLASS  TEXT,
                $COL_IE_MESSAGE      TEXT
            )
            """.trimIndent()
        )
    }

    /** Attempt FTS5 creation. Returns true on success. */
    private fun tryCreateFts5(db: SQLiteDatabase): Boolean {
        return try {
            db.execSQL(
                "CREATE VIRTUAL TABLE IF NOT EXISTS $TABLE_CHUNKS_FTS " +
                "USING fts5(content, chunk_id UNINDEXED)"
            )
            true
        } catch (_: Exception) {
            false
        }
    }

    /** Fallback: attempt FTS4 creation (widely available on Android). Returns true on success. */
    private fun tryCreateFts4(db: SQLiteDatabase): Boolean {
        return try {
            // FTS4 column layout: content text, chunk_id text (notindexed)
            db.execSQL(
                "CREATE VIRTUAL TABLE IF NOT EXISTS $TABLE_CHUNKS_FTS " +
                "USING fts4(content, chunk_id)"
            )
            true
        } catch (_: Exception) {
            false
        }
    }
}
