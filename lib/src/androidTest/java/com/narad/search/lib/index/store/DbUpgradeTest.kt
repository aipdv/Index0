package com.narad.search.lib.index.store

// NEEDS-DEVICE: ./gradlew :lib:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.narad.search.lib.index.store.DbUpgradeTest

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented DB upgrade tests.
 *
 * Test 1 (v2 -> v4): fully destructive -- ALL tables are dropped and recreated.
 *   Proves chunks, file_index, image_vectors are empty and config has EG2 values.
 *
 * Test 2 (v3 -> v4): additive -- only favorites and index_events are added.
 *   Proves existing chunks, file_index, image_vectors and index_config rows survive (no data loss).
 */
@RunWith(AndroidJUnit4::class)
class DbUpgradeTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private lateinit var dbName: String

    @Before
    fun setUp() {
        dbName = "narad_upgrade_test_${System.currentTimeMillis()}.db"
    }

    @After
    fun tearDown() {
        context.deleteDatabase(dbName)
    }

    // -------------------------------------------------------------------------
    // Test 1: v2 -> v4 (destructive)
    // -------------------------------------------------------------------------

    @Test
    fun upgrade_v2_to_v4_clears_all_tables_including_image_vectors() {
        // ---- Phase 1: create a v2 DB and seed rows manually ----------------
        val seedDb = context.openOrCreateDatabase(dbName, 0, null)
        seedDb.use { db ->
            // v2 schema (Granite 384-dim text + old 512-dim image embeddings):
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS chunks (" +
                "id TEXT PRIMARY KEY, file_path TEXT, chunk_offset INTEGER, " +
                "model_id TEXT, dim INTEGER, vector BLOB, content TEXT)"
            )
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS file_index (" +
                "file_path TEXT PRIMARY KEY, mime_type TEXT, last_modified INTEGER, " +
                "last_indexed INTEGER, chunk_count INTEGER, model_id TEXT, dim INTEGER)"
            )
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS index_config (key TEXT PRIMARY KEY, value TEXT)"
            )
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS image_vectors (" +
                "id TEXT PRIMARY KEY, file_path TEXT, vector BLOB, model_id TEXT, dim INTEGER)"
            )

            // Seed: one chunk row (Granite 384-dim)
            val chunkCv = ContentValues().apply {
                put("id", "old_chunk#0")
                put("file_path", "/old/granite_doc.pdf")
                put("chunk_offset", 0)
                put("model_id", "granite-97m-r2")
                put("dim", 384)
                put("vector", ByteArray(384 * 4) { 0 })
                put("content", "Granite era content")
            }
            db.insert("chunks", null, chunkCv)

            // Seed: one file_index row
            val fiCv = ContentValues().apply {
                put("file_path", "/old/granite_doc.pdf")
                put("mime_type", "application/pdf")
                put("last_modified", 1000L)
                put("last_indexed", 2000L)
                put("chunk_count", 1)
                put("model_id", "granite-97m-r2")
                put("dim", 384)
            }
            db.insert("file_index", null, fiCv)

            // Seed: index_config (v2 Granite values)
            db.execSQL("INSERT INTO index_config VALUES ('text_model_id', 'granite-97m-r2')")
            db.execSQL("INSERT INTO index_config VALUES ('text_dim', '384')")

            // Seed: one image_vectors row (old 512-dim data that must NOT survive upgrade)
            val ivCv = ContentValues().apply {
                put("id", "image#1")
                put("file_path", "/photos/cat.jpg")
                put("vector", ByteArray(512 * 4) { 1 })
                put("model_id", "old-image-model")
                put("dim", 512)
            }
            db.insert("image_vectors", null, ivCv)

            // Mark as v2 so SQLiteOpenHelper calls onUpgrade (not onCreate).
            db.version = 2
        }

        // ---- Phase 2: verify pre-upgrade counts ----------------------------
        val preSeedDb = context.openOrCreateDatabase(dbName, 0, null)
        preSeedDb.use { db ->
            assertEquals("chunks must have 1 row before upgrade", 1, countRows(db, "chunks"))
            assertEquals("image_vectors must have 1 row before upgrade", 1, countRows(db, "image_vectors"))
        }

        // ---- Phase 3: open via IndexDatabase (v4) -- triggers onUpgrade ----
        val v4Db = IndexDatabase(context, dbName)
        v4Db.readableDatabase  // force open + upgrade
        v4Db.close()

        // ---- Phase 4: verify post-upgrade state ----------------------------
        val postDb = context.openOrCreateDatabase(dbName, 0, null)
        postDb.use { db ->
            // All text tables must be empty after destructive upgrade.
            assertEquals("chunks must be empty after upgrade from v2", 0, countRows(db, "chunks"))
            assertEquals("file_index must be empty after upgrade from v2", 0, countRows(db, "file_index"))

            // index_config must have the new EG2 model ID seeded.
            val textModelId = db.rawQuery(
                "SELECT value FROM index_config WHERE key='text_model_id'", null
            ).use { c -> if (c.moveToFirst()) c.getString(0) else null }
            assertEquals(
                "index_config.text_model_id must be 'embeddinggemma-2' after upgrade",
                "embeddinggemma-2", textModelId
            )

            // image_vectors must be EMPTY after upgrade (old data was dropped and recreated).
            assertEquals(
                "image_vectors must be recreated empty after upgrade from v2",
                0,
                countRows(db, "image_vectors")
            )

            // index_config must have the new image_model_id = 'embeddinggemma-2'
            val imageModelId = db.rawQuery(
                "SELECT value FROM index_config WHERE key='image_model_id'", null
            ).use { c -> if (c.moveToFirst()) c.getString(0) else null }
            assertEquals(
                "index_config.image_model_id must be 'embeddinggemma-2' after upgrade",
                "embeddinggemma-2", imageModelId
            )

            // index_config must have image_dim = 256 (not the old 512)
            val imageDim = db.rawQuery(
                "SELECT value FROM index_config WHERE key='image_dim'", null
            ).use { c -> if (c.moveToFirst()) c.getString(0) else null }
            assertEquals(
                "index_config.image_dim must be '256' after upgrade",
                "256", imageDim
            )

            // v4 tables must exist after a v2->v4 upgrade.
            assertTrue("favorites table must exist after v2->v4 upgrade", tableExists(db, "favorites"))
            assertTrue("index_events table must exist after v2->v4 upgrade", tableExists(db, "index_events"))
        }
    }

    // -------------------------------------------------------------------------
    // Test 2: v3 -> v4 (additive -- no data loss)
    // -------------------------------------------------------------------------

    @Test
    fun upgrade_v3_to_v4_adds_new_tables_without_clearing_existing_data() {
        // ---- Phase 1: create a v3 DB with data in existing tables ----------
        val seedDb = context.openOrCreateDatabase(dbName, 0, null)
        seedDb.use { db ->
            // v3 schema
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS chunks (" +
                "id TEXT PRIMARY KEY, file_path TEXT, chunk_offset INTEGER, " +
                "model_id TEXT, dim INTEGER, vector BLOB, content TEXT)"
            )
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS file_index (" +
                "file_path TEXT PRIMARY KEY, mime_type TEXT, last_modified INTEGER, " +
                "last_indexed INTEGER, chunk_count INTEGER, model_id TEXT, dim INTEGER)"
            )
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS index_config (key TEXT PRIMARY KEY, value TEXT)"
            )
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS image_vectors (" +
                "id TEXT PRIMARY KEY, file_path TEXT, vector BLOB, model_id TEXT, dim INTEGER)"
            )

            // Seed a chunk (EG2 256-dim) -- must survive the upgrade
            val chunkCv = ContentValues().apply {
                put("id", "doc#0")
                put("file_path", "/docs/contract.pdf")
                put("chunk_offset", 0)
                put("model_id", "embeddinggemma-2")
                put("dim", 256)
                put("vector", ByteArray(256 * 4) { 1 })
                put("content", "Contract terms and penalty clause")
            }
            db.insert("chunks", null, chunkCv)

            // Seed a file_index row -- must survive the upgrade
            val fiCv = ContentValues().apply {
                put("file_path", "/docs/contract.pdf")
                put("mime_type", "application/pdf")
                put("last_modified", 12345L)
                put("last_indexed", 67890L)
                put("chunk_count", 1)
                put("model_id", "embeddinggemma-2")
                put("dim", 256)
            }
            db.insert("file_index", null, fiCv)

            // Seed an image vector -- must survive
            val ivCv = ContentValues().apply {
                put("id", "photo#1")
                put("file_path", "/photos/sunset.jpg")
                put("vector", ByteArray(256 * 4) { 2 })
                put("model_id", "embeddinggemma-2")
                put("dim", 256)
            }
            db.insert("image_vectors", null, ivCv)

            // v3 config -- must survive
            db.execSQL("INSERT INTO index_config VALUES ('text_model_id', 'embeddinggemma-2')")
            db.execSQL("INSERT INTO index_config VALUES ('text_dim', '256')")

            // Mark as v3 so SQLiteOpenHelper calls onUpgrade.
            db.version = 3
        }

        // ---- Phase 2: open via IndexDatabase (v4) -- triggers v3->v4 path ---
        val v4Db = IndexDatabase(context, dbName)
        v4Db.readableDatabase  // force open + upgrade
        v4Db.close()

        // ---- Phase 3: verify data was preserved and new tables exist --------
        val postDb = context.openOrCreateDatabase(dbName, 0, null)
        postDb.use { db ->
            // Existing data must be intact.
            assertEquals("chunks must still have 1 row after v3->v4 upgrade", 1, countRows(db, "chunks"))
            assertEquals("file_index must still have 1 row after v3->v4 upgrade", 1, countRows(db, "file_index"))
            assertEquals("image_vectors must still have 1 row after v3->v4 upgrade", 1, countRows(db, "image_vectors"))

            // index_config must be unchanged -- specifically the EG2 model id.
            val textModelId = db.rawQuery(
                "SELECT value FROM index_config WHERE key='text_model_id'", null
            ).use { c -> if (c.moveToFirst()) c.getString(0) else null }
            assertEquals(
                "index_config.text_model_id must survive v3->v4 upgrade unchanged",
                "embeddinggemma-2", textModelId
            )

            // New tables must exist and be empty.
            assertTrue("favorites table must exist after v3->v4 upgrade", tableExists(db, "favorites"))
            assertEquals("favorites must be empty initially", 0, countRows(db, "favorites"))

            assertTrue("index_events table must exist after v3->v4 upgrade", tableExists(db, "index_events"))
            assertEquals("index_events must be empty initially", 0, countRows(db, "index_events"))
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private fun countRows(db: SQLiteDatabase, table: String): Int {
        return db.rawQuery("SELECT COUNT(*) FROM $table", null).use { c ->
            if (c.moveToFirst()) c.getInt(0) else 0
        }
    }

    private fun tableExists(db: SQLiteDatabase, table: String): Boolean {
        return db.rawQuery(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            arrayOf(table)
        ).use { c -> c.moveToFirst() }
    }
}
