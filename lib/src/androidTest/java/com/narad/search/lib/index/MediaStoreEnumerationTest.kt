package com.narad.search.lib.index

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.work.ListenableWorker
import androidx.work.WorkerParameters
import androidx.work.testing.TestListenableWorkerBuilder
import androidx.work.workDataOf
import com.narad.search.lib.index.enumeration.MediaStoreEnumerator
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository
import com.narad.search.lib.index.worker.IndexWorker
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.IOException

/**
 * Instrumented tests for B5: MediaStore document enumeration.
 *
 * Proves the PRODUCTION path end-to-end per L13:
 *  - Test 1: MediaStoreEnumerator.enumerateDocuments() returns the seeded file.
 *  - Test 2: IndexWorker with only KEY_MODEL_PATH (no KEY_FILE_PATHS) enumerates
 *            MediaStore, extracts content, and persists chunks to IndexRepository.
 *
 * Both tests seed a real document into MediaStore.Downloads, then clean up in @After.
 *
 * URI note: insertion via MediaStore.Downloads.EXTERNAL_CONTENT_URI yields a "downloads"
 * authority URI (e.g. content://media/external/downloads/<id>), but the enumerator
 * builds URIs via MediaStore.Files.getContentUri(VOLUME_EXTERNAL) which produces
 * content://media/external/file/<id> with the same _ID.  Both point to the same file.
 * Tests match on display name (Test 1) and the enumerator's produced URI (Test 2)
 * rather than the raw insertion URI to stay resilient to this URI-form difference.
 */
@RunWith(AndroidJUnit4::class)
class MediaStoreEnumerationTest {

    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    /** URI returned by MediaStore.Downloads.insert - used for deletion in @After. */
    private var seededInsertUri: Uri? = null

    /** Unique display name for this test run so sequential runs don't collide. */
    private val displayName = "narad_b5_test_${System.currentTimeMillis()}.txt"

    /** Content written into the seeded document. */
    private val documentContent = "penalty clause in the rental contract"

    @Before
    fun clearIndexBeforeTest() {
        // Clear any leftover rows from a prior test run (or a crashed test) so count
        // assertions cannot pass spuriously on pre-existing data.
        try {
            val db = IndexDatabase(context)
            IndexRepository(db).clearAll()
            db.close()
        } catch (e: Exception) {
            // Best-effort; test will catch real pollution via its own assertions.
        }
    }

    @Before
    fun grantStoragePermissions() {
        // Grant storage permissions so IndexWorker does not stop at WaitingForPermission.
        val auto = InstrumentationRegistry.getInstrumentation().uiAutomation
        val pkg  = context.packageName
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            auto.grantRuntimePermission(pkg, Manifest.permission.READ_MEDIA_IMAGES)
            auto.grantRuntimePermission(pkg, Manifest.permission.READ_MEDIA_VIDEO)
        } else {
            auto.grantRuntimePermission(pkg, Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        // Read to EOF so the appop is set before any test body runs.
        android.os.ParcelFileDescriptor.AutoCloseInputStream(
            auto.executeShellCommand("appops set $pkg MANAGE_EXTERNAL_STORAGE allow")
        ).use { it.readBytes() }
    }

    @Before
    fun seedMediaStoreDocument() {
        val resolver = context.contentResolver

        val cv = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, displayName)
            put(MediaStore.Downloads.MIME_TYPE, "text/plain")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Downloads.RELATIVE_PATH, "Download/")
            }
        }

        val insertUri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv)
            ?: throw IOException("Failed to insert test document into MediaStore.Downloads")

        seededInsertUri = insertUri

        resolver.openOutputStream(insertUri)?.use { stream ->
            stream.write(documentContent.toByteArray(Charsets.UTF_8))
        } ?: throw IOException("Failed to open output stream for seeded MediaStore URI $insertUri")
    }

    @After
    fun cleanUp() {
        // Delete the seeded MediaStore file using the original Downloads URI.
        seededInsertUri?.let { uri ->
            try {
                context.contentResolver.delete(uri, null, null)
            } catch (e: Exception) {
                // Best-effort cleanup; test failure is not caused by cleanup issues.
            }
        }
        // Clear the index DB so this test does not bleed state into other tests.
        try {
            val db = IndexDatabase(context)
            IndexRepository(db).clearAll()
            db.close()
        } catch (e: Exception) {
            // Best-effort cleanup.
        }
    }

    // -------------------------------------------------------------------------
    // Test 1: MediaStoreEnumerator.enumerateDocuments() finds the seeded file.
    // -------------------------------------------------------------------------

    /**
     * Verifies that [MediaStoreEnumerator.enumerateDocuments] returns the document
     * inserted into MediaStore.Downloads in [seedMediaStoreDocument].
     *
     * Matches on display name because the enumerator returns a Files-collection URI
     * (content://media/external/file/<id>) while the insertion returns a Downloads URI
     * (content://media/external/downloads/<id>); the _ID is the same but the base path
     * differs.
     *
     * Asserts:
     *  - A result whose display name equals [displayName] is present.
     *  - That result has mimeType == "text/plain".
     *  - That result has lastModified > 0 (DATE_MODIFIED is populated).
     */
    @Test
    fun test1_enumerateDocuments_findsSeedFile() {
        // No sleep needed: MediaStore.insert() + openOutputStream().use{} are synchronous,
        // and the _ID-resolution query below acts as a natural sync barrier.

        // Enumerate; find the seeded file by display_name.
        // We query DISPLAY_NAME separately via ContentResolver to find the expected ID.
        val insertUri = seededInsertUri ?: error("seededInsertUri was not set by @Before")

        // Resolve the _ID of the insertion so we can look it up in the enumerator results.
        val seededId = context.contentResolver.query(
            insertUri,
            arrayOf(MediaStore.Downloads._ID),
            null, null, null
        )?.use { c -> if (c.moveToFirst()) c.getLong(0) else null }
            ?: error("Could not resolve _ID for seeded URI $insertUri")

        // The enumerator produces URIs on the Files collection: content://media/external/file/<id>
        val expectedEnumeratorUri = android.content.ContentUris.withAppendedId(
            MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL),
            seededId
        ).toString()

        val targets = MediaStoreEnumerator().enumerateDocuments(context)

        val match = targets.find { it.uri == expectedEnumeratorUri }

        assertNotNull(
            "MediaStoreEnumerator should return the seeded file as URI $expectedEnumeratorUri; " +
                "got ${targets.size} results: ${targets.map { it.uri }.take(5)}",
            match
        )

        assertEquals(
            "Seeded file should have mimeType 'text/plain'",
            "text/plain",
            match!!.mimeType
        )

        assertTrue(
            "Seeded file should have lastModified > 0 (DATE_MODIFIED populated by MediaStore)",
            match.lastModified > 0L
        )
    }

    // -------------------------------------------------------------------------
    // Test 2: End-to-end - IndexWorker with only KEY_MODEL_PATH enumerates,
    // extracts, and persists. This is the L13 production-path proof.
    // -------------------------------------------------------------------------

    /**
     * L13 production path proof: drives [IndexWorker.doWork] with ONLY [IndexWorker.KEY_MODEL_PATH]
     * (no [IndexWorker.KEY_FILE_PATHS]), exactly as the real producers do.
     *
     * Flow:
     *  1. Worker sees no KEY_FILE_PATHS → falls into Branch B → calls MediaStoreEnumerator.
     *  2. Enumerator returns the seeded file (and possibly other docs on the device).
     *  3. Worker calls DocumentExtractor.extract(uri, context) for each target.
     *  4. FakeEmbedder produces deterministic 384-dim vectors (no real model needed).
     *  5. Chunks are persisted to IndexRepository.
     *
     * Asserts:
     *  - Worker returns Result.success().
     *  - IndexRepository.chunkCount() > 0 (at least the seeded file was processed).
     *  - A file_index row exists for the seeded file URI (enumerator's Files-collection form).
     */
    @Test
    fun test2_indexWorkerWithoutFilePaths_enumeratesAndPersistsSeededDocument() {
        val insertUri = seededInsertUri ?: error("seededInsertUri was not set by @Before")

        // Resolve the _ID so we can compute the Files-collection URI the enumerator will use.
        val seededId = context.contentResolver.query(
            insertUri,
            arrayOf(MediaStore.Downloads._ID),
            null, null, null
        )?.use { c -> if (c.moveToFirst()) c.getLong(0) else null }
            ?: error("Could not resolve _ID for seeded URI $insertUri")

        val expectedEnumeratorUri = android.content.ContentUris.withAppendedId(
            MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL),
            seededId
        ).toString()

        // Build IndexWorker with FakeEmbedder injected via Factory.
        // embedderFactory != null → model-file guard skipped (consistent with IndexWorkerTest).
        val worker = TestListenableWorkerBuilder<IndexWorker>(context)
            .setWorkerFactory(object : androidx.work.WorkerFactory() {
                override fun createWorker(
                    appContext: Context,
                    workerClassName: String,
                    workerParameters: WorkerParameters
                ): androidx.work.ListenableWorker? {
                    return if (workerClassName == IndexWorker::class.java.name) {
                        IndexWorker(appContext, workerParameters) { _ -> FakeEmbedder() }
                    } else null
                }
            })
            .setInputData(
                // Only KEY_MODEL_PATH - no KEY_FILE_PATHS.
                // This is the exact shape produced by IndexService / MediaStoreObserver / BootReceiver.
                workDataOf(IndexWorker.KEY_MODEL_PATH to "/fake/model.onnx")
            )
            .build()

        val result = runBlocking { worker.doWork() }

        assertTrue(
            "IndexWorker should return Result.success() when it self-enumerates via MediaStore",
            result is ListenableWorker.Result.Success
        )

        val db = IndexDatabase(context)
        val repository = IndexRepository(db)
        try {
            val chunkCount = repository.chunkCount()
            assertTrue(
                "At least 1 chunk should be persisted after MediaStore enumeration " +
                    "(seeded file contains: '$documentContent'). Got chunkCount=$chunkCount",
                chunkCount > 0
            )

            val indexedCount = repository.indexedFileCount()
            assertTrue(
                "file_index should have at least 1 row after end-to-end run. Got $indexedCount",
                indexedCount > 0
            )

            // Confirm the seeded URI (in enumerator's Files-collection form) appears in file_index.
            val directCursor = db.readableDatabase.query(
                IndexDatabase.TABLE_FILE_INDEX,
                arrayOf(IndexDatabase.COL_FI_FILE_PATH),
                "${IndexDatabase.COL_FI_FILE_PATH} = ?",
                arrayOf(expectedEnumeratorUri),
                null, null, null
            )
            val seededRowExists = directCursor.use { it.moveToFirst() }
            assertTrue(
                "file_index should contain a row for the seeded URI $expectedEnumeratorUri. " +
                    "indexedFileCount=$indexedCount, chunkCount=$chunkCount",
                seededRowExists
            )

            // Verifier-required proof: confirm at least one chunk was produced FROM the
            // seeded file specifically (not merely from some other document on the device).
            val chunksCursor = db.readableDatabase.query(
                IndexDatabase.TABLE_CHUNKS,
                arrayOf("COUNT(*)"),
                "${IndexDatabase.COL_FILE_PATH} = ?",
                arrayOf(expectedEnumeratorUri),
                null, null, null
            )
            val chunksForSeededFile = chunksCursor.use { c ->
                if (c.moveToFirst()) c.getInt(0) else 0
            }
            assertTrue(
                "At least 1 chunk should exist in '${IndexDatabase.TABLE_CHUNKS}' for the seeded " +
                    "URI $expectedEnumeratorUri. Got chunksForSeededFile=$chunksForSeededFile",
                chunksForSeededFile > 0
            )
        } finally {
            db.close()
        }
    }
}
