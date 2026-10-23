package com.narad.search.lib.index

// NEEDS-DEVICE: ./gradlew :lib:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.narad.search.lib.index.IndexRepositoryTest

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.narad.search.lib.index.chunking.Chunker
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository
import com.narad.search.lib.index.worker.Embedder
import com.narad.search.lib.index.worker.Indexer
import com.narad.search.lib.query.store.KeywordStore
import com.narad.search.lib.query.store.VectorStore
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID

/**
 * Instrumented persistence and orchestration tests for U4.
 *
 * Tests 1-3 + 5 from the test gate specification.
 * Tests 4 (IndexWorker model-absent) and 6 (BootReceiver re-enqueue)
 * are in separate test classes.
 *
 * Tests added for W7 (v4 schema, M7): favorites and index_events via the repository API.
 */
@RunWith(AndroidJUnit4::class)
class IndexRepositoryTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    // Fix item 9: use UUID for isolation -- avoids timestamp collisions in parallel runs
    private val dbName = "test_index_repo_${UUID.randomUUID()}.db"
    private lateinit var db: IndexDatabase
    private lateinit var repository: IndexRepository
    private val embedder: Embedder = FakeEmbedder()
    private val filesDir get() = context.filesDir

    @Before
    fun setUp() {
        db = IndexDatabase(context, dbName)
        repository = IndexRepository(db)
    }

    @After
    fun tearDown() {
        db.close()
        // Delete the test DB file from the app's databases directory
        context.deleteDatabase(dbName)
        // Clean up test text files
        filesDir.listFiles()?.filter { it.name.startsWith("test_doc_") }?.forEach { it.delete() }
    }

    // -------------------------------------------------------------------------
    // Test 1: Index 3 files -> exactly 3 files indexed
    // Fix item 9: changed "at least 3" to exact count assertion
    // -------------------------------------------------------------------------
    @Test
    fun test1_indexThreeFiles_threeFilesIndexed() {
        // Write 3 known text files to filesDir
        val doc1 = File(filesDir, "test_doc_contract.txt").also {
            it.writeText("This rental contract includes a penalty clause for late payment. The tenant must pay deposit.")
        }
        val doc2 = File(filesDir, "test_doc_weather.txt").also {
            it.writeText("Today's weather forecast shows sunny skies with mild temperatures across the region.")
        }
        val doc3 = File(filesDir, "test_doc_invoice.txt").also {
            it.writeText("Invoice #1234: Payment due in 30 days. Total amount owed for services rendered.")
        }

        val files = listOf(doc1, doc2, doc3)

        val indexer = buildIndexer(repository, embedder)
        indexer.indexFiles(files.map { it.absolutePath })

        // Each short doc fits in 1 chunk with the default chunker settings (max 550 chars).
        // So after indexing 3 single-chunk docs, chunkCount = 3.
        val chunkCount = repository.chunkCount()
        assertEquals("Should have exactly 3 chunks for 3 single-chunk documents", 3, chunkCount)

        // Assert file_index has 3 rows
        val fileCount = repository.indexedFileCount()
        assertEquals("file_index should have 3 rows", 3, fileCount)

        // Rehydrate in-memory stores from the persisted SQLite rows (this is the
        // normal startup flow: index -> persist -> rehydrate on next query).
        val vectorStore = VectorStore()
        val keywordStore = KeywordStore()
        repository.rehydrate(vectorStore, keywordStore)

        // Fix item 9: exact count (3 chunks)
        assertEquals("VectorStore should have exactly 3 entries after rehydration", 3, vectorStore.size)

        // Assert ranking via BM25 KeywordStore
        val keywordResults = keywordStore.search("penalty clause", topK = 10)
        assertFalse("Keyword search should return results", keywordResults.isEmpty())
        val topKeywordResult = keywordResults.first()
        assertTrue(
            "Contract file should rank #1 for 'penalty clause' keyword query, got: ${topKeywordResult.filePath}",
            topKeywordResult.filePath.contains("contract")
        )
    }

    // -------------------------------------------------------------------------
    // Test 2: Dedup / idempotency
    // Fix item 9: replaced tautological assertion with a real rehydrate-into-fresh-store check.
    // The old test created vectorStore2/keywordStore2, passed them to buildIndexer (which
    // suppressed them via @Suppress UNUSED_PARAMETER and did NOT pass to Indexer), so
    // vectorStore2.size could never change -- the assertion was always true regardless of
    // whether dedup actually worked.
    // New approach: after the second run, rehydrate into a FRESH store and assert its
    // size equals the post-first-run rehydrated size.
    // -------------------------------------------------------------------------
    @Test
    fun test2_dedupIdempotency_secondRunSkipsUnchangedFiles() {
        val doc = File(filesDir, "test_doc_dedup.txt").also {
            it.writeText("Idempotency test: this document about penalty clause and deposit.")
        }
        val files = listOf(doc.absolutePath)

        // First run
        val indexer1 = buildIndexer(repository, embedder)
        indexer1.indexFiles(files)
        val countAfterFirst = repository.chunkCount()
        assertTrue("Should have at least 1 chunk after first run", countAfterFirst >= 1)

        // Rehydrate into fresh stores to capture post-first-run in-memory state
        val vsAfterFirst = VectorStore()
        val ksAfterFirst = KeywordStore()
        repository.rehydrate(vsAfterFirst, ksAfterFirst)
        val sizeAfterFirstRun = vsAfterFirst.size

        // Second run: same file unchanged -> should be skipped by fileNeedsIndex check
        val indexer2 = buildIndexer(repository, embedder)
        indexer2.indexFiles(files)
        val countAfterSecond = repository.chunkCount()
        assertEquals(
            "Chunk count should not grow on second run of unchanged file",
            countAfterFirst, countAfterSecond
        )

        // Rehydrate into a FRESH store after the second run -- real dedup check (fix item 9)
        val vsAfterSecond = VectorStore()
        val ksAfterSecond = KeywordStore()
        repository.rehydrate(vsAfterSecond, ksAfterSecond)
        assertEquals(
            "Rehydrated VectorStore after second run should equal first-run size (no duplicates)",
            sizeAfterFirstRun, vsAfterSecond.size
        )
    }

    // -------------------------------------------------------------------------
    // Test 3: Persistence round-trip (kill/restart simulation)
    // -------------------------------------------------------------------------
    @Test
    fun test3_persistenceRoundTrip_rehydrateRestoresStores() {
        val doc = File(filesDir, "test_doc_persist.txt").also {
            it.writeText("Contract with penalty clause for non-payment of deposit amount.")
        }

        // First instance: index + rehydrate to populate in-memory stores
        val vs1 = VectorStore()
        val ks1 = KeywordStore()
        val indexer1 = buildIndexer(repository, embedder)
        indexer1.indexFiles(listOf(doc.absolutePath))
        // Rehydrate vs1/ks1 from the SQLite rows (normal startup flow)
        repository.rehydrate(vs1, ks1)

        val sizeAfterIndex = vs1.size
        val kwSizeAfterIndex = ks1.size
        assertTrue("Should have indexed at least one chunk", sizeAfterIndex > 0)

        // Close the first DB instance to simulate app kill
        db.close()

        // "Restart": open a fresh DB + repository on the same named DB
        val db2 = IndexDatabase(context, dbName)
        val repo2 = IndexRepository(db2)
        val vs2 = VectorStore()
        val ks2 = KeywordStore()

        repo2.rehydrate(vs2, ks2)
        db2.close()

        // Assert stores are repopulated
        assertEquals("VectorStore should be fully rehydrated", sizeAfterIndex, vs2.size)
        assertEquals("KeywordStore should be fully rehydrated", kwSizeAfterIndex, ks2.size)

        // Assert search still finds the content after rehydration using keyword search
        val kwResults = ks2.search("penalty clause", topK = 5)
        assertFalse("Keyword search should return results after rehydration", kwResults.isEmpty())
        assertTrue(
            "Rehydrated KeywordStore should rank 'persist' document #1, got: ${kwResults.firstOrNull()?.filePath}",
            kwResults.first().filePath.contains("persist")
        )
    }

    // -------------------------------------------------------------------------
    // Test 5: Checkpoint every 50 files
    // -------------------------------------------------------------------------
    @Test
    fun test5_checkpointEvery50_progressCallbackFiredAt50() {
        val testFiles = (1..62).map { i ->
            File(filesDir, "test_doc_ckpt_$i.txt").also { f ->
                f.writeText("Document $i about topics such as contracts, invoices, weather or reports.")
            }
        }

        val checkpointValues = mutableListOf<Int>()

        val indexer = Indexer(
            embedder = embedder,
            extractText = { path ->
                try { File(path).readText() } catch (e: Exception) { null }
            },
            chunkText = { text -> Chunker().chunk(text) },
            repository = repository,
            checkpointEvery = 50,
            onProgress = { processed, _ -> checkpointValues.add(processed) }
        )

        indexer.indexFiles(testFiles.map { it.absolutePath })

        // 62 files: checkpoint fires at 50, then at 62 (final via "last batch not on boundary")
        assertTrue(
            "Progress callback should have been called at least twice (50 and final)",
            checkpointValues.size >= 2
        )
        // The 50-file checkpoint must appear
        assertTrue(
            "Progress callback should include a checkpoint at exactly 50 files, got: $checkpointValues",
            checkpointValues.any { it == 50 }
        )

        // Cleanup
        testFiles.forEach { it.delete() }
    }

    // -------------------------------------------------------------------------
    // Test 7: Orphan chunk cleanup on re-index with fewer chunks
    // Fix item 6: verify deleteChunksForFile removes stale rows on re-index
    // -------------------------------------------------------------------------
    @Test
    fun test7_orphanChunkCleanup_reindexWithFewerChunksLeavesNoOrphans() {
        val doc = File(filesDir, "test_doc_orphan.txt")

        // First indexing run: write content that produces multiple chunks.
        // We override chunkText to force 5 chunks regardless of content.
        val indexer1 = Indexer(
            embedder = embedder,
            extractText = { path -> try { File(path).readText() } catch (e: Exception) { null } },
            chunkText = { _ ->
                // Force exactly 5 chunks
                listOf("chunk one", "chunk two", "chunk three", "chunk four", "chunk five")
            },
            repository = repository
        )
        doc.writeText("original content that produces 5 chunks")
        indexer1.indexFiles(listOf(doc.absolutePath))

        val countAfter5 = repository.chunkCount()
        assertEquals("Should have exactly 5 chunks after first run", 5, countAfter5)

        // Simulate file edit: update last-modified so fileNeedsIndex returns true
        // (touch the file timestamp)
        doc.writeText("updated content now produces only 2 chunks")
        // Small sleep to ensure lastModified changes
        Thread.sleep(100)
        doc.setLastModified(System.currentTimeMillis())

        // Second indexing run: same file now yields only 2 chunks
        val indexer2 = Indexer(
            embedder = embedder,
            extractText = { path -> try { File(path).readText() } catch (e: Exception) { null } },
            chunkText = { _ ->
                // Force exactly 2 chunks (fewer than before)
                listOf("new chunk A", "new chunk B")
            },
            repository = repository
        )
        indexer2.indexFiles(listOf(doc.absolutePath))

        val countAfterReindex = repository.chunkCount()
        assertEquals(
            "After re-indexing with 2 chunks, should have exactly 2 chunks (no orphans from old run of 5)",
            2,
            countAfterReindex
        )

        // Rehydrate into a fresh store and verify no orphan entries
        val vs = VectorStore()
        val ks = KeywordStore()
        repository.rehydrate(vs, ks)
        assertEquals(
            "Rehydrated VectorStore should have exactly 2 entries (no orphan chunks)",
            2,
            vs.size
        )

        doc.delete()
    }

    // -------------------------------------------------------------------------
    // chunkContent: hit and miss (fix 3)
    // -------------------------------------------------------------------------

    /**
     * chunkContent returns the correct text when the chunk exists (hit).
     * Uses upsertChunk directly so no embedder is needed.
     */
    @Test
    fun chunkContent_hit_returnsCorrectText() {
        val filePath = "/test/chunkContent_hit.txt"
        val chunkOffset = 3
        val content = "This is the content of chunk 3."
        val id = "$filePath#$chunkOffset"
        val fakeVector = FloatArray(256) { 0f }.also { it[0] = 1f }
        repository.upsertChunk(
            id = id,
            filePath = filePath,
            chunkOffset = chunkOffset,
            modelId = "test-model",
            dim = 256,
            vector = fakeVector,
            content = content
        )
        val result = repository.chunkContent(filePath, chunkOffset)
        assertEquals(
            "chunkContent must return the inserted text for a known (filePath, chunkOffset)",
            content,
            result
        )
    }

    /**
     * chunkContent returns null when no matching chunk exists (miss).
     */
    @Test
    fun chunkContent_miss_returnsNull() {
        val result = repository.chunkContent("/does/not/exist.txt", 999)
        assertNull(
            "chunkContent must return null for a (filePath, chunkOffset) that was never inserted",
            result
        )
    }

    // -------------------------------------------------------------------------
    // W7 / M7: Favorites (KTD16)
    // -------------------------------------------------------------------------

    @Test
    fun favorites_addTwice_keepsOneRow_withOriginalAddedAt() {
        // CONFLICT_IGNORE: second add is a no-op; the original row (and its timestamp) stays.
        val firstUri = "content://media/external/images/media/42"
        val secondUri = "content://media/external/images/media/43"
        repository.addFavorite(firstUri)
        Thread.sleep(50)
        repository.addFavorite(secondUri)
        assertEquals(listOf(secondUri, firstUri), repository.listFavorites())

        // Second add must be silently ignored.
        Thread.sleep(50)
        repository.addFavorite(firstUri)

        assertEquals("Adding the same URI twice must keep exactly two rows", 2, repository.favoriteCount())
        assertEquals(listOf(secondUri, firstUri), repository.listFavorites())
    }

    @Test
    fun favorites_remove_deletesRow() {
        val uri = "content://media/external/images/media/99"
        repository.addFavorite(uri)
        assertEquals(1, repository.favoriteCount())
        repository.removeFavorite(uri)
        assertEquals("removeFavorite must delete the row", 0, repository.favoriteCount())
    }

    @Test
    fun favorites_removeNonExistent_isNoOp() {
        // Should not throw
        repository.removeFavorite("content://media/external/images/media/0")
        assertEquals(0, repository.favoriteCount())
    }

    @Test
    fun favorites_listFavorites_newestFirst() {
        val uri1 = "content://media/external/images/media/1"
        val uri2 = "content://media/external/images/media/2"
        repository.addFavorite(uri1)
        Thread.sleep(50)  // ensure distinct timestamps; 10ms is too short on coarse-clock devices
        repository.addFavorite(uri2)
        val list = repository.listFavorites()
        assertEquals("listFavorites must return all 2 entries", 2, list.size)
        assertEquals("Most recently added URI must be first", uri2, list[0])
        assertEquals("Earlier URI must be second", uri1, list[1])
    }

    @Test
    fun favorites_clearAll_keepsFavorites() {
        // Favorites are user data and must survive a clearAll (reindex).
        repository.addFavorite("content://media/external/images/media/7")
        repository.addFavorite("content://media/external/images/media/8")
        assertEquals(2, repository.favoriteCount())
        repository.clearAll()
        assertEquals("clearAll must NOT remove favorites (they are user data)", 2, repository.favoriteCount())
    }

    // -------------------------------------------------------------------------
    // W7 / M7: Index events (KTD18)
    // -------------------------------------------------------------------------

    @Test
    fun events_appendEvent_writesRow() {
        repository.appendEvent(
            runId = "run-001",
            stage = "Reading",
            level = "INFO",
            fileName = "contract.pdf",
            errorClass = null,
            message = "Indexed successfully"
        )
        val events = repository.listEvents()
        assertEquals("appendEvent must write exactly one row", 1, events.size)
        val e = events[0]
        assertEquals("run-001", e.runId)
        assertEquals("Reading", e.stage)
        assertEquals("INFO", e.level)
        assertEquals("contract.pdf", e.fileName)
        assertNull(e.errorClass)
        assertEquals("Indexed successfully", e.message)
    }

    @Test
    fun events_appendEvent_nullFieldsAreNullInResult() {
        repository.appendEvent(runId = "r", stage = "Done", level = "INFO")
        val e = repository.listEvents().first()
        assertNull("fileName should be null when not provided", e.fileName)
        assertNull("errorClass should be null when not provided", e.errorClass)
        assertNull("message should be null when not provided", e.message)
    }

    @Test
    fun events_appendEvent_ktd18_fileNameStrippedToBaseName() {
        // KTD18: appendEvent must store only the bare file name, not the full path.
        repository.appendEvent(
            runId = "r",
            stage = "Reading",
            level = "WARN",
            fileName = "/sdcard/Documents/reports/q3.pdf"
        )
        val e = repository.listEvents().first()
        assertEquals("appendEvent must store only the bare file name (KTD18)", "q3.pdf", e.fileName)
    }

    @Test
    fun events_appendEvent_ktd18_messageCappedAt200Chars() {
        val longMsg = "x".repeat(300)
        repository.appendEvent(runId = "r", stage = "Done", level = "INFO", message = longMsg)
        val e = repository.listEvents().first()
        assertEquals("appendEvent must cap message at 200 chars (KTD18)", 200, e.message?.length)
    }

    @Test
    fun events_listEvents_newestFirst() {
        repository.appendEvent(runId = "r1", stage = "Finding", level = "INFO")
        Thread.sleep(5)
        repository.appendEvent(runId = "r2", stage = "Done", level = "INFO")
        val list = repository.listEvents()
        assertEquals(2, list.size)
        assertEquals("listEvents must return newest event first", "r2", list[0].runId)
        assertEquals("r1", list[1].runId)
    }

    @Test
    fun events_pruneEvents_byAge_removesOldRows() {
        // Insert a row with a very old timestamp by manipulating the DB directly.
        val w = db.writableDatabase
        val cv = android.content.ContentValues().apply {
            // timestamp in the distant past (more than 7 days ago)
            put(IndexDatabase.COL_IE_TIMESTAMP_MS, System.currentTimeMillis() - 8L * 24 * 3600 * 1000)
            put(IndexDatabase.COL_IE_RUN_ID, "old-run")
            put(IndexDatabase.COL_IE_STAGE, "Done")
            put(IndexDatabase.COL_IE_LEVEL, "INFO")
        }
        w.insert(IndexDatabase.TABLE_INDEX_EVENTS, null, cv)

        // Insert a recent row.
        repository.appendEvent(runId = "new-run", stage = "Done", level = "INFO")

        assertEquals("Should have 2 rows before prune", 2, repository.listEvents().size)

        repository.pruneEvents()

        val remaining = repository.listEvents()
        assertEquals("pruneEvents must remove rows older than 7 days", 1, remaining.size)
        assertEquals("new-run", remaining[0].runId)
    }

    @Test
    fun events_pruneEvents_byCount_keepsNewest500() {
        // Insert 510 events, all recent.
        for (i in 1..510) {
            repository.appendEvent(runId = "r$i", stage = "Reading", level = "INFO")
        }
        assertEquals(510, repository.listEvents(600).size)

        repository.pruneEvents()

        val remaining = repository.listEvents(600)
        assertEquals("pruneEvents must keep at most 500 rows", 500, remaining.size)
        // Newest 500: r510 down to r11. r1..r10 should be gone.
        val runIds = remaining.map { it.runId }.toSet()
        assertTrue("r510 (newest) must be in the remaining 500", runIds.contains("r510"))
        assertFalse("r1 (oldest) must have been pruned", runIds.contains("r1"))
    }

    @Test
    fun events_clearAll_doesNotClearEvents() {
        // clearAll is for reindexing; events are an audit log and must not be cleared.
        repository.appendEvent(runId = "r", stage = "Done", level = "INFO")
        repository.clearAll()
        assertEquals("clearAll must NOT remove index_events rows", 1, repository.listEvents().size)
    }

    // -------------------------------------------------------------------------
    // U8 / R13: prune. SQL is only exercisable on a device (no Robolectric on the host).
    // The access and finished-pass gates are unit-tested in PruneGateTest.
    // -------------------------------------------------------------------------

    private val primary = setOf("external_primary")
    private fun docUri(n: Int) = "content://media/external/file/$n"
    private fun photoUri(n: Int) = "content://media/external/file/${1000 + n}"

    private fun seedDoc(uri: String) {
        repository.upsertChunksForFile(
            uri, listOf(Triple(0, FloatArray(256) { 0.1f }, "text of $uri")),
            IndexDatabase.MODEL_ID, IndexDatabase.DIM_VALUE
        )
        repository.markFileIndexed(uri, null, 1L, 1)
    }

    private fun seedPhoto(uri: String, favorite: Boolean = false) {
        repository.upsertImageVector(uri, uri, FloatArray(256) { 0.1f })
        repository.markFileIndexed(
            uri, null, 1L, 1, IndexDatabase.IMAGE_MODEL_ID, IndexDatabase.IMAGE_DIM_VALUE
        )
        if (favorite) repository.addFavorite(uri)
    }

    @Test
    fun prune_documents_deletedFileLeavesIndex_otherKept() {
        seedDoc(docUri(1)); seedDoc(docUri(2))

        val removed = repository.pruneDocuments(setOf(docUri(1)), primary)

        assertEquals(1, removed)
        assertEquals("only A is left in file_index", 1, repository.indexedFileCount())
        assertEquals("only A's chunk is left", 1, repository.chunkCount())
        assertNotNull(repository.chunkContent(docUri(1), 0))
        assertNull("B's chunk is gone", repository.chunkContent(docUri(2), 0))
        assertTrue("B no longer needs a file_index row, so it would be re-indexed if it came back",
            repository.fileNeedsIndex(docUri(2), 1L))
    }

    @Test
    fun prune_photos_deletedPhotoLeavesVectorsFileIndexAndFavorites() {
        seedPhoto(photoUri(1), favorite = true); seedPhoto(photoUri(2), favorite = true)

        val removed = repository.prunePhotos(setOf(photoUri(1)), primary)

        assertEquals(1, removed)
        assertEquals(1, repository.imageVectorCount())
        assertEquals(1, repository.indexedFileCount())
        assertEquals(listOf(photoUri(1)), repository.listFavorites())
    }

    @Test
    fun prune_documentsAndPhotosAreSeparate() {
        seedDoc(docUri(1)); seedPhoto(photoUri(1))

        // The photo is not in the document enumeration, but it is not a document row.
        assertEquals(0, repository.pruneDocuments(setOf(docUri(1)), primary))
        assertEquals(1, repository.imageVectorCount())
        assertEquals(2, repository.indexedFileCount())

        // And the document is not in the photo enumeration, but it is not a photo row.
        assertEquals(0, repository.prunePhotos(setOf(photoUri(1)), primary))
        assertEquals(1, repository.chunkCount())
        assertEquals(2, repository.indexedFileCount())
    }

    @Test
    fun prune_emptyEnumeration_deletesNothing() {
        seedDoc(docUri(1)); seedPhoto(photoUri(1), favorite = true)

        assertEquals(0, repository.pruneDocuments(emptySet(), primary))
        assertEquals(0, repository.prunePhotos(emptySet(), primary))

        assertEquals(2, repository.indexedFileCount())
        assertEquals(1, repository.chunkCount())
        assertEquals(1, repository.imageVectorCount())
        assertEquals(1, repository.favoriteCount())
    }

    @Test
    fun prune_noVolumesListed_deletesNothing() {
        seedDoc(docUri(1)); seedDoc(docUri(2))
        assertEquals(0, repository.pruneDocuments(setOf(docUri(1)), emptySet()))
        assertEquals(2, repository.indexedFileCount())
    }

    @Test
    fun prune_recordedVolumeAbsent_skipsAndDeletesNothing_thenResumesWhenBack() {
        seedDoc(docUri(1)); seedDoc(docUri(2)); seedPhoto(photoUri(1)); seedPhoto(photoUri(2))
        val withCard = primary + "1A2B-3C4D"

        // Run with the card mounted: everything is seen, nothing to prune, volumes recorded.
        assertEquals(0, repository.pruneDocuments(setOf(docUri(1), docUri(2)), withCard))

        // Card unmounted: its files vanish from the enumeration. Both prunes must be skipped.
        assertEquals(0, repository.pruneDocuments(setOf(docUri(1)), primary))
        assertEquals(0, repository.prunePhotos(setOf(photoUri(1)), primary))
        assertEquals(4, repository.indexedFileCount())
        assertEquals(2, repository.chunkCount())
        assertEquals(2, repository.imageVectorCount())

        // Card back, and B really is gone: now it is pruned.
        assertEquals(1, repository.pruneDocuments(setOf(docUri(1)), withCard))
        assertNull(repository.chunkContent(docUri(2), 0))
    }

    @Test
    fun prune_volumeRecordedAtListing_blocksPruneAfterStoppedRunAndCardRemoval() {
        seedDoc(docUri(1)); seedDoc(docUri(2))
        val withCard = primary + "1A2B-3C4D"
        // An earlier finished run records {primary}.
        assertEquals(0, repository.pruneDocuments(setOf(docUri(1), docUri(2)), primary))
        // A later run sees the card and records it at listing time, then is stopped: no prune.
        repository.recordVolumes(withCard)
        // The card is removed. The next finished run must not delete the card's rows.
        assertEquals(0, repository.pruneDocuments(setOf(docUri(1)), primary))
        assertNotNull("d2 is kept", repository.chunkContent(docUri(2), 0))
    }

    @Test
    fun recordVolumes_unionsAndIgnoresEmpty() {
        repository.recordVolumes(primary)
        repository.recordVolumes(setOf("1A2B-3C4D"))
        repository.recordVolumes(emptySet())
        assertEquals(setOf("external_primary", "1A2B-3C4D"), repository.getConfig("prune_volumes")!!.split('\n').toSet())
    }

    @Test
    fun prune_newVolumeAppearing_isRecordedAndDoesNotBlock() {
        seedDoc(docUri(1)); seedDoc(docUri(2))
        repository.pruneDocuments(setOf(docUri(1), docUri(2)), primary)

        assertEquals(1, repository.pruneDocuments(setOf(docUri(1)), primary + "1A2B-3C4D"))
    }

    @Test
    fun prune_neverTouchesNonMediaKeys() {
        val plain = File(filesDir, "test_doc_plain.txt").also { it.writeText("plain path file text") }
        buildIndexer(repository, embedder).indexFiles(listOf(plain.absolutePath))
        seedDoc(docUri(1))

        repository.pruneDocuments(setOf(docUri(1)), primary)

        assertFalse("explicit-path row must stay", repository.fileNeedsIndex(plain.absolutePath, plain.lastModified()))
    }

    @Test
    fun prune_clearAllResetsVolumeRecord() {
        seedDoc(docUri(1))
        repository.pruneDocuments(setOf(docUri(1)), primary + "1A2B-3C4D")
        assertNotNull(repository.getConfig("prune_volumes"))

        repository.clearAll()

        assertNull("a reindex starts with no recorded volumes", repository.getConfig("prune_volumes"))
    }

    // -------------------------------------------------------------------------
    // Private helper: build a standard Indexer using the test embedder
    // -------------------------------------------------------------------------
    private fun buildIndexer(
        repo: IndexRepository,
        emb: Embedder,
        checkpointEvery: Int = 50
    ): Indexer {
        return Indexer(
            embedder = emb,
            extractText = { path ->
                try { File(path).readText() } catch (e: Exception) { null }
            },
            chunkText = { text -> Chunker().chunk(text) },
            repository = repo,
            checkpointEvery = checkpointEvery,
            onProgress = { _, _ -> /* no-op */ }
        )
    }
}
