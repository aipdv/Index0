package me.zhanghai.android.files.index

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import android.os.SystemClock
import android.util.Log
import com.narad.search.lib.device.DeviceTierDetector
import com.narad.search.lib.index.service.IndexService
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository
import com.narad.search.lib.query.engine.ImageResult
import com.narad.search.lib.query.engine.QueryEngine
import com.narad.search.lib.query.model.DjlTokenizer
import com.narad.search.lib.query.model.EmbeddingModel
import com.narad.search.lib.query.store.KeywordStore
import com.narad.search.lib.query.store.VectorStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

// ---------------------------------------------------------------------------
// Readiness
// ---------------------------------------------------------------------------

sealed class SearchReadiness {
    object NotProvisioned : SearchReadiness()
    object Loading : SearchReadiness()
    object Ready : SearchReadiness()
    data class Error(val message: String) : SearchReadiness()
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/**
 * Enriched document result for display in the UI.
 *
 * All fields are resolved by [RealSearchController] on Dispatchers.IO after the
 * underlying [QueryEngine.search] call; no enrichment is done for the whole index.
 */
data class DocRow(
    val filePath: String,
    val score: Float,
    val chunkOffset: Int,
    val displayName: String,
    val folderName: String,
    val dateModified: Long,
    val fileKind: FileKind,
    val snippet: String,
    /** Strong match for the query (see [strongDocs]). Weak hits go under "Less related". */
    val strong: Boolean = true
)

/**
 * Ranked image result with no snippet because images have no text chunk.
 */
data class PhotoRow(
    val filePath: String,
    val score: Float,
    /** Strong match for the query (see [strongMask]). Weak hits go under "Less related". */
    val strong: Boolean = true
)

/**
 * Combined search output: document results + photo results as separate lists.
 *
 * Photos list is empty (not an error) when the vision model is absent.
 */
data class SearchOutput(
    val docs: List<DocRow>,
    val photos: List<PhotoRow>
) {
    companion object {
        val EMPTY = SearchOutput(emptyList(), emptyList())
    }
}

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

interface SearchController {
    val readiness: SearchReadiness
    suspend fun search(query: String, topK: Int = 20): SearchOutput
    suspend fun invalidate()
    /** Load the model and run one throwaway embed ahead of the first search (search field focus). Default no-op. */
    suspend fun warmUp() {}
    /** Recent indexed documents, newest first. Default returns empty (for fakes/tests). */
    suspend fun recentDocuments(limit: Int): List<DocRow> = emptyList()
    /** Recent indexed photos, newest-inserted first. Default returns empty (for fakes/tests). */
    suspend fun recentPhotos(limit: Int): List<PhotoRow> = emptyList()
    /** Total number of files in the index. Default returns 0. */
    suspend fun indexedFileCount(): Int = 0
}

// ---------------------------------------------------------------------------
// Real implementation
// ---------------------------------------------------------------------------

/**
 * Production [SearchController].
 *
 * Lazily builds the EmbeddingGemma 2 query stack on first use (off-main, IO).
 *
 * - If the text model files are absent → [SearchReadiness.NotProvisioned].
 * - If the vision encoder is absent → photos list is empty (not an error).
 *
 * @param context           Application context.
 * @param modelDirProvider  Test override: flat dir containing `model_quantized.onnx` +
 *                          `model_quantized.onnx_data` + `tokenizer.json`.
 *                          When null, uses [IndexService.getDefaultModelPath].
 */
class RealSearchController(
    private val context: Context,
    private val modelDirProvider: (() -> String)? = null
) : SearchController {

    companion object {
        private const val TAG = "RealSearchController"
        private const val SNIPPET_MAX_LEN = 160
        // ponytail: 30s floor on freshness rebuilds; per-file incremental updates if throughput matters
        private const val REBUILD_RATE_LIMIT_MS = 30_000L
    }

    @Volatile
    override var readiness: SearchReadiness = SearchReadiness.Loading
        private set

    private val initMutex = Mutex()

    // One handle for the controller's life (SQLiteOpenHelper is thread-safe). It used to open
    // and close one per call at six sites. The process owns it, so it is never closed.
    private val db: IndexDatabase by lazy { IndexDatabase(context) }

    @Volatile
    private var engine: QueryEngine? = null

    @Volatile
    private var embeddingModel: EmbeddingModel? = null

    @Volatile
    private var fastSearchOnly: Boolean = false

    // The throwaway embed ran for the current model. Reset when the model is rebuilt.
    @Volatile
    private var warmedUp: Boolean = false

    // Freshness stamps are set after a successful store rebuild and compared in checkFreshness().
    @Volatile private var lastSeenTimestamp: Long? = null
    @Volatile private var lastSeenChunkCount: Int = -1
    // Rate-limit: only in checkFreshness path; init 0 so the first freshness rebuild always runs
    @Volatile private var lastFreshnessRebuildMs: Long = 0L

    // ---- Public API --------------------------------------------------------

    override suspend fun search(query: String, topK: Int): SearchOutput {
        if (query.isBlank()) return SearchOutput.EMPTY

        ensureInitialised()

        // Cheap freshness check: rebuild stores (not ONNX model) if index grew
        if (readiness == SearchReadiness.Ready) checkFreshness()

        return when (readiness) {
            SearchReadiness.Ready -> withContext(Dispatchers.IO) { runSearch(query, topK) }
            else                  -> SearchOutput.EMPTY
        }
    }

    // Best effort: a failure here (for example an OrtException) must never reach the caller,
    // which runs in a LaunchedEffect on focus. The first real search reports any real problem.
    override suspend fun warmUp() {
        try {
            ensureInitialised()
            if (readiness != SearchReadiness.Ready || warmedUp) return
            withContext(Dispatchers.IO) { engine?.warmUp() }
            warmedUp = true
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "Warm-up failed, ignoring", e)
        } catch (e: LinkageError) {
            Log.w(TAG, "Warm-up failed, ignoring", e)
        }
    }

    override suspend fun invalidate() {
        withContext(Dispatchers.IO) {
            initMutex.withLock {
                warmedUp = false
                embeddingModel?.close()
                embeddingModel = null
                engine = null
                readiness = SearchReadiness.Loading
                lastSeenTimestamp = null
                lastSeenChunkCount = -1
                lastFreshnessRebuildMs = 0L
            }
        }
        ensureInitialised()
    }

    // ---- Init --------------------------------------------------------------

    private suspend fun ensureInitialised() {
        if (engine != null) return
        withContext(Dispatchers.IO) {
            initMutex.withLock {
                if (engine != null) return@withLock
                buildEngine()
            }
        }
    }

    private fun buildEngine() {
        fastSearchOnly = modelDirProvider == null &&
            DeviceTierDetector.detect(context).fastSearchOnly
        if (fastSearchOnly) {
            if (rebuildStores()) {
                readiness = SearchReadiness.Ready
                Log.i(TAG, "QueryEngine ready in FTS-only mode")
            } else {
                readiness = SearchReadiness.Error("Index load failed")
            }
            return
        }

        val (onnxPath, tokPath) = resolvePaths()

        if (!File(onnxPath).exists() || !File(tokPath).exists()) {
            Log.w(TAG, "EG2 text model absent. NotProvisioned. onnx=$onnxPath exists=${File(onnxPath).exists()}")
            readiness = SearchReadiness.NotProvisioned
            return
        }

        val model: EmbeddingModel = try {
            // Fix: reuse existing model to avoid creating a 300 MB native object on each retry
            embeddingModel ?: EmbeddingModel(onnxPath, DjlTokenizer(tokPath))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to build EmbeddingModel", e)
            readiness = SearchReadiness.NotProvisioned
            return
        }

        embeddingModel = model
        if (rebuildStores()) {
            readiness = SearchReadiness.Ready
            Log.i(TAG, "QueryEngine ready")
        } else {
            readiness = SearchReadiness.Error("Index load failed")
        }
    }

    /**
     * Rebuild in-memory stores from the current DB state, reusing [embeddingModel].
     * Called both on first init and on freshness-triggered rebuilds.
     *
     * Fix (stamp-before-reload): reads lastIndexedAt/chunkCount BEFORE rehydrating so a
     * write landing during reload appears as a fresh stamp on the next checkFreshness call.
     *
     * Does NOT touch [readiness]. Callers decide what state a failure means.
     * Returns true on success, false on failure (old engine and stores are preserved).
     */
    private fun rebuildStores(): Boolean {
        val model = embeddingModel
        if (!fastSearchOnly && model == null) return false
        val textVec  = VectorStore()
        val kwStore  = KeywordStore()
        val imageVec = VectorStore()

        var ts: Long? = null
        var cc: Int = 0
        try {
            val repo = IndexRepository(db)
            // Read stamp BEFORE rehydrating: a write landing mid-reload will be seen next check
            ts = repo.lastIndexedAt()
            cc = repo.chunkCount()

            if (fastSearchOnly) {
                repo.rehydrateKeywords(kwStore)
            } else {
                repo.rehydrate(textVec, kwStore)
            }
            Log.i(TAG, "Text index rehydrated: ${textVec.size} vectors, ${kwStore.size} keywords")

            // Vision model is optional. An empty image store is not an error.
            if (!fastSearchOnly && modelDirProvider == null) {
                val visionPath = IndexService.getVisionEncoderPath(context)
                if (File(visionPath).exists()) {
                    try {
                        repo.rehydrateImages(imageVec)
                        Log.i(TAG, "Image index rehydrated: ${imageVec.size} vectors")
                    } catch (e: Exception) {
                        Log.w(TAG, "Image rehydration failed. Image results empty", e)
                    }
                } else {
                    Log.i(TAG, "Vision encoder absent. Image results will be empty")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Index load failed", e)
            return false
        }

        engine = QueryEngine(
            embeddingModel = model,
            vectorStore = textVec,
            keywordStore = kwStore,
            imageVectorStore = imageVec,
            alpha = 0.6f,
            fastSearchOnly = fastSearchOnly
        )
        lastSeenTimestamp = ts
        lastSeenChunkCount = cc
        return true
    }

    /**
     * Cheap two-query freshness check: if [lastIndexedAt] or [chunkCount] changed,
     * acquires [initMutex] and calls [rebuildStores] (reuses the loaded ONNX model).
     * Thread-safe: double-checked inside the lock.
     *
     * Fix (rate-limit): at most one rebuild per [REBUILD_RATE_LIMIT_MS] ms.
     * Uses [SystemClock.elapsedRealtime] (monotonic) so wall-clock changes (NTP, DST)
     * do not accidentally suppress the rate limit.
     *
     * Fix (keep-on-fail): if rebuildStores fails, old engine/stores are preserved and
     * readiness stays Ready. The rate-limit timestamp IS still updated on failure so
     * the next retry waits the full window, preventing a tight retry loop on persistent
     * failures (e.g. corrupt DB). Retried on the next stamp change after the window expires.
     */
    private suspend fun checkFreshness() = withContext(Dispatchers.IO) {
        val (newTs, newCc) = quickStamp()
        if (newTs == lastSeenTimestamp && newCc == lastSeenChunkCount) return@withContext
        initMutex.withLock {
            val now = SystemClock.elapsedRealtime()
            if (now - lastFreshnessRebuildMs < REBUILD_RATE_LIMIT_MS) return@withLock
            val (ts2, cc2) = quickStamp()
            if (ts2 == lastSeenTimestamp && cc2 == lastSeenChunkCount) return@withLock
            Log.i(TAG, "Index stale (ts=$ts2 cc=$cc2 prevTs=$lastSeenTimestamp prevCc=$lastSeenChunkCount). Rebuilding stores")
            // Update rate-limit stamp regardless of outcome (prevents tight retry loop on failure).
            lastFreshnessRebuildMs = SystemClock.elapsedRealtime()
            if (rebuildStores()) {
                readiness = SearchReadiness.Ready
            } else {
                // Keep old engine; stay Ready; retry after rate-limit window when stamp changes again
                Log.w(TAG, "Store rebuild failed during freshness check. Keeping old engine, will retry")
            }
        }
    }

    private fun quickStamp(): Pair<Long?, Int> {
        val repo = IndexRepository(db)
        return repo.lastIndexedAt() to repo.chunkCount()
    }

    private fun resolvePaths(): Pair<String, String> {
        return if (modelDirProvider != null) {
            // Test override: flat dir with EG2 files at root level
            val dir = modelDirProvider.invoke()
            "$dir/model_quantized.onnx" to "$dir/tokenizer.json"
        } else {
            val onnxPath = IndexService.getDefaultModelPath(context)
            val dir = File(onnxPath).parent ?: context.filesDir.absolutePath
            onnxPath to "$dir/tokenizer.json"
        }
    }

    // ---- Search + enrich ---------------------------------------------------

    /**
     * Run QueryEngine.search + searchImages, then enrich results with display metadata.
     * Called on Dispatchers.IO.
     */
    private fun runSearch(query: String, topK: Int): SearchOutput {
        val eng = engine ?: return SearchOutput.EMPTY

        // 1. Raw file-level results from one shared query embedding
        val raw = eng.searchAll(query, topK)
        val rawDocs = raw.documents
        val rawPhotos = raw.images

        // 2. Meaningful query words (shared stopword list) drive snippets and strong matches
        val terms = meaningfulTerms(query)

        val docStrong   = strongDocs(rawDocs.map { it.cosine }, rawDocs.map { it.matchedTerms }, terms)
        val photoStrong = strongPhotos(rawPhotos.map { it.score }, terms)
        val (docBucket, photoBucket) = bucketHits(
            rawDocs.mapIndexed { i, d -> Hit(docStrong[i], d.cosine, d.keywordHit) },
            rawPhotos.mapIndexed { i, p -> Hit(photoStrong[i], p.score, false) },
        )

        // 3. Enrich documents
        val repo = IndexRepository(db)
        val enrichedDocs = rawDocs.withIndex().filter { docBucket[it.index] != Bucket.DROP }.map { (i, fr) ->
            val chunkText = repo.chunkContent(fr.filePath, fr.chunkOffset)
            val snippet   = if (chunkText != null) buildSnippet(chunkText, terms, SNIPPET_MAX_LEN, startsMidText = fr.chunkOffset > 0) else ""
            val meta      = resolveFileMeta(fr.filePath)
            DocRow(
                filePath     = fr.filePath,
                score        = fr.score,
                chunkOffset  = fr.chunkOffset,
                displayName  = meta.displayName,
                folderName   = meta.folderName,
                dateModified = meta.dateModified,
                fileKind     = fileKindFromMime(meta.mimeType),
                snippet      = snippet,
                strong       = docBucket[i] == Bucket.STRONG
            )
        }

        // 4. Photos need no enrichment. Display is done via thumbnail.
        val photoRows = rawPhotos.withIndex().filter { photoBucket[it.index] != Bucket.DROP }
            .map { (i, p) -> PhotoRow(p.filePath, p.score, photoBucket[i] == Bucket.STRONG) }

        return SearchOutput(enrichedDocs, photoRows)
    }

    // ---- Home-screen queries (lightweight DB reads, no ONNX needed) --------

    /**
     * Most recently indexed documents ordered by last_indexed DESC.
     *
     * Queries only [IndexDatabase.TABLE_FILE_INDEX] rows NOT present in
     * [IndexDatabase.TABLE_IMAGE_VECTORS] (images do not have text chunks and are
     * shown in the photo carousel, not the doc list). Runs on [Dispatchers.IO].
     */
    override suspend fun recentDocuments(limit: Int): List<DocRow> = withContext(Dispatchers.IO) {
        try {
            val results = mutableListOf<DocRow>()
            db.readableDatabase.rawQuery(
                """SELECT ${IndexDatabase.COL_FI_FILE_PATH}
                   FROM ${IndexDatabase.TABLE_FILE_INDEX}
                   WHERE ${IndexDatabase.COL_FI_FILE_PATH} NOT IN
                       (SELECT ${IndexDatabase.COL_IV_FILE_PATH} FROM ${IndexDatabase.TABLE_IMAGE_VECTORS})
                   ORDER BY ${IndexDatabase.COL_FI_LAST_INDEXED} DESC LIMIT ?""",
                arrayOf(limit.toString())
            ).use { c ->
                while (c.moveToNext()) {
                    val path = c.getString(0)
                    val meta = resolveFileMeta(path)
                    results += DocRow(
                        filePath     = path,
                        score        = 0f,
                        chunkOffset  = 0,
                        displayName  = meta.displayName,
                        folderName   = meta.folderName,
                        dateModified = meta.dateModified,
                        fileKind     = fileKindFromMime(meta.mimeType),
                        snippet      = ""
                    )
                }
            }
            results
        } catch (e: Exception) {
            Log.w(TAG, "recentDocuments query failed", e)
            emptyList()
        }
    }

    /**
     * Most recently inserted photo URIs from [IndexDatabase.TABLE_IMAGE_VECTORS],
     * using rowid as an insertion-order proxy (no date column in image_vectors).
     */
    override suspend fun recentPhotos(limit: Int): List<PhotoRow> = withContext(Dispatchers.IO) {
        try {
            val results = mutableListOf<PhotoRow>()
            db.readableDatabase.rawQuery(
                "SELECT ${IndexDatabase.COL_IV_FILE_PATH} FROM ${IndexDatabase.TABLE_IMAGE_VECTORS} ORDER BY rowid DESC LIMIT ?",
                arrayOf(limit.toString())
            ).use { c ->
                while (c.moveToNext()) results += PhotoRow(c.getString(0), 0f)
            }
            results
        } catch (e: Exception) {
            Log.w(TAG, "recentPhotos query failed", e)
            emptyList()
        }
    }

    override suspend fun indexedFileCount(): Int = withContext(Dispatchers.IO) {
        try { IndexRepository(db).indexedFileCount() } catch (e: Exception) { 0 }
    }

    // ---- Metadata resolution -----------------------------------------------

    private data class FileMeta(
        val displayName: String,
        val folderName: String,
        val dateModified: Long,
        val mimeType: String?
    )

    /**
     * Resolve display name, folder, date, and MIME type for [pathOrUri].
     *
     * - For `content://` URIs: queries MediaStore (single row, projection only).
     * - For plain file paths: derived from the path string without IO.
     */
    private fun resolveFileMeta(pathOrUri: String): FileMeta {
        if (pathOrUri.startsWith("content://")) {
            val uri = Uri.parse(pathOrUri)
            try {
                val projection = arrayOf(
                    MediaStore.MediaColumns.DISPLAY_NAME,
                    MediaStore.MediaColumns.RELATIVE_PATH,
                    MediaStore.MediaColumns.DATE_MODIFIED,
                    MediaStore.MediaColumns.MIME_TYPE
                )
                context.contentResolver.query(uri, projection, null, null, null)?.use { c ->
                    if (c.moveToFirst()) {
                        val displayName  = c.getString(0) ?: lastSegment(pathOrUri)
                        val relativePath = c.getString(1) ?: ""
                        val dateModified = c.getLong(2) * 1000L  // seconds → millis
                        val mimeType     = c.getString(3)
                        val folder = relativePath.trimEnd('/').substringAfterLast('/').ifEmpty { relativePath }
                        return FileMeta(displayName, folder, dateModified, mimeType)
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "MediaStore query failed for $pathOrUri", e)
            }
            return FileMeta(lastSegment(pathOrUri), "", 0L, null)
        } else {
            val file = File(pathOrUri)
            val mime = mimeFromExtension(file.extension)
            return FileMeta(
                displayName  = file.name,
                folderName   = file.parentFile?.name ?: "",
                dateModified = file.lastModified(),
                mimeType     = mime
            )
        }
    }

    private fun lastSegment(pathOrUri: String): String =
        pathOrUri.trimEnd('/').substringAfterLast('/').ifEmpty { pathOrUri }

    private fun mimeFromExtension(ext: String): String? = when (ext.lowercase()) {
        "pdf"  -> "application/pdf"
        "doc"  -> "application/msword"
        "docx" -> "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        "txt"  -> "text/plain"
        "md"   -> "text/markdown"
        "jpg", "jpeg" -> "image/jpeg"
        "png"  -> "image/png"
        "webp" -> "image/webp"
        else   -> null
    }
}
