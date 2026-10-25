package me.zhanghai.android.files.index

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.narad.search.lib.index.provision.ProvisionState
import com.narad.search.lib.index.service.BootReceiver
import com.narad.search.lib.index.service.IndexService
import com.narad.search.lib.index.service.MediaStoreObserver
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexEvent
import com.narad.search.lib.index.store.IndexRepository
import com.narad.search.lib.index.worker.IndexStage
import com.narad.search.lib.index.worker.IndexWorker
import com.narad.search.lib.index.worker.PHOTOS_DEFERRED_STATUS
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID

// ---- Pure state logic (JVM tested in IndexingViewModelTest) ----------------------------

/** One-time user work and background work, split the way [isIndexingActive] expects. */
internal data class IndexWork(val user: List<WorkInfo>, val background: List<WorkInfo>) {
    val all: List<WorkInfo> get() = user + background
}

/**
 * What the Indexing screen shows as "now".
 *
 * [stage] is the live stage while work runs, the last terminal stage (Done or Failed) when
 * idle, or null when no stage is known. [running] is the [isIndexingActive] rule: one-time
 * work counts while unfinished, periodic and MediaStore deltas only while RUNNING.
 * [photosDeferred]: the last run skipped photos for low memory (idle only).
 */
internal data class IndexingStatus(
    val stage: IndexStage? = null,
    val running: Boolean = false,
    val lastRunAt: Long? = null,
    val lastRunDurationMs: Long? = null,
    val photosDeferred: Boolean = false,
)

internal fun resolveIndexingStatus(work: IndexWork): IndexingStatus {
    // Newest finished run. WorkManager returns EMPTY progress for finished work, so read outputData.
    val last = work.all
        .filter { it.state == WorkInfo.State.SUCCEEDED || it.state == WorkInfo.State.FAILED }
        .maxByOrNull { it.outputData.getLong(IndexWorker.KEY_FINISHED_AT, 0L) }
    val finishedAt = last?.outputData?.getLong(IndexWorker.KEY_FINISHED_AT, 0L)?.takeIf { it > 0L }
    val startedAt = last?.outputData?.getLong(IndexWorker.KEY_STARTED_AT, 0L)?.takeIf { it > 0L }
    val duration = if (finishedAt != null && startedAt != null && finishedAt >= startedAt) {
        finishedAt - startedAt
    } else null

    if (isIndexingActive(work.user, work.background)) {
        // RUNNING work first. An ENQUEUED one-time run (waiting for the model or a retry) may
        // still carry a stage in its progress.
        val live = pickRunningWork(work.all)
            ?: work.user.firstOrNull { !it.state.isFinished }
        // A Failed stage left in progress while WorkManager waits to retry is not final.
        val stage = live?.let { IndexStage.fromData(it.progress) }?.takeUnless { it is IndexStage.Failed }
        // E26: keep the last finished run's time and duration while a new run is going.
        return IndexingStatus(stage = stage, running = true, lastRunAt = finishedAt, lastRunDurationMs = duration)
    }
    if (last == null) return IndexingStatus()
    return IndexingStatus(
        stage = IndexStage.fromData(last.outputData),
        running = false,
        lastRunAt = finishedAt,
        lastRunDurationMs = duration,
        photosDeferred = last.outputData.getString(IndexWorker.KEY_STATUS) == PHOTOS_DEFERRED_STATUS,
    )
}

/**
 * Pure: ids of finished work. Key effects on this, not on the WorkInfo list: every progress
 * tick changes the list, but a finished id is added only once (review-033).
 */
internal fun finishedWorkIds(infos: List<WorkInfo>): Set<UUID> =
    infos.filter { it.state.isFinished }.map { it.id }.toSet()

/** Ring fill 0..1, or null when the progress is not known (indeterminate). */
internal fun ringFraction(stage: IndexStage?): Float? = when (stage) {
    is IndexStage.Reading -> if (stage.total > 0) (stage.done.toFloat() / stage.total).coerceIn(0f, 1f) else null
    is IndexStage.Done -> 1f
    is IndexStage.WaitingForModel -> stage.percent?.let { (it / 100f).coerceIn(0f, 1f) }
    else -> null
}

/** What the status row says about the photo model instead of "Up to date". */
internal sealed class PhotoModelStatus {
    object Missing : PhotoModelStatus()
    data class Downloading(val percent: Int) : PhotoModelStatus()
    data class Failed(val reason: String) : PhotoModelStatus()
}

/**
 * Pure: null when the phone has no photo search ([imagesOn] false, Tier C) or the photo model
 * is ready. Percent is per file, the same as the text model row.
 */
internal fun photoModelStatus(imagesOn: Boolean, state: ProvisionState): PhotoModelStatus? {
    if (!imagesOn) return null
    return when (state) {
        ProvisionState.Ready -> null
        ProvisionState.NotStarted -> PhotoModelStatus.Missing
        is ProvisionState.Downloading -> PhotoModelStatus.Downloading(
            if (state.bytesTotal > 0) (state.bytesDownloaded * 100 / state.bytesTotal).toInt().coerceIn(0, 100) else 0
        )
        is ProvisionState.Verifying -> PhotoModelStatus.Downloading(100)
        is ProvisionState.Failed -> PhotoModelStatus.Failed(state.reason)
    }
}

/**
 * Pure: the part of a model download failure that is safe to show, or null for the plain
 * "check your connection" text. Raw reasons carry a "[file]" prefix and exception text.
 */
internal fun plainDownloadFailure(reason: String): String? {
    // ponytail: matches our own lib text (ModelProvisioner U8); add a typed reason to
    // ProvisionState.Failed if more cases appear.
    val at = reason.indexOf("Not enough free space")
    return if (at >= 0) reason.substring(at).trimEnd('.') else null
}

/** Progress of one type row: [total] is null when only an indexed count is known. */
internal data class TypeProgress(val done: Int, val total: Int?)

internal const val KIND_PHOTOS = "photos"
internal const val KIND_DOCUMENTS = "documents"

internal fun typeProgress(kind: String, stage: IndexStage?, indexedCount: Int?): TypeProgress? {
    if (stage is IndexStage.Reading && stage.kind == kind && stage.total > 0) {
        return TypeProgress(stage.done, stage.total)
    }
    return indexedCount?.let { TypeProgress(it, null) }
}

internal data class IndexingActions(val start: Boolean, val reindex: Boolean, val tryAgain: Boolean)

/** Which buttons the screen offers. WaitingForModel offers none: the download must finish first. */
internal fun availableActions(status: IndexingStatus): IndexingActions {
    val stage = status.stage
    return when {
        stage is IndexStage.WaitingForModel -> IndexingActions(start = false, reindex = false, tryAgain = false)
        stage is IndexStage.Failed && !status.running -> IndexingActions(start = false, reindex = true, tryAgain = true)
        status.running -> IndexingActions(start = false, reindex = true, tryAgain = false)
        else -> IndexingActions(start = true, reindex = true, tryAgain = false)
    }
}

internal enum class FailureKind { MODEL, STORAGE, OTHER }

/** Raw failure reasons are technical and may echo file details. The screen shows only the kind. */
internal fun classifyFailure(reason: String): FailureKind {
    val r = reason.lowercase(Locale.ROOT)
    return when {
        "enospc" in r || "no space" in r || "disk full" in r || "storage" in r -> FailureKind.STORAGE
        "model" in r || "onnx" in r || "tokenizer" in r -> FailureKind.MODEL
        else -> FailureKind.OTHER
    }
}

internal enum class ProblemKind { PDF, WORD, TEXT, DOCUMENT, PHOTO, RUN, OTHER }

internal fun classifyProblem(event: IndexEvent): ProblemKind {
    val m = event.message.orEmpty()
    return when {
        "PDF" in m -> ProblemKind.PDF
        "DOCX" in m || "SAX" in m -> ProblemKind.WORD
        "Plain text" in m -> ProblemKind.TEXT
        event.stage == "extraction" -> ProblemKind.DOCUMENT
        event.stage == "photos" -> ProblemKind.PHOTO
        event.stage == "worker" -> ProblemKind.RUN
        else -> ProblemKind.OTHER
    }
}

internal data class ProblemItem(val name: String?, val count: Int)
internal data class ProblemGroup(val kind: ProblemKind, val items: List<ProblemItem>)

/**
 * Pure: WARN and ERROR events grouped by problem kind (newest event first keeps its order).
 * [names] maps a stored event file name (a MediaStore id) to a readable name.
 * Rows with the same name collapse into one with a count.
 */
internal fun groupProblems(events: List<IndexEvent>, names: Map<String, String>): List<ProblemGroup> =
    events
        .filter { it.level == "WARN" || it.level == "ERROR" }
        .groupBy { classifyProblem(it) }
        .map { (kind, list) ->
            val items = list
                .groupBy { e -> e.fileName?.let { names[it] ?: it } }
                .map { (name, same) -> ProblemItem(name, same.size) }
            ProblemGroup(kind, items)
        }

/**
 * Pure: the text "Copy diagnostics" puts on the clipboard. Stage names, counts and event
 * lines only. Never file content: events carry a file name, an error class and a short
 * capped message (KTD18), and a Failed reason is reduced to its kind.
 */
internal fun buildDiagnostics(
    status: IndexingStatus,
    photoCount: Int?,
    documentCount: Int?,
    events: List<IndexEvent>,
    names: Map<String, String>,
): String = buildString {
    appendLine("Narad indexing diagnostics")
    appendLine("Stage: ${stageName(status.stage)}")
    appendLine("Running: ${status.running}")
    status.lastRunAt?.let { appendLine("Last run: ${isoTime(it)}") }
    status.lastRunDurationMs?.let { appendLine("Last run duration: ${it / 1000} s") }
    photoCount?.let { appendLine("Photos indexed: $it") }
    documentCount?.let { appendLine("Documents indexed: $it") }
    appendLine("Events: ${events.size}")
    events.forEach { e ->
        val file = e.fileName?.let { names[it] ?: it } ?: "-"
        appendLine("${isoTime(e.timestampMs)} ${e.level} ${e.stage} ${e.errorClass ?: "-"} $file ${e.message ?: ""}".trimEnd())
    }
}

private fun stageName(stage: IndexStage?): String = when (stage) {
    null -> "none"
    is IndexStage.WaitingForModel -> "WaitingForModel" + (stage.percent?.let { " $it%" } ?: "")
    is IndexStage.WaitingForPermission -> "WaitingForPermission"
    is IndexStage.Finding -> "Finding ${stage.kind}"
    is IndexStage.Reading -> "Reading ${stage.kind} ${stage.done}/${stage.total}"
    is IndexStage.Done -> "Done ${stage.count}"
    is IndexStage.Failed -> "Failed (${classifyFailure(stage.reason)})"
    IndexStage.Paused -> "Paused"
}

private fun isoTime(ms: Long): String =
    SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        .apply { timeZone = TimeZone.getTimeZone("UTC") }
        .format(Date(ms))

// ---- Work observation ------------------------------------------------------------------

/** Unique-work names for one-time user work (initial index, app-start delta). */
internal fun userWorkInfosFlow(wm: WorkManager): Flow<List<WorkInfo>> =
    combine(
        wm.getWorkInfosForUniqueWorkFlow(IndexService.INITIAL_INDEX_WORK_NAME),
        wm.getWorkInfosForUniqueWorkFlow(BootReceiver.FOREGROUND_DELTA_WORK_NAME),
    ) { a, b -> a + b }

/** Periodic delta and MediaStore delta: ENQUEUED almost always, so only RUNNING counts. */
internal fun backgroundWorkInfosFlow(wm: WorkManager): Flow<List<WorkInfo>> =
    combine(
        wm.getWorkInfosForUniqueWorkFlow(BootReceiver.DELTA_WORK_NAME),
        wm.getWorkInfosForUniqueWorkFlow(MediaStoreObserver.DELTA_WORK_NAME),
    ) { a, b -> a + b }

internal fun indexWorkFlow(wm: WorkManager): Flow<IndexWork> =
    combine(userWorkInfosFlow(wm), backgroundWorkInfosFlow(wm)) { u, b -> IndexWork(u, b) }

// ---- ViewModel -------------------------------------------------------------------------

/** What the DB loader returns. Loaded on IO by the injected lambda. */
internal data class IndexingData(
    val events: List<IndexEvent>,
    val photoCount: Int,
    val documentCount: Int,
)

internal data class IndexingUiState(
    val status: IndexingStatus = IndexingStatus(),
    val photoCount: Int? = null,
    val documentCount: Int? = null,
    val problems: List<ProblemGroup> = emptyList(),
    val loaded: Boolean = false,
    // Kept for the diagnostics text, so it does not re-query the DB on the main thread.
    val events: List<IndexEvent> = emptyList(),
    val names: Map<String, String> = emptyMap(),
)

/**
 * ViewModel for the Indexing screen.
 *
 * All Android access is injected as lambdas (application context only, never an Activity):
 * [work] observes WorkManager, [loadData] reads the DB on IO, [resolveNames] maps stored
 * MediaStore ids to display names on IO and returns an empty map on any failure.
 *
 * The data reload keys on [finishedWorkIds], not on the work list, and a newer finish
 * cancels an older reload, so an old read can never overwrite a newer one.
 */
internal class IndexingViewModel(
    private val work: Flow<IndexWork>,
    private val loadData: suspend () -> IndexingData,
    private val resolveNames: suspend (Set<String>) -> Map<String, String>,
) : ViewModel() {

    private val _uiState = MutableStateFlow(IndexingUiState())
    val uiState: StateFlow<IndexingUiState> = _uiState.asStateFlow()

    private var lastFinished: Set<UUID>? = null
    private var reloadJob: Job? = null

    init {
        viewModelScope.launch {
            work.collect { w ->
                _uiState.update { it.copy(status = resolveIndexingStatus(w)) }
                // Everything up to launch() runs on the main thread with no suspension point,
                // so the check-then-act on lastFinished / reloadJob cannot interleave.
                val ids = finishedWorkIds(w.all)
                if (ids != lastFinished) {
                    lastFinished = ids
                    reloadJob?.cancel()
                    reloadJob = launch { reload() }
                }
            }
        }
    }

    private suspend fun reload() {
        val data = try { loadData() } catch (e: kotlinx.coroutines.CancellationException) { throw e } catch (_: Exception) { null }
        if (data == null) {
            _uiState.update { it.copy(loaded = true) }
            return
        }
        val names = try {
            resolveNames(data.events.mapNotNull { it.fileName }.toSet())
        } catch (e: kotlinx.coroutines.CancellationException) {
            throw e
        } catch (_: Exception) {
            emptyMap()
        }
        _uiState.update {
            it.copy(
                photoCount = data.photoCount,
                documentCount = data.documentCount,
                problems = groupProblems(data.events, names),
                events = data.events,
                names = names,
                loaded = true,
            )
        }
    }

    fun diagnosticsText(): String = _uiState.value.let {
        buildDiagnostics(it.status, it.photoCount, it.documentCount, it.events, it.names)
    }

    companion object {
        /** Real wiring. Only the application context is captured. */
        internal fun create(appContext: Context): IndexingViewModel {
            val wm = WorkManager.getInstance(appContext)
            return IndexingViewModel(
                work = indexWorkFlow(wm),
                loadData = { loadIndexingData(appContext) },
                resolveNames = { ids -> resolveMediaStoreNames(appContext.contentResolver, ids) },
            )
        }
    }
}

private suspend fun loadIndexingData(context: Context): IndexingData = withContext(Dispatchers.IO) {
    // Index0: reading status must not create the index file before the user starts indexing.
    if (!context.getDatabasePath(IndexDatabase.DB_NAME).exists()) {
        return@withContext IndexingData(events = emptyList(), photoCount = 0, documentCount = 0)
    }
    val db = IndexDatabase(context)
    try {
        val repo = IndexRepository(db)
        IndexingData(
            events = repo.listEvents(),
            photoCount = repo.imageVectorCount(),
            // ponytail: file_index holds photos too (ImageIndexer marks them), so subtract the
            // image vectors. Off by the photos that failed to embed; a per-kind query if it matters.
            documentCount = (repo.indexedFileCount() - repo.imageVectorCount()).coerceAtLeast(0),
        )
    } finally {
        db.close()
    }
}

/**
 * review-032 / M21: event file names for MediaStore targets are the numeric MediaStore id
 * (`IndexTarget` has no display name). Rebuild the stored URI (the enumerator's Files
 * collection plus the id) and read DISPLAY_NAME. Off the main thread, one query per batch.
 * Any failure (missing permission, deleted file, bad id) leaves that name out, so the caller
 * falls back to the stored name. Non-numeric names are real file names and pass through.
 */
internal suspend fun resolveMediaStoreNames(
    resolver: ContentResolver,
    storedNames: Set<String>,
): Map<String, String> = withContext(Dispatchers.IO) {
    val ids = storedNames.filter { it.isNotEmpty() && it.all(Char::isDigit) }
    if (ids.isEmpty()) return@withContext emptyMap()
    val out = HashMap<String, String>()
    val collection: Uri = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL)
    // SQLite caps bound variables at 999, listEvents caps at 500, chunk anyway.
    ids.chunked(400).forEach { chunk ->
        try {
            resolver.query(
                collection,
                arrayOf(MediaStore.MediaColumns._ID, MediaStore.MediaColumns.DISPLAY_NAME),
                "${MediaStore.MediaColumns._ID} IN (${chunk.joinToString(",") { "?" }})",
                chunk.toTypedArray(),
                null,
            )?.use { c ->
                while (c.moveToNext()) {
                    val name = c.getString(1)
                    if (!name.isNullOrBlank()) out[c.getLong(0).toString()] = name
                }
            }
        } catch (_: Exception) {
            // SecurityException (permission revoked) or any provider error: keep stored names.
        }
    }
    out
}
