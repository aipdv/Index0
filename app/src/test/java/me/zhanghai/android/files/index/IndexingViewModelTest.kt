package me.zhanghai.android.files.index

import androidx.work.Data
import androidx.work.WorkInfo
import androidx.work.WorkInfo.State
import com.narad.search.lib.index.provision.ProvisionState
import com.narad.search.lib.index.store.IndexEvent
import com.narad.search.lib.index.worker.IndexStage
import com.narad.search.lib.index.worker.IndexWorker
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.UUID

/**
 * JVM tests for the Indexing screen state (U26, review-032, review-033).
 * Gate: :app:testDebugUnitTest --tests 'me.zhanghai.android.files.index.IndexingViewModelTest'
 */
@OptIn(ExperimentalCoroutinesApi::class)
class IndexingViewModelTest {

    private val dispatcher = StandardTestDispatcher()

    @Before fun setUp() { Dispatchers.setMain(dispatcher) }
    @After fun tearDown() { Dispatchers.resetMain() }

    private fun info(
        state: State,
        progress: Data = Data.EMPTY,
        output: Data = Data.EMPTY,
        id: UUID = UUID.randomUUID(),
    ) = WorkInfo(id, state, emptySet(), output, progress)

    private fun event(
        id: Long,
        stage: String = "extraction",
        level: String = "WARN",
        file: String? = "a.pdf",
        errorClass: String? = "IOException",
        message: String? = "PDF extraction failed",
    ) = IndexEvent(id, 1_700_000_000_000L + id, "run1", stage, level, file, errorClass, message)

    private fun doneOutput(count: Int, startedAt: Long, finishedAt: Long): Data =
        Data.Builder().putAll(IndexStage.Done(count).toData())
            .putLong(IndexWorker.KEY_STARTED_AT, startedAt)
            .putLong(IndexWorker.KEY_FINISHED_AT, finishedAt).build()

    // ---- status resolution ----------------------------------------------------------

    @Test
    fun `two RUNNING jobs, the one with a stage wins even when listed second`() {
        val waiting = info(State.RUNNING)
        val reading = info(State.RUNNING, progress = IndexStage.Reading("photos", 4, 10).toData())
        val status = resolveIndexingStatus(IndexWork(listOf(waiting, reading), emptyList()))
        assertEquals(IndexStage.Reading("photos", 4, 10), status.stage)
    }

    @Test
    fun `Reading photos 4 of 10 gives stage, 40 percent ring and per-type progress`() {
        val running = info(State.RUNNING, progress = IndexStage.Reading("photos", 4, 10).toData())
        val status = resolveIndexingStatus(IndexWork(listOf(running), emptyList()))

        assertTrue(status.running)
        assertEquals(IndexStage.Reading("photos", 4, 10), status.stage)
        assertEquals(0.4f, ringFraction(status.stage)!!, 0.0001f)
        assertEquals(TypeProgress(4, 10), typeProgress(KIND_PHOTOS, status.stage, indexedCount = 99))
        // The other type has no live total, so it shows the indexed count.
        assertEquals(TypeProgress(7, null), typeProgress(KIND_DOCUMENTS, status.stage, indexedCount = 7))
        assertNull(typeProgress(KIND_DOCUMENTS, status.stage, indexedCount = null))
    }

    @Test
    fun `ring is indeterminate without a total and full when done`() {
        assertNull(ringFraction(IndexStage.Reading("photos", 0, 0)))
        assertNull(ringFraction(IndexStage.Finding("documents")))
        assertNull(ringFraction(null))
        assertEquals(1f, ringFraction(IndexStage.Done(5))!!, 0f)
        assertEquals(1f, ringFraction(IndexStage.Reading("photos", 12, 10))!!, 0f)
    }

    @Test
    fun `WaitingForModel shows the percent and offers no actions`() {
        val running = info(State.RUNNING, progress = IndexStage.WaitingForModel(30).toData())
        val status = resolveIndexingStatus(IndexWork(listOf(running), emptyList()))

        assertEquals(IndexStage.WaitingForModel(30), status.stage)
        assertEquals(0.3f, ringFraction(status.stage)!!, 0.0001f)
        assertEquals(IndexingActions(start = false, reindex = false, tryAgain = false), availableActions(status))
    }

    @Test
    fun `idle offers start and reindex, running offers reindex only`() {
        val idle = IndexingStatus()
        assertEquals(IndexingActions(start = true, reindex = true, tryAgain = false), availableActions(idle))
        val running = IndexingStatus(stage = IndexStage.Reading("photos", 1, 5), running = true)
        assertEquals(IndexingActions(start = false, reindex = true, tryAgain = false), availableActions(running))
    }

    @Test
    fun `terminal Failed offers Try again and no Start`() {
        val failedOut = Data.Builder().putAll(IndexStage.Failed("boom").toData())
            .putLong(IndexWorker.KEY_FINISHED_AT, 500L).build()
        val status = resolveIndexingStatus(IndexWork(listOf(info(State.FAILED, output = failedOut)), emptyList()))

        assertEquals(IndexStage.Failed("boom"), status.stage)
        assertFalse(status.running)
        assertEquals(IndexingActions(start = false, reindex = true, tryAgain = true), availableActions(status))
    }

    @Test
    fun `Failed left in progress while WorkManager waits to retry is not shown as final`() {
        val retrying = info(State.ENQUEUED, progress = IndexStage.Failed("boom").toData())
        val status = resolveIndexingStatus(IndexWork(listOf(retrying), emptyList()))
        assertTrue(status.running)
        assertNull(status.stage)
    }

    @Test
    fun `last run time and duration come from outputData of finished work`() {
        val done = info(State.SUCCEEDED, output = doneOutput(count = 12, startedAt = 1_000L, finishedAt = 126_000L))
        val status = resolveIndexingStatus(IndexWork(listOf(done), emptyList()))

        assertEquals(IndexStage.Done(12), status.stage)
        assertEquals(126_000L, status.lastRunAt)
        assertEquals(125_000L, status.lastRunDurationMs)
    }

    @Test
    fun `E26 running status still carries the last finished run time and duration`() {
        val old = info(State.SUCCEEDED, output = doneOutput(count = 12, startedAt = 1_000L, finishedAt = 126_000L))
        val running = info(State.RUNNING, progress = IndexStage.Reading("photos", 4, 10).toData())
        val status = resolveIndexingStatus(IndexWork(listOf(old, running), emptyList()))

        assertTrue(status.running)
        assertEquals(IndexStage.Reading("photos", 4, 10), status.stage)
        assertEquals(126_000L, status.lastRunAt)
        assertEquals(125_000L, status.lastRunDurationMs)
    }

    @Test
    fun `E26 running with no finished run has no last run time`() {
        val running = info(State.RUNNING, progress = IndexStage.Reading("photos", 4, 10).toData())
        val status = resolveIndexingStatus(IndexWork(listOf(running), emptyList()))

        assertTrue(status.running)
        assertNull(status.lastRunAt)
        assertNull(status.lastRunDurationMs)
    }

    @Test
    fun `progress of finished work is ignored, WorkManager clears it`() {
        val progressOnly = info(State.SUCCEEDED, progress = IndexStage.Done(9).toData())
        val status = resolveIndexingStatus(IndexWork(listOf(progressOnly), emptyList()))
        assertNull(status.stage)
        assertNull(status.lastRunAt)
    }

    @Test
    fun `E19 idle status shows WaitingForPermission after a permission-wait run, a newer Done replaces it`() {
        val wait = info(
            State.SUCCEEDED,
            output = Data.Builder()
                .putAll(IndexStage.WaitingForPermission("android.permission.READ_MEDIA_IMAGES").toData())
                .putLong(IndexWorker.KEY_FINISHED_AT, 100L).build()
        )
        val idle = resolveIndexingStatus(IndexWork(listOf(wait), emptyList()))
        assertEquals(IndexStage.WaitingForPermission("android.permission.READ_MEDIA_IMAGES"), idle.stage)
        assertFalse(idle.running)

        val done = info(State.SUCCEEDED, output = doneOutput(count = 7, startedAt = 150L, finishedAt = 200L))
        val after = resolveIndexingStatus(IndexWork(listOf(wait, done), emptyList()))
        assertEquals(IndexStage.Done(7), after.stage)
    }

    @Test
    fun `latest finished run wins across success and failure`() {
        val oldDone = info(State.SUCCEEDED, output = doneOutput(5, 1L, 100L))
        val newFailed = info(
            State.FAILED,
            output = Data.Builder().putAll(IndexStage.Failed("x").toData())
                .putLong(IndexWorker.KEY_FINISHED_AT, 900L).build()
        )
        val status = resolveIndexingStatus(IndexWork(listOf(oldDone, newFailed), emptyList()))
        assertEquals(IndexStage.Failed("x"), status.stage)
    }

    @Test
    fun `periodic delta sitting ENQUEUED is not indexing but a RUNNING one is`() {
        val periodic = info(State.ENQUEUED)
        assertFalse(resolveIndexingStatus(IndexWork(emptyList(), listOf(periodic))).running)
        val running = info(State.RUNNING, progress = IndexStage.Finding("documents").toData())
        val s = resolveIndexingStatus(IndexWork(emptyList(), listOf(running)))
        assertTrue(s.running)
        assertEquals(IndexStage.Finding("documents"), s.stage)
    }

    @Test
    fun `enqueued one-time work counts as indexing`() {
        assertTrue(resolveIndexingStatus(IndexWork(listOf(info(State.ENQUEUED)), emptyList())).running)
    }

    @Test
    fun `no work at all is idle with no stage`() {
        assertEquals(IndexingStatus(), resolveIndexingStatus(IndexWork(emptyList(), emptyList())))
    }

    // ---- review-033: key on finished ids, not on the list ---------------------------

    @Test
    fun `progress ticks do not change the finished id set, a new finish does`() {
        val id = UUID.randomUUID()
        val tick1 = listOf(info(State.RUNNING, IndexStage.Reading("photos", 1, 10).toData(), id = id))
        val tick2 = listOf(info(State.RUNNING, IndexStage.Reading("photos", 2, 10).toData(), id = id))
        val finished = listOf(info(State.SUCCEEDED, output = IndexStage.Done(10).toData(), id = id))

        assertEquals(finishedWorkIds(tick1), finishedWorkIds(tick2))
        assertTrue(finishedWorkIds(tick1).isEmpty())
        assertNotEquals(finishedWorkIds(tick2), finishedWorkIds(finished))
        assertEquals(setOf(id), finishedWorkIds(finished))
    }

    // ---- problems -------------------------------------------------------------------

    @Test
    fun `no events gives no problem groups`() {
        assertTrue(groupProblems(emptyList(), emptyMap()).isEmpty())
    }

    @Test
    fun `problems group by kind, collapse repeated names and skip INFO`() {
        val events = listOf(
            event(5, file = "b.pdf"),
            event(4, file = "a.pdf"),
            event(3, file = "a.pdf"),
            event(2, stage = "photos", message = "Failed to load image", file = "99", errorClass = "IOException"),
            event(1, stage = "worker", level = "INFO", message = "Model not ready", file = null),
        )
        val groups = groupProblems(events, emptyMap())

        assertEquals(listOf(ProblemKind.PDF, ProblemKind.PHOTO), groups.map { it.kind })
        assertEquals(listOf(ProblemItem("b.pdf", 1), ProblemItem("a.pdf", 2)), groups[0].items)
        assertEquals(listOf(ProblemItem("99", 1)), groups[1].items)
    }

    @Test
    fun `resolved display name replaces the MediaStore id, unresolved ids fall back`() {
        val events = listOf(
            event(2, stage = "photos", message = "Failed to load image", file = "123"),
            event(1, stage = "photos", message = "Failed to load image", file = "456"),
        )
        val items = groupProblems(events, mapOf("123" to "IMG_0001.jpg")).single().items
        assertEquals(listOf("IMG_0001.jpg", "456"), items.map { it.name })
    }

    @Test
    fun `failure reasons classify without leaking the raw text`() {
        assertEquals(FailureKind.STORAGE, classifyFailure("write failed: ENOSPC (No space left on device)"))
        assertEquals(FailureKind.MODEL, classifyFailure("Failed to load model.onnx"))
        assertEquals(FailureKind.OTHER, classifyFailure("anything else"))
        assertEquals(FailureKind.OTHER, classifyFailure(""))
    }

    // ---- diagnostics ----------------------------------------------------------------

    @Test
    fun `diagnostics holds stages and event lines and no file content`() {
        val secret = "SECRET_BODY_TEXT_731"
        val status = IndexingStatus(
            stage = IndexStage.Failed("could not read $secret"),
            running = false,
            lastRunAt = 1_700_000_000_000L,
            lastRunDurationMs = 65_000L,
        )
        val text = buildDiagnostics(
            status, photoCount = 3, documentCount = 8,
            events = listOf(event(1, file = "123", message = "PDF extraction failed")),
            names = mapOf("123" to "report.pdf"),
        )

        assertTrue(text.contains("Stage: Failed (OTHER)"))
        assertTrue(text.contains("Photos indexed: 3"))
        assertTrue(text.contains("Documents indexed: 8"))
        assertTrue(text.contains("WARN extraction IOException report.pdf PDF extraction failed"))
        assertTrue(text.contains("Last run duration: 65 s"))
        assertFalse("raw failure reason must not be copied", text.contains(secret))
    }

    // ---- ViewModel wiring -----------------------------------------------------------

    private fun makeVm(
        work: MutableStateFlow<IndexWork>,
        load: suspend () -> IndexingData,
        names: suspend (Set<String>) -> Map<String, String> = { emptyMap() },
    ) = IndexingViewModel(work = work, loadData = load, resolveNames = names)

    @Test
    fun `reload keys on finished ids, progress ticks do not re-query the database`() = runTest(dispatcher) {
        val id = UUID.randomUUID()
        fun tick(n: Int) = IndexWork(
            listOf(info(State.RUNNING, IndexStage.Reading("photos", n, 10).toData(), id = id)), emptyList()
        )
        val work = MutableStateFlow(tick(0))
        var loads = 0
        val vm = makeVm(work, load = { loads++; IndexingData(emptyList(), photoCount = 2, documentCount = 3) })
        advanceUntilIdle()
        assertEquals(1, loads)

        work.value = tick(1); advanceUntilIdle()
        work.value = tick(2); advanceUntilIdle()
        assertEquals("progress ticks must not reload", 1, loads)
        assertEquals(IndexStage.Reading("photos", 2, 10), vm.uiState.value.status.stage)

        work.value = IndexWork(
            listOf(info(State.SUCCEEDED, output = IndexStage.Done(5).toData(), id = id)), emptyList()
        )
        advanceUntilIdle()
        assertEquals("a newly finished run reloads once", 2, loads)
    }

    @Test
    fun `display names resolve in the ViewModel and a resolver failure falls back to the stored name`() =
        runTest(dispatcher) {
            val events = listOf(event(1, stage = "photos", message = "Failed to load image", file = "123"))
            val data = IndexingData(events, 0, 0)

            val okVm = makeVm(MutableStateFlow(IndexWork(emptyList(), emptyList())), { data },
                names = { ids -> ids.associateWith { "IMG_$it.jpg" } })
            advanceUntilIdle()
            assertEquals("IMG_123.jpg", okVm.uiState.value.problems.single().items.single().name)

            val failVm = makeVm(MutableStateFlow(IndexWork(emptyList(), emptyList())), { data },
                names = { throw SecurityException("no permission") })
            advanceUntilIdle()
            val state = failVm.uiState.value
            assertTrue(state.loaded)
            assertEquals("123", state.problems.single().items.single().name)
        }

    @Test
    fun `a failing database read still marks the screen loaded and keeps the status`() = runTest(dispatcher) {
        val vm = makeVm(
            MutableStateFlow(IndexWork(listOf(info(State.ENQUEUED)), emptyList())),
            load = { throw IllegalStateException("db locked") },
        )
        advanceUntilIdle()
        assertTrue(vm.uiState.value.loaded)
        assertTrue(vm.uiState.value.status.running)
    }

    @Test
    fun `an older slow reload cannot overwrite a newer one`() = runTest(dispatcher) {
        val firstGate = CompletableDeferred<Unit>()
        var call = 0
        val work = MutableStateFlow(IndexWork(listOf(info(State.ENQUEUED)), emptyList()))
        val vm = makeVm(work, load = {
            val n = ++call
            if (n == 1) firstGate.await()
            IndexingData(emptyList(), photoCount = n, documentCount = n)
        })
        advanceUntilIdle()                    // first reload is suspended on its gate
        work.value = IndexWork(
            listOf(info(State.SUCCEEDED, output = IndexStage.Done(1).toData())), emptyList()
        )
        advanceUntilIdle()                    // second reload completes
        firstGate.complete(Unit)
        advanceUntilIdle()

        assertEquals(2, vm.uiState.value.photoCount)
    }

    @Test
    fun `diagnosticsText reflects the loaded state`() = runTest(dispatcher) {
        val vm = makeVm(
            MutableStateFlow(IndexWork(emptyList(), emptyList())),
            load = { IndexingData(listOf(event(1, file = "x.pdf")), photoCount = 4, documentCount = 6) },
        )
        advanceUntilIdle()
        val text = vm.diagnosticsText()
        assertTrue(text.contains("Photos indexed: 4"))
        assertTrue(text.contains("x.pdf"))
    }

    @Test fun photoModelStatus_coversMissingDownloadingFailed_onlyWithPhotoSearch() {
        assertEquals(PhotoModelStatus.Missing, photoModelStatus(true, ProvisionState.NotStarted))
        assertEquals(
            PhotoModelStatus.Downloading(25),
            photoModelStatus(true, ProvisionState.Downloading(0, 2, "vision.onnx", 50, 200)),
        )
        assertEquals(
            PhotoModelStatus.Downloading(0),
            photoModelStatus(true, ProvisionState.Downloading(0, 2, "vision.onnx", 50, -1)),
        )
        assertEquals(PhotoModelStatus.Downloading(100), photoModelStatus(true, ProvisionState.Verifying("vision.onnx")))
        assertEquals(PhotoModelStatus.Failed("no space"), photoModelStatus(true, ProvisionState.Failed("no space")))
        assertNull(photoModelStatus(true, ProvisionState.Ready))
        // Tier C has no photo search: never ask for the photo model.
        assertNull(photoModelStatus(false, ProvisionState.NotStarted))
        assertNull(photoModelStatus(false, ProvisionState.Failed("no space")))
    }
}
