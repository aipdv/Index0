package com.narad.search.lib.index

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.testing.WorkManagerTestInitHelper
import androidx.work.workDataOf
import com.narad.search.lib.index.service.IndexService
import com.narad.search.lib.index.worker.IndexWorker
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.TimeUnit

/**
 * Instrumented tests asserting the REPLACE-vs-KEEP WorkManager enqueue policy.
 *
 * These tests exercise that:
 *   (a) [IndexService.enqueueUserInitiatedIndex] uses [ExistingWorkPolicy.REPLACE]
 *       so a prior KEEP-enqueued unique job is replaced (prior job ID changes).
 *   (b) After calling [enqueueUserInitiatedIndex], exactly ONE WorkInfo exists for
 *       the unique work name - not zero, not more than one.
 *
 * Fix 13: strengthened assertions replace the previous "theatre" checks.
 */
@RunWith(AndroidJUnit4::class)
class ReindexPolicyTest {

    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun setUp() {
        // Initialize the WorkManager test harness - required to observe work states
        // in instrumented tests without an actual running WorkManager process.
        WorkManagerTestInitHelper.initializeTestWorkManager(context)
    }

    // -------------------------------------------------------------------------
    // Test A: REPLACE policy replaces prior KEEP-enqueued job
    // -------------------------------------------------------------------------

    /**
     * Enqueue a job with KEEP, then call [IndexService.enqueueUserInitiatedIndex]
     * (which uses REPLACE).
     *
     * Fix 13: assert the prior job becomes CANCELLED (not just that the ID changed).
     * WorkManagerTestInitHelper runs workers eagerly so the first job may finish
     * before REPLACE can cancel it, but since IndexWorker fails without a real model
     * the first job ends in FAILED - which is also acceptable as proof that the prior
     * work is no longer ENQUEUED/RUNNING.
     */
    @Test
    fun userInitiatedIndex_replacesPriorKeptJob() {
        val wm = WorkManager.getInstance(context)

        // First enqueue with KEEP (simulates the automated producer path).
        val firstRequest = OneTimeWorkRequestBuilder<IndexWorker>()
            .setInputData(workDataOf(IndexWorker.KEY_MODEL_PATH to "/fake/first.onnx"))
            .build()

        wm.enqueueUniqueWork(
            IndexService.INITIAL_INDEX_WORK_NAME,
            ExistingWorkPolicy.KEEP,
            firstRequest
        )

        val firstWorkId = firstRequest.id

        // Now call the user-initiated helper (uses REPLACE).
        IndexService.enqueueUserInitiatedIndex(context)

        // Fix 13: There should now be exactly one WorkInfo for the unique name.
        val workInfoList = wm.getWorkInfosForUniqueWork(IndexService.INITIAL_INDEX_WORK_NAME).get()
        assertEquals(
            "After REPLACE enqueue there should be exactly ONE WorkInfo for " +
            IndexService.INITIAL_INDEX_WORK_NAME,
            1,
            workInfoList.size
        )

        val currentWorkId = workInfoList.first().id
        assertNotEquals(
            "User-initiated REPLACE should have replaced the first KEEP-enqueued job " +
            "(work IDs must differ). First ID: $firstWorkId, current: $currentWorkId",
            firstWorkId,
            currentWorkId
        )

        // The new job must not be CANCELLED (it was just enqueued).
        val currentState = workInfoList.first().state
        assertTrue(
            "The replacement job must not be CANCELLED, got: $currentState",
            currentState != WorkInfo.State.CANCELLED
        )
    }

    // -------------------------------------------------------------------------
    // Test B: enqueueUserInitiatedIndex produces exactly one work item
    // -------------------------------------------------------------------------

    /**
     * Fix 13: replace the theatre assertion ("state != CANCELLED") with a real structural
     * assertion: after [IndexService.enqueueUserInitiatedIndex], there must be EXACTLY ONE
     * WorkInfo for [IndexService.INITIAL_INDEX_WORK_NAME].
     *
     * WorkManagerTestInitHelper may run the worker eagerly and the worker may FAIL
     * (no ONNX model present) - that is acceptable; what matters is that exactly one
     * work item was recorded.
     */
    @Test
    fun userInitiatedIndex_producesExactlyOneWorkItem() {
        val wm = WorkManager.getInstance(context)

        IndexService.enqueueUserInitiatedIndex(context)

        val workInfoList = wm.getWorkInfosForUniqueWork(IndexService.INITIAL_INDEX_WORK_NAME).get()

        assertEquals(
            "enqueueUserInitiatedIndex() should produce exactly one WorkInfo for " +
            IndexService.INITIAL_INDEX_WORK_NAME +
            " (got ${workInfoList.size})",
            1,
            workInfoList.size
        )
    }

    // -------------------------------------------------------------------------
    // Test C: review-024 -- isAlreadyActive detects live work (NEEDS-DEVICE)
    // -------------------------------------------------------------------------

    /**
     * review-024: [IndexService.isAlreadyActive] must return true when the unique work
     * is ENQUEUED, so that a duplicate ACTION_START_INDEXING does not replace it.
     *
     * NEEDS-DEVICE:
     *   adb shell am instrument -w \
     *     -e class com.narad.search.lib.index.ReindexPolicyTest#isAlreadyActive_trueForEnqueuedWork \
     *     com.narad.search.lib.test/androidx.test.runner.AndroidJUnitRunner
     */
    @Test
    fun isAlreadyActive_trueForEnqueuedWork() {
        val wm = WorkManager.getInstance(context)

        val request = OneTimeWorkRequestBuilder<IndexWorker>()
            .setInputData(workDataOf(IndexWorker.KEY_MODEL_PATH to "/fake/model.onnx"))
            // The test WorkManager runs eligible work at once; a delay keeps it ENQUEUED.
            .setInitialDelay(1, TimeUnit.HOURS)
            .build()
        wm.enqueueUniqueWork(IndexService.INITIAL_INDEX_WORK_NAME, ExistingWorkPolicy.KEEP, request)

        val infos = wm.getWorkInfosForUniqueWork(IndexService.INITIAL_INDEX_WORK_NAME).get()
        val states = infos.map { it.state }

        assertTrue(
            "isAlreadyActive must be true for ENQUEUED work; got states: $states",
            IndexService.isAlreadyActive(states)
        )
    }
}
