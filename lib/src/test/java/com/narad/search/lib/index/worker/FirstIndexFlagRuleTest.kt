package com.narad.search.lib.index.worker

import androidx.work.ListenableWorker.Result
import androidx.work.workDataOf
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** E19: the first-index flag is set only by a Done or WaitingForPermission success. */
class FirstIndexFlagRuleTest {

    @Test fun `Done success sets the flag`() {
        assertTrue(shouldMarkFirstIndexDone(Result.success(IndexStage.Done(count = 12).toData())))
    }

    @Test fun `WaitingForPermission success sets the flag`() {
        val out = IndexStage.WaitingForPermission("android.permission.READ_MEDIA_IMAGES").toData()
        assertTrue(shouldMarkFirstIndexDone(Result.success(out)))
    }

    @Test fun `success with no stage (the Paused return) does not set the flag`() {
        val paused = Result.success(
            workDataOf(IndexWorker.KEY_STARTED_AT to 1L, IndexWorker.KEY_FINISHED_AT to 2L)
        )
        assertFalse(shouldMarkFirstIndexDone(paused))
    }

    @Test fun `success carrying a Paused stage does not set the flag`() {
        assertFalse(shouldMarkFirstIndexDone(Result.success(IndexStage.Paused.toData())))
    }

    @Test fun `retry and failure do not set the flag`() {
        assertFalse(shouldMarkFirstIndexDone(Result.retry()))
        assertFalse(shouldMarkFirstIndexDone(Result.failure()))
        assertFalse(shouldMarkFirstIndexDone(Result.failure(IndexStage.Done(count = 1).toData())))
    }
}
