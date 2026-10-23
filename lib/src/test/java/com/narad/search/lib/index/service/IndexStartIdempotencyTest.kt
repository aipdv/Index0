package com.narad.search.lib.index.service

import androidx.work.WorkInfo
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM unit tests for [IndexService.isAlreadyActive] (review-024).
 *
 * The function is the idempotency gate in [IndexService.startIndexing]: if initial-index
 * work is already ENQUEUED or RUNNING, the ordinary start path observes the existing job
 * rather than re-enqueueing with REPLACE (which would restart it).
 *
 * Mutation check (paste in report): change [isAlreadyActive] body to `return false`.
 * The `isAlreadyActive true for ENQUEUED` and `isAlreadyActive true for RUNNING` tests fail.
 *
 * Test gate: :lib:testDebugUnitTest --tests '*.IndexStartIdempotencyTest'
 */
class IndexStartIdempotencyTest {

    @Test
    fun `isAlreadyActive true for ENQUEUED`() {
        // mutation: change body to return false -> this test fails
        assertTrue(IndexService.isAlreadyActive(listOf(WorkInfo.State.ENQUEUED)))
    }

    @Test
    fun `isAlreadyActive true for RUNNING`() {
        assertTrue(IndexService.isAlreadyActive(listOf(WorkInfo.State.RUNNING)))
    }

    @Test
    fun `isAlreadyActive true when list has both ENQUEUED and RUNNING`() {
        assertTrue(
            IndexService.isAlreadyActive(
                listOf(WorkInfo.State.ENQUEUED, WorkInfo.State.RUNNING)
            )
        )
    }

    @Test
    fun `isAlreadyActive false for SUCCEEDED`() {
        assertFalse(IndexService.isAlreadyActive(listOf(WorkInfo.State.SUCCEEDED)))
    }

    @Test
    fun `isAlreadyActive false for FAILED`() {
        assertFalse(IndexService.isAlreadyActive(listOf(WorkInfo.State.FAILED)))
    }

    @Test
    fun `isAlreadyActive false for CANCELLED`() {
        assertFalse(IndexService.isAlreadyActive(listOf(WorkInfo.State.CANCELLED)))
    }

    @Test
    fun `isAlreadyActive false for empty list`() {
        assertFalse(IndexService.isAlreadyActive(emptyList()))
    }

    @Test
    fun `isAlreadyActive false when only terminal states present`() {
        assertFalse(
            IndexService.isAlreadyActive(
                listOf(WorkInfo.State.SUCCEEDED, WorkInfo.State.FAILED, WorkInfo.State.CANCELLED)
            )
        )
    }

    @Test
    fun `reindex requires model and at least one readable source`() {
        assertFalse(IndexService.canReindex(modelReady = false, documentAccess = true, photoAccess = true))
        assertFalse(IndexService.canReindex(modelReady = true, documentAccess = false, photoAccess = false))
        assertTrue(IndexService.canReindex(modelReady = true, documentAccess = true, photoAccess = false))
        assertTrue(IndexService.canReindex(modelReady = true, documentAccess = false, photoAccess = true))
    }
}
