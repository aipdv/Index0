package com.narad.search.lib.index

import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import android.content.Context
import androidx.work.Configuration
import androidx.work.CoroutineWorker
import androidx.work.ListenableWorker
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.WorkerFactory
import androidx.work.WorkerParameters
import androidx.work.testing.WorkManagerTestInitHelper
import com.narad.search.lib.index.service.BootReceiver
import com.narad.search.lib.index.service.IndexService
import com.narad.search.lib.index.worker.IndexWorker
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Test 6: Boot re-enqueue.
 *
 * Uses WorkManagerTestInitHelper so WorkManager tasks can be queried
 * synchronously. Calls BootReceiver.onReceive with ACTION_BOOT_COMPLETED
 * and asserts the periodic delta work is enqueued.
 */
@RunWith(AndroidJUnit4::class)
class BootReceiverTest {

    private val fakeModel = FakeModelFiles(InstrumentationRegistry.getInstrumentation().targetContext)

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        // The test WorkManager runs eligible work at once. A real IndexWorker would finish in a
        // later test and write first_index_done (E18), so IndexWorker is swapped for a no-op.
        WorkManagerTestInitHelper.initializeTestWorkManager(
            context,
            Configuration.Builder().setWorkerFactory(NoopIndexWorkerFactory).build()
        )
        // The delta enqueue is guarded on the model being on disk (E6).
        fakeModel.create()
        // ...and on the first full index having finished (W29).
        IndexService.setFirstIndexDone(context, true)
    }

    @After
    fun tearDown() {
        fakeModel.delete()
        IndexService.setFirstIndexDone(
            InstrumentationRegistry.getInstrumentation().targetContext, false
        )
    }

    @Test
    fun deltaWork_notEnqueuedBeforeFirstIndexDone() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        IndexService.setFirstIndexDone(context, false)
        val wm = WorkManager.getInstance(context)
        wm.cancelUniqueWork(BootReceiver.DELTA_WORK_NAME).result.get()
        wm.cancelUniqueWork(BootReceiver.FOREGROUND_DELTA_WORK_NAME).result.get()
        wm.pruneWork().result.get()

        BootReceiver.enqueueForegroundDeltaWork(context)
        BootReceiver.enqueuePeriodicDeltaWork(context)

        assertTrue(
            "foreground delta must not be enqueued before the first full index finished",
            wm.getWorkInfosForUniqueWork(BootReceiver.FOREGROUND_DELTA_WORK_NAME).get().isEmpty()
        )
        assertTrue(
            "periodic delta must not be enqueued before the first full index finished",
            wm.getWorkInfosForUniqueWork(BootReceiver.DELTA_WORK_NAME).get().isEmpty()
        )
    }

    @Test
    fun deltaWork_notEnqueuedWhenModelMissing() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        fakeModel.delete()
        val wm = WorkManager.getInstance(context)
        wm.cancelUniqueWork(BootReceiver.DELTA_WORK_NAME).result.get()
        wm.cancelUniqueWork(BootReceiver.FOREGROUND_DELTA_WORK_NAME).result.get()
        wm.pruneWork().result.get()
        // A real model already on the device would make the guard pass; fail loudly, not vacuously.
        assertFalse(
            "a real model is on this device, the missing-model case cannot be exercised",
            java.io.File(com.narad.search.lib.index.service.IndexService.getDefaultModelPath(context)).exists()
        )

        BootReceiver.enqueueForegroundDeltaWork(context)
        BootReceiver.enqueuePeriodicDeltaWork(context)

        assertTrue(
            "foreground delta must not be enqueued before the model exists",
            wm.getWorkInfosForUniqueWork(BootReceiver.FOREGROUND_DELTA_WORK_NAME).get().isEmpty()
        )
        assertTrue(
            "periodic delta must not be enqueued before the model exists",
            wm.getWorkInfosForUniqueWork(BootReceiver.DELTA_WORK_NAME).get().isEmpty()
        )
    }

    @Test
    fun test6_bootReceiver_enqueuesPeriodicDeltaWork() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Trigger the boot receiver
        val receiver = BootReceiver()
        receiver.onReceive(context, Intent(Intent.ACTION_BOOT_COMPLETED))

        // Verify the delta work was enqueued
        val workManager = WorkManager.getInstance(context)
        val workInfos = workManager
            .getWorkInfosForUniqueWork(BootReceiver.DELTA_WORK_NAME)
            .get()

        assertFalse(
            "BootReceiver should enqueue periodic delta work on boot",
            workInfos.isEmpty()
        )
        // The work should be ENQUEUED or RUNNING (not cancelled/failed)
        val allValidStates = workInfos.all { info ->
            info.state == WorkInfo.State.ENQUEUED ||
            info.state == WorkInfo.State.RUNNING ||
            info.state == WorkInfo.State.BLOCKED
        }
        // Fix item 9: use assertTrue (not assert) - JVM assert is disabled on ART;
        // JUnit's assertTrue always evaluates regardless of -ea flag.
        assertTrue(
            "All enqueued work should be in ENQUEUED/RUNNING/BLOCKED state, got: ${workInfos.map { it.state }}",
            allValidStates
        )
    }

    @Test
    fun periodicDeltaWork_runsOvernightWithoutCharging_hourly() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        BootReceiver.enqueuePeriodicDeltaWork(context)

        val info = WorkManager.getInstance(context)
            .getWorkInfosForUniqueWork(BootReceiver.DELTA_WORK_NAME).get().single()
        assertTrue("must require device idle", info.constraints.requiresDeviceIdle())
        assertTrue("must require battery not low", info.constraints.requiresBatteryNotLow())
        assertFalse("must NOT require charging", info.constraints.requiresCharging())
        assertEquals(
            "periodic catch-up must run hourly",
            java.util.concurrent.TimeUnit.HOURS.toMillis(1),
            info.periodicityInfo!!.repeatIntervalMillis
        )
    }

    @Test
    fun userInitiatedIndex_alsoSchedulesOvernightWork() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        WorkManager.getInstance(context).cancelUniqueWork(BootReceiver.DELTA_WORK_NAME).result.get()
        WorkManager.getInstance(context).pruneWork().result.get()

        com.narad.search.lib.index.service.IndexService.enqueueUserInitiatedIndex(context)

        val infos = WorkManager.getInstance(context)
            .getWorkInfosForUniqueWork(BootReceiver.DELTA_WORK_NAME).get()
        assertTrue(
            "starting an index must schedule the overnight job without waiting for a reboot",
            infos.any { it.state == WorkInfo.State.ENQUEUED }
        )
    }
}

/** Hands out [NoopWorker] for [IndexWorker] so no test leaves a live indexing run behind. */
private object NoopIndexWorkerFactory : WorkerFactory() {
    override fun createWorker(
        appContext: Context,
        workerClassName: String,
        workerParameters: WorkerParameters
    ): ListenableWorker? =
        if (workerClassName == IndexWorker::class.java.name) NoopWorker(appContext, workerParameters) else null
}

private class NoopWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {
    override suspend fun doWork(): Result = Result.success()
}
