package com.narad.search.lib.index.service

import android.content.Context
import android.database.ContentObserver
import android.os.Handler
import android.os.Looper
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.narad.search.lib.index.worker.IndexWorker

/**
 * [ContentObserver] that watches MediaStore for new / modified files and
 * enqueues a delta [IndexWorker] job with a 2-second debounce.
 *
 * The debounce prevents duplicate enqueues when the OS fires multiple
 * callbacks for a single file-save (Bug Watch: OS buffering).
 *
 * WorkManager constraints for the delta job: [BootReceiver.overnightConstraints]
 * (device idle + battery not low; charging not required).
 *
 * The debounce logic is isolated in [Debouncer] so it can be unit-tested
 * without a running Android ContentObserver.
 */
class MediaStoreObserver(
    private val context: Context,
    private val modelPath: String,
    debounceMs: Long = 2_000L
) : ContentObserver(Handler(Looper.getMainLooper())) {

    private val debouncer = Debouncer(debounceMs) { enqueueDeltaWork() }

    override fun onChange(selfChange: Boolean) {
        debouncer.trigger()
    }

    /**
     * Cancel any pending debounced work.
     *
     * Fix item 7: this method delegates to the real [Debouncer.cancel] so that
     * pending callbacks are cancelled when [IndexService.onDestroy] / stopIndexing
     * unregisters the observer. The previous implementation in IndexService was a
     * private no-op extension function that never reached this debouncer.
     */
    fun cancel() {
        debouncer.cancel()
    }

    private fun enqueueDeltaWork() = BootReceiver.enqueueIfModelReady(modelPath, IndexService.isFirstIndexDone(context)) {
        val request = OneTimeWorkRequestBuilder<IndexWorker>()
            .setConstraints(BootReceiver.overnightConstraints())
            .setInputData(workDataOf(IndexWorker.KEY_MODEL_PATH to modelPath))
            .build()

        WorkManager.getInstance(context).enqueueUniqueWork(
            DELTA_WORK_NAME,
            ExistingWorkPolicy.REPLACE,
            request
        )
    }

    companion object {
        const val DELTA_WORK_NAME = "narad_delta_index"
    }
}

/**
 * Pure, testable debounce helper.
 *
 * When [trigger] is called, any previously pending action is cancelled and
 * a new one is scheduled to fire after [delayMs] milliseconds.
 */
class Debouncer(
    private val delayMs: Long,
    private val action: () -> Unit
) {
    private val handler = Handler(Looper.getMainLooper())
    private val runnable = Runnable { action() }

    fun trigger() {
        handler.removeCallbacks(runnable)
        handler.postDelayed(runnable, delayMs)
    }

    fun cancel() {
        handler.removeCallbacks(runnable)
    }
}
