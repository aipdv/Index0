package com.narad.search.lib.index.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.narad.search.lib.index.worker.IndexWorker
import com.narad.search.lib.index.worker.modelFilesReady
import java.util.concurrent.TimeUnit

/**
 * [BroadcastReceiver] for [Intent.ACTION_BOOT_COMPLETED].
 *
 * Re-enqueues the periodic delta [IndexWorker] after device reboot so the
 * index stays current without requiring the user to manually restart indexing.
 *
 * The periodic work runs hourly while the phone is idle and the battery is not low
 * (overnight indexing, with or without charging). See [overnightConstraints].
 */
class BootReceiver : BroadcastReceiver() {

    companion object {
        const val DELTA_WORK_NAME = "narad_periodic_delta_index"

        /**
         * One-time foreground-delta work name.
         *
         * Enqueued on every [Activity.onActivityStarted] so the index catches up
         * promptly when the user opens the app, without waiting for the overnight
         * idle window. KEEP policy: if a run is already queued or running, the
         * second enqueue is a no-op.
         */
        const val FOREGROUND_DELTA_WORK_NAME = "narad_foreground_delta"

        /**
         * Runs [enqueue] only when the text model is on disk and the first full index has
         * finished ([IndexService.isFirstIndexDone]).
         *
         * The first full index is the user-started foreground run; a delta that fired earlier
         * would take INDEX_RUN_LOCK and do that first read under background limits (W29).
         *
         * The delta jobs only refresh an existing index. Before the user accepts Gemma
         * Terms there is no model, and a worker would wait for it and backoff-retry forever.
         * Every delta enqueue (periodic, foreground, [MediaStoreObserver]) routes through here.
         */
        internal inline fun enqueueIfModelReady(
            modelPath: String,
            firstIndexDone: Boolean,
            enqueue: (String) -> Unit,
        ) {
            if (modelFilesReady(modelPath) && firstIndexDone) enqueue(modelPath)
        }

        // Index0: user toggle for the overnight job, stored with the other engine flags.
        private const val LIB_PREFS = "narad_lib_prefs"
        private const val KEY_OVERNIGHT_ENABLED = "overnight_enabled"

        fun isOvernightEnabled(context: Context): Boolean =
            context.getSharedPreferences(LIB_PREFS, Context.MODE_PRIVATE)
                .getBoolean(KEY_OVERNIGHT_ENABLED, true)

        fun setOvernightEnabled(context: Context, enabled: Boolean) {
            context.getSharedPreferences(LIB_PREFS, Context.MODE_PRIVATE).edit()
                .putBoolean(KEY_OVERNIGHT_ENABLED, enabled).apply()
            if (enabled) {
                enqueuePeriodicDeltaWork(context)
            } else {
                WorkManager.getInstance(context).cancelUniqueWork(DELTA_WORK_NAME)
            }
        }

        /**
         * Index0: the overnight toggle guard. Start, every finished run and boot all enqueue
         * the periodic job through [enqueuePeriodicDeltaWork], so this one check holds for all.
         */
        internal inline fun enqueuePeriodicIfAllowed(
            overnightEnabled: Boolean,
            modelPath: String,
            firstIndexDone: Boolean,
            enqueue: (String) -> Unit,
        ) {
            if (overnightEnabled) enqueueIfModelReady(modelPath, firstIndexDone, enqueue)
        }

        /**
         * Enqueue (or replace) the periodic delta work.
         *
         * Exposed as a static method so both [BootReceiver.onReceive] and
         * [IndexService] can call it without duplicating logic.
         */
        /** Shared by the periodic job and [MediaStoreObserver] delta jobs. */
        fun overnightConstraints(): Constraints = Constraints.Builder()
            .setRequiresDeviceIdle(true)
            .setRequiresBatteryNotLow(true)
            .build()

        fun enqueuePeriodicDeltaWork(context: Context) {
            // Overnight indexing, with or without charging: run while the phone sits idle
            // (screen off, unused) and the battery is not low. Android stops each run after
            // ~10 min or when the phone wakes; the hourly period lets the next idle window
            // resume where the last run stopped (per-file persistence + fileNeedsIndex).
            val constraints = overnightConstraints()

            // Default model path, the same path used by IndexService
            enqueuePeriodicIfAllowed(
                isOvernightEnabled(context),
                IndexService.getDefaultModelPath(context),
                IndexService.isFirstIndexDone(context)
            ) { modelPath ->
                val request = PeriodicWorkRequestBuilder<IndexWorker>(1, TimeUnit.HOURS)
                    .setConstraints(constraints)
                    .setInputData(workDataOf(IndexWorker.KEY_MODEL_PATH to modelPath))
                    .build()

                WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                    DELTA_WORK_NAME,
                    ExistingPeriodicWorkPolicy.KEEP,
                    request
                )
            }
        }

        /**
         * Enqueue a one-time foreground delta with no idle constraint.
         *
         * Called from [NaradApplication.onActivityStarted] so the index catches up
         * as soon as the user opens the app. KEEP policy: if a run is already in
         * progress or queued, this is a no-op (at most one foreground delta at a time).
         *
         * No idle/charging constraint - runs immediately in the foreground so the
         * user sees fresh results without waiting for an overnight idle window.
         */
        fun enqueueForegroundDeltaWork(context: Context) {
            enqueueIfModelReady(IndexService.getDefaultModelPath(context), IndexService.isFirstIndexDone(context)) { modelPath ->
                val request = OneTimeWorkRequestBuilder<IndexWorker>()
                    .setInputData(workDataOf(IndexWorker.KEY_MODEL_PATH to modelPath))
                    .build()
                WorkManager.getInstance(context).enqueueUniqueWork(
                    FOREGROUND_DELTA_WORK_NAME,
                    ExistingWorkPolicy.KEEP,
                    request
                )
            }
        }
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            enqueuePeriodicDeltaWork(context)
        }
    }
}
