package com.narad.search.lib.index.worker

import androidx.work.Data

/**
 * Lifecycle stages emitted by [IndexWorker] via [androidx.work.ListenableWorker.setProgress]
 * (key: [KEY_STAGE]).
 *
 * Published at every state transition so the status chip (U20), the indexing screen (U26),
 * and the event log (U27) can render the current phase without polling
 * [androidx.work.WorkInfo.State].
 *
 * Serialized into [androidx.work.Data] via [toData]; deserialized via [fromData].
 *
 * The [Reading] variant also writes the legacy [KEY_PROGRESS_PROCESSED] and
 * [KEY_PROGRESS_TOTAL] keys so the existing chip in [SearchScreenContent] keeps working
 * until W17/U20 migrates it to [fromData].
 */
sealed class IndexStage {

    /** Model files absent or still downloading. [percent] is 0-100 or null if unknown. */
    data class WaitingForModel(val percent: Int? = null) : IndexStage()

    /** Storage (or notification) permission has not been granted yet. */
    data class WaitingForPermission(val which: String) : IndexStage()

    /** MediaStore enumeration is in progress before any file count is known. */
    data class Finding(val kind: String) : IndexStage()

    /**
     * Files are being read and embedded.
     *
     * [done] and [total] are the file counts for the current [kind] pass.
     */
    data class Reading(val kind: String, val done: Int, val total: Int) : IndexStage()

    /**
     * Run completed successfully.
     *
     * [count] = total files indexed across all passes.
     * [reason] is set when [count] == 0 (e.g. "Nothing to read yet").
     */
    data class Done(val count: Int, val reason: String? = null) : IndexStage()

    /** Run completed with a permanent failure (model corrupt, I/O error). */
    data class Failed(val reason: String) : IndexStage()

    /** Worker was stopped mid-run ([androidx.work.ListenableWorker.isStopped]). */
    object Paused : IndexStage()

    // -------------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------------

    fun toData(): Data {
        val b = Data.Builder()
        b.putString(KEY_STAGE, typeName())
        when (this) {
            is WaitingForModel -> percent?.let { b.putInt(KEY_PERCENT, it) }
            is WaitingForPermission -> b.putString(KEY_WHICH, which)
            is Finding -> b.putString(KEY_KIND, kind)
            is Reading -> {
                b.putString(KEY_KIND, kind)
                b.putInt(KEY_DONE, done)
                b.putInt(KEY_TOTAL, total)
                // ponytail: backward-compat keys for SearchScreenContent chip until U20 migrates
                b.putInt(KEY_PROGRESS_PROCESSED_COMPAT, done)
                b.putInt(KEY_PROGRESS_TOTAL_COMPAT, total)
            }
            is Done -> {
                b.putInt(KEY_COUNT, count)
                reason?.let { b.putString(KEY_REASON, it) }
            }
            is Failed -> b.putString(KEY_REASON, reason)
            Paused -> { /* no extra keys */ }
        }
        return b.build()
    }

    private fun typeName(): String = when (this) {
        is WaitingForModel    -> TYPE_WAITING_FOR_MODEL
        is WaitingForPermission -> TYPE_WAITING_FOR_PERMISSION
        is Finding            -> TYPE_FINDING
        is Reading            -> TYPE_READING
        is Done               -> TYPE_DONE
        is Failed             -> TYPE_FAILED
        Paused                -> TYPE_PAUSED
    }

    companion object {

        /** Primary key - put in every [Data] payload produced by [toData]. */
        const val KEY_STAGE = "stage"

        private const val KEY_PERCENT = "stage_percent"
        private const val KEY_WHICH   = "stage_which"
        private const val KEY_KIND    = "stage_kind"
        private const val KEY_DONE    = "stage_done"
        private const val KEY_TOTAL   = "stage_total"
        private const val KEY_COUNT   = "stage_count"
        private const val KEY_REASON  = "stage_reason"

        // Backward-compat keys matched to IndexWorker.KEY_PROGRESS_* values
        private const val KEY_PROGRESS_PROCESSED_COMPAT = "processed"
        private const val KEY_PROGRESS_TOTAL_COMPAT     = "total"

        private const val NO_INT = Int.MIN_VALUE

        // Discriminator string constants
        private const val TYPE_WAITING_FOR_MODEL      = "WaitingForModel"
        private const val TYPE_WAITING_FOR_PERMISSION = "WaitingForPermission"
        private const val TYPE_FINDING                = "Finding"
        private const val TYPE_READING                = "Reading"
        private const val TYPE_DONE                   = "Done"
        private const val TYPE_FAILED                 = "Failed"
        private const val TYPE_PAUSED                 = "Paused"

        /**
         * Deserialize an [IndexStage] from a [Data] payload, or return null if the
         * [KEY_STAGE] key is absent or unrecognized.
         */
        fun fromData(data: Data): IndexStage? = when (data.getString(KEY_STAGE)) {
            TYPE_WAITING_FOR_MODEL ->
                WaitingForModel(
                    percent = data.getInt(KEY_PERCENT, NO_INT).let { if (it == NO_INT) null else it }
                )
            TYPE_WAITING_FOR_PERMISSION ->
                WaitingForPermission(which = data.getString(KEY_WHICH) ?: "")
            TYPE_FINDING ->
                Finding(kind = data.getString(KEY_KIND) ?: "")
            TYPE_READING ->
                Reading(
                    kind  = data.getString(KEY_KIND) ?: "",
                    done  = data.getInt(KEY_DONE, 0),
                    total = data.getInt(KEY_TOTAL, 0)
                )
            TYPE_DONE ->
                Done(
                    count  = data.getInt(KEY_COUNT, 0),
                    reason = data.getString(KEY_REASON)
                )
            TYPE_FAILED ->
                Failed(reason = data.getString(KEY_REASON) ?: "")
            TYPE_PAUSED -> Paused
            else -> null
        }
    }
}
