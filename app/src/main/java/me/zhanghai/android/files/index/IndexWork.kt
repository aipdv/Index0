package me.zhanghai.android.files.index

import androidx.work.WorkInfo
import com.narad.search.lib.index.worker.IndexStage

// Pure helpers copied from narad StatusChip.kt (the Compose parts stay in Narad).

/**
 * Pure: is indexing under way?
 *
 * [userWork] is the initial build and the app-start delta. Both are one-time and run at once,
 * so any unfinished state counts (ENQUEUED waiting for model or retry, RUNNING, BLOCKED).
 * [backgroundWork] is the hourly periodic delta and the MediaStore delta. Both carry idle or
 * charging constraints, and the periodic one sits ENQUEUED between runs almost all the time,
 * so only RUNNING counts there. Otherwise the empty state could never show.
 */
internal fun isIndexingActive(userWork: List<WorkInfo>, backgroundWork: List<WorkInfo>): Boolean =
    userWork.any { !it.state.isFinished } || backgroundWork.any { it.state == WorkInfo.State.RUNNING }

/**
 * Pure: the RUNNING job to show. At first index two jobs run at once: the initial build waits
 * on the index lock with no progress, the foreground delta does the real Reading. Prefer the
 * one that reports a stage, else the first RUNNING.
 */
internal fun pickRunningWork(infos: List<WorkInfo>): WorkInfo? =
    infos.filter { it.state == WorkInfo.State.RUNNING }.let { running ->
        running.firstOrNull { IndexStage.fromData(it.progress) != null } ?: running.firstOrNull()
    }
