package com.narad.search.lib.index.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import androidx.core.app.NotificationCompat
import com.narad.search.lib.R
import com.narad.search.lib.index.worker.IndexStage

/**
 * Builds the foreground-service progress notification for [IndexService].
 *
 * Creates the notification channel on API 26+ (required since Oreo).
 */
object IndexNotifications {

    const val CHANNEL_ID = "narad_indexing"
    const val NOTIFICATION_ID = 1001

    fun createChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Search indexing",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows search index progress"
                setShowBadge(false)
            }
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }

    /**
     * Build a notification for the given [IndexStage].
     *
     * Each stage produces human-readable text for the status bar. Indeterminate
     * progress is shown when no file count is available yet.
     */
    fun buildStageNotification(context: Context, stage: IndexStage): Notification {
        val (text, processed, total) = when (stage) {
            is IndexStage.WaitingForModel -> {
                val pct = stage.percent
                val t = if (pct != null) "Waiting for the search model ($pct%)" else "Waiting for the search model"
                Triple(t, 0, 0)
            }
            is IndexStage.WaitingForPermission -> Triple("Storage access needed", 0, 0)
            is IndexStage.Finding -> Triple("Finding files...", 0, 0)
            is IndexStage.Reading -> Triple("${stage.done} of ${stage.total} files read", stage.done, stage.total)
            is IndexStage.Done -> Triple("Index up to date", 0, 0)
            is IndexStage.Failed -> Triple("Index failed", 0, 0)
            IndexStage.Paused -> Triple("Index paused", 0, 0)
        }
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle("Building search index")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_stat_narad)
            .setOngoing(true)
            .setProgress(total, processed, total == 0)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    /**
     * Build a progress notification (kept for backward compatibility).
     *
     * @param context     Android context.
     * @param processed   Files processed so far.
     * @param total       Total files to process (0 = indeterminate progress).
     */
    fun buildProgressNotification(
        context: Context,
        processed: Int = 0,
        total: Int = 0
    ): Notification = buildStageNotification(
        context,
        if (total > 0) IndexStage.Reading("documents", processed, total)
        else IndexStage.Finding("documents")
    )
}
