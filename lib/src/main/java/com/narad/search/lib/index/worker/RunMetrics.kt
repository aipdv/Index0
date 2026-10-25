package com.narad.search.lib.index.worker

import android.content.Context
import android.net.Uri
import android.os.BatteryManager
import android.os.Debug
import android.os.PowerManager
import android.provider.OpenableColumns
import android.util.Log
import java.io.File
import java.time.Instant

// ---- Run-level event text (on in every build, no file names) ---------------------------

/** Events kept by [com.narad.search.lib.index.store.IndexRepository.pruneEvents]. */
internal fun eventCap(metrics: Boolean): Int = if (metrics) METRICS_EVENT_CAP else 500

/** Events the metrics export reads (the UI list keeps its own 500). */
const val METRICS_EVENT_CAP = 5_000

internal fun runStartMessage(
    workerId: String,
    attempt: Int,
    foreground: Boolean,
    charging: Boolean?,
    batteryPct: Int?,
    idle: Boolean?,
): String = "start worker=$workerId attempt=$attempt fg=$foreground" +
    " charging=${charging ?: "-"} battery=${batteryPct ?: "-"} idle=${idle ?: "-"}"

internal fun gateMessage(availBytes: Long, thresholdBytes: Long, lowMemory: Boolean, totalBytes: Long, fits: Boolean): String =
    (if (fits) "gate" else "photos deferred: low memory") +
        " availMem=$availBytes threshold=$thresholdBytes lowMemory=$lowMemory totalMem=$totalBytes" +
        " decision=${if (fits) "run" else "defer"}"

/** [stopReason] is null when the run was not stopped or the API is below 31. */
internal fun runEndMessage(
    outcome: String,
    docsSeen: Int,
    docsTotal: Int,
    photosSeen: Int,
    photosTotal: Int,
    read: Int,
    ms: Long,
    stopReason: Int?,
): String = "end outcome=$outcome docs=$docsSeen/$docsTotal photos=$photosSeen/$photosTotal" +
    " read=$read skipped=${(docsSeen + photosSeen - read).coerceAtLeast(0)} ms=$ms stop=${stopReason ?: "-"}"

// ---- Per-file results ------------------------------------------------------------------

internal const val RESULT_INDEXED = "indexed"
internal const val RESULT_EMPTY = "empty"
internal const val RESULT_EXTRACT_FAIL = "extract_fail"
internal const val RESULT_DECODE_FAIL = "decode_fail"
internal const val RESULT_ENCODE_FAIL = "encode_fail"

/** Pure: a document's result from its chunk count and whether extraction warned. */
internal fun docResult(chunks: Int, warned: Boolean): String = when {
    chunks > 0 -> RESULT_INDEXED
    warned -> RESULT_EXTRACT_FAIL
    else -> RESULT_EMPTY
}

// ---- Metrics file (NARAD_METRICS builds only) ------------------------------------------

internal const val METRICS_FILE_CAP_BYTES = 10L * 1024 * 1024

const val METRICS_TSV_HEADER = "ts_utc\trun_id\tkind\tname\ttype\tsize_bytes\tms\tchunks\tresult\tpss_kb"

/** Append-only, outside the event prune and outside backups. */
fun metricsFile(context: Context): File = File(context.noBackupFilesDir, "metrics/index-metrics.tsv")

/** Pure: one TSV row. Null is "-"; tabs and line breaks become spaces; names keep the part after the last '/'. */
internal fun tsvRow(
    tsMs: Long,
    runId: String,
    kind: String,
    name: String? = null,
    type: String? = null,
    sizeBytes: Long? = null,
    ms: Long? = null,
    chunks: Int? = null,
    result: String? = null,
    pssKb: Long? = null,
): String = listOf(
    Instant.ofEpochMilli(tsMs).toString(), runId, kind, name?.substringAfterLast('/'), type,
    sizeBytes, ms, chunks, result, pssKb,
).joinToString("\t") { cell ->
    cell?.toString()?.replace(Regex("[\t\r\n]"), " ")?.ifEmpty { null } ?: "-"
}

/**
 * Writes one run's rows to [file]. At [capBytes] it stops writing and calls [onFull] once
 * (a file already at the cap from an earlier run calls nothing). Write errors are logged only.
 */
internal class RunMetrics(
    private val file: File,
    private val runId: String,
    private val capBytes: Long = METRICS_FILE_CAP_BYTES,
    private val now: () -> Long = System::currentTimeMillis,
    private val onFull: () -> Unit,
) {
    private var full = file.length() >= capBytes

    fun append(row: String) {
        if (full) return
        try {
            val header = if (file.length() == 0L) METRICS_TSV_HEADER + "\n" else ""
            val bytes = (header + row + "\n").toByteArray()
            if (file.length() + bytes.size > capBytes) {
                full = true
                onFull()
                return
            }
            file.parentFile?.mkdirs()
            file.appendBytes(bytes)
        } catch (e: Exception) {
            Log.w("RunMetrics", "Failed to write metrics row", e)
        }
    }

    fun pss(point: String) = append(tsvRow(now(), runId, "pss", name = point, pssKb = Debug.getPss()))

    fun file(name: String?, type: String?, sizeBytes: Long?, ms: Long, chunks: Int?, result: String) =
        append(tsvRow(now(), runId, "file", name, type, sizeBytes, ms, chunks, result))
}

/** Display name and size of a content URI (or a plain path). Nulls when unknown. */
internal fun nameAndSize(context: Context, target: String): Pair<String?, Long?> {
    if (!target.startsWith("content:")) {
        val f = File(target)
        return f.name to f.length().takeIf { f.exists() }
    }
    return try {
        context.contentResolver.query(
            Uri.parse(target), arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null
        )?.use { c ->
            if (!c.moveToFirst()) null else
                c.getString(0) to (if (c.isNull(1)) null else c.getLong(1))
        } ?: (null to null)
    } catch (e: Exception) {
        null to null
    }
}

/** Charging, battery percent and device idle. Nulls when a service is missing. */
internal data class DeviceState(val charging: Boolean?, val batteryPct: Int?, val idle: Boolean?)

internal fun deviceState(context: Context): DeviceState {
    val battery = context.getSystemService(BatteryManager::class.java)
    val power = context.getSystemService(PowerManager::class.java)
    return DeviceState(
        charging = battery?.isCharging,
        batteryPct = battery?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)?.takeIf { it in 0..100 },
        idle = power?.isDeviceIdleMode,
    )
}
