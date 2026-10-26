package com.narad.search.lib.index.worker

import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository

// ponytail: head of the file only (~14 KB of text at 512-char chunks): the first pages are enough
// to find a file, and a 32-chunk cap keeps the first index fast. Text past it is not searchable.
// Upgrade path: deepen later (a second pass over long files, or chunks spread across the file).
internal const val MAX_CHUNKS_PER_FILE = 32

/**
 * When to publish progress: at every [everyFiles]-th file, or when [everyMs] passed since the
 * last publish. A count-only tick left "0 of 331" on screen for 2.5 min (stress test, 2026-10-25).
 */
internal class ProgressTicker(
    private val everyFiles: Int = 50,
    private val everyMs: Long = 1_000,
    private val nowMs: () -> Long = { System.nanoTime() / 1_000_000 },
) {
    private var lastCount = 0
    private var lastAt = nowMs()

    fun due(count: Int): Boolean {
        val now = nowMs()
        if (count <= 0 || (count % everyFiles != 0 && now - lastAt < everyMs)) return false
        lastCount = count
        lastAt = now
        return true
    }

    /** True when [count] was not published yet (the final call). */
    fun pending(count: Int): Boolean = count != lastCount
}

/**
 * Plain, injectable, testable indexing pipeline.
 *
 * Dependencies are constructor-injected so tests can substitute a [FakeEmbedder]
 * and a simple file-read lambda without an Android runtime.
 *
 * Algorithm (per file):
 *  1. Check [repository.fileNeedsIndex] - skip if unchanged (dedup/idempotency).
 *  2. Extract text via [extractText].
 *  3. Chunk via [chunkText].
 *  4. Embed each chunk via [embedder].
 *  5. Fix item 6: call [repository.upsertChunksForFile] which deletes stale orphan
 *     chunks BEFORE writing the new set, then writes all chunks in one transaction
 *     (fix item 11: transaction batching).
 *  6. After all chunks for a file: [repository.markFileIndexed].
 *  7. Every [checkpointEvery] files or 1 s ([ProgressTicker]): call [onProgress] with (processed, total).
 *     A final progress call is always emitted after the last file.
 *
 * @param embedder          Embedding seam ([OnnxEmbedder] in production; [FakeEmbedder] in tests).
 * @param extractText       Lambda: filePath → extracted text (or null to skip).
 * @param chunkText         Lambda: text → list of chunk strings.
 * @param repository        Durable index storage.
 * @param checkpointEvery   Emit a progress callback every N files (default 50; Android 14 KTD-8).
 * @param onProgress        Called with (filesProcessedSoFar, totalFiles) at each checkpoint.
 * @param lastModifiedOf    Lambda: filePath → last-modified timestamp used for dedup.
 *                          Default reads [java.io.File.lastModified] (millis) - preserves
 *                          existing behaviour for the File-based path.
 *                          When MediaStore URIs are indexed, the caller passes a lambda that
 *                          returns MediaStore DATE_MODIFIED (seconds) instead, keeping dedup
 *                          per-key consistent without unit-mixing.
 */
class Indexer(
    private val embedder: Embedder,
    private val extractText: (filePath: String) -> String?,
    private val chunkText: (text: String) -> List<String>,
    private val repository: IndexRepository,
    private val checkpointEvery: Int = 50,
    private val onProgress: (processed: Int, total: Int) -> Unit = { _, _ -> },
    private val lastModifiedOf: (filePath: String) -> Long = { java.io.File(it).lastModified() },
    // Checked before each file. Each file is persisted on its own, so stopping between
    // files is safe and the next run resumes via fileNeedsIndex().
    private val shouldStop: () -> Boolean = { false },
    // Called after each file that was read (not for unchanged files): path, ms, chunks written.
    private val onFileDone: (filePath: String, ms: Long, chunks: Int) -> Unit = { _, _, _ -> }
) {

    /**
     * Index all files in [filePaths].
     *
     * Files that have not changed since last indexing are skipped.
     * Progress is reported every [checkpointEvery] files or 1 s, and once at the end.
     *
     * @return true when every file was visited; false when [shouldStop] ended the pass early.
     *         The prune (U8) must only run after a true.
     */
    fun indexFiles(filePaths: List<String>): Boolean {
        val total = filePaths.size
        var processed = 0
        var finished = true
        val ticker = ProgressTicker(checkpointEvery)
        fun maybeCheckpoint() {
            if (ticker.due(processed)) onProgress(processed, total)
        }

        for (filePath in filePaths) {
            if (shouldStop()) { finished = false; break }
            val lastModified = lastModifiedOf(filePath)

            // Dedup: skip if not changed
            if (!repository.fileNeedsIndex(filePath, lastModified)) {
                processed++
                maybeCheckpoint()
                continue
            }

            val fileStart = System.nanoTime()
            fun fileDone(chunks: Int) = onFileDone(filePath, (System.nanoTime() - fileStart) / 1_000_000, chunks)

            // Extract text
            val text = extractText(filePath)
            if (text.isNullOrBlank()) {
                // Mark as indexed even if no text (avoids repeated attempts on binary files).
                // Fix item 6: delete any stale chunks for this file before marking.
                repository.deleteChunksForFile(filePath)
                repository.markFileIndexed(
                    filePath = filePath,
                    mimeType = null,
                    lastModified = lastModified,
                    chunkCount = 0
                )
                fileDone(0)
                processed++
                maybeCheckpoint()
                continue
            }

            // Chunk and embed. Only the first MAX_CHUNKS_PER_FILE chunks: a 1 MB log or a long
            // DOCX made ~2,300 chunks and held the whole run for minutes (stress test, 2026-10-25).
            val chunks = chunkText(text).take(MAX_CHUNKS_PER_FILE)
            val chunkData = mutableListOf<Triple<Int, FloatArray, String>>()
            for ((chunkIndex, chunkContent) in chunks.withIndex()) {
                // Large files: stop mid-file too. The file stays unmarked, so the next run redoes it.
                if (shouldStop()) return false
                val vector = embedder.embed(chunkContent)
                chunkData.add(Triple(chunkIndex, vector, chunkContent))
            }

            // Fix items 6 + 11: upsertChunksForFile deletes orphan chunks first,
            // then writes all chunks in a single transaction.
            repository.upsertChunksForFile(
                filePath = filePath,
                chunks = chunkData,
                modelId = IndexDatabase.MODEL_ID,
                dim = IndexDatabase.DIM_VALUE
            )

            repository.markFileIndexed(
                filePath = filePath,
                mimeType = null,
                lastModified = lastModified,
                chunkCount = chunkData.size
            )
            fileDone(chunkData.size)

            processed++
            maybeCheckpoint()
        }

        // Always emit a final progress event if the last file was not published yet
        if (total > 0 && ticker.pending(processed)) {
            onProgress(processed, total)
        }
        return finished
    }
}
