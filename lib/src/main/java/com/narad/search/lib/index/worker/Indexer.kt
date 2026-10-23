package com.narad.search.lib.index.worker

import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository

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
 *  7. Every [checkpointEvery] files: call [onProgress] with (processed, total).
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
    private val shouldStop: () -> Boolean = { false }
) {

    /**
     * Index all files in [filePaths].
     *
     * Files that have not changed since last indexing are skipped.
     * Progress is reported every [checkpointEvery] files and once at the end.
     *
     * @return true when every file was visited; false when [shouldStop] ended the pass early.
     *         The prune (U8) must only run after a true.
     */
    fun indexFiles(filePaths: List<String>): Boolean {
        val total = filePaths.size
        var processed = 0
        var finished = true

        for (filePath in filePaths) {
            if (shouldStop()) { finished = false; break }
            val lastModified = lastModifiedOf(filePath)

            // Dedup: skip if not changed
            if (!repository.fileNeedsIndex(filePath, lastModified)) {
                processed++
                maybeCheckpoint(processed, total)
                continue
            }

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
                processed++
                maybeCheckpoint(processed, total)
                continue
            }

            // Chunk and embed
            val chunks = chunkText(text)
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

            processed++
            maybeCheckpoint(processed, total)
        }

        // Always emit a final progress event if last batch did not land on checkpoint boundary
        if (total > 0 && processed % checkpointEvery != 0) {
            onProgress(processed, total)
        }
        return finished
    }

    private fun maybeCheckpoint(processed: Int, total: Int) {
        if (processed > 0 && processed % checkpointEvery == 0) {
            onProgress(processed, total)
        }
    }
}
