package com.narad.search.lib.query.store

import com.narad.search.lib.query.engine.VectorResult

/**
 * In-memory vector store with brute-force cosine KNN.
 *
 * Vectors must be **L2-normalised** before insertion. Under that invariant,
 * cosine similarity reduces to a plain dot product, which avoids a second
 * normalisation on every query.
 *
 * U1 (KTD3): id-to-slot HashMap + one contiguous FloatArray.
 *   - add() is O(1) amortised (replaces O(N) indexOfFirst scan).
 *   - search() dot-product loop walks a single flat FloatArray for cache locality.
 *   - Capacity doubles on overflow; metadata stays in parallel ArrayLists.
 *
 * Thread safety: not thread-safe. The caller (QueryEngine) must synchronise
 * externally if needed.
 *
 * On-device persistence is deferred to U4 (Decision B2).
 */
class VectorStore {

    // ponytail: id-to-slot HashMap + flat FloatArray replaces mutableListOf<Entry>()
    // O(1) add vs O(N) indexOfFirst. The O(N) scan stays, but is now cache-local.
    private val idToSlot    = HashMap<String, Int>()
    private val slotIds     = ArrayList<String>()
    private val slotPaths   = ArrayList<String>()
    private val slotOffsets = ArrayList<Int>()
    private val slotModels  = ArrayList<String>()

    private var dim         = -1         // set on first add; all entries share one dim
    private var data        = FloatArray(0) // flat: slot s occupies [s*dim, (s+1)*dim)
    private var count       = 0          // slots used

    /**
     * Add (or update) an embedding record.
     *
     * If an entry with the same [id] already exists it is replaced, so
     * repeated indexing of the same chunk is idempotent.
     *
     * @param id          Unique chunk identifier (e.g. `"<filePath>#<offset>"`).
     * @param filePath    Absolute (or content-URI) path to the source file.
     * @param chunkOffset Byte or token offset of this chunk inside the file.
     * @param modelId     Model that produced the embedding (e.g. `"embeddinggemma-2"`).
     * @param vector      L2-normalised embedding. All entries in one store must share the same dim.
     */
    fun add(
        id: String,
        filePath: String,
        chunkOffset: Int,
        modelId: String,
        vector: FloatArray
    ) {
        if (dim == -1) {
            dim  = vector.size
            data = FloatArray(dim * 64) // initial capacity: 64 slots
        }
        require(vector.size == dim) { "Dimension mismatch: expected $dim, got ${vector.size}" }

        val existing = idToSlot[id]
        if (existing != null) {
            // Replace in-place: overwrite the vector region and all metadata.
            vector.copyInto(data, existing * dim)
            slotPaths[existing]   = filePath
            slotOffsets[existing] = chunkOffset
            slotModels[existing]  = modelId
        } else {
            // New entry: grow the flat array if needed, then append.
            val slot = count
            if (count == data.size / dim) {
                data = data.copyOf(maxOf(data.size * 2, dim * 64))
            }
            vector.copyInto(data, slot * dim)
            slotIds.add(id)
            slotPaths.add(filePath)
            slotOffsets.add(chunkOffset)
            slotModels.add(modelId)
            idToSlot[id] = slot
            count++
        }
    }

    /**
     * Return the [topK] entries most similar to [queryVector] by cosine
     * similarity (dot product of L2-normalised vectors), sorted descending.
     *
     * Returns an empty list when the store contains no entries.
     */
    fun search(queryVector: FloatArray, topK: Int): List<VectorResult> {
        if (count == 0) return emptyList()
        require(queryVector.size == dim) {
            "Query dimension ${queryVector.size} does not match store dimension $dim"
        }

        data class Scored(val slot: Int, val score: Float)

        val scored = ArrayList<Scored>(count)
        for (slot in 0 until count) {
            var s    = 0f
            val base = slot * dim
            for (i in 0 until dim) s += data[base + i] * queryVector[i]
            scored.add(Scored(slot, s))
        }
        scored.sortByDescending { it.score }

        return scored.take(topK).map { (slot, score) ->
            VectorResult(
                id          = slotIds[slot],
                filePath    = slotPaths[slot],
                chunkOffset = slotOffsets[slot],
                modelId     = slotModels[slot],
                score       = score
            )
        }
    }

    /** Number of vectors currently stored. */
    val size: Int get() = count

    /** Remove all entries. */
    fun clear() {
        idToSlot.clear()
        slotIds.clear()
        slotPaths.clear()
        slotOffsets.clear()
        slotModels.clear()
        dim   = -1
        data  = FloatArray(0)
        count = 0
    }
}
