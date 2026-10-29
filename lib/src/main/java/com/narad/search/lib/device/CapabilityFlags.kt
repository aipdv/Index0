package com.narad.search.lib.device

/**
 * Immutable capability flags for a device, derived from its hardware tier.
 *
 * Serialisation contract (pure Map<String,String> - NO Android imports):
 *   [toEntries]    - produces a flat String→String map suitable for any KV store
 *   [fromEntries]  - reconstructs a CapabilityFlags from that map
 *
 * The caller is responsible for persisting the map (e.g. to SharedPreferences via
 * a thin adapter in the Android layer).  Keeping this class Android-free allows
 * the full U6 test gate to run on the host JVM.
 *
 * Field semantics (G1/G4):
 *   tier                  - classified hardware tier (A / B / C)
 *   textModelId           - ONNX model identifier string
 *                             All tiers → "embeddinggemma-2" (M1: EmbeddingGemma 2 swap)
 *   imageEmbeddingEnabled - EG2 image search (Tier A and B only)
 *   npuAvailable          - NNAPI/QNN/NPU was successfully probed at detection time
 *   totalRamBytes         - physical RAM from ActivityManager.MemoryInfo.totalMem (Bug Watch: use physical, not available)
 *   fastSearchOnly        - FTS5-only query path, skip ANN vector search (Tier C only)
 *
 * NO Android imports below this comment.
 */
data class CapabilityFlags(
    val tier: Tier,
    val textModelId: String,
    val imageEmbeddingEnabled: Boolean,
    val npuAvailable: Boolean,
    val totalRamBytes: Long,
    val fastSearchOnly: Boolean
) {

    /**
     * Serialise to a flat [Map]<[String], [String]> so the value can be persisted
     * in any key-value store (SharedPreferences, DataStore, etc.).
     *
     * All values are encoded as plain strings - Booleans as "true"/"false", Long as
     * decimal string, Tier as its [Tier.name].
     */
    fun toEntries(): Map<String, String> = mapOf(
        KEY_TIER                   to tier.name,
        KEY_TEXT_MODEL_ID          to textModelId,
        KEY_IMAGE_EMBEDDING        to imageEmbeddingEnabled.toString(),
        KEY_NPU_AVAILABLE          to npuAvailable.toString(),
        KEY_TOTAL_RAM_BYTES        to totalRamBytes.toString(),
        KEY_FAST_SEARCH_ONLY       to fastSearchOnly.toString()
    )

    companion object {
        // Map keys - stable across serialisation versions
        internal const val KEY_TIER                = "tier"
        internal const val KEY_TEXT_MODEL_ID       = "textModelId"
        internal const val KEY_IMAGE_EMBEDDING     = "imageEmbeddingEnabled"
        internal const val KEY_NPU_AVAILABLE       = "npuAvailable"
        internal const val KEY_TOTAL_RAM_BYTES     = "totalRamBytes"
        internal const val KEY_FAST_SEARCH_ONLY    = "fastSearchOnly"

        /**
         * Reconstruct a [CapabilityFlags] from a map previously produced by [toEntries].
         *
         * @throws IllegalArgumentException if a required key is missing or a value cannot
         *   be parsed (indicates the persisted data is corrupt or from an incompatible version).
         */
        fun fromEntries(entries: Map<String, String>): CapabilityFlags {
            fun require(key: String): String =
                entries[key] ?: throw IllegalArgumentException(
                    "CapabilityFlags.fromEntries: missing required key '$key'"
                )

            return CapabilityFlags(
                tier                  = Tier.valueOf(require(KEY_TIER)),
                textModelId           = require(KEY_TEXT_MODEL_ID),
                imageEmbeddingEnabled = require(KEY_IMAGE_EMBEDDING).toBooleanStrict(),
                npuAvailable          = require(KEY_NPU_AVAILABLE).toBooleanStrict(),
                totalRamBytes         = require(KEY_TOTAL_RAM_BYTES).toLong(),
                fastSearchOnly        = require(KEY_FAST_SEARCH_ONLY).toBooleanStrict()
            )
        }
    }
}
