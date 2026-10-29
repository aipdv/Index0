package com.narad.search.lib.device

/**
 * Pure-function device tier classifier.
 *
 * Implements G4 (authoritative - overrides the original plan):
 *   minSdk floor  = 29 (Android 10)
 *   Physical RAM  = ActivityManager.MemoryInfo.totalMem  (Bug Watch: physical, not available)
 *
 *   Tier A: totalRamBytes ≥ 3.4 GiB  AND  npuAvailable == true
 *   Tier B: totalRamBytes ≥ 3.4 GiB  AND  npuAvailable == false
 *   Tier C: totalRamBytes <  3.4 GiB  (covers 2–3.4 GiB and degraded <2 GB - must not crash)
 *
 * NO Android imports - this file must remain host-JVM-testable.
 *
 * The [classify] and [capabilitiesFor] functions are stateless pure functions:
 * same inputs always produce the same output (idempotent by design, no mutable state).
 */
object TierClassifier {

    /**
     * Physical RAM threshold in bytes that gates Tier A / Tier B from Tier C.
     *
     * 3.4 GiB, not 4 GiB: an advertised "4 GB" phone reports about 3.6 to 3.8 GiB in
     * totalMem (the kernel and GPU reserve the rest), so a 4 GiB cut-off would put every
     * real 4 GB phone in Tier C and switch photo search off.
     */
    private val MIN_PHOTO_RAM: Long = 34L * 1024L * 1024L * 1024L / 10L

    /**
     * Classify a device into a [Tier] given its hardware characteristics.
     *
     * @param totalRamBytes  Physical total RAM in bytes (from ActivityManager.MemoryInfo.totalMem).
     *                       Zero or negative values are treated as below the threshold (Tier C), never crash.
     * @param npuAvailable   True if an NPU / NNAPI delegate was successfully probed (G4).
     * @param apiLevel       Build.VERSION.SDK_INT. Values below 29 are out of scope (G4 floor)
     *                       but the function still returns Tier C gracefully rather than throwing.
     * @return The detected [Tier].
     */
    fun classify(
        totalRamBytes: Long,
        npuAvailable: Boolean,
        @Suppress("UNUSED_PARAMETER") apiLevel: Int
    ): Tier = when {
        totalRamBytes >= MIN_PHOTO_RAM && npuAvailable -> Tier.A
        totalRamBytes >= MIN_PHOTO_RAM                -> Tier.B
        else                                    -> Tier.C  // <3.4 GiB, including <2 GB degraded
    }

    /**
     * Derive a full [CapabilityFlags] from a [Tier] and the raw hardware inputs.
     *
     * Model ID mapping (M1: all tiers use EmbeddingGemma 2):
     *   Tier A → "embeddinggemma-2"
     *   Tier B → "embeddinggemma-2"
     *   Tier C → "embeddinggemma-2"
     *
     * Image embedding (EG2 vision encoder, M2):
     *   Enabled on Tier A and B; disabled on Tier C.
     *
     * fastSearchOnly (R10):
     *   True on Tier C (FTS5-only, skip ANN vector search for latency).
     *
     * @param tier           The tier returned by [classify].
     * @param npuAvailable   Passed through for recording in the flags.
     * @param totalRamBytes  Passed through for recording in the flags.
     */
    fun capabilitiesFor(
        tier: Tier,
        npuAvailable: Boolean,
        totalRamBytes: Long
    ): CapabilityFlags = when (tier) {
        Tier.A -> CapabilityFlags(
            tier                  = Tier.A,
            textModelId           = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable          = npuAvailable,
            totalRamBytes         = totalRamBytes,
            fastSearchOnly        = false
        )
        Tier.B -> CapabilityFlags(
            tier                  = Tier.B,
            textModelId           = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable          = npuAvailable,
            totalRamBytes         = totalRamBytes,
            fastSearchOnly        = false
        )
        Tier.C -> CapabilityFlags(
            tier                  = Tier.C,
            textModelId           = "embeddinggemma-2",
            imageEmbeddingEnabled = false,
            npuAvailable          = npuAvailable,
            totalRamBytes         = totalRamBytes,
            fastSearchOnly        = true
        )
    }
}
