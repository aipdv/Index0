package com.narad.search.lib.device

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

/**
 * TierClassifierTest - U6 Test Gate (JVM-only, no Android device needed).
 *
 * Covers the G4-authoritative tier boundaries:
 *   Tier A: ≥3.4 GiB RAM AND npuAvailable=true, apiLevel ≥ 29
 *   Tier B: ≥3.4 GiB RAM, no NPU
 *   Tier C: <3.4 GiB RAM (including <2 GB degraded)
 *
 * The threshold is 3.4 GiB (U2), not 4 GiB: advertised 4 GB phones report ~3.6 to 3.8 GiB.
 */
class TierClassifierTest {

    private val GB = 1024L * 1024L * 1024L

    /** 3.4 GiB, the Tier A/B floor. */
    private val MIN_PHOTO_RAM = 34L * GB / 10L

    // An advertised "4 GB" phone reports about 3.6 to 3.8 GiB in totalMem (U2).
    @Test
    fun `advertised 4GB phone reporting 3_6GiB returns Tier B`() {
        assertEquals(Tier.B, TierClassifier.classify(36L * GB / 10L, npuAvailable = false, apiLevel = 33))
    }

    @Test
    fun `advertised 4GB phone reporting 3_8GiB with NPU returns Tier A`() {
        assertEquals(Tier.A, TierClassifier.classify(38L * GB / 10L, npuAvailable = true, apiLevel = 33))
    }

    @Test
    fun `exactly 3_4GiB returns Tier B`() {
        assertEquals(Tier.B, TierClassifier.classify(MIN_PHOTO_RAM, npuAvailable = false, apiLevel = 33))
    }

    @Test
    fun `3_0GiB returns Tier C`() {
        assertEquals(Tier.C, TierClassifier.classify(3L * GB, npuAvailable = false, apiLevel = 33))
    }

    // -------------------------------------------------------------------------
    // Tier A: ≥4 GB AND NPU available
    // -------------------------------------------------------------------------

    @Test
    fun `flagship 8GB plus NPU returns Tier A`() {
        val tier = TierClassifier.classify(
            totalRamBytes = 8L * GB,
            npuAvailable = true,
            apiLevel = 33
        )
        assertEquals(Tier.A, tier)
    }

    @Test
    fun `exactly 4GB plus NPU returns Tier A`() {
        val tier = TierClassifier.classify(
            totalRamBytes = 4L * GB,
            npuAvailable = true,
            apiLevel = 31
        )
        assertEquals(Tier.A, tier)
    }

    @Test
    fun `4GB plus NPU at minSdk floor API 29 returns Tier A`() {
        val tier = TierClassifier.classify(
            totalRamBytes = 4L * GB,
            npuAvailable = true,
            apiLevel = 29
        )
        assertEquals(Tier.A, tier)
    }

    // -------------------------------------------------------------------------
    // Tier B: ≥4 GB RAM, no NPU
    // -------------------------------------------------------------------------

    @Test
    fun `exactly 4GB no NPU returns Tier B`() {
        val tier = TierClassifier.classify(
            totalRamBytes = 4L * GB,
            npuAvailable = false,
            apiLevel = 31
        )
        assertEquals(Tier.B, tier)
    }

    @Test
    fun `8GB no NPU returns Tier B`() {
        val tier = TierClassifier.classify(
            totalRamBytes = 8L * GB,
            npuAvailable = false,
            apiLevel = 30
        )
        assertEquals(Tier.B, tier)
    }

    @Test
    fun `6GB no NPU returns Tier B`() {
        val tier = TierClassifier.classify(
            totalRamBytes = 6L * GB,
            npuAvailable = false,
            apiLevel = 33
        )
        assertEquals(Tier.B, tier)
    }

    // -------------------------------------------------------------------------
    // Tier C: <4 GB RAM (covers 2–4 GB range and degraded <2 GB)
    // -------------------------------------------------------------------------

    @Test
    fun `just below 3_4GiB no NPU returns Tier C`() {
        val tier = TierClassifier.classify(
            totalRamBytes = MIN_PHOTO_RAM - 1L,
            npuAvailable = false,
            apiLevel = 30
        )
        assertEquals(Tier.C, tier)
    }

    @Test
    fun `just below 3_4GiB with NPU returns Tier C`() {
        // Having NPU but too little RAM → Tier C (Tier A requires BOTH conditions)
        val tier = TierClassifier.classify(
            totalRamBytes = MIN_PHOTO_RAM - 1L,
            npuAvailable = true,
            apiLevel = 30
        )
        assertEquals(Tier.C, tier)
    }

    @Test
    fun `exactly 2GB returns Tier C`() {
        val tier = TierClassifier.classify(
            totalRamBytes = 2L * GB,
            npuAvailable = false,
            apiLevel = 29
        )
        assertEquals(Tier.C, tier)
    }

    @Test
    fun `3GB mid-range no NPU returns Tier C`() {
        val tier = TierClassifier.classify(
            totalRamBytes = 3L * GB,
            npuAvailable = false,
            apiLevel = 30
        )
        assertEquals(Tier.C, tier)
    }

    @Test
    fun `degraded sub-2GB returns Tier C without crash`() {
        // G4: <2 GB is classified as C degraded - must not crash
        val tier = TierClassifier.classify(
            totalRamBytes = 1L * GB,
            npuAvailable = false,
            apiLevel = 29
        )
        assertEquals(Tier.C, tier)
    }

    @Test
    fun `zero RAM edge case returns Tier C without crash`() {
        val tier = TierClassifier.classify(
            totalRamBytes = 0L,
            npuAvailable = false,
            apiLevel = 29
        )
        assertEquals(Tier.C, tier)
    }

    // -------------------------------------------------------------------------
    // Idempotency: same inputs → same result
    // -------------------------------------------------------------------------

    @Test
    fun `classify is idempotent for Tier A inputs`() {
        val input = Triple(8L * GB, true, 33)
        val result1 = TierClassifier.classify(input.first, input.second, input.third)
        val result2 = TierClassifier.classify(input.first, input.second, input.third)
        assertEquals("classify must return identical result on repeated calls", result1, result2)
    }

    @Test
    fun `classify is idempotent for Tier B inputs`() {
        val result1 = TierClassifier.classify(6L * GB, false, 31)
        val result2 = TierClassifier.classify(6L * GB, false, 31)
        assertEquals(result1, result2)
    }

    @Test
    fun `classify is idempotent for Tier C inputs`() {
        val result1 = TierClassifier.classify(2L * GB, false, 29)
        val result2 = TierClassifier.classify(2L * GB, false, 29)
        assertEquals(result1, result2)
    }

    // -------------------------------------------------------------------------
    // capabilitiesFor: tier → model mapping (G1/G4)
    // -------------------------------------------------------------------------

    @Test
    fun `Tier A maps to EmbeddingGemma2 model`() {
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.A,
            npuAvailable = true,
            totalRamBytes = 8L * GB
        )
        assertEquals(
            "Tier A must use embeddinggemma-2 (M1: all tiers unified)",
            "embeddinggemma-2",
            flags.textModelId
        )
    }

    @Test
    fun `Tier A enables image embedding`() {
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.A,
            npuAvailable = true,
            totalRamBytes = 8L * GB
        )
        assertEquals(true, flags.imageEmbeddingEnabled)
    }

    @Test
    fun `Tier A does not set fastSearchOnly`() {
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.A,
            npuAvailable = true,
            totalRamBytes = 8L * GB
        )
        assertEquals(false, flags.fastSearchOnly)
    }

    @Test
    fun `Tier A sets npuAvailable true`() {
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.A,
            npuAvailable = true,
            totalRamBytes = 8L * GB
        )
        assertEquals(true, flags.npuAvailable)
    }

    @Test
    fun `Tier B maps to EmbeddingGemma2 model`() {
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.B,
            npuAvailable = false,
            totalRamBytes = 6L * GB
        )
        assertEquals(
            "Tier B must use embeddinggemma-2 (M1: all tiers unified)",
            "embeddinggemma-2",
            flags.textModelId
        )
    }

    @Test
    fun `Tier B enables image embedding`() {
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.B,
            npuAvailable = false,
            totalRamBytes = 6L * GB
        )
        assertEquals(true, flags.imageEmbeddingEnabled)
    }

    @Test
    fun `Tier B does not set fastSearchOnly`() {
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.B,
            npuAvailable = false,
            totalRamBytes = 6L * GB
        )
        assertEquals(false, flags.fastSearchOnly)
    }

    @Test
    fun `Tier C maps to EmbeddingGemma2 model`() {
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.C,
            npuAvailable = false,
            totalRamBytes = 2L * GB
        )
        assertEquals(
            "Tier C must use embeddinggemma-2 (M1: all tiers unified)",
            "embeddinggemma-2",
            flags.textModelId
        )
    }

    @Test
    fun `Tier C disables image embedding`() {
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.C,
            npuAvailable = false,
            totalRamBytes = 2L * GB
        )
        assertEquals(false, flags.imageEmbeddingEnabled)
    }

    @Test
    fun `Tier C sets fastSearchOnly true`() {
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.C,
            npuAvailable = false,
            totalRamBytes = 2L * GB
        )
        assertEquals(true, flags.fastSearchOnly)
    }

    @Test
    fun `Tier C preserves totalRamBytes in flags`() {
        val ram = 2L * GB
        val flags = TierClassifier.capabilitiesFor(
            tier = Tier.C,
            npuAvailable = false,
            totalRamBytes = ram
        )
        assertEquals(ram, flags.totalRamBytes)
    }

    @Test
    fun `capabilitiesFor result is non-null for all tiers`() {
        assertNotNull(TierClassifier.capabilitiesFor(Tier.A, true, 8L * GB))
        assertNotNull(TierClassifier.capabilitiesFor(Tier.B, false, 6L * GB))
        assertNotNull(TierClassifier.capabilitiesFor(Tier.C, false, 2L * GB))
    }
}
