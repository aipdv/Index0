package com.narad.search.lib.device

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * CapabilityFlagsTest - U6 Test Gate (JVM-only, no Android device needed).
 *
 * Verifies:
 *   1. Round-trip: fromEntries(flags.toEntries()) == flags for all tiers
 *   2. No data loss (all fields preserved after serialisation)
 *   3. Non-ASCII-safe model IDs survive round-trip
 *   4. toEntries() produces a Map<String,String> - no Android SharedPreferences import
 */
class CapabilityFlagsTest {

    private val GB = 1024L * 1024L * 1024L

    // -------------------------------------------------------------------------
    // Round-trip tests: fromEntries(flags.toEntries()) == flags
    // -------------------------------------------------------------------------

    @Test
    fun `Tier A flags round-trip without data loss`() {
        val original = CapabilityFlags(
            tier = Tier.A,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable = true,
            totalRamBytes = 8L * GB,
            fastSearchOnly = false
        )
        val roundTripped = CapabilityFlags.fromEntries(original.toEntries())
        assertEquals(original, roundTripped)
    }

    @Test
    fun `Tier B flags round-trip without data loss`() {
        val original = CapabilityFlags(
            tier = Tier.B,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable = false,
            totalRamBytes = 6L * GB,
            fastSearchOnly = false
        )
        val roundTripped = CapabilityFlags.fromEntries(original.toEntries())
        assertEquals(original, roundTripped)
    }

    @Test
    fun `Tier C flags round-trip without data loss`() {
        val original = CapabilityFlags(
            tier = Tier.C,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = false,
            npuAvailable = false,
            totalRamBytes = 2L * GB,
            fastSearchOnly = true
        )
        val roundTripped = CapabilityFlags.fromEntries(original.toEntries())
        assertEquals(original, roundTripped)
    }

    @Test
    fun `Tier C degraded sub-2GB flags round-trip`() {
        val original = CapabilityFlags(
            tier = Tier.C,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = false,
            npuAvailable = false,
            totalRamBytes = 1L * GB,
            fastSearchOnly = true
        )
        val roundTripped = CapabilityFlags.fromEntries(original.toEntries())
        assertEquals(original, roundTripped)
    }

    // -------------------------------------------------------------------------
    // Non-ASCII model ID round-trip
    // -------------------------------------------------------------------------

    @Test
    fun `non-ASCII model id round-trips correctly`() {
        // Model IDs could theoretically contain Unicode characters (e.g. future models
        // with non-Latin names). Verify the Map<String,String> serialisation handles them.
        val original = CapabilityFlags(
            tier = Tier.A,
            textModelId = "embedding-gemma-中文-308m",  // contains CJK chars
            imageEmbeddingEnabled = true,
            npuAvailable = true,
            totalRamBytes = 8L * GB,
            fastSearchOnly = false
        )
        val roundTripped = CapabilityFlags.fromEntries(original.toEntries())
        assertEquals(original.textModelId, roundTripped.textModelId)
        assertEquals(original, roundTripped)
    }

    @Test
    fun `Devanagari model id survives round-trip`() {
        val original = CapabilityFlags(
            tier = Tier.B,
            textModelId = "granite-ग्रेनाइट-97m",
            imageEmbeddingEnabled = true,
            npuAvailable = false,
            totalRamBytes = 4L * GB,
            fastSearchOnly = false
        )
        val roundTripped = CapabilityFlags.fromEntries(original.toEntries())
        assertEquals(original.textModelId, roundTripped.textModelId)
    }

    // -------------------------------------------------------------------------
    // Individual field preservation
    // -------------------------------------------------------------------------

    @Test
    fun `tier field is preserved through serialisation`() {
        listOf(Tier.A, Tier.B, Tier.C).forEach { tier ->
            val flags = CapabilityFlags(
                tier = tier,
                textModelId = "test-model",
                imageEmbeddingEnabled = false,
                npuAvailable = false,
                totalRamBytes = 4L * GB,
                fastSearchOnly = false
            )
            assertEquals(tier, CapabilityFlags.fromEntries(flags.toEntries()).tier)
        }
    }

    @Test
    fun `imageEmbeddingEnabled true is preserved`() {
        val flags = CapabilityFlags(
            tier = Tier.A,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable = true,
            totalRamBytes = 8L * GB,
            fastSearchOnly = false
        )
        assertTrue(CapabilityFlags.fromEntries(flags.toEntries()).imageEmbeddingEnabled)
    }

    @Test
    fun `imageEmbeddingEnabled false is preserved`() {
        val flags = CapabilityFlags(
            tier = Tier.C,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = false,
            npuAvailable = false,
            totalRamBytes = 2L * GB,
            fastSearchOnly = true
        )
        assertFalse(CapabilityFlags.fromEntries(flags.toEntries()).imageEmbeddingEnabled)
    }

    @Test
    fun `fastSearchOnly true is preserved`() {
        val flags = CapabilityFlags(
            tier = Tier.C,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = false,
            npuAvailable = false,
            totalRamBytes = 2L * GB,
            fastSearchOnly = true
        )
        assertTrue(CapabilityFlags.fromEntries(flags.toEntries()).fastSearchOnly)
    }

    @Test
    fun `fastSearchOnly false is preserved`() {
        val flags = CapabilityFlags(
            tier = Tier.A,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable = true,
            totalRamBytes = 8L * GB,
            fastSearchOnly = false
        )
        assertFalse(CapabilityFlags.fromEntries(flags.toEntries()).fastSearchOnly)
    }

    @Test
    fun `npuAvailable true is preserved`() {
        val flags = CapabilityFlags(
            tier = Tier.A,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable = true,
            totalRamBytes = 8L * GB,
            fastSearchOnly = false
        )
        assertTrue(CapabilityFlags.fromEntries(flags.toEntries()).npuAvailable)
    }

    @Test
    fun `npuAvailable false is preserved`() {
        val flags = CapabilityFlags(
            tier = Tier.B,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable = false,
            totalRamBytes = 6L * GB,
            fastSearchOnly = false
        )
        assertFalse(CapabilityFlags.fromEntries(flags.toEntries()).npuAvailable)
    }

    @Test
    fun `totalRamBytes is preserved through serialisation`() {
        val ram = 12L * GB + 345_678L
        val flags = CapabilityFlags(
            tier = Tier.A,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable = true,
            totalRamBytes = ram,
            fastSearchOnly = false
        )
        assertEquals(ram, CapabilityFlags.fromEntries(flags.toEntries()).totalRamBytes)
    }

    // -------------------------------------------------------------------------
    // toEntries() shape: all expected keys present
    // -------------------------------------------------------------------------

    @Test
    fun `toEntries produces a map with all expected keys`() {
        val flags = CapabilityFlags(
            tier = Tier.B,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable = false,
            totalRamBytes = 6L * GB,
            fastSearchOnly = false
        )
        val entries = flags.toEntries()
        val expectedKeys = setOf(
            "tier",
            "textModelId",
            "imageEmbeddingEnabled",
            "npuAvailable",
            "totalRamBytes",
            "fastSearchOnly"
        )
        assertEquals(
            "toEntries() must emit exactly the expected keys",
            expectedKeys,
            entries.keys
        )
    }

    @Test
    fun `toEntries values are all non-null strings`() {
        val flags = CapabilityFlags(
            tier = Tier.A,
            textModelId = "embeddinggemma-2",
            imageEmbeddingEnabled = true,
            npuAvailable = true,
            totalRamBytes = 8L * GB,
            fastSearchOnly = false
        )
        flags.toEntries().forEach { (key, value) ->
            assertTrue("Value for key '$key' must not be blank", value.isNotBlank())
        }
    }
}
