package com.narad.search.lib.device

/**
 * Device capability tier for Narad.
 *
 * Defined by G4 (overrides the plan's SD835/Android-8 tiers):
 *   Tier A - ≥3.4 GiB physical RAM AND NPU/NNAPI usable: EmbeddingGemma 2, images on.
 *   Tier B - ≥3.4 GiB RAM, no usable NPU: EmbeddingGemma 2 (CPU), images on.
 *   Tier C - <3.4 GiB RAM (2–3.4 GiB range, or degraded <2 GB): EmbeddingGemma 2, fastSearchOnly=true, images off.
 *
 * NO Android imports - this file must remain host-JVM-testable.
 */
enum class Tier {
    A,
    B,
    C
}
