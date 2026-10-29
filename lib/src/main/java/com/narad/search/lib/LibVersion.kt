package com.narad.search.lib

/**
 * Library version constants for the Narad :lib module.
 *
 * Placeholder for U1 - real library code (QueryEngine, VectorStore, etc.)
 * is added in U2 and U3.
 *
 * NOTE: This file contains no Android-framework imports so that the :lib
 * module's main source remains JVM-testable on the host machine (build-decisions
 * §2d). Keep all Android-SDK imports out of :lib main source.
 */
object LibVersion {
    const val NAME    = "narad-lib"
    const val VERSION = "0.1.0"

    /** Stored embedding dimension: EmbeddingGemma 2 output truncated (Matryoshka) to 256. */
    const val TEXT_EMBEDDING_DIM = 256

    /** EG2 image embedding dimension - same 256-dim Matryoshka space as text (Tier A/B only). */
    const val IMAGE_EMBEDDING_DIM = 256
}
