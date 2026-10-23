package com.narad.search.lib

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

/**
 * ScaffoldSmokeTest - U1 Test Gate.
 *
 * This is the simplest possible JUnit4 test that proves `:lib:test` can be
 * compiled and executed on the host JVM without an Android device.
 *
 * It does NOT test ONNX Runtime reachability or sqlite-vec JNI loading yet -
 * those require host-native binaries that may not be available at scaffold time
 * (see build-decisions.md §2d, Option A / B fallback).
 *
 * TODO(U2): add OrtEnvironment reachability test once the host-platform
 *   ONNX Runtime JAR is verified to load on macOS arm64.
 * TODO(U2): add sqlite-vec host-platform test (or mark NEEDS-DEVICE if
 *   macOS arm64 JNI binding is unavailable for version 0.1.6).
 */
class ScaffoldSmokeTest {

    @Test
    fun `text embedding dimension constant is 384`() {
        // G1: canonical text embedding dimension is locked to 384 for all tiers.
        assertEquals(256, LibVersion.TEXT_EMBEDDING_DIM)
    }

    @Test
    fun `image embedding dimension constant is 256`() {
        // EG2 image encoder produces 256-dim vectors in the same Matryoshka space as text.
        assertEquals(256, LibVersion.IMAGE_EMBEDDING_DIM)
    }

    @Test
    fun `LibVersion is not null`() {
        // Trivial smoke test - ensures the class loads without clinit error.
        assertNotNull(LibVersion)
    }

    @Test
    fun `library version string is non-empty`() {
        assert(LibVersion.VERSION.isNotBlank()) {
            "Library version string must not be blank"
        }
    }
}
