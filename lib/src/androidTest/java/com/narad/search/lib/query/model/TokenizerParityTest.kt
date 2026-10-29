package com.narad.search.lib.query.model

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.*
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device parity test: proves that the production [DjlTokenizer] (backed by
 * `libdjl_tokenizer.so` from `ai.djl.android:tokenizer-native`) produces byte-
 * for-byte identical token IDs to the JVM golden reference in
 * [com.narad.search.lib.query.model.TokenizerGoldenReferenceTest].
 *
 * This satisfies Learning L13: a green fake-driven gate can hide a no-op
 * production path. Here we exercise the REAL production [DjlTokenizer] class
 * on the Android device and assert exact parity with the host-native result.
 *
 * Pre-requisite: push the EG2 tokenizer.json to the device before running:
 *   adb push models/embeddinggemma-2/tokenizer.json /data/local/tmp/narad_tokenizer.json
 *
 * The path `/data/local/tmp/narad_tokenizer.json` is world-readable, so the
 * instrumented test APK can open it without special permissions.
 */
@RunWith(AndroidJUnit4::class)
class TokenizerParityTest {

    companion object {
        // Same sentences as TokenizerGoldenReferenceTest - must not diverge.
        private const val SENTENCE_EN = "The tenant must pay a penalty for late rent."
        private const val SENTENCE_HI = "किरायेदार को देर से किराया देने पर जुर्माना देना होगा।"

        // Tokenizer.json pushed to this path before the test run.
        private const val TOKENIZER_PATH = "/data/local/tmp/narad_tokenizer.json"

        // -------------------------------------------------------------------------
        // Golden expected arrays - MUST match TokenizerGoldenReferenceTest constants.
        // EmbeddingGemma 2 (Gemma 2 vocab, SentencePiece BPE), generated 2026-10-08.
        // -------------------------------------------------------------------------

        // EN: "The tenant must pay a penalty for late rent." - 12 tokens (BOS + 10 + EOS)
        private val EXPECTED_EN_IDS: LongArray = longArrayOf(
            2, 818, 28715, 1921, 2350, 496, 20256, 573, 5226, 7801, 236761, 1
        )

        // HI: "किरायेदार को देर से किराया देने पर जुर्माना देना होगा।" - 16 tokens
        private val EXPECTED_HI_IDS: LongArray = longArrayOf(
            2, 4286, 2208, 6251, 10782, 1198, 40525, 1248, 154635, 15327, 1539, 198151, 25227, 7582, 236890, 1
        )

        /** BOS: `<bos>` token ID for Gemma 2 vocab. */
        private const val BOS_TOKEN_ID = 2L

        /** EOS: `<eos>` token ID for Gemma 2 vocab. */
        private const val EOS_TOKEN_ID = 1L

        private lateinit var tokenizer: DjlTokenizer

        @JvmStatic
        @BeforeClass
        fun setUpClass() {
            check(java.io.File(TOKENIZER_PATH).exists()) {
                "EG2 tokenizer.json not found at $TOKENIZER_PATH - run: " +
                "adb push models/embeddinggemma-2/tokenizer.json $TOKENIZER_PATH"
            }
            tokenizer = try {
                DjlTokenizer(TOKENIZER_PATH)
            } catch (e: Throwable) {
                throw AssertionError(
                    "libdjl_tokenizer.so failed to load or tokenizer.json unreadable: ${e.message}", e
                )
            }
        }
    }

    @Test
    fun parityTest_EN_tokenIds_matchHostGoldenReference() {
        val enc = tokenizer.encode(SENTENCE_EN)
        println("ON-DEVICE EN inputIds (${enc.inputIds.size} tokens) = ${enc.inputIds.toList()}")

        assertEquals(
            "ON-DEVICE EN: token count mismatch - EG2 tokenizer parity BROKEN",
            EXPECTED_EN_IDS.size,
            enc.inputIds.size
        )
        assertArrayEquals(
            "ON-DEVICE EN: token IDs differ from JVM golden reference",
            EXPECTED_EN_IDS,
            enc.inputIds
        )
    }

    @Test
    fun parityTest_HI_tokenIds_matchHostGoldenReference() {
        val enc = tokenizer.encode(SENTENCE_HI)
        println("ON-DEVICE HI inputIds (${enc.inputIds.size} tokens) = ${enc.inputIds.toList()}")

        assertEquals(
            "ON-DEVICE HI: token count mismatch - EG2 tokenizer parity BROKEN",
            EXPECTED_HI_IDS.size,
            enc.inputIds.size
        )
        assertArrayEquals(
            "ON-DEVICE HI: token IDs differ from JVM golden reference",
            EXPECTED_HI_IDS,
            enc.inputIds
        )
    }

    @Test
    fun parityTest_EN_prefixIsBos_suffixIsEos() {
        val enc = tokenizer.encode(SENTENCE_EN)
        assertEquals(
            "ON-DEVICE EN: first token must be BOS (<bos>=$BOS_TOKEN_ID), got ${enc.inputIds.firstOrNull()}",
            BOS_TOKEN_ID,
            enc.inputIds.first()
        )
        assertEquals(
            "ON-DEVICE EN: last token must be EOS (<eos>=$EOS_TOKEN_ID), got ${enc.inputIds.lastOrNull()}",
            EOS_TOKEN_ID,
            enc.inputIds.last()
        )
    }

    @Test
    fun parityTest_HI_prefixIsBos_suffixIsEos() {
        val enc = tokenizer.encode(SENTENCE_HI)
        assertEquals(
            "ON-DEVICE HI: first token must be BOS (<bos>=$BOS_TOKEN_ID), got ${enc.inputIds.firstOrNull()}",
            BOS_TOKEN_ID,
            enc.inputIds.first()
        )
        assertEquals(
            "ON-DEVICE HI: last token must be EOS (<eos>=$EOS_TOKEN_ID), got ${enc.inputIds.lastOrNull()}",
            EOS_TOKEN_ID,
            enc.inputIds.last()
        )
    }
}
