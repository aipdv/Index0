package com.narad.search.lib.query.model

import org.junit.Assert.assertEquals
import org.junit.Test

/** DJL tokenizer builds must not phone home. Class init sets both off switches. */
class DjlTokenizerPrivacyTest {

    @Test
    fun initialisingDjlTokenizer_setsTelemetryOffSwitches() {
        // Initialises the class (companion init) without needing a tokenizer.json.
        Class.forName(DjlTokenizer::class.java.name, true, DjlTokenizer::class.java.classLoader)

        assertEquals("true", System.getProperty("OPT_OUT_TRACKING"))
        assertEquals("true", System.getProperty("ai.djl.offline"))
    }
}
