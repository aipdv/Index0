package me.zhanghai.android.files.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AnalyticsTest {

    @Test fun `an event that is not on the allowlist is dropped`() {
        assertNull(filterAnalyticsEvent("\$screen", mapOf("\$screen_name" to "FileListActivity")))
        assertNull(filterAnalyticsEvent("search_query", mapOf("query" to "tax 2024")))
    }

    @Test fun `only allowed and kept properties survive, with no location`() {
        val sent = filterAnalyticsEvent(
            "file_transfer_queued",
            mapOf(
                "file_count" to 2, "operation" to "copy", "path" to "/sdcard/Download/a.pdf",
                "distinct_id" to "anon", "\$process_person_profile" to false,
                "\$app_version" to "0.1.0", "\$os_version" to "16",
                "\$device_model" to "Pixel 7a", "\$device_manufacturer" to "Google",
                "\$locale" to "en-US", "\$timezone" to "Asia/Kolkata",
                "\$network_carrier" to "Jio", "\$user_agent" to "Dalvik", "\$session_id" to "s"
            )
        )
        assertEquals(
            mapOf(
                "file_count" to 2, "operation" to "copy", "distinct_id" to "anon",
                "\$process_person_profile" to false, "\$app_version" to "0.1.0",
                "\$os_version" to "16", "\$geoip_disable" to true
            ),
            sent
        )
    }

    @Test fun `lifecycle events keep their versions but drop the deep link`() {
        val sent = filterAnalyticsEvent(
            "Application Opened",
            mapOf("version" to "0.1.0", "build" to 40, "from_background" to false,
                "url" to "content://x/secret.pdf", "referring_application" to "com.x")
        )
        assertEquals(
            mapOf("version" to "0.1.0", "build" to 40, "from_background" to false,
                "\$geoip_disable" to true),
            sent
        )
    }

    @Test fun `the ask shows only from the 3rd launch, once, and only with a key`() {
        assertFalse(shouldAskAnalytics(available = true, asked = false, launchCount = 1))
        assertFalse(shouldAskAnalytics(available = true, asked = false, launchCount = 2))
        assertTrue(shouldAskAnalytics(available = true, asked = false, launchCount = 3))
        assertFalse(shouldAskAnalytics(available = true, asked = true, launchCount = 4))
        assertFalse(shouldAskAnalytics(available = false, asked = false, launchCount = 3))
    }

    @Test fun `the SDK starts only after opt-in in a build with a key`() {
        assertFalse(shouldStartAnalytics(available = true, enabled = false))
        assertFalse(shouldStartAnalytics(available = false, enabled = true))
        assertTrue(shouldStartAnalytics(available = true, enabled = true))
    }
}
