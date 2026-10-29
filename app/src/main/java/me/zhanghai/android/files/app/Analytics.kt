package me.zhanghai.android.files.app

import android.app.Dialog
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatDialogFragment
import androidx.core.content.edit
import androidx.core.text.parseAsHtml
import androidx.fragment.app.DialogFragment
import androidx.fragment.app.Fragment
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.posthog.PersonProfiles
import com.posthog.PostHog
import com.posthog.android.PostHogAndroid
import com.posthog.android.PostHogAndroidConfig
import me.zhanghai.android.files.BuildConfig
import me.zhanghai.android.files.R
import me.zhanghai.android.files.util.setSpanClickable
import me.zhanghai.android.files.util.show

// ponytail: one place for the policy link; pr/5 may rename the repo.
const val PRIVACY_POLICY_URL = "https://github.com/aipdv/Index0/blob/master/PRIVACY.md"

/** Event name to the only properties it may carry. Anything else is dropped before the queue. */
internal val ANALYTICS_EVENTS: Map<String, Set<String>> = mapOf(
    "Application Installed" to setOf("version", "build"),
    "Application Updated" to setOf("version", "build", "previous_version", "previous_build"),
    "Application Opened" to setOf("version", "build", "from_background"),
    "file_delete_queued" to setOf("file_count"),
    "file_archive_queued" to setOf("file_count"),
    "file_transfer_queued" to setOf("file_count", "operation"),
    "file_renamed" to emptySet(),
    "files_shared" to setOf("file_count"),
    "bookmark_added" to emptySet(),
    "file_created" to setOf("item_type"),
    "meaning_search_used" to emptySet(),
    "indexing_started" to emptySet(),
    "reindex_started" to emptySet(),
    "model_download_started" to emptySet(),
    "model_download_completed" to emptySet(),
    "model_download_failed" to emptySet(),
    "overnight_indexing_toggled" to setOf("enabled"),
    "storage_added" to setOf("type")
)

/** SDK properties kept on every event. Device, locale, network and screen properties are dropped. */
internal val ANALYTICS_KEPT_SDK_PROPERTIES = setOf(
    "distinct_id", "\$process_person_profile", "\$lib", "\$lib_version", "\$app_version",
    "\$app_build", "\$os_name", "\$os_version", "\$is_emulator"
)

/** Pure: the properties to send, or null when [event] is not on the allowlist. */
internal fun filterAnalyticsEvent(event: String, properties: Map<String, Any>?): Map<String, Any>? {
    val allowed = ANALYTICS_EVENTS[event] ?: return null
    return properties.orEmpty().filterKeys { it in allowed || it in ANALYTICS_KEPT_SDK_PROPERTIES } +
        ("\$geoip_disable" to true)
}

internal const val ANALYTICS_ASK_AT_LAUNCH = 3

/** Pure: ask once, from the 3rd main launch on, and only in builds that have a key. */
internal fun shouldAskAnalytics(available: Boolean, asked: Boolean, launchCount: Int): Boolean =
    available && !asked && launchCount >= ANALYTICS_ASK_AT_LAUNCH

/** Pure: the SDK starts only in a build with a key and after the user opted in. */
internal fun shouldStartAnalytics(available: Boolean, enabled: Boolean): Boolean =
    available && enabled

/** Opt-in PostHog EU analytics. Before opt-in nothing is set up, stored or sent. */
object Analytics {
    private const val KEY_LAUNCH_COUNT = "analytics_launch_count"
    private const val KEY_ASKED = "analytics_asked"

    val isAvailable =
        BuildConfig.POSTHOG_API_KEY.isNotBlank() && BuildConfig.POSTHOG_HOST.isNotBlank()

    private var isStarted = false

    private val enabledKey: String
        get() = application.getString(R.string.pref_key_analytics_enabled)

    private val isEnabled: Boolean
        get() = defaultSharedPreferences.getBoolean(enabledKey, false)

    // Strong reference: SharedPreferences holds listeners weakly.
    private val listener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
        if (key == enabledKey) {
            onEnabledChanged()
        }
    }

    fun initialize() {
        if (!isAvailable) {
            return
        }
        defaultSharedPreferences.registerOnSharedPreferenceChangeListener(listener)
        onEnabledChanged()
    }

    fun setEnabled(enabled: Boolean) {
        defaultSharedPreferences.edit { putBoolean(enabledKey, enabled) }
    }

    private fun onEnabledChanged() {
        if (shouldStartAnalytics(isAvailable, isEnabled)) {
            if (!isStarted) {
                PostHogAndroid.setup(application, createConfig())
                isStarted = true
            }
            // A past opt-out is persisted by the SDK.
            PostHog.optIn()
        } else if (isStarted) {
            PostHog.optOut()
            // A new anonymous ID on the next opt-in.
            PostHog.reset()
        }
    }

    private fun createConfig(): PostHogAndroidConfig =
        PostHogAndroidConfig(
            projectToken = BuildConfig.POSTHOG_API_KEY, host = BuildConfig.POSTHOG_HOST
        ).apply {
            personProfiles = PersonProfiles.NEVER
            setDefaultPersonProperties = false
            remoteConfig = false
            preloadFeatureFlags = false
            sendFeatureFlagEvent = false
            sessionReplay = false
            surveys = false
            captureScreenViews = false
            captureDeepLinks = false
            capturePushNotificationSubscriptions = false
            capturePushNotificationOpened = false
            errorTrackingConfig.autoCapture = false
            addBeforeSend { event ->
                filterAnalyticsEvent(event.event, event.properties)
                    ?.let { event.copy(properties = it.toMutableMap()) }
            }
        }

    fun capture(event: String, properties: Map<String, Any> = emptyMap()) {
        if (isStarted && isEnabled) {
            PostHog.capture(event, properties = properties)
        }
    }

    /** Call on a fresh start of the main screen from the launcher. */
    fun onMainLaunch() {
        if (!isAvailable) {
            return
        }
        defaultSharedPreferences.edit {
            putInt(KEY_LAUNCH_COUNT, defaultSharedPreferences.getInt(KEY_LAUNCH_COUNT, 0) + 1)
        }
    }

    /** Shows the one-time ask when it is due and no other dialog is open in [fragment]. */
    fun maybeAsk(fragment: Fragment) {
        val asked = defaultSharedPreferences.getBoolean(KEY_ASKED, false)
        val launchCount = defaultSharedPreferences.getInt(KEY_LAUNCH_COUNT, 0)
        if (!shouldAskAnalytics(isAvailable, asked, launchCount)) {
            return
        }
        if (fragment.childFragmentManager.fragments.any { it is DialogFragment }) {
            return
        }
        // Shown once ever. A dismiss counts as No (the pref stays off).
        defaultSharedPreferences.edit { putBoolean(KEY_ASKED, true) }
        AnalyticsConsentDialogFragment().show(fragment)
    }
}

class AnalyticsConsentDialogFragment : AppCompatDialogFragment() {
    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
        MaterialAlertDialogBuilder(requireContext(), theme)
            .setTitle(R.string.analytics_dialog_title)
            .setMessage(getString(R.string.analytics_dialog_message, PRIVACY_POLICY_URL).parseAsHtml())
            .setPositiveButton(R.string.analytics_dialog_share) { _, _ -> Analytics.setEnabled(true) }
            .setNegativeButton(R.string.analytics_dialog_no) { _, _ -> Analytics.setEnabled(false) }
            .create()
            .apply {
                setOnShowListener { findViewById<TextView>(android.R.id.message)?.setSpanClickable() }
            }
}
