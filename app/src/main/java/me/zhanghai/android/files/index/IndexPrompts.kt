package me.zhanghai.android.files.index

import android.annotation.SuppressLint
import android.app.Application
import android.app.Dialog
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings
import androidx.appcompat.app.AppCompatDialogFragment
import androidx.core.content.edit
import androidx.fragment.app.DialogFragment
import androidx.fragment.app.Fragment
import androidx.work.WorkManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.narad.search.lib.index.service.BootReceiver
import com.narad.search.lib.index.service.IndexService
import kotlinx.coroutines.flow.first
import me.zhanghai.android.files.R
import me.zhanghai.android.files.app.application
import me.zhanghai.android.files.util.show
import me.zhanghai.android.files.util.startActivitySafe

// ---- Pure rules (JVM tested in IndexPromptsTest) ------------------------------------------

/** The heat notice: once, while the first index runs. */
internal fun shouldShowHeatNotice(firstIndexRunning: Boolean, firstIndexDone: Boolean, shown: Boolean): Boolean =
    firstIndexRunning && !firstIndexDone && !shown

/**
 * Overnight indexing is on by default, so the battery ask comes once after the first index,
 * never at first launch. [exempt] is a binder call: read last.
 */
internal fun shouldAskBatteryAfterFirstIndex(
    overnightOn: Boolean,
    firstIndexDone: Boolean,
    asked: Boolean,
    exempt: () -> Boolean,
): Boolean = overnightOn && firstIndexDone && !asked && !exempt()

/** Each switch to on asks while the app is not exempt. A decline is never asked again by itself. */
internal fun shouldAskBatteryOnToggle(turnedOn: Boolean, exempt: () -> Boolean): Boolean =
    turnedOn && !exempt()

/** Better Files: the one-time heat notice and the battery exemption ask for overnight indexing. */
object IndexPrompts {
    private const val KEY_HEAT_SHOWN = "heat_notice_shown"
    private const val KEY_BATTERY_ASKED = "battery_exemption_asked"

    private val prefs
        get() = application.getSharedPreferences(IndexEngine.PREFS, Application.MODE_PRIVATE)

    private fun isExempt(): Boolean =
        application.getSystemService(PowerManager::class.java)
            .isIgnoringBatteryOptimizations(application.packageName)

    /** The first index is queued or running (one-time work, so ENQUEUED counts too). */
    suspend fun isFirstIndexRunning(): Boolean = try {
        !IndexService.isFirstIndexDone(application) &&
            WorkManager.getInstance(application)
                .getWorkInfosForUniqueWorkFlow(IndexService.INITIAL_INDEX_WORK_NAME).first()
                .any { !it.state.isFinished }
    } catch (e: Exception) {
        false
    }

    /**
     * Shows at most one prompt in [fragment]: the heat notice while the first index runs, else
     * the battery ask after it. Returns true when it showed one, so the caller shows nothing else.
     */
    fun maybeShow(fragment: Fragment, firstIndexRunning: Boolean): Boolean {
        if (fragment.childFragmentManager.fragments.any { it is DialogFragment }) {
            return false
        }
        val firstIndexDone = IndexService.isFirstIndexDone(application)
        if (shouldShowHeatNotice(firstIndexRunning, firstIndexDone, prefs.getBoolean(KEY_HEAT_SHOWN, false))) {
            prefs.edit { putBoolean(KEY_HEAT_SHOWN, true) }
            HeatNoticeDialogFragment().show(fragment)
            return true
        }
        if (shouldAskBatteryAfterFirstIndex(
                BootReceiver.isOvernightEnabled(application), firstIndexDone,
                prefs.getBoolean(KEY_BATTERY_ASKED, false), ::isExempt
            )) {
            requestBatteryExemption(fragment)
            return true
        }
        return false
    }

    fun onOvernightToggled(fragment: Fragment, enabled: Boolean) {
        if (shouldAskBatteryOnToggle(enabled, ::isExempt)) {
            requestBatteryExemption(fragment)
        }
    }

    private fun requestBatteryExemption(fragment: Fragment) {
        // Set first: the system dialog returns to onResume, which must not ask again.
        prefs.edit { putBoolean(KEY_BATTERY_ASKED, true) }
        fragment.startActivitySafe(batteryExemptionIntent())
    }

    // ponytail: the direct system dialog needs REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, which Play
    // allows only for some app types. For a Play build, return
    // Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS) and drop the manifest permission.
    @SuppressLint("BatteryLife")
    private fun batteryExemptionIntent(): Intent =
        Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
            .setData(Uri.fromParts("package", application.packageName, null))
}

class HeatNoticeDialogFragment : AppCompatDialogFragment() {
    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
        MaterialAlertDialogBuilder(requireContext(), theme)
            .setMessage(R.string.indexing_heat_notice)
            .setPositiveButton(android.R.string.ok, null)
            .create()
}
