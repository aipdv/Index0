package me.zhanghai.android.files.index

import android.annotation.SuppressLint
import android.app.Application
import android.app.Dialog
import android.content.ActivityNotFoundException
import android.content.ComponentName
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings
import android.util.Log
import androidx.appcompat.app.AppCompatDialogFragment
import androidx.core.content.edit
import androidx.fragment.app.DialogFragment
import androidx.fragment.app.Fragment
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.narad.search.lib.index.service.BootReceiver
import com.narad.search.lib.index.service.IndexService
import kotlinx.parcelize.Parcelize
import me.zhanghai.android.files.R
import me.zhanghai.android.files.app.application
import me.zhanghai.android.files.util.ParcelableArgs
import me.zhanghai.android.files.util.args
import me.zhanghai.android.files.util.putArgs
import me.zhanghai.android.files.util.show
import me.zhanghai.android.files.util.startActivitySafe

// ---- Pure rules (JVM tested in IndexPromptsTest) ------------------------------------------

/** What the Indexing screen shows: the heat notice, the background ask, both in one dialog, or nothing. */
internal enum class IndexPrompt { HEAT_AND_BACKGROUND, HEAT, BACKGROUND }

internal fun isVivo(manufacturer: String): Boolean = manufacturer.equals("vivo", ignoreCase = true)

/**
 * Only while files are read (the model is in place), so never at first launch. Heat: once, during
 * the first index. Background: once, on any reading run, so a phone whose first index never ends
 * (vivo kills it every few minutes) is still asked. On vivo the system exemption does not stop
 * com.vivo.abe, so exempt does not block the ask there. [exempt] is a binder call: read last.
 */
internal fun indexPrompt(
    reading: Boolean,
    firstIndexDone: Boolean,
    heatShown: Boolean,
    overnightOn: Boolean,
    asked: Boolean,
    manufacturer: String,
    exempt: () -> Boolean,
): IndexPrompt? {
    if (!reading) return null
    val heat = !firstIndexDone && !heatShown
    val background = overnightOn && !asked && (isVivo(manufacturer) || !exempt())
    return when {
        heat && background -> IndexPrompt.HEAT_AND_BACKGROUND
        heat -> IndexPrompt.HEAT
        background -> IndexPrompt.BACKGROUND
        else -> null
    }
}

internal enum class AllowStep { VIVO_POWER_PAGE, SYSTEM_DIALOG }

/**
 * What Allow opens, bottom to top. vivo: the vivo power page, with the system dialog on top while
 * not exempt, so the user answers the system dialog first, then sees the vivo page.
 */
internal fun allowSteps(manufacturer: String, exempt: Boolean): List<AllowStep> = listOfNotNull(
    AllowStep.VIVO_POWER_PAGE.takeIf { isVivo(manufacturer) },
    AllowStep.SYSTEM_DIALOG.takeIf { !exempt },
)

/** Each switch to on asks while the app is not exempt. A decline is never asked again by itself. */
internal fun shouldAskBatteryOnToggle(turnedOn: Boolean, exempt: () -> Boolean): Boolean =
    turnedOn && !exempt()

/** Better Files: the one-time heat notice and the background ask, on the Indexing screen only. */
object IndexPrompts {
    private const val TAG = "IndexPrompts"
    private const val KEY_HEAT_SHOWN = "heat_notice_shown"
    private const val KEY_BATTERY_ASKED = "battery_exemption_asked"

    // ponytail: untested on a real vivo (the PO tests it). The page where vivo allows high background
    // power use; vivo (com.vivo.abe) kills background work even when the app is battery exempt.
    private val VIVO_POWER_PAGE = ComponentName(
        "com.vivo.abe", "com.vivo.applicationbehaviorengine.ui.ExcessivePowerManagerActivity"
    )

    private val prefs
        get() = application.getSharedPreferences(IndexEngine.PREFS, Application.MODE_PRIVATE)

    private fun isExempt(): Boolean =
        application.getSystemService(PowerManager::class.java)
            .isIgnoringBatteryOptimizations(application.packageName)

    /** Shows at most one dialog in [fragment] while [reading] files. Returns true when it showed one. */
    fun maybeShow(fragment: Fragment, reading: Boolean): Boolean {
        if (fragment.childFragmentManager.fragments.any { it is DialogFragment }) {
            return false
        }
        val prompt = indexPrompt(
            reading, IndexService.isFirstIndexDone(application), prefs.getBoolean(KEY_HEAT_SHOWN, false),
            BootReceiver.isOvernightEnabled(application), prefs.getBoolean(KEY_BATTERY_ASKED, false),
            Build.MANUFACTURER, ::isExempt
        ) ?: return false
        val heat = prompt != IndexPrompt.BACKGROUND
        val background = prompt != IndexPrompt.HEAT
        // Set first, so a rotation, Not now or the system dialog's return never shows it again.
        prefs.edit {
            if (heat) putBoolean(KEY_HEAT_SHOWN, true)
            if (background) putBoolean(KEY_BATTERY_ASKED, true)
        }
        val vivoLine = background && isVivo(Build.MANUFACTURER)
        IndexPromptDialogFragment().putArgs(IndexPromptDialogFragment.Args(heat, background, vivoLine)).show(fragment)
        return true
    }

    fun onOvernightToggled(fragment: Fragment, enabled: Boolean) {
        if (shouldAskBatteryOnToggle(enabled, ::isExempt)) {
            prefs.edit { putBoolean(KEY_BATTERY_ASKED, true) }
            fragment.startActivitySafe(batteryExemptionIntent())
        }
    }

    /** Allow in the dialog: the [allowSteps] in one go. The vivo page falls back to app details. */
    internal fun allow(fragment: Fragment) {
        val steps = allowSteps(Build.MANUFACTURER, isExempt())
        val system = batteryExemptionIntent().takeIf { AllowStep.SYSTEM_DIALOG in steps }
        if (AllowStep.VIVO_POWER_PAGE in steps) {
            val appDetails = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                .setData(Uri.fromParts("package", application.packageName, null))
            for (page in listOf(Intent().setComponent(VIVO_POWER_PAGE), appDetails)) {
                try {
                    fragment.requireActivity().startActivities(listOfNotNull(page, system).toTypedArray())
                    return
                } catch (e: ActivityNotFoundException) {
                    Log.w(TAG, "No ${page.component ?: page.action}", e)
                } catch (e: SecurityException) {
                    Log.w(TAG, "Not allowed: ${page.component ?: page.action}", e)
                }
            }
        }
        system?.let { fragment.startActivitySafe(it) }
    }

    // ponytail: the direct system dialog needs REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, which Play
    // allows only for some app types. For a Play build, return
    // Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS) and drop the manifest permission.
    @SuppressLint("BatteryLife")
    private fun batteryExemptionIntent(): Intent =
        Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
            .setData(Uri.fromParts("package", application.packageName, null))
}

/** The heat line, the background line and the vivo line, as due. One dialog, never two in a row. */
class IndexPromptDialogFragment : AppCompatDialogFragment() {
    private val args by args<Args>()

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val message = listOfNotNull(
            getString(R.string.indexing_heat_notice).takeIf { args.heat },
            getString(R.string.indexing_background_ask).takeIf { args.background },
            getString(R.string.indexing_background_ask_vivo).takeIf { args.vivo },
        ).joinToString("\n\n")
        val builder = MaterialAlertDialogBuilder(requireContext(), theme).setMessage(message)
        if (args.background) {
            builder.setPositiveButton(R.string.indexing_background_allow) { _, _ ->
                IndexPrompts.allow(requireParentFragment())
            }
            builder.setNegativeButton(R.string.indexing_background_not_now, null)
        } else {
            builder.setPositiveButton(android.R.string.ok, null)
        }
        return builder.create()
    }

    @Parcelize
    class Args(val heat: Boolean, val background: Boolean, val vivo: Boolean) : ParcelableArgs
}
