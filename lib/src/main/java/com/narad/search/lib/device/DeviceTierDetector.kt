package com.narad.search.lib.device

import android.app.ActivityManager
import android.content.Context
import android.os.Build

/**
 * Android entry point for device tier detection.
 *
 * Gathers real hardware inputs and delegates to the pure-function [TierClassifier]:
 *   1. Physical RAM via [ActivityManager.MemoryInfo.totalMem]  (Bug Watch: physical, not availMem)
 *   2. NPU / NNAPI probe via reflection on android.os.NeuralNetworks.getRuntimeFeatureLevel()
 *      (the class is hidden/NDK-only; reflection is the safe Java-layer approach)
 *   3. API level from [Build.VERSION.SDK_INT]
 *
 * [detect] is idempotent: identical runtime hardware → identical [CapabilityFlags].
 * There is no mutable global state; callers may cache the result themselves
 * (e.g. write to SharedPreferences via a thin adapter).
 *
 * IMPORTANT: This file uses Android SDK imports.  Its full runtime behaviour on real
 * hardware is NEEDS-DEVICE.  All pure classification logic lives in [TierClassifier]
 * (no Android imports) and is fully tested on the host JVM.
 *
 * Bug Watch (SPRINTS.md U6):
 *   - RAM detection uses totalMem (physical), not availMem (fluctuates).
 *   - NNAPI availability check may lie on some OEM builds; we accept the probe result
 *     as the primary signal. Cross-referencing with Build.HARDWARE is a future
 *     hardening task if field reports show false positives.
 *   - This function MUST be called from a background thread (Dispatchers.IO) to avoid
 *     blocking startup on the main thread.
 */
object DeviceTierDetector {

    /**
     * Detect the device's [CapabilityFlags] from real hardware characteristics.
     *
     * This function is pure with respect to its inputs: calling it multiple times on
     * the same physical device will return an equal [CapabilityFlags] each time
     * (idempotent - no mutable side effects inside this function).
     *
     * @param context  Any [Context] - used only to access system services.
     * @return         Fully populated [CapabilityFlags] for this device.
     *
     * NEEDS-DEVICE: runtime behaviour verified on physical hardware only.
     */
    fun detect(context: Context): CapabilityFlags {
        val totalRamBytes = readTotalRam(context)
        val npuAvailable  = probeNnapi()
        val apiLevel      = Build.VERSION.SDK_INT

        val tier = TierClassifier.classify(
            totalRamBytes = totalRamBytes,
            npuAvailable  = npuAvailable,
            apiLevel      = apiLevel
        )
        return TierClassifier.capabilitiesFor(
            tier          = tier,
            npuAvailable  = npuAvailable,
            totalRamBytes = totalRamBytes
        )
    }

    // -------------------------------------------------------------------------
    // Internal helpers - internal visibility for unit-test stubbing if needed
    // -------------------------------------------------------------------------

    /**
     * Read physical total RAM.
     *
     * [ActivityManager.MemoryInfo.totalMem] is the kernel-reported physical memory
     * and is stable across the device lifetime (G4 / Bug Watch: use physical, not available).
     */
    internal fun readTotalRam(context: Context): Long {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)
        return memInfo.totalMem
    }

    /**
     * Probe whether a usable NNAPI / NPU accelerator is present.
     *
     * [android.os.NeuralNetworks] is a hidden class accessible only via the NDK.
     * We probe it through reflection: if getRuntimeFeatureLevel() returns ≥ 1 (FEATURE_LEVEL_1),
     * the runtime has a functional NNAPI path.  Any exception (NoSuchMethod, SecurityException,
     * missing library) is caught and treated as "no NPU" - graceful degradation.
     *
     * Only checked on API 30+ where getRuntimeFeatureLevel() was added (API 31 in NDK,
     * present from API 30 on some builds).  Devices below API 30 conservatively return false.
     *
     * Note: some OEM builds (Xiaomi, MediaTek) report NNAPI available but have
     * incomplete or broken driver implementations.  Cross-referencing with
     * Build.HARDWARE is left as a future hardening task (Bug Watch item).
     */
    internal fun probeNnapi(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            // API < 30: skip NNAPI probe; Tier A requires API ≥ 29 (G4) but we're
            // conservative here - NPU detection requires a reliable probe.
            return false
        }
        return try {
            val clazz  = Class.forName("android.os.NeuralNetworks")
            val method = clazz.getMethod("getRuntimeFeatureLevel")
            val level  = method.invoke(null) as? Int ?: 0
            // FEATURE_LEVEL_1 == 1: basic NNAPI operations supported
            level >= 1
        } catch (_: ClassNotFoundException) {
            false
        } catch (_: NoSuchMethodException) {
            false
        } catch (_: Exception) {
            // SecurityException, InvocationTargetException, etc.
            false
        } catch (_: LinkageError) {
            false
        }
    }
}
