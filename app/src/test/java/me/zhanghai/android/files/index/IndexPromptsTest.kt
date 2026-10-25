package me.zhanghai.android.files.index

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class IndexPromptsTest {
    private val exempt = { true }
    private val notExempt = { false }
    private val mustNotCall: () -> Boolean = { throw AssertionError("exempt read too early") }

    @Test
    fun heatNoticeShowsOnceWhileTheFirstIndexRuns() {
        assertTrue(shouldShowHeatNotice(firstIndexRunning = true, firstIndexDone = false, shown = false))
        assertFalse(shouldShowHeatNotice(firstIndexRunning = true, firstIndexDone = false, shown = true))
    }

    @Test
    fun heatNoticeNeverShowsBeforeOrAfterTheFirstIndex() {
        assertFalse(shouldShowHeatNotice(firstIndexRunning = false, firstIndexDone = false, shown = false))
        // A later delta run is not the first index.
        assertFalse(shouldShowHeatNotice(firstIndexRunning = true, firstIndexDone = true, shown = false))
    }

    @Test
    fun batteryAskComesOnceAfterTheFirstIndex() {
        assertTrue(shouldAskBatteryAfterFirstIndex(overnightOn = true, firstIndexDone = true, asked = false, exempt = notExempt))
        // Declined or allowed once: never again by itself.
        assertFalse(shouldAskBatteryAfterFirstIndex(overnightOn = true, firstIndexDone = true, asked = true, exempt = mustNotCall))
    }

    @Test
    fun batteryAskNeverAtFirstLaunch() {
        assertFalse(shouldAskBatteryAfterFirstIndex(overnightOn = true, firstIndexDone = false, asked = false, exempt = mustNotCall))
    }

    @Test
    fun batteryAskSkippedWhenOvernightOffOrAlreadyExempt() {
        assertFalse(shouldAskBatteryAfterFirstIndex(overnightOn = false, firstIndexDone = true, asked = false, exempt = mustNotCall))
        assertFalse(shouldAskBatteryAfterFirstIndex(overnightOn = true, firstIndexDone = true, asked = false, exempt = exempt))
    }

    @Test
    fun toggleAsksOnlyWhenTurnedOnAndNotExempt() {
        assertTrue(shouldAskBatteryOnToggle(turnedOn = true, exempt = notExempt))
        assertFalse(shouldAskBatteryOnToggle(turnedOn = true, exempt = exempt))
        assertFalse(shouldAskBatteryOnToggle(turnedOn = false, exempt = mustNotCall))
    }
}
