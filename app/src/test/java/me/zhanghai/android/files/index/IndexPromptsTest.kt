package me.zhanghai.android.files.index

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class IndexPromptsTest {
    private val exempt = { true }
    private val notExempt = { false }
    private val mustNotCall: () -> Boolean = { throw AssertionError("exempt read too early") }

    private fun prompt(
        reading: Boolean = true,
        firstIndexDone: Boolean = false,
        heatShown: Boolean = false,
        overnightOn: Boolean = true,
        asked: Boolean = false,
        manufacturer: String = "Google",
        exempt: () -> Boolean = notExempt,
    ) = indexPrompt(reading, firstIndexDone, heatShown, overnightOn, asked, manufacturer, exempt)

    @Test
    fun nothingBeforeFilesAreRead() {
        // First launch, model download, finding files: no prompt, and no binder call.
        assertNull(prompt(reading = false, exempt = mustNotCall))
    }

    @Test
    fun firstIndexReadingShowsHeatAndBackgroundInOneDialog() {
        assertEquals(IndexPrompt.HEAT_AND_BACKGROUND, prompt())
    }

    @Test
    fun heatAlreadyShownGivesTheBackgroundAskAlone() {
        assertEquals(IndexPrompt.BACKGROUND, prompt(heatShown = true))
    }

    @Test
    fun heatAloneWhenTheBackgroundAskIsNotDue() {
        assertEquals(IndexPrompt.HEAT, prompt(asked = true, exempt = mustNotCall))
        assertEquals(IndexPrompt.HEAT, prompt(overnightOn = false, exempt = mustNotCall))
        assertEquals(IndexPrompt.HEAT, prompt(exempt = exempt))
    }

    @Test
    fun askedOnceNeverAgain() {
        // Allowed or Not now: the flag is set when the dialog shows, so it never loops.
        assertNull(prompt(heatShown = true, asked = true, exempt = mustNotCall))
    }

    @Test
    fun exemptOrOvernightOffMeansNoBackgroundAsk() {
        assertNull(prompt(heatShown = true, exempt = exempt))
        assertNull(prompt(heatShown = true, overnightOn = false, exempt = mustNotCall))
    }

    @Test
    fun laterRunAfterTheFirstIndexStillAsksOnceButNoHeat() {
        assertEquals(IndexPrompt.BACKGROUND, prompt(firstIndexDone = true))
        assertNull(prompt(firstIndexDone = true, asked = true, exempt = mustNotCall))
    }

    @Test
    fun vivoAsksEvenWhenExempt() {
        // com.vivo.abe kills the app even with the system exemption.
        assertEquals(IndexPrompt.BACKGROUND, prompt(heatShown = true, manufacturer = "vivo", exempt = mustNotCall))
        assertEquals(IndexPrompt.BACKGROUND, prompt(heatShown = true, manufacturer = "VIVO", exempt = exempt))
        assertNull(prompt(heatShown = true, manufacturer = "vivo", asked = true, exempt = mustNotCall))
        assertNull(prompt(heatShown = true, manufacturer = "vivo", overnightOn = false, exempt = mustNotCall))
    }

    @Test
    fun allowOpensTheSystemDialogAndOnVivoThePowerPageUnderIt() {
        assertEquals(listOf(AllowStep.SYSTEM_DIALOG), allowSteps("Google", exempt = false))
        assertEquals(listOf(AllowStep.VIVO_POWER_PAGE, AllowStep.SYSTEM_DIALOG), allowSteps("vivo", exempt = false))
        assertEquals(listOf(AllowStep.VIVO_POWER_PAGE), allowSteps("vivo", exempt = true))
    }

    @Test
    fun toggleAsksOnlyWhenTurnedOnAndNotExempt() {
        assertTrue(shouldAskBatteryOnToggle(turnedOn = true, exempt = notExempt))
        assertFalse(shouldAskBatteryOnToggle(turnedOn = true, exempt = exempt))
        assertFalse(shouldAskBatteryOnToggle(turnedOn = false, exempt = mustNotCall))
    }
}
