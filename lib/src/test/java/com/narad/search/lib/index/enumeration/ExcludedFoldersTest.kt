package com.narad.search.lib.index.enumeration

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ExcludedFoldersTest {

    private fun excluded(path: String, user: List<String> = emptyList()) =
        ExcludedFolders.isExcluded(path, user)

    @Test
    fun `android folder is excluded by default, any case`() {
        assertTrue(excluded("/storage/emulated/0/Android/data/com.x/files/x.pdf"))
        assertTrue(excluded("/storage/emulated/0/android/OBB/com.x/main.obb"))
        assertTrue(excluded("/storage/emulated/0/Android/x.pdf"))
        assertTrue(excluded("/storage/1A2B-3C4D/Android/data/x.jpg"))
        assertTrue(excluded("/storage/emulated/0/Android/mediaX/x.pdf"))
    }

    @Test
    fun `android media is kept, so WhatsApp and Telegram documents are indexed`() {
        assertFalse(excluded("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Documents/bill.pdf"))
        assertFalse(excluded("/storage/emulated/0/android/Media/org.telegram.messenger/Telegram/Telegram Documents/a.pdf"))
        assertFalse(excluded("/storage/1A2B-3C4D/Android/media/x.pdf"))
        // Hidden folders inside it still go (the dot rule).
        assertTrue(excluded("/storage/emulated/0/Android/media/x/.hidden/a.pdf"))
        assertTrue(excluded("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/.Statuses/s.jpg"))
    }

    @Test
    fun `android below the volume root is not the system folder`() {
        assertFalse(excluded("/storage/emulated/0/Download/Android/notes.pdf"))
        assertFalse(excluded("/storage/emulated/0/AndroidDev/notes.pdf"))
    }

    @Test
    fun `hidden folders are excluded, hidden files in visible folders are kept`() {
        assertTrue(excluded("/storage/emulated/0/Download/.hidden/a.pdf"))
        assertFalse(excluded("/storage/emulated/0/Download/.notes.txt"))
        assertFalse(excluded("/storage/emulated/0/Download/a.pdf"))
    }

    @Test
    fun `user folder matches on a slash boundary`() {
        val user = listOf("/storage/emulated/0/Work")
        assertTrue(excluded("/storage/emulated/0/Work/a.pdf", user))
        assertTrue(excluded("/storage/emulated/0/Work/sub/b.pdf", user))
        assertFalse(excluded("/storage/emulated/0/Workshop/c.pdf", user))
    }

    @Test
    fun `user folder with a trailing slash behaves the same`() {
        val user = listOf("/storage/emulated/0/Work/")
        assertTrue(excluded("/storage/emulated/0/Work/a.pdf", user))
        assertFalse(excluded("/storage/emulated/0/Workshop/c.pdf", user))
    }
}
