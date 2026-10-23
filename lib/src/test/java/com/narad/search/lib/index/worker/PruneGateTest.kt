package com.narad.search.lib.index.worker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** U8: the prune is a data-loss path, so its gates are pinned here without a device. */
class PruneGateTest {

    private val api33 = 33
    private val api34 = 34
    private val api32 = 32

    @Test
    fun `api 33 with only READ_MEDIA_IMAGES reads as full photo access`() {
        // Video is not requested any more: images alone must be enough.
        assertEquals(
            PhotoAccess.FULL,
            photoAccess(api33, mediaImages = true, userSelected = false, readExternal = false)
        )
    }

    @Test
    fun `api 34 user-selected grant alone is partial not full`() {
        assertEquals(
            PhotoAccess.PARTIAL,
            photoAccess(api34, mediaImages = false, userSelected = true, readExternal = false)
        )
    }

    @Test
    fun `api 34 full grant wins over user-selected`() {
        assertEquals(
            PhotoAccess.FULL,
            photoAccess(api34, mediaImages = true, userSelected = true, readExternal = false)
        )
    }

    @Test
    fun `no grant is NONE on every api`() {
        for (sdk in listOf(29, api32, api33, api34, 36)) {
            assertEquals(
                "sdk $sdk",
                PhotoAccess.NONE,
                photoAccess(sdk, mediaImages = false, userSelected = false, readExternal = false)
            )
        }
    }

    @Test
    fun `api 32 and below use READ_EXTERNAL_STORAGE`() {
        assertEquals(PhotoAccess.FULL, photoAccess(api32, false, false, readExternal = true))
        assertEquals(PhotoAccess.NONE, photoAccess(api32, true, true, readExternal = false))
    }

    @Test
    fun `api 33 ignores a stray user-selected flag`() {
        // READ_MEDIA_VISUAL_USER_SELECTED does not exist below API 34.
        assertEquals(PhotoAccess.NONE, photoAccess(api33, false, userSelected = true, readExternal = false))
    }

    @Test
    fun `photo prune runs only with full access at listing and now, and a finished pass`() {
        val full = PhotoAccess.FULL
        assertTrue(shouldPrunePhotos(full, full, passFinished = true))
        assertFalse("pass stopped", shouldPrunePhotos(full, full, passFinished = false))
        assertFalse("user-selected-only grant", shouldPrunePhotos(PhotoAccess.PARTIAL, PhotoAccess.PARTIAL, true))
        assertFalse("access revoked", shouldPrunePhotos(PhotoAccess.NONE, PhotoAccess.NONE, true))
        assertFalse("revoked after listing", shouldPrunePhotos(full, PhotoAccess.NONE, true))
        assertFalse("downgraded after listing", shouldPrunePhotos(full, PhotoAccess.PARTIAL, true))
    }

    @Test
    fun `partial at listing then full now does not prune`() {
        // The user switched "Select photos" to "Allow all" while the pass ran: the listing is a subset.
        assertFalse(shouldPrunePhotos(PhotoAccess.PARTIAL, PhotoAccess.FULL, passFinished = true))
        assertFalse(shouldPrunePhotos(PhotoAccess.NONE, PhotoAccess.FULL, passFinished = true))
    }

    @Test
    fun `document prune needs access at listing and now, and a finished pass`() {
        assertTrue(shouldPruneDocuments(atListing = true, now = true, passFinished = true))
        assertFalse("pass stopped", shouldPruneDocuments(true, true, passFinished = false))
        assertFalse("access revoked", shouldPruneDocuments(false, false, true))
        assertFalse("granted mid-pass, listing was empty or partial", shouldPruneDocuments(false, true, true))
        assertFalse("revoked after listing", shouldPruneDocuments(true, false, true))
    }

    @Test
    fun `document and photo rules are independent`() {
        // Documents granted and finished, photos only partial: documents prune, photos do not.
        assertTrue(shouldPruneDocuments(true, true, true))
        assertFalse(shouldPrunePhotos(PhotoAccess.PARTIAL, PhotoAccess.PARTIAL, true))
    }
}
