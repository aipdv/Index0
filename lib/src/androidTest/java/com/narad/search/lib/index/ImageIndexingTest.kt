package com.narad.search.lib.index

import android.Manifest
import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.work.ListenableWorker
import androidx.work.WorkerParameters
import androidx.work.testing.TestListenableWorkerBuilder
import androidx.work.workDataOf
import com.narad.search.lib.index.enumeration.MediaStoreEnumerator
import com.narad.search.lib.index.store.IndexDatabase
import com.narad.search.lib.index.store.IndexRepository
import com.narad.search.lib.index.worker.ImageEncoder
import com.narad.search.lib.index.worker.IndexWorker
import com.narad.search.lib.query.store.VectorStore
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.IOException
import kotlin.math.sqrt

/**
 * Instrumented test for the image indexing pipeline end-to-end (L13 production path).
 *
 * **What is tested:**
 *  1. Seeded image (JPEG inserted into MediaStore) is enumerated by [MediaStoreEnumerator.enumerateImages].
 *  2. [IndexWorker] Branch B with fake image encoder factory → image lands in `image_vectors` table.
 *  3. Dedup idempotency: second run does NOT duplicate the image vector row.
 *
 * **Fake image encoder:** [FakeImageEncoder] implements [ImageEncoder] interface and returns a
 * deterministic 256-dim L2-normalised vector computed from the bitmap's width/height.
 * The real ONNX model is NOT loaded - no model files needed on device for this test.
 *
 * **Capability gating forced on:** [imageEncoderFactory] is injected (non-null) so the
 * production capability check ([DeviceTierDetector]) is bypassed - the image branch always runs.
 *
 * **L13 compliance:** drives the real [IndexWorker.doWork] entry point with the real
 * [MediaStoreEnumerator.enumerateImages] enumeration path.
 */
@RunWith(AndroidJUnit4::class)
class ImageIndexingTest {

    companion object {
        private const val TAG = "ImageIndexingTest"
    }

    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    /** URI returned by MediaStore insert - used for deletion in @After. */
    private var insertedImageUri: Uri? = null

    /** The enum-form URI that [MediaStoreEnumerator.enumerateImages] will produce. */
    private lateinit var imageEnumUri: String

    /** Unique display name per run to avoid cross-test pollution. */
    private val displayName = "narad_imgtest_${System.currentTimeMillis()}.jpg"

    @Before
    fun setUp() {
        // Grant storage permissions so IndexWorker does not stop at WaitingForPermission.
        val auto = InstrumentationRegistry.getInstrumentation().uiAutomation
        val pkg  = context.packageName
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            auto.grantRuntimePermission(pkg, Manifest.permission.READ_MEDIA_IMAGES)
            auto.grantRuntimePermission(pkg, Manifest.permission.READ_MEDIA_VIDEO)
        } else {
            auto.grantRuntimePermission(pkg, Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        // MANAGE_EXTERNAL_STORAGE is an appop, not a runtime permission.
        // Read to EOF so the appop is set before any test body runs.
        android.os.ParcelFileDescriptor.AutoCloseInputStream(
            auto.executeShellCommand("appops set $pkg MANAGE_EXTERNAL_STORAGE allow")
        ).use { it.readBytes() }

        // Clear any leftover state.
        try {
            val db = IndexDatabase(context)
            IndexRepository(db).clearAll()
            db.close()
        } catch (e: Exception) {
            Log.w(TAG, "clearAll() in @Before threw - continuing", e)
        }

        // Seed a JPEG into MediaStore.Images.
        imageEnumUri = seedImageAndResolveEnumUri()
        Log.i(TAG, "Seeded image → $imageEnumUri")
    }

    @After
    fun tearDown() {
        insertedImageUri?.let { uri ->
            try { context.contentResolver.delete(uri, null, null) } catch (e: Exception) {
                Log.w(TAG, "Failed to delete seeded image URI $uri", e)
            }
        }
        try {
            val db = IndexDatabase(context)
            IndexRepository(db).clearAll()
            db.close()
        } catch (e: Exception) {
            Log.w(TAG, "clearAll() in @After threw - ignoring", e)
        }
    }

    // -------------------------------------------------------------------------
    // Test 1: image enumeration returns the seeded image
    // -------------------------------------------------------------------------

    /**
     * Verifies that [MediaStoreEnumerator.enumerateImages] returns the seeded JPEG.
     */
    @Test
    fun enumerateImages_returnsSeededImage() {
        val enumerator = MediaStoreEnumerator()
        val targets = enumerator.enumerateImages(context)
        Log.i(TAG, "enumerateImages returned ${targets.size} image(s)")

        val match = targets.find { it.uri == imageEnumUri }
        assertNotNull(
            "enumerateImages should return the seeded image as URI $imageEnumUri; " +
            "got ${targets.size} results: ${targets.map { it.uri }.take(5)}",
            match
        )
        assertTrue(
            "Seeded image should have lastModified > 0",
            match!!.lastModified > 0L
        )
        assertEquals(
            "Seeded image should have mimeType=image/jpeg",
            "image/jpeg",
            match.mimeType
        )
    }

    // -------------------------------------------------------------------------
    // Test 2: image indexing end-to-end - lands in image_vectors table
    // -------------------------------------------------------------------------

    /**
     * Drives the real [IndexWorker.doWork] Branch B with:
     *  - FakeEmbedder (text side - no real text model needed).
     *  - FakeImageEncoder (image side - no real ONNX model needed).
     *    Capability is "forced on" because imageEncoderFactory is non-null.
     *
     * Asserts:
     *  - Worker returns Result.success().
     *  - The seeded image URI appears in `image_vectors` (imageVectorCount() >= 1).
     *  - [rehydrateImages] restores the vector and the image appears in VectorStore.
     */
    @Test
    fun imageIndexing_seedImageUri_appearsInImageVectors() {
        val worker = buildWorkerWithFakeEncoders()
        val result = runBlocking { worker.doWork() }

        assertTrue(
            "IndexWorker should return Result.success() with fake encoders",
            result is ListenableWorker.Result.Success
        )

        val db = IndexDatabase(context)
        val repo = IndexRepository(db)
        try {
            val count = repo.imageVectorCount()
            Log.i(TAG, "imageVectorCount() = $count after indexing run")
            assertTrue(
                "imageVectorCount() should be >= 1 after indexing seeded image. Got $count",
                count >= 1
            )

            // Rehydrate and confirm the seeded URI is in the image VectorStore.
            val imageStore = VectorStore()
            repo.rehydrateImages(imageStore)
            Log.i(TAG, "imageStore.size after rehydration = ${imageStore.size}")
            assertTrue(
                "VectorStore should have >= 1 image entry after rehydration. Got ${imageStore.size}",
                imageStore.size >= 1
            )

            // Verify the seeded image URI appears in VectorStore search results.
            // Use a unit vector along dim-0 as a query (FakeImageEncoder returns that for dim-0 bitmaps).
            val queryVector = FloatArray(256).also { it[0] = 1.0f }
            val results = imageStore.search(queryVector, topK = imageStore.size)
            Log.i(TAG, "VectorStore search results: ${results.map { it.filePath }}")
            val found = results.find { it.filePath == imageEnumUri }
            assertNotNull(
                "The seeded image URI $imageEnumUri must appear in VectorStore search results. " +
                "Results: ${results.map { it.filePath }}",
                found
            )
        } finally {
            db.close()
        }
    }

    // -------------------------------------------------------------------------
    // Test 3: dedup idempotency - second run does NOT duplicate
    // -------------------------------------------------------------------------

    /**
     * Proves idempotency: running the worker twice on the same seeded image
     * should NOT create duplicate rows in `image_vectors`.
     *
     * [IndexRepository.upsertImageVector] uses INSERT OR REPLACE - the count must
     * remain stable on the second run (same URI + same DATE_MODIFIED → dedup fires).
     */
    @Test
    fun imageIndexing_secondRun_doesNotDuplicate() {
        // First run.
        val worker1 = buildWorkerWithFakeEncoders()
        runBlocking { worker1.doWork() }

        val db1 = IndexDatabase(context)
        val countAfterFirst = IndexRepository(db1).imageVectorCount()
        db1.close()
        Log.i(TAG, "imageVectorCount() after 1st run = $countAfterFirst")
        assertTrue("Count should be >= 1 after first run", countAfterFirst >= 1)

        // Second run - same seeded image, same DATE_MODIFIED → dedup should fire.
        val worker2 = buildWorkerWithFakeEncoders()
        runBlocking { worker2.doWork() }

        val db2 = IndexDatabase(context)
        val countAfterSecond = IndexRepository(db2).imageVectorCount()
        db2.close()
        Log.i(TAG, "imageVectorCount() after 2nd run = $countAfterSecond")

        assertEquals(
            "imageVectorCount() must be the same after the second run (dedup is idempotent). " +
            "Got $countAfterSecond; expected $countAfterFirst.",
            countAfterFirst,
            countAfterSecond
        )
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Build an [IndexWorker] with fake text (FakeEmbedder) and fake image (FakeImageEncoder)
     * factories so no ONNX models are needed on the emulator.
     *
     * [imageEncoderFactory] is non-null → production capability check is bypassed
     * (image branch always runs in this test).
     */
    private fun buildWorkerWithFakeEncoders(): IndexWorker {
        return TestListenableWorkerBuilder<IndexWorker>(context)
            .setWorkerFactory(
                IndexWorker.Factory.withBothEncoders(
                    embedderFactory     = { _ -> FakeEmbedder() },
                    imageEncoderFactory = { _ -> FakeImageEncoder() }
                )
            )
            .setInputData(
                workDataOf(IndexWorker.KEY_MODEL_PATH to "/fake/model.onnx")
            )
            .build()
    }

    /**
     * Insert a programmatically-created JPEG (64×64 solid-colour bitmap) into
     * MediaStore.Images and return its enumerator-form URI.
     *
     * The enumerator form uses the Files collection:
     * `content://media/external/file/<id>` - the same as [MediaStoreEnumerator.enumerateImages].
     */
    private fun seedImageAndResolveEnumUri(): String {
        val resolver = context.contentResolver

        // Create a minimal 64×64 JPEG in memory.
        val bmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        bmp.eraseColor(Color.rgb(100, 150, 200))
        val baos = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 90, baos)
        bmp.recycle()
        val jpegBytes = baos.toByteArray()

        val cv = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
        }

        val insertUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv)
            ?: throw IOException("Failed to insert test image into MediaStore.Images")

        insertedImageUri = insertUri

        resolver.openOutputStream(insertUri)?.use { stream ->
            stream.write(jpegBytes)
        } ?: throw IOException("Failed to write JPEG bytes to $insertUri")

        // Commit the write: clear IS_PENDING and set DATE_MODIFIED so lastModified > 0.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val commitCv = ContentValues().apply {
                put(MediaStore.MediaColumns.IS_PENDING, 0)
                put(MediaStore.MediaColumns.DATE_MODIFIED, System.currentTimeMillis() / 1000L)
            }
            resolver.update(insertUri, commitCv, null, null)
        }

        // Resolve _ID to compute the Files-collection URI form.
        val id: Long = resolver.query(
            insertUri,
            arrayOf(MediaStore.Images.Media._ID),
            null, null, null
        )?.use { c -> if (c.moveToFirst()) c.getLong(0) else null }
            ?: error("Could not resolve _ID for inserted image URI $insertUri")

        return ContentUris.withAppendedId(
            MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL),
            id
        ).toString()
    }
}

// -------------------------------------------------------------------------
// FakeImageEncoder - deterministic 256-dim ImageEncoder for orchestration tests
// -------------------------------------------------------------------------

/**
 * Fake [ImageEncoder] that returns a deterministic 256-dim L2-normalised vector.
 * No ONNX model is needed - purely deterministic, based on bitmap dimensions.
 *
 * This implements the [ImageEncoder] interface (not [com.narad.search.lib.query.model.image.EG2BitmapImageEncoder]
 * which is a concrete class requiring a real ONNX path at construction).
 *
 * Used in [ImageIndexingTest] to test the full image indexing pipeline
 * (enumerate → encode → persist → rehydrate) without the real ONNX model.
 */
class FakeImageEncoder : ImageEncoder {

    companion object {
        private const val IMAGE_DIM = 256

        /**
         * Deterministic 256-dim L2-normalised vector from a seed value.
         * Used to produce consistent vectors for round-trip tests.
         */
        fun deterministicVector(seed: Long): FloatArray {
            val raw = FloatArray(IMAGE_DIM) { i ->
                val h = (seed xor (i.toLong() * 2654435761L)).toInt()
                (h and 0xFFFF).toFloat() / 0xFFFF.toFloat() - 0.5f
            }
            var norm = 0.0f
            for (x in raw) norm += x * x
            norm = sqrt(norm.toDouble()).toFloat()
            return if (norm < 1e-12f) raw else FloatArray(raw.size) { i -> raw[i] / norm }
        }
    }

    override fun encode(bitmap: Bitmap): FloatArray {
        // Deterministic based on bitmap size (the test always seeds a 64×64 bitmap).
        val seed = (bitmap.width.toLong() * 1000L) + bitmap.height.toLong()
        return deterministicVector(seed)
    }

    override fun close() { /* no resources to release */ }
}
