package com.narad.search.lib.index.provision

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

// ---------------------------------------------------------------------------
// Manifest types
// ---------------------------------------------------------------------------

/**
 * A single file in the model manifest.
 *
 * @param relativeDest  Relative path under [Context.filesDir]. E.g. `models/embeddinggemma-2/model_quantized.onnx`.
 * @param url           Full URL, OR a relative path resolved against the baseUrl.
 *                      Ignored when [assetPath] is non-null.
 * @param sha256        Lowercase hex SHA-256 of the correct file contents.
 * @param sizeBytes     Expected byte count of the fully-downloaded file.
 * @param assetPath     If non-null, this file is copied from the APK asset at this path
 *                      instead of being downloaded via HTTP. The sha256/sizeBytes are still
 *                      verified after the copy.
 */
data class ModelFile(
    val url: String,
    val relativeDest: String,
    val sha256: String,
    val sizeBytes: Long,
    val assetPath: String? = null
)

/**
 * A named bundle of [ModelFile] entries that must all be present for the model to be usable.
 */
data class ModelManifest(
    val modelId: String,
    val files: List<ModelFile>
)

// ---------------------------------------------------------------------------
// ProvisionState - sealed class for StateFlow
// ---------------------------------------------------------------------------

/** Progress states emitted by [ModelProvisioner.stateFlow]. */
sealed class ProvisionState {
    /** Initial state - provision() has not been called yet. */
    object NotStarted : ProvisionState()

    /**
     * One file is being streamed down.
     *
     * @param fileIndex        0-based index of the file currently downloading.
     * @param fileCount        Total number of files in the manifest.
     * @param fileName         Display-friendly file name (last path segment of [relativeDest]).
     * @param bytesDownloaded  Bytes received so far for this file.
     * @param bytesTotal       Expected total bytes for this file (-1 if unknown).
     */
    data class Downloading(
        val fileIndex: Int,
        val fileCount: Int,
        val fileName: String,
        val bytesDownloaded: Long,
        val bytesTotal: Long
    ) : ProvisionState()

    /**
     * Download complete; computing/verifying SHA-256.
     *
     * @param fileName  File being verified.
     */
    data class Verifying(val fileName: String) : ProvisionState()

    /** All files are present, sized correctly, and (if freshly downloaded) hash-verified. */
    object Ready : ProvisionState()

    /**
     * An unrecoverable error occurred.
     *
     * @param reason  Human-readable explanation for logging / display.
     */
    data class Failed(val reason: String) : ProvisionState()
}

// ---------------------------------------------------------------------------
// EmbeddingGemma 2 manifest (M1)
// ---------------------------------------------------------------------------

/**
 * EmbeddingGemma 2 manifest.
 *
 * The **device layout** is FLAT:
 * `models/embeddinggemma-2/{model_quantized.onnx, model_quantized.onnx_data, tokenizer.json}`.
 *
 * The tuned ONNX graph (`model_quantized.onnx`, 498 KB) ships IN THE APK under
 * `assets/models/embeddinggemma-2/model_quantized.onnx` and is copied to device
 * storage on first provision. Only the large weight file and the tokenizer are
 * downloaded from Hugging Face.
 *
 * Pinned HF revision: `daa72c51243991dfcaf9f9137d2c573d8f7790c0` (tuned graph commit).
 */
object EmbeddingGemma2Manifest {
    const val MODEL_ID = "embeddinggemma-2"

    /** Pinned HuggingFace base URL for downloadable files. */
    const val BASE_URL =
        "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/daa72c51243991dfcaf9f9137d2c573d8f7790c0/"

    val manifest = ModelManifest(
        modelId = MODEL_ID,
        files = listOf(
            // Tuned ONNX graph: copies from APK asset (no HTTP download).
            // model_quantized.onnx_data must sit alongside in the SAME dir (graph-internal name).
            ModelFile(
                url          = "",  // unused - assetPath takes precedence
                relativeDest = "models/embeddinggemma-2/model_quantized.onnx",
                sha256       = "4bd9649778ba6f6460858103b138df6a70ddb22fb4c6ac844fe6fd20e56d691c",
                sizeBytes    = 498_523L,
                assetPath    = "models/embeddinggemma-2/model_quantized.onnx"
            ),
            // Large weight file: downloaded from HuggingFace.
            ModelFile(
                url          = "onnx/model_quantized.onnx_data",
                relativeDest = "models/embeddinggemma-2/model_quantized.onnx_data",
                sha256       = "278a7ff1248c3618e4bd11a607fc54f7bdc7778854230f3956d3f86bd9db4f3b",
                sizeBytes    = 313_724_928L
            ),
            // Tokenizer: downloaded from HuggingFace.
            ModelFile(
                url          = "tokenizer.json",
                relativeDest = "models/embeddinggemma-2/tokenizer.json",
                sha256       = "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
                sizeBytes    = 32_170_510L
            )
        )
    )
}

// ---------------------------------------------------------------------------
// EmbeddingGemma 2 vision encoder manifest (M2)
// ---------------------------------------------------------------------------

/**
 * Vision encoder manifest for EmbeddingGemma 2.
 *
 * Provisioned SEPARATELY from the text model ([EmbeddingGemma2Manifest]) so that
 * Tier C phones (where `imageEmbeddingEnabled=false`) never download the 195 MB
 * weight file. The call-site gates on [CapabilityFlags.imageEmbeddingEnabled].
 *
 * The **device layout** is FLAT, co-located with the LM files:
 * `models/embeddinggemma-2/{vision_encoder_quantized.onnx, vision_encoder_quantized.onnx_data}`
 *
 * The graph (165 KB) ships IN THE APK under
 * `assets/models/embeddinggemma-2/vision_encoder_quantized.onnx`.
 * The weight file (195 MB) is downloaded from Hugging Face at the same pinned revision
 * as the LM data.
 */
object EmbeddingGemma2VisionManifest {
    /** Same model id as the text model - both live in the same directory. */
    const val MODEL_ID = "embeddinggemma-2"

    val manifest = ModelManifest(
        modelId = MODEL_ID,
        files = listOf(
            // Vision graph: copies from APK asset (no HTTP download).
            // vision_encoder_quantized.onnx_data must sit alongside (graph-internal name).
            ModelFile(
                url          = "",   // unused - assetPath takes precedence
                relativeDest = "models/embeddinggemma-2/vision_encoder_quantized.onnx",
                sha256       = "14a8e62f96456c7d42091fe73b5d5c24b8798cde92a5f82411e15e6c0d628259",
                sizeBytes    = 165_094L,
                assetPath    = "models/embeddinggemma-2/vision_encoder_quantized.onnx"
            ),
            // Large vision weight file: downloaded from HuggingFace (pinned revision).
            ModelFile(
                url          = "onnx/vision_encoder_quantized.onnx_data",
                relativeDest = "models/embeddinggemma-2/vision_encoder_quantized.onnx_data",
                sha256       = "3dabd69c0a36e9a8771ad82030dde74daa5a0e02b7047a5d3f3382b1137bab89",
                sizeBytes    = 195_228_672L
            )
        )
    )
}

// ---------------------------------------------------------------------------
// ModelProvisioner
// ---------------------------------------------------------------------------

/**
 * Downloads and verifies model files onto the device's internal storage.
 *
 * ### Usage
 * ```kotlin
 * val provisioner = ModelProvisioner(manifest = EmbeddingGemma2Manifest.manifest)
 * // Observe progress:
 * lifecycleScope.launch {
 *     provisioner.stateFlow.collect { state -> /* update UI */ }
 * }
 * // Trigger download (no-op if already provisioned):
 * provisioner.provision(context)
 * ```
 *
 * ### B6 Production path
 * [NaradApplication] holds a singleton [ModelProvisioner] instance.
 * the Settings tab observes [stateFlow] and calls [provision] when the user
 * taps "Start Indexing" and the model is absent.
 *
 * ### Download protocol
 * Uses [HttpURLConnection] only - **no new production dependency**.
 * Downloads happen on [Dispatchers.IO].
 *
 * ### Atomic install
 * Each file is streamed to `<finalPath>.tmp`, SHA-256 is computed inline
 * while streaming, then size AND sha256 are verified before
 * `File.renameTo(finalPath)`.  The final path is never left in a partial state.
 *
 * ### Resume (U8)
 * An IO error or cancellation KEEPS the `.tmp`. The next [provision] sends
 * `Range: bytes=<tmp length>-`: a 206 appends, a 200 means the server ignored the range, so the
 * `.tmp` is deleted and the body restarts from zero. The `.tmp` is also deleted on a size or
 * SHA-256 mismatch. Free space is checked before any byte downloads.
 *
 * ### Extending for additional models (M2+)
 * Construct a [ModelProvisioner] with a different [ModelManifest] (e.g.
 * [EmbeddingGemma2VisionManifest.manifest]) to provision the vision model independently.
 *
 * @param manifest  Describes the files to download.
 * @param destRootProvider  Lambda that returns the root [File] for final paths.
 *   Defaults to [Context.filesDir].  Override in tests for isolation.
 */
class ModelProvisioner(
    private val manifest: ModelManifest = EmbeddingGemma2Manifest.manifest,
    private val destRootProvider: ((Context) -> File)? = null
) {

    // U8: seam so tests can fake a full disk. Production reads the real free space. A property,
    // not a constructor parameter, so the trailing-lambda call ModelProvisioner(manifest) { root }
    // keeps binding to destRootProvider.
    internal var usableSpaceOf: (File) -> Long = { it.usableSpace }

    companion object {
        private const val TAG = "ModelProvisioner"

        /**
         * Default base URL for [provision] calls that do not supply an explicit URL.
         *
         * Points at the pinned EmbeddingGemma 2 HuggingFace revision used by
         * [EmbeddingGemma2Manifest]. HttpURLConnection follows the 302 redirect
         * to the HF CDN automatically (verified: `HttpURLConnection.setInstanceFollowRedirects`
         * defaults to true).
         *
         * The URL must end with `"/"` so relative file names are resolved correctly.
         */
        const val DEFAULT_MODEL_BASE_URL = EmbeddingGemma2Manifest.BASE_URL

        private const val BUFFER_SIZE = 64 * 1024  // 64 KB streaming buffer
        private const val CONNECT_TIMEOUT_MS = 15_000
        private const val READ_TIMEOUT_MS    = 30_000

        /** Free space kept on top of the bytes still to download (temp files, DB, user data). */
        internal const val FREE_SPACE_MARGIN_BYTES = 100L * 1024 * 1024
        private const val MB = 1024L * 1024
        private const val HTTP_RANGE_NOT_SATISFIABLE = 416
    }

    private val _stateFlow = MutableStateFlow<ProvisionState>(ProvisionState.NotStarted)
    private val provisionMutex = Mutex()

    /** Observe download and verification progress. Thread-safe. */
    val stateFlow: StateFlow<ProvisionState> = _stateFlow.asStateFlow()

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Provisions all files in [manifest] to [Context.filesDir] (or the test-overridden root).
     *
     * - **Already provisioned (size check passes):** emits [ProvisionState.Ready] and returns
     *   immediately without making any network requests.
     * - **Partial / missing files:** downloads each missing file from [baseUrl].
     * - **Hash / size mismatch on download:** emits [ProvisionState.Failed]; `.tmp` file
     *   is deleted; final path is never written.
     *
     * This function is safe to call from any coroutine context - it always moves work
     * onto [Dispatchers.IO] internally.
     *
     * @param context  Any [Context] - used only to locate [Context.filesDir].
     * @param baseUrl  CDN base URL.  Defaults to [DEFAULT_MODEL_BASE_URL] (pinned Hugging Face revision).
     */
    suspend fun provision(
        context: Context,
        baseUrl: String = DEFAULT_MODEL_BASE_URL
    ) = provisionMutex.withLock {
        provision(context, destRootProvider?.invoke(context) ?: context.filesDir, baseUrl)
    }

    internal suspend fun provision(
        destRoot: File,
        baseUrl: String = DEFAULT_MODEL_BASE_URL
    ) = provisionMutex.withLock {
        provision(null, destRoot, baseUrl)
    }

    private suspend fun provision(
        context: Context?,
        destRoot: File,
        baseUrl: String
    ) = withContext(Dispatchers.IO) {

        if (isProvisioned(destRoot)) {
            Log.i(TAG, "[${manifest.modelId}] Already provisioned - skipping download")
            _stateFlow.value = ProvisionState.Ready
            return@withContext
        }

        val totalFiles = manifest.files.size

        // U8: refuse to start a download the disk cannot hold. Bytes already in a kept .tmp
        // count as downloaded, so a resume is not blocked by its own partial file.
        val bytesNeeded = manifest.files.sumOf { mf ->
            val finalFile = File(destRoot, mf.relativeDest)
            if (finalFile.exists() && finalFile.length() == mf.sizeBytes) {
                0L
            } else {
                val tmp = File("${finalFile.absolutePath}.tmp")
                (mf.sizeBytes - (if (tmp.exists()) tmp.length() else 0L)).coerceAtLeast(0L)
            }
        }
        destRoot.mkdirs()
        val free = usableSpaceOf(destRoot)
        if (free < bytesNeeded + FREE_SPACE_MARGIN_BYTES) {
            val reason = "Not enough free space: need ${(bytesNeeded + FREE_SPACE_MARGIN_BYTES) / MB} MB, " +
                "have ${free / MB} MB"
            Log.w(TAG, "[${manifest.modelId}] $reason")
            _stateFlow.value = ProvisionState.Failed(reason)
            return@withContext
        }

        for ((index, modelFile) in manifest.files.withIndex()) {
            val finalFile = File(destRoot, modelFile.relativeDest)
            val tmpFile   = File("${finalFile.absolutePath}.tmp")
            val fileName  = finalFile.name

            // Skip if this specific file is already correctly sized.
            if (finalFile.exists() && finalFile.length() == modelFile.sizeBytes) {
                Log.d(TAG, "[$fileName] Already present with correct size - skipping")
                continue
            }

            // Ensure parent directory exists.
            finalFile.parentFile?.mkdirs()

            // Resolve the download URL.
            val downloadUrl = resolveUrl(baseUrl, modelFile.url)
            Log.i(TAG, "[$fileName] Downloading from $downloadUrl")

            try {
                if (modelFile.assetPath != null) {
                    // Copy from APK asset instead of downloading.
                    copyAssetAtomic(
                        context   = requireNotNull(context) {
                            "Context is required to provision asset-backed model files"
                        },
                        assetPath = modelFile.assetPath,
                        finalFile = finalFile,
                        tmpFile   = tmpFile,
                        modelFile = modelFile,
                        fileIndex = index,
                        fileCount = totalFiles
                    )
                } else {
                    downloadFileAtomic(
                        url       = downloadUrl,
                        finalFile = finalFile,
                        tmpFile   = tmpFile,
                        modelFile = modelFile,
                        fileIndex = index,
                        fileCount = totalFiles,
                        isActive  = { isActive }
                    )
                }
            } catch (e: CancellationException) {
                // U8: keep the partial .tmp so the next provision resumes it. Go back to idle so
                // observers do not show a download that is no longer running.
                _stateFlow.value = ProvisionState.NotStarted
                throw e
            } catch (e: Exception) {
                val reason = "[$fileName] Install failed: ${e.message}"
                Log.e(TAG, reason, e)
                // U8: the .tmp stays on IOException (resume). Mismatch paths delete it themselves.
                _stateFlow.value = ProvisionState.Failed(reason)
                return@withContext
            }
        }

        _stateFlow.value = ProvisionState.Ready
        Log.i(TAG, "[${manifest.modelId}] Provisioning complete")
    }

    /**
     * Quick existence + size check.  Does NOT re-verify SHA-256 (that is done on
     * fresh downloads only - avoids re-hashing the 94 MB model on every app start).
     *
     * @return `true` iff all manifest files exist at their final paths AND their
     *   byte count equals the expected [ModelFile.sizeBytes].
     */
    fun isProvisioned(context: Context): Boolean {
        val destRoot = destRootProvider?.invoke(context) ?: context.filesDir
        return isProvisioned(destRoot)
    }

    /**
     * Seeds [stateFlow] from disk after a process restart: files already installed mean
     * [ProvisionState.Ready], same as the first step of [provision]. Size check only.
     * Uses compareAndSet so it never overwrites Downloading, Verifying or Failed.
     * Does file IO: call off the main thread.
     */
    fun refreshFromDisk(context: Context) {
        if (isProvisioned(context)) {
            _stateFlow.compareAndSet(ProvisionState.NotStarted, ProvisionState.Ready)
        }
    }

    private fun isProvisioned(destRoot: File): Boolean {
        return manifest.files.all { modelFile ->
            val f = File(destRoot, modelFile.relativeDest)
            f.exists() && f.length() == modelFile.sizeBytes
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /**
     * Copy [assetPath] from the APK assets to [tmpFile], verify size + SHA-256, then
     * atomically rename to [finalFile].
     *
     * Emits [ProvisionState.Downloading] (reusing the progress state) while streaming.
     * Emits [ProvisionState.Verifying] before the hash check.
     * Throws on any verification failure so the caller can emit [ProvisionState.Failed].
     */
    private fun copyAssetAtomic(
        context: Context,
        assetPath: String,
        finalFile: File,
        tmpFile: File,
        modelFile: ModelFile,
        fileIndex: Int,
        fileCount: Int
    ) {
        val fileName = finalFile.name
        val digest = MessageDigest.getInstance("SHA-256")
        var bytesWritten = 0L

        context.assets.open(assetPath).use { input ->
            tmpFile.outputStream().buffered(BUFFER_SIZE).use { out ->
                val buf = ByteArray(BUFFER_SIZE)
                var read: Int
                while (input.read(buf).also { read = it } != -1) {
                    out.write(buf, 0, read)
                    digest.update(buf, 0, read)
                    bytesWritten += read
                    _stateFlow.value = ProvisionState.Downloading(
                        fileIndex       = fileIndex,
                        fileCount       = fileCount,
                        fileName        = fileName,
                        bytesDownloaded = bytesWritten,
                        bytesTotal      = modelFile.sizeBytes
                    )
                }
            }
        }

        val actualSize = tmpFile.length()
        if (actualSize != modelFile.sizeBytes) {
            tmpFile.delete()
            throw IOException(
                "[$fileName] Asset size mismatch: expected ${modelFile.sizeBytes}, got $actualSize"
            )
        }

        _stateFlow.value = ProvisionState.Verifying(fileName)

        val actualHash = digest.digest().joinToString("") { "%02x".format(it) }
        if (actualHash != modelFile.sha256) {
            tmpFile.delete()
            throw IOException(
                "[$fileName] Asset SHA-256 mismatch: expected ${modelFile.sha256}, got $actualHash"
            )
        }

        if (!tmpFile.renameTo(finalFile)) {
            tmpFile.delete()
            throw IOException("[$fileName] renameTo from asset copy failed")
        }

        Log.i(TAG, "[$fileName] Installed from asset at ${finalFile.absolutePath} ($actualSize bytes)")
    }

    /**
     * Download [url] to [tmpFile], verify size + SHA-256, then atomically rename to [finalFile].
     *
     * Resumes a kept [tmpFile] with an HTTP Range request (U8):
     *  - 206: the existing bytes are hashed, then the body is appended.
     *  - 416 to a Range request: [tmpFile] is deleted and the download restarts from zero.
     *  - 200 to a Range request: the server ignored the range, so [tmpFile] is deleted and the
     *    full body is written from zero.
     *  - [tmpFile] already at full size: no request, straight to verification.
     * Size or SHA-256 mismatch deletes [tmpFile]. An IOException or cancellation keeps it.
     *
     * Emits [ProvisionState.Downloading] progress callbacks while streaming.
     * Emits [ProvisionState.Verifying] before the hash check.
     * Throws on any verification failure so the caller can emit [ProvisionState.Failed].
     * Throws [CancellationException] when [isActive] turns false.
     */
    private fun downloadFileAtomic(
        url: String,
        finalFile: File,
        tmpFile: File,
        modelFile: ModelFile,
        fileIndex: Int,
        fileCount: Int,
        isActive: () -> Boolean = { true }
    ) {
        val fileName = finalFile.name

        // A kept .tmp longer than the expected size cannot be a valid prefix: start over.
        if (tmpFile.exists() && tmpFile.length() > modelFile.sizeBytes) tmpFile.delete()
        val resumeFrom = if (tmpFile.exists()) tmpFile.length() else 0L

        val digest = MessageDigest.getInstance("SHA-256")
        var bytesDownloaded = 0L

        // Fix 7: always disconnect in a finally block to release the underlying socket.
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.connectTimeout = CONNECT_TIMEOUT_MS
        connection.readTimeout    = READ_TIMEOUT_MS
        try {
            if (resumeFrom == modelFile.sizeBytes) {
                // The whole file was written but the process died before the rename.
                bytesDownloaded = hashExisting(tmpFile, digest)
            } else {
                if (resumeFrom > 0) connection.setRequestProperty("Range", "bytes=$resumeFrom-")
                connection.connect()

                val responseCode = connection.responseCode
                if (resumeFrom > 0 && responseCode == HTTP_RANGE_NOT_SATISFIABLE) {
                    // The kept .tmp is not a usable prefix (file changed, or offset past the end).
                    // Without this, every retry would send the same Range and get 416 again.
                    if (!tmpFile.delete()) throw IOException("could not delete ${tmpFile.name} after 416")
                    return downloadFileAtomic(url, finalFile, tmpFile, modelFile, fileIndex, fileCount, isActive)
                }
                val append = when {
                    resumeFrom > 0 && responseCode == HttpURLConnection.HTTP_PARTIAL -> true
                    responseCode == HttpURLConnection.HTTP_OK -> {
                        // Fresh download, or the server ignored the Range header: from zero.
                        if (resumeFrom > 0) tmpFile.delete()
                        false
                    }
                    else -> throw IOException("HTTP $responseCode for $url")
                }
                if (append) bytesDownloaded = hashExisting(tmpFile, digest)

                val contentLength = connection.contentLength.toLong()  // -1 if unknown
                val totalBytes = when {
                    append -> modelFile.sizeBytes
                    contentLength > 0 -> contentLength
                    else -> modelFile.sizeBytes
                }

                // Fix 9: throttle Downloading emissions to ~1% steps so the UI is not
                // flooded with 1500+ events for the 94 MB ONNX model.
                var lastEmittedPct = -1
                var received = 0L

                java.io.FileOutputStream(tmpFile, append).buffered(BUFFER_SIZE).use { out ->
                    connection.inputStream.use { input ->
                        val buf = ByteArray(BUFFER_SIZE)
                        var read: Int
                        while (input.read(buf).also { read = it } != -1) {
                            // U8: stop on cancellation. use{} flushes, so the .tmp keeps every byte read.
                            if (!isActive()) throw CancellationException("Model download cancelled")
                            out.write(buf, 0, read)
                            digest.update(buf, 0, read)
                            bytesDownloaded += read
                            received += read

                            // Emit only when percentage has advanced by at least 1 point.
                            val pct = if (totalBytes > 0)
                                (bytesDownloaded * 100 / totalBytes).toInt()
                            else 0
                            if (pct > lastEmittedPct) {
                                lastEmittedPct = pct
                                _stateFlow.value = ProvisionState.Downloading(
                                    fileIndex        = fileIndex,
                                    fileCount        = fileCount,
                                    fileName         = fileName,
                                    bytesDownloaded  = bytesDownloaded,
                                    bytesTotal       = totalBytes
                                )
                            }
                        }
                    }
                }

                // HttpURLConnection reports a connection that dropped mid-body as a normal end of
                // stream. A body shorter than its own Content-Length is a network drop, not a
                // wrong file: fail as an IOException so the .tmp is kept and resumed.
                val declared = connection.contentLengthLong
                if (declared > 0 && received < declared) {
                    throw IOException("[$fileName] Connection ended early: got $received of $declared bytes")
                }
            }

            // Verify size
            val actualSize = tmpFile.length()
            if (actualSize != modelFile.sizeBytes) {
                tmpFile.delete()
                throw IOException(
                    "[$fileName] Size mismatch: expected ${modelFile.sizeBytes}, got $actualSize"
                )
            }

            // Emit verifying state
            _stateFlow.value = ProvisionState.Verifying(fileName)

            // Verify SHA-256
            val actualHash = digest.digest().joinToString("") { "%02x".format(it) }
            if (actualHash != modelFile.sha256) {
                tmpFile.delete()
                throw IOException(
                    "[$fileName] SHA-256 mismatch: expected ${modelFile.sha256}, got $actualHash"
                )
            }

            // Fix 10: atomic rename - throw on failure rather than falling back to copyTo
            // which can leave a partial final file on-disk if interrupted.
            if (!tmpFile.renameTo(finalFile)) {
                tmpFile.delete()
                throw IOException(
                    "[$fileName] renameTo failed - tmp and final paths may be on different mount points; " +
                    "filesDir is internal storage and should never cross mounts"
                )
            }

            Log.i(TAG, "[$fileName] Installed at ${finalFile.absolutePath} ($actualSize bytes)")
        } finally {
            // Fix 7: release socket regardless of success or failure.
            connection.disconnect()
        }
    }

    /** Feeds the bytes already in [file] into [digest]; returns how many there were. */
    private fun hashExisting(file: File, digest: MessageDigest): Long {
        var total = 0L
        file.inputStream().buffered(BUFFER_SIZE).use { input ->
            val buf = ByteArray(BUFFER_SIZE)
            var read: Int
            while (input.read(buf).also { read = it } != -1) {
                digest.update(buf, 0, read)
                total += read
            }
        }
        return total
    }

    /**
     * Resolve a possibly-relative [fileUrl] against [baseUrl].
     *
     * If [fileUrl] starts with `"http"` it is used verbatim.
     * Otherwise it is appended to [baseUrl] with a single `/` separator.
     */
    private fun resolveUrl(baseUrl: String, fileUrl: String): String {
        return if (fileUrl.startsWith("http")) {
            fileUrl
        } else {
            val base = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"
            "$base$fileUrl"
        }
    }
}
