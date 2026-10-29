package me.zhanghai.android.files.index

import android.content.ContentUris
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import com.narad.search.lib.index.enumeration.ExcludedFolders
import com.narad.search.lib.index.provision.ProvisionState
import java8.nio.file.Path
import java8.nio.file.Paths
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import me.zhanghai.android.files.app.application
import me.zhanghai.android.files.provider.linux.isLinuxPath

/** Meaning-based search for the file list. Results are local file paths under the current folder. */
object SemanticSearch {
    private const val TAG = "SemanticSearch"
    private const val TOP_K = 20

    /**
     * Strong semantic hits under [folder], best first. Empty when [folder] is not on local
     * storage, the model is not ready, or anything fails: name search must never break.
     */
    fun search(folder: Path, query: String): List<Path> {
        val scope = if (folder.isLinuxPath) folder.toString() else null
        if (semanticScope(scope) == null || query.isBlank()) return emptyList()
        // No model yet: do not touch the engine (it would create an empty index file).
        if (IndexEngine.provisioner.stateFlow.value != ProvisionState.Ready) return emptyList()
        return try {
            val output = runBlocking { IndexEngine.searchController.search(query, TOP_K) }
            val uris = output.docs.filter { it.strong }.map { it.filePath } +
                output.photos.filter { it.strong }.map { it.filePath }
            val paths = resolvePaths(uris)
            scopeHits(uris.map { paths[it] }, scope, ExcludedFolders.userFolders(application))
                .map { Paths.get(it) }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "Semantic search failed, name search only", e)
            emptyList()
        } catch (e: LinkageError) {
            Log.w(TAG, "Semantic search failed, name search only", e)
            emptyList()
        }
    }

    /** `content://media/.../<id>` (or a raw path) to its file path, one MediaStore query. */
    private fun resolvePaths(urisOrPaths: List<String>): Map<String, String> {
        val out = HashMap<String, String>()
        val idToUri = HashMap<String, String>()
        for (item in urisOrPaths) {
            if (item.startsWith("content://")) {
                idToUri[ContentUris.parseId(Uri.parse(item)).toString()] = item
            } else {
                out[item] = item
            }
        }
        if (idToUri.isEmpty()) return out
        val ids = idToUri.keys.toList()
        application.contentResolver.query(
            MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL),
            arrayOf(MediaStore.Files.FileColumns._ID, MediaStore.Files.FileColumns.DATA),
            "${MediaStore.Files.FileColumns._ID} IN (${ids.joinToString(",") { "?" }})",
            ids.toTypedArray(),
            null
        )?.use { cursor ->
            while (cursor.moveToNext()) {
                val path = cursor.getString(1) ?: continue
                idToUri[cursor.getLong(0).toString()]?.let { out[it] = path }
            }
        }
        return out
    }
}

/**
 * Pure: the folder semantic search runs in, or null. Only shared storage volumes are indexed,
 * so the device root, app folders and remote or archive paths (null) get name search only.
 */
internal fun semanticScope(folder: String?): String? =
    folder?.takeIf { it.startsWith("/storage/") }?.trimEnd('/')

/**
 * Pure: keep resolved hits (null = did not resolve) that sit under [folder] on a `/`
 * boundary and are not excluded. Order kept, duplicates dropped.
 */
internal fun scopeHits(resolved: List<String?>, folder: String?, userExcluded: Collection<String>): List<String> {
    val scope = semanticScope(folder) ?: return emptyList()
    return resolved.asSequence()
        .filterNotNull()
        .filter { it.startsWith("$scope/") }
        .filterNot { ExcludedFolders.isExcluded(it, userExcluded) }
        .distinct()
        .toList()
}
