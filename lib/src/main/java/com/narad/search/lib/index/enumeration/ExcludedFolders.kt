package com.narad.search.lib.index.enumeration

import android.content.Context

/**
 * Index0: folders the index never reads, like Spotlight Privacy on macOS.
 *
 * Default rules: everything under `<volume root>/Android/` (app data and system files), and
 * every folder whose name starts with `.` (hidden folders). A dot-file in a visible folder is
 * still indexed. User entries are absolute folder paths, matched on a `/` boundary.
 *
 * One matcher ([isExcluded]) serves enumeration and the search-result filter.
 *
 * ponytail: an exclusion change drops rows through the normal prune of the next full pass.
 * The prune skips an empty listing (IndexRepository.pruneDocuments), so excluding every folder
 * that has documents leaves old rows until something is listed again. The search filter still
 * hides them. Per-folder delete if that matters.
 */
object ExcludedFolders {

    private const val LIB_PREFS = "narad_lib_prefs"
    private const val KEY_USER_FOLDERS = "excluded_folders"

    private val ANDROID_DIR = Regex("^/storage/(emulated/\\d+|[^/]+)/android(/|$)", RegexOption.IGNORE_CASE)

    fun userFolders(context: Context): List<String> =
        context.getSharedPreferences(LIB_PREFS, Context.MODE_PRIVATE)
            .getStringSet(KEY_USER_FOLDERS, emptySet()).orEmpty().sorted()

    fun add(context: Context, folder: String) = edit(context) { it += normalize(folder) }

    fun remove(context: Context, folder: String) = edit(context) { it -= normalize(folder) }

    private fun edit(context: Context, change: (MutableSet<String>) -> Unit) {
        val prefs = context.getSharedPreferences(LIB_PREFS, Context.MODE_PRIVATE)
        // getStringSet returns a set that must not be modified; copy it.
        val folders = prefs.getStringSet(KEY_USER_FOLDERS, emptySet()).orEmpty().toMutableSet()
        change(folders)
        prefs.edit().putStringSet(KEY_USER_FOLDERS, folders).apply()
    }

    internal fun normalize(folder: String): String = folder.trimEnd('/').ifEmpty { "/" }

    /** Pure: true when the file at [path] must not be indexed or shown. */
    fun isExcluded(path: String, userFolders: Collection<String>): Boolean {
        if (ANDROID_DIR.containsMatchIn(path)) return true
        // Every segment except the last (the file name) is a folder.
        val folders = path.split('/').dropLast(1)
        if (folders.any { it.startsWith('.') }) return true
        return userFolders.any { entry ->
            val folder = normalize(entry)
            folder == "/" || path == folder || path.startsWith("$folder/")
        }
    }
}
