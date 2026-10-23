/*
 * Copyright (c) 2019 Hai Zhang <dreaming.in.code.zh@gmail.com>
 * All Rights Reserved.
 */

package me.zhanghai.android.files.filelist

import android.os.AsyncTask
import java8.nio.file.Path
import me.zhanghai.android.files.file.FileItem
import me.zhanghai.android.files.file.loadFileItem
import me.zhanghai.android.files.index.SemanticSearch
import me.zhanghai.android.files.provider.common.search
import me.zhanghai.android.files.util.CloseableLiveData
import me.zhanghai.android.files.util.Failure
import me.zhanghai.android.files.util.Loading
import me.zhanghai.android.files.util.Stateful
import me.zhanghai.android.files.util.Success
import me.zhanghai.android.files.util.valueCompat
import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Future

class SearchFileListLiveData(
    private val path: Path,
    private val query: String
) : CloseableLiveData<Stateful<List<FileItem>>>() {
    private var future: Future<Unit>? = null

    init {
        loadValue()
    }

    fun loadValue() {
        future?.cancel(true)
        value = Loading(emptyList())
        future = (AsyncTask.THREAD_POOL_EXECUTOR as ExecutorService).submit<Unit> {
            val fileList = mutableListOf<FileItem>()
            try {
                // Index0: meaning matches first, then name matches. A file in both shows once.
                val semanticPaths = SemanticSearch.search(path, query)
                for (semanticPath in semanticPaths) {
                    fileList.add(
                        try {
                            semanticPath.loadFileItem()
                        } catch (e: IOException) {
                            // Stale index row (file moved or deleted).
                            continue
                        }
                    )
                }
                if (Thread.interrupted()) {
                    return@submit
                }
                if (fileList.isNotEmpty()) {
                    postValue(Loading(fileList.toList()))
                }
                val semanticPathSet = semanticPaths.toSet()
                path.search(query, INTERVAL_MILLIS) { paths: List<Path> ->
                    for (path in paths) {
                        if (path in semanticPathSet) {
                            continue
                        }
                        val fileItem = try {
                            path.loadFileItem()
                        } catch (e: IOException) {
                            e.printStackTrace()
                            // TODO: Support file without information.
                            continue
                        }
                        fileList.add(fileItem)
                    }
                    postValue(Loading(fileList.toList()))
                }
                postValue(Success(fileList))
            } catch (e: Exception) {
                // TODO: Retrieval of previous value is racy.
                postValue(Failure(valueCompat.value, e))
            }
        }
    }

    override fun close() {
        future?.cancel(true)
    }

    companion object {
        private const val INTERVAL_MILLIS = 500L
    }
}
