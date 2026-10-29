package com.narad.search.lib.index

import android.content.Context
import com.narad.search.lib.index.service.IndexService
import java.io.File

/**
 * Dummy artefacts at [IndexService.getDefaultModelPath] so `modelFilesReady` is true and the
 * delta enqueue guard lets the enqueue through. Only files that were missing are created and
 * removed again, so a real model already on the device is never touched.
 */
class FakeModelFiles(context: Context) {
    private val graph = File(IndexService.getDefaultModelPath(context))
    private val created = mutableListOf<File>()

    fun create() {
        graph.parentFile!!.mkdirs()
        listOf(graph, File(graph.path + "_data"), graph.resolveSibling("tokenizer.json"))
            .filter { !it.exists() }
            .forEach { it.writeBytes(ByteArray(1)); created += it }
    }

    fun delete() {
        created.forEach { it.delete() }
        created.clear()
    }
}
