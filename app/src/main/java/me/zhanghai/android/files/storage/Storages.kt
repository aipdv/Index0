/*
 * Copyright (c) 2019 Hai Zhang <dreaming.in.code.zh@gmail.com>
 * All Rights Reserved.
 */

package me.zhanghai.android.files.storage

import me.zhanghai.android.files.app.Analytics
import me.zhanghai.android.files.settings.Settings
import me.zhanghai.android.files.util.removeFirst
import me.zhanghai.android.files.util.valueCompat

object Storages {
    fun addOrReplace(storage: Storage) {
        val storages = Settings.STORAGES.valueCompat.toMutableList().apply {
            val index = indexOfFirst { it.id == storage.id }
            if (index != -1) {
                this[index] = storage
            } else {
                this += storage
                captureStorageAdded(storage)
            }
        }
        Settings.STORAGES.putValue(storages)
    }

    // Index0: the server type only, never a host, user name or path.
    private fun captureStorageAdded(storage: Storage) {
        val type = when (storage) {
            is SmbServer -> "smb"
            is SftpServer -> "sftp"
            is FtpServer -> "ftp"
            is WebDavServer -> "webdav"
            else -> return
        }
        Analytics.capture("storage_added", mapOf("type" to type))
    }

    fun replace(storage: Storage) {
        val storages = Settings.STORAGES.valueCompat.toMutableList()
            .apply { this[indexOfFirst { it.id == storage.id }] = storage }
        Settings.STORAGES.putValue(storages)
    }

    fun move(fromPosition: Int, toPosition: Int) {
        val bookmarkDirectories = Settings.STORAGES.valueCompat.toMutableList()
            .apply { add(toPosition, removeAt(fromPosition)) }
        Settings.STORAGES.putValue(bookmarkDirectories)
    }

    fun remove(storage: Storage) {
        val bookmarkDirectories = Settings.STORAGES.valueCompat.toMutableList()
            .apply { removeFirst { it.id == storage.id } }
        Settings.STORAGES.putValue(bookmarkDirectories)
    }
}
