/*
 * Copyright (c) 2022 Hai Zhang <dreaming.in.code.zh@gmail.com>
 * All Rights Reserved.
 */

package me.zhanghai.android.files.util

inline fun <reified T : Throwable> Throwable.findCauseByClass(): T? {
    var current: Throwable? = this
    do {
        if (current is T) {
            return current
        }
        current = current!!.cause
    } while (current != null)
    return null
}

// Index0: error text for the UI. Throwable.toString() shows the package name.
fun Throwable.toUiText(): String {
    val name = javaClass.simpleName
    val message = localizedMessage
    return if (message.isNullOrBlank()) name else "$name: $message"
}
