package me.zhanghai.android.files.index

import com.narad.search.lib.query.store.queryTerms

// Pure helpers copied from narad SearchScreenContent.kt (the Compose parts stay in Narad).

/**
 * Pure: merged character ranges of [snippet] to set in bold for the query [terms].
 * A term must start a word ("train" marks "Train" and "trains", never "restrain"), and the mark
 * runs to the end of that word, so "rent" marks all of "Rental" (KTD9). Terms of one or two
 * letters must be the whole word, so Hinglish "ka" does not mark "kaju" or "katli".
 */
internal fun highlightRanges(snippet: String, terms: List<String>): List<IntRange> {
    val ranges = mutableListOf<IntRange>()
    for (term in terms) {
        if (term.isBlank() || term.length > snippet.length) continue
        for (idx in 0..snippet.length - term.length) {
            if (!snippet.regionMatches(idx, term, 0, term.length, ignoreCase = true)) continue
            var end = idx + term.length
            val startsWord = idx == 0 || !isWordCodePoint(snippet.codePointBefore(idx))
            val endsWord = end == snippet.length || !isWordCodePoint(snippet.codePointAt(end))
            if (startsWord && (term.codePointCount(0, term.length) > 2 || endsWord)) {
                while (end < snippet.length && isWordCodePoint(snippet.codePointAt(end))) {
                    end += Character.charCount(snippet.codePointAt(end))
                }
                ranges += idx until end
            }
        }
    }
    return ranges.sortedBy { it.first }.fold(mutableListOf()) { acc, r ->
        if (acc.isNotEmpty() && r.first <= acc.last().last + 1) {
            acc[acc.lastIndex] = acc.last().first..maxOf(acc.last().last, r.last)
        } else {
            acc += r
        }
        acc
    }
}

internal fun isWordCodePoint(codePoint: Int): Boolean {
    if (Character.isLetterOrDigit(codePoint)) return true
    return when (Character.getType(codePoint)) {
        Character.NON_SPACING_MARK.toInt(),
        Character.COMBINING_SPACING_MARK.toInt(),
        Character.ENCLOSING_MARK.toInt() -> true
        else -> false
    }
}

/**
 * Pure: the query words worth showing in bold, lowercased, in
 * query order. Uses the one shared stopword list in lib ([queryTerms]), so "the", "by" and
 * Hinglish "ka" are left out here and in the keyword ranking alike.
 */
internal fun meaningfulTerms(query: String): List<String> = queryTerms(query)
