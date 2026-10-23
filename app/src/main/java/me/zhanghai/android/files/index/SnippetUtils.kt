package me.zhanghai.android.files.index

/** File type for display (file-type tile colours in the UI). */
enum class FileKind { PDF, DOC, TXT, IMAGE }

/**
 * Map a MIME type string to a [FileKind].
 *
 * null or unknown MIME -> TXT (safe fallback).
 */
fun fileKindFromMime(mime: String?): FileKind {
    if (mime == null) return FileKind.TXT
    return when {
        mime == "application/pdf"                           -> FileKind.PDF
        mime.startsWith("application/msword")               -> FileKind.DOC
        mime.startsWith("application/vnd.openxmlformats-officedocument.wordprocessingml") -> FileKind.DOC
        mime.startsWith("application/vnd.oasis.opendocument.text") -> FileKind.DOC
        mime.startsWith("image/")                           -> FileKind.IMAGE
        mime.startsWith("text/")                            -> FileKind.TXT
        else                                                -> FileKind.TXT
    }
}

/** How many characters of context the snippet shows before the first matched word. */
private const val SNIPPET_LEAD = 25

/**
 * Build a snippet of at most [maxLen] characters that starts about [SNIPPET_LEAD] characters
 * before the first hit of any [queryTerms] in [content]. Pass meaningful terms only (see
 * [meaningfulTerms]): a hit is a word start, the same rule as [highlightRanges], so "train"
 * never anchors on "restrain". No hit means a prefix window.
 *
 * All operations are char-index based (Kotlin String is UTF-16), so Devanagari and
 * other multi-byte scripts are handled correctly -- no byte-index arithmetic.
 *
 * Bug Watch: "Hindi/Hinglish snippet centering breaks on Devanagari."
 *
 * @param content    Full chunk text.
 * @param queryTerms Terms to search for (case-insensitive). Empty list means prefix window.
 * @param maxLen     Maximum character length of the returned snippet.
 * @param startsMidText True for a chunk after the first: its first characters are a raw cut, so
 *                   the snippet drops the partial first word and opens with "…".
 * @return Snippet string, trimmed, never longer than [maxLen] chars.
 */
fun buildSnippet(content: String, queryTerms: List<String>, maxLen: Int = 160, startsMidText: Boolean = false): String {
    if (content.isEmpty()) return ""
    if (content.length <= maxLen && !startsMidText) return content.trim()

    val hit = highlightRanges(content, queryTerms).firstOrNull()?.first
    val firstHit = hit ?: 0

    // Never open mid-word: step back to the word start. A hit near the end of the text gives a
    // shorter snippet, so the hit stays in view at the front. When the text itself begins at a raw
    // cut ([startsMidText]), position 0 is a cut too: skip the partial word (never past the hit).
    // Give up after a few chars (a very long word): the leading "…" marks the hard cut.
    val reach = maxLen / 4
    var start = (firstHit - minOf(SNIPPET_LEAD, reach)).coerceAtLeast(0)
    fun midWord(i: Int) =
        i in 1 until content.length && isWordCodePoint(content.codePointBefore(i)) && isWordCodePoint(content.codePointAt(i))
    var s = start
    while (s > start - reach && s > 0 && midWord(s)) s--
    if (!midWord(s)) start = s
    if (startsMidText && start == 0) {
        // No hit (meaning-only match, stopword query): the cap is the reach alone.
        val cap = if (hit == null) reach else minOf(reach, hit)
        while (start < cap && isWordCodePoint(content.codePointAt(start))) start++
        // Then skip the punctuation and spaces after the partial word: "ure, from" opens at "from".
        while (start < cap && !isWordCodePoint(content.codePointAt(start))) start++
    }

    val lead = if (start > 0 || startsMidText) "\u2026" else ""
    return lead + content.substring(start, (start + maxLen - lead.length).coerceAtMost(content.length)).trim()
}
