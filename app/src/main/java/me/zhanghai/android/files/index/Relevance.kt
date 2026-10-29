package me.zhanghai.android.files.index

// ponytail: fixed floors calibrated on the narad36 sample set (28 files, EmbeddingGemma 2,
// 13 Oct 2026). True matches scored 0.70 to 0.81, noise 0.55 to 0.68. Re-tune on a real phone
// corpus. A miss costs one tap, not a lost file: weak hits still show under "Less related".
internal const val DOC_FLOOR   = 0.69f
internal const val DOC_GAP     = 0.035f
internal const val PHOTO_FLOOR = 0.69f
internal const val PHOTO_GAP   = 0.05f

/**
 * Pure: which hits are strong matches. A hit is strong when its raw cosine clears both the
 * absolute [floor] and the band [gap] below the best hit, so one clear winner is not padded
 * with near-misses and a query with no real match returns no strong hits at all.
 */
internal fun strongMask(cosines: List<Float>, floor: Float, gap: Float): List<Boolean> {
    val top = cosines.maxOrNull() ?: return emptyList()
    val cut = maxOf(floor, top - gap)
    return cosines.map { it >= cut }
}

/**
 * Pure: strong documents. A document is strong when its cosine passes [strongMask], or when its
 * keyword matches cover every meaningful query word (in any chunks). A query with no meaningful
 * [terms] ("the", "a") marks nothing strong, so the cosine of a stopword never claims a match.
 * Keyword-only search (low-memory tier) has no cosines, so there only the coverage rule applies.
 */
internal fun strongDocs(cosines: List<Float>, matched: List<Set<String>>, terms: List<String>): List<Boolean> {
    if (terms.isEmpty()) return cosines.map { false }
    val mask = strongMask(cosines, DOC_FLOOR, DOC_GAP)
    return mask.mapIndexed { i, byCosine -> byCosine || matched[i].containsAll(terms) }
}

/** Pure: strong photos. Same empty-terms rule as [strongDocs]; photos have no keyword signal. */
internal fun strongPhotos(cosines: List<Float>, terms: List<String>): List<Boolean> =
    if (terms.isEmpty()) cosines.map { false } else strongMask(cosines, PHOTO_FLOOR, PHOTO_GAP)

/** Weak floor (KTD4): a row under this cosine with no keyword hit is dropped, not shown. */
internal const val WEAK_FLOOR = 0.62f

/** Cap on "Less related" rows, documents and photos together. */
internal const val LESS_RELATED_MAX = 10

internal enum class Bucket { STRONG, WEAK, DROP }

internal data class Hit(val strong: Boolean, val cosine: Float, val keywordHit: Boolean)

/**
 * Pure: sort every hit into STRONG, WEAK (shown under "Less related") or DROP. Weak means cosine
 * at least [WEAK_FLOOR] or any keyword hit. Only the best [LESS_RELATED_MAX] weak rows across
 * documents and photos survive; a keyword-only hit ranks as if it sat at the floor.
 */
internal fun bucketHits(docs: List<Hit>, photos: List<Hit>): Pair<List<Bucket>, List<Bucket>> {
    fun bucket(h: Hit) = when {
        h.strong -> Bucket.STRONG
        h.cosine >= WEAK_FLOOR || h.keywordHit -> Bucket.WEAK
        else -> Bucket.DROP
    }
    val d = docs.map(::bucket).toMutableList()
    val p = photos.map(::bucket).toMutableList()
    fun rank(h: Hit) = if (h.keywordHit) maxOf(h.cosine, WEAK_FLOOR) else h.cosine
    // Candidates are (isDoc, index, rank); sortedByDescending is stable, so docs win ties.
    val weak = docs.indices.filter { d[it] == Bucket.WEAK }.map { Triple(true, it, rank(docs[it])) } +
        photos.indices.filter { p[it] == Bucket.WEAK }.map { Triple(false, it, rank(photos[it])) }
    for ((isDoc, i, _) in weak.sortedByDescending { it.third }.drop(LESS_RELATED_MAX)) {
        if (isDoc) d[i] = Bucket.DROP else p[i] = Bucket.DROP
    }
    return d to p
}

/** The rows ResultsContent shows: main (strong) lists, and the folded "Less related" lists. */
internal data class RowSplit(
    val mainDocs: List<DocRow>,
    val mainPhotos: List<PhotoRow>,
    val lessDocs: List<DocRow>,
    val lessPhotos: List<PhotoRow>,
) {
    /** Nothing passed the strong cutoff: show "No close matches" and fold every row. */
    val noStrong: Boolean get() = mainDocs.isEmpty() && mainPhotos.isEmpty()
}

/** Pure: chip counts and the main list are strong rows only. Everything else is "Less related". */
internal fun splitRows(docs: List<DocRow>, photos: List<PhotoRow>) = RowSplit(
    mainDocs = docs.filter { it.strong },
    mainPhotos = photos.filter { it.strong },
    lessDocs = docs.filterNot { it.strong },
    lessPhotos = photos.filterNot { it.strong },
)
