package com.narad.search.lib.index.worker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files

class RunMetricsTest {

    private val workerId = "123e4567-e89b-12d3-a456-426614174000"

    // ---- TSV rows -------------------------------------------------------------------------

    @Test
    fun `a row has the 12 header columns and nulls are dashes`() {
        assertEquals(
            "ts_utc\trun_id\tkind\tname\ttype\tsize_bytes\tms\tchunks\tresult\tpss_kb\textract_ms\tembed_ms",
            METRICS_TSV_HEADER,
        )
        val row = tsvRow(0L, "run1", "pss", name = "start", pssKb = 512)
        assertEquals(listOf("1970-01-01T00:00:00Z", "run1", "pss", "start", "-", "-", "-", "-", "-", "512", "-", "-"), row.split('\t'))
    }

    @Test
    fun `a file row puts extract and embed times in the last 2 columns`() {
        val row = tsvRow(0L, "run1", "file", name = "a.pdf", type = "application/pdf", sizeBytes = 10, ms = 9,
            chunks = 2, result = RESULT_INDEXED, extractMs = 3, embedMs = 5)
        assertEquals(listOf("a.pdf", "application/pdf", "10", "9", "2", "indexed", "-", "3", "5"), row.split('\t').drop(3))
    }

    @Test
    fun `tabs and line breaks in a name keep 12 columns, and only the part after the last slash stays`() {
        val row = tsvRow(0L, "run1", "file", name = "/sdcard/Docs/a\tb\nc\rd.pdf", type = "application/pdf",
            sizeBytes = 10, ms = 5, chunks = 2, result = RESULT_INDEXED)
        val cells = row.split('\t')
        assertEquals(12, cells.size)
        assertEquals("a b c d.pdf", cells[3])
        assertFalse(row.contains('\n') || row.contains('\r') || row.contains('/'.toString() + "sdcard"))
    }

    // ---- File cap ---------------------------------------------------------------------------

    @Test
    fun `appends stop at the cap and ask for the full warning once`() {
        val file = Files.createTempDirectory("metrics").resolve("m/index-metrics.tsv").toFile()
        var warnings = 0
        val m = RunMetrics(file, "run1", capBytes = 1024, now = { 0L }) { warnings++ }
        repeat(100) { m.file("doc$it.pdf", "application/pdf", 100, 3, 1, RESULT_INDEXED, 1, 2) }
        assertEquals(1, warnings)
        assertTrue(file.length() <= 1024)
        val lines = file.readLines()
        assertEquals(METRICS_TSV_HEADER, lines.first())
        assertTrue(lines.drop(1).all { it.split('\t').size == 12 })

        // A later run on a full file writes nothing and does not warn again.
        val before = file.length()
        RunMetrics(file, "run2", capBytes = 1024, now = { 0L }) { warnings++ }.file("x", null, null, 1, 0, RESULT_EMPTY, 1, 0)
        assertEquals(1, warnings)
        assertEquals(before, file.length())
    }

    @Test
    fun `event cap is 5000 in a metrics build and 500 otherwise`() {
        assertEquals(5_000, eventCap(metrics = true))
        assertEquals(500, eventCap(metrics = false))
    }

    // ---- Results ------------------------------------------------------------------------------

    @Test
    fun `document result comes from chunks and the extraction warning`() {
        assertEquals(RESULT_INDEXED, docResult(chunks = 3, warned = false))
        assertEquals(RESULT_INDEXED, docResult(chunks = 3, warned = true))
        assertEquals(RESULT_EXTRACT_FAIL, docResult(chunks = 0, warned = true))
        assertEquals(RESULT_EMPTY, docResult(chunks = 0, warned = false))
    }

    // ---- Run-level event text -------------------------------------------------------------

    @Test
    fun `run lines fit the 200 character event cap with every field`() {
        val start = runStartMessage(workerId, 3, foreground = true, charging = false, batteryPct = 87, idle = true)
        assertEquals("start worker=$workerId attempt=3 fg=true charging=false battery=87 idle=true", start)

        val gate = gateMessage(Long.MAX_VALUE, Long.MAX_VALUE, false, Long.MAX_VALUE, fits = true)
        assertTrue(gate.startsWith("gate availMem="))
        assertTrue(gate.endsWith("decision=run"))
        val defer = gateMessage(1L, 2L, true, 3L, fits = false)
        assertEquals("photos deferred: low memory availMem=1 threshold=2 lowMemory=true totalMem=3 decision=defer", defer)

        val end = runEndMessage("deferred", 331, 331, 0, 900, 12, Long.MAX_VALUE, stopReason = 1)
        assertEquals("end outcome=deferred docs=331/331 photos=0/900 read=12 skipped=319 ms=${Long.MAX_VALUE} stop=1", end)

        listOf(start, gate, defer, end).forEach { assertTrue(it, it.length <= 200) }
    }

    @Test
    fun `unknown device state and no stop reason print dashes`() {
        assertTrue(runStartMessage(workerId, 1, false, null, null, null).endsWith("charging=- battery=- idle=-"))
        assertTrue(runEndMessage("complete", 0, 0, 0, 0, 0, 5, stopReason = null).endsWith("stop=-"))
    }
}
