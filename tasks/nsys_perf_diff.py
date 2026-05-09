"""Quantify where the new pipeline spends extra wall time vs legacy.

Reads two nsys SQLite reports and reports per-rank-averaged:
  * Total host time spent in CUDA Runtime API (CPU dispatch budget)
  * Top runtime APIs by cumulative ns
  * Cross-stream sync API counts (cudaStreamWaitEvent vs cudaStreamSynchronize)
  * NVTX range time per top symbol (HSTU stage timings)
  * Empty-stream gap time: max(end) - min(start) - sum(kernel busy on stream)
"""

from __future__ import annotations

import sqlite3
import sys


def q(con, sql, *args):
    cur = con.cursor()
    cur.execute(sql, args)
    return cur.fetchall()


def runtime_api_top(con, n=15):
    return q(
        con,
        """
        SELECT s.value AS api,
               COUNT(*) AS calls,
               SUM(end - start) AS total_ns,
               AVG(end - start) AS avg_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        GROUP BY s.value
        ORDER BY total_ns DESC
        LIMIT ?
        """,
        n,
    )


def per_rank_runtime_total(con):
    """Total host time the process(es) spent inside any CUDA runtime
    API call. Higher = more CPU-side dispatch work / sync wait."""
    rows = q(
        con,
        """
        SELECT COUNT(DISTINCT globalTid) AS ranks,
               SUM(end - start) AS total_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        """,
    )
    return rows[0]


def stream_idle_gap(con):
    """For each stream, idle time = capture window - sum(kernel busy).
    A higher idle gap means the GPU has more bubbles between kernels.
    """
    rows = q(
        con,
        """
        SELECT MIN(start) AS t0, MAX(end) AS t1
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        """,
    )
    t0, t1 = rows[0]
    span = t1 - t0
    out = q(
        con,
        """
        SELECT streamId, SUM(end - start) AS busy_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY streamId
        ORDER BY busy_ns DESC
        """,
    )
    rows = []
    for sid, busy in out:
        rows.append((sid, span, busy, span - busy))
    return rows


def nvtx_per_step(con):
    """NVTX ranges with `step` in their text — wall time per training
    iteration as recorded by torch.cuda.nvtx.range_push("step N")."""
    rows = q(
        con,
        """
        SELECT s.value AS name,
               COUNT(*) AS n,
               SUM(e.end - e.start) AS total_ns,
               AVG(e.end - e.start) AS avg_ns
        FROM NVTX_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE s.value LIKE 'step %'
          AND e.eventType IN (59, 60, 70, 75)
        GROUP BY s.value
        ORDER BY MIN(e.start)
        """,
    )
    return rows


def empty_stream_blocks(con, threshold_us=100):
    """Largest contiguous gaps on the busiest stream (compute) where no
    kernel is running. These are the GPU bubbles we want to eliminate.
    """
    # Pick the busiest stream (by total kernel time)
    busy = q(
        con,
        """
        SELECT streamId, SUM(end - start) AS s
        FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY streamId
        ORDER BY s DESC LIMIT 1
        """,
    )
    if not busy:
        return []
    sid = busy[0][0]
    rows = q(
        con,
        """
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE streamId = ?
        ORDER BY start
        """,
        sid,
    )
    gaps = []
    prev_end = rows[0][1]
    for s, e in rows[1:]:
        if s > prev_end:
            gap = s - prev_end
            if gap > threshold_us * 1000:
                gaps.append(gap)
        if e > prev_end:
            prev_end = e
    gaps.sort(reverse=True)
    return sid, gaps


def report(con, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    ranks, total = per_rank_runtime_total(con)
    print(f"Distinct host TIDs in runtime: {ranks}")
    print(f"Total host time in runtime API: {total / 1e6:.2f} ms (sum across TIDs)")
    print(f"  Per-TID avg:                  {total / ranks / 1e6:.2f} ms")

    print("\nTop runtime APIs by total host time:")
    for api, calls, total_ns, avg_ns in runtime_api_top(con, n=12):
        print(
            f"  {api:50s}  calls={calls:>8d}  "
            f"total={total_ns / 1e6:>9.2f} ms  avg={avg_ns:>10.0f} ns"
        )

    print("\nPer-stream idle gap (capture window − sum kernel busy):")
    for sid, span, busy, idle in stream_idle_gap(con)[:6]:
        print(
            f"  stream {sid}: span={span / 1e6:.2f} ms  "
            f"busy={busy / 1e6:.2f} ms  idle={idle / 1e6:.2f} ms  "
            f"({idle / span * 100:.1f}%)"
        )

    sid_gaps = empty_stream_blocks(con, threshold_us=100)
    if sid_gaps:
        sid, gaps = sid_gaps
        print(f"\nLargest GPU bubbles on busiest stream {sid} " f"(>100us, top 10):")
        for g in gaps[:10]:
            print(f"  {g / 1e6:.3f} ms")
        print(
            f"  total bubbles >100us: count={len(gaps)} "
            f"sum={sum(gaps) / 1e6:.2f} ms"
        )

    iters = nvtx_per_step(con)
    if iters:
        # Sample a few steady-state iters (skip first 5)
        sample = iters[5:25]
        if sample:
            avg = sum(r[2] for r in sample) / len(sample) / 1e6
            print(
                f"\nNVTX 'step N' per-iter (steady, avg of {len(sample)}): "
                f"{avg:.2f} ms"
            )


def main():
    legacy_db = sys.argv[1]
    new_db = sys.argv[2]
    with sqlite3.connect(f"file:{legacy_db}?mode=ro", uri=True) as con:
        report(con, f"LEGACY: {legacy_db}")
    with sqlite3.connect(f"file:{new_db}?mode=ro", uri=True) as con:
        report(con, f"NEW:    {new_db}")


if __name__ == "__main__":
    main()
