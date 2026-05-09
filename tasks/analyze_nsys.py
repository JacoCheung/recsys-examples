"""Compare nsys reports for legacy vs new pipeline.

Probes three structural goals:
  1. Multiple threads submitting different streams in parallel
  2. Effective overlap (every comm has a kernel/memcpy below)
  3. Sync not blocking host from submitting subsequent kernels

Usage:
  python tasks/analyze_nsys.py <legacy.sqlite> <new.sqlite>
"""

from __future__ import annotations

import sqlite3
import sys


def q(con, sql, *args):
    cur = con.cursor()
    cur.execute(sql, args)
    return cur.fetchall()


def list_tables(con):
    return [r[0] for r in q(con, "SELECT name FROM sqlite_master WHERE type='table'")]


def stream_count(con):
    """Distinct CUDA streams used in capture range."""
    rows = q(
        con,
        "SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL",
    )
    return rows[0][0] if rows else 0


def streams_with_kernels(con):
    """For each stream, count kernels + total time."""
    return q(
        con,
        """
        SELECT streamId, COUNT(*) AS n, SUM(end - start) AS total_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY streamId
        ORDER BY total_ns DESC
        """,
    )


def runtime_threads(con):
    """Distinct host thread IDs that issued CUDA runtime calls."""
    rows = q(
        con,
        """
        SELECT COUNT(DISTINCT globalTid)
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        """,
    )
    return rows[0][0] if rows else 0


def threads_per_stream(con):
    """How many distinct host threads launch onto each CUDA stream?
    A stream getting work from >=2 threads means the executor is
    multi-threading kernel submission to that stream — i.e. CPU side
    isn't bottlenecked on a single thread.
    """
    return q(
        con,
        """
        SELECT k.streamId,
               COUNT(DISTINCT r.globalTid) AS thread_count,
               COUNT(*) AS n
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
          ON k.correlationId = r.correlationId
        GROUP BY k.streamId
        ORDER BY n DESC
        """,
    )


def stream_time_overlap(con):
    """Total kernel busy time per stream + capture-range duration so we
    can see whether multiple streams are simultaneously active.

    Heuristic for overlap: if SUM(per-stream busy) > capture range
    duration, multiple streams ran concurrently for non-trivial time.
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
    streams = q(
        con,
        """
        SELECT streamId, SUM(end - start) AS busy_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY streamId
        ORDER BY busy_ns DESC
        """,
    )
    total_busy = sum(b for _, b in streams)
    return span, streams, total_busy


def comm_with_compute_neighbors(con, comm_substr="ncclDevKernel"):
    """For each comm kernel on stream X, find concurrent compute on
    stream Y at the same wall-clock window. Returns the fraction of
    comm time that overlaps with at least one compute kernel.
    """
    # Comm kernels (NCCL)
    comms = q(
        con,
        """
        SELECT k.start, k.end, k.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE s.value LIKE ?
        ORDER BY k.start
        """,
        f"%{comm_substr}%",
    )
    if not comms:
        return 0.0, 0
    # Compute kernels (everything not NCCL)
    computes = q(
        con,
        """
        SELECT k.start, k.end, k.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE s.value NOT LIKE ?
        ORDER BY k.start
        """,
        f"%{comm_substr}%",
    )

    # For overlap, sweep-line interval intersection per comm.
    # To keep it cheap, bucket computes per stream and walk.
    overlapped_ns = 0
    total_comm_ns = 0
    # Pre-sort once.
    cstart = [c[0] for c in computes]
    cend = [c[1] for c in computes]
    n_compute = len(computes)
    # Simple linear scan per comm; comms are usually << computes so OK.
    j = 0
    for cs, ce, sid in comms:
        total_comm_ns += ce - cs
        # Find first compute that ends after cs
        while j < n_compute and cend[j] < cs:
            j += 1
        # Walk forward to find every compute starting before ce
        ovl = 0
        k = j
        while k < n_compute and cstart[k] < ce:
            comp_s = cstart[k]
            comp_e = cend[k]
            comp_sid = computes[k][2]
            # Skip same-stream (same stream forces serial)
            if comp_sid != sid:
                lo = max(cs, comp_s)
                hi = min(ce, comp_e)
                if hi > lo:
                    ovl += hi - lo
            k += 1
        # Cap overlap at comm window
        ovl = min(ovl, ce - cs)
        overlapped_ns += ovl
    pct = (overlapped_ns / total_comm_ns * 100.0) if total_comm_ns else 0.0
    return pct, len(comms)


def host_sync_blocking(con):
    """Detect host-blocking syncs that bottleneck submission.

    A cudaStreamSynchronize / cudaDeviceSynchronize / cudaEventSynchronize
    that lasts >> the time required for the kernel queue to drain
    indicates the host is sitting idle waiting on the GPU. Healthy
    pattern: brief syncs only.
    """
    return q(
        con,
        """
        SELECT s.value AS api, COUNT(*) AS n,
               SUM(r.end - r.start) AS total_ns,
               AVG(r.end - r.start) AS avg_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE 'cuda%Synchronize%'
           OR s.value LIKE 'cudaEventSynchronize%'
           OR s.value LIKE 'cudaDeviceSynchronize%'
        GROUP BY s.value
        ORDER BY total_ns DESC
        """,
    )


def nvtx_top(con, n=20):
    """Top NVTX named ranges (by count) to confirm engine task names
    appear (e.g. h2d, forward, backward, prefetch_embeddings)."""
    return q(
        con,
        """
        SELECT s.value, COUNT(*) AS n,
               SUM(e.end - e.start) AS total_ns
        FROM NVTX_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE e.eventType IN (59, 60, 70, 75)
        GROUP BY s.value
        ORDER BY n DESC
        LIMIT ?
        """,
        n,
    )


def report(con, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    span, streams, total_busy = stream_time_overlap(con)
    print(f"Capture span:           {span/1e6:.2f} ms")
    print(f"Total kernel busy time: {total_busy/1e6:.2f} ms")
    print(
        f"Cross-stream overlap:   {(total_busy - span) / span * 100:+.1f}% "
        f"(>= 0% means multi-stream concurrency)"
    )
    print(f"\nDistinct streams: {len(streams)}")
    for sid, busy in streams[:8]:
        print(f"  stream {sid}: {busy/1e6:.2f} ms ({busy/total_busy*100:.1f}%)")

    print(f"\nDistinct host threads issuing CUDA: {runtime_threads(con)}")
    print(f"Threads per stream (from runtime↔kernel correlation):")
    for sid, n_threads, n_kernels in threads_per_stream(con)[:8]:
        print(f"  stream {sid}: {n_threads} thread(s), {n_kernels} kernel(s)")

    pct, n_comm = comm_with_compute_neighbors(con, "ncclDevKernel")
    print(f"\nNCCL kernels: {n_comm}")
    print(f"NCCL time overlapped by compute on other streams: {pct:.1f}%")

    print(f"\nHost-blocking sync API calls:")
    for api, n, total_ns, avg_ns in host_sync_blocking(con):
        print(
            f"  {api}: count={n}, total={total_ns/1e6:.2f} ms, " f"avg={avg_ns:.0f} ns"
        )

    print(f"\nTop NVTX ranges:")
    for name, n, total_ns in nvtx_top(con):
        print(f"  {name}: count={n}, total={total_ns/1e6:.2f} ms")


def main():
    legacy_db = sys.argv[1]
    new_db = sys.argv[2]
    with sqlite3.connect(f"file:{legacy_db}?mode=ro", uri=True) as con:
        report(con, f"LEGACY: {legacy_db}")
    with sqlite3.connect(f"file:{new_db}?mode=ro", uri=True) as con:
        report(con, f"NEW:    {new_db}")


if __name__ == "__main__":
    main()
