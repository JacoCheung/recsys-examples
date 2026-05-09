"""Check whether 'step N' NVTX ranges appear in legacy + new nsys."""
import sqlite3
import sys

for path in sys.argv[1:]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute(
        """
        SELECT s.value, COUNT(*) AS n, AVG(e.end - e.start)/1e6 AS avg_ms
        FROM NVTX_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE (s.value LIKE 'step%' OR s.value LIKE '#%' OR s.value LIKE '%step%')
        GROUP BY s.value
        ORDER BY n DESC
        LIMIT 20
        """
    )
    print(f"\n=== {path} ===")
    rows = cur.fetchall()
    if not rows:
        print("(no step-related NVTX ranges)")
    for v, n, avg_ms in rows:
        print(f"  {v!r:40s}  count={n:>5d}  avg={avg_ms:.2f} ms")

    # also check eventType distribution
    cur.execute(
        """
        SELECT eventType, COUNT(*) FROM NVTX_EVENTS GROUP BY eventType
        """
    )
    print("  NVTX eventType distribution:", cur.fetchall())

    # Look specifically for "step" anywhere in name
    print("  --- raw match for 'step' ---")
    cur.execute(
        """
        SELECT s.value, COUNT(*)
        FROM NVTX_EVENTS e JOIN StringIds s ON e.textId = s.id
        WHERE s.value GLOB 'step*' OR s.value GLOB '*step*'
        GROUP BY s.value
        ORDER BY MIN(e.start)
        LIMIT 10
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("    none")
    for r in rows:
        print(f"    {r}")

    # Each step range has a unique numeric suffix → not matched by GROUP BY
    print("  --- count of distinct 'step *' strings ---")
    cur.execute(
        """
        SELECT COUNT(DISTINCT s.value)
        FROM NVTX_EVENTS e JOIN StringIds s ON e.textId = s.id
        WHERE s.value GLOB 'step *'
        """
    )
    print(f"    distinct=", cur.fetchone()[0])
    cur.execute(
        """
        SELECT COUNT(*)
        FROM NVTX_EVENTS e JOIN StringIds s ON e.textId = s.id
        WHERE s.value GLOB 'step *'
        """
    )
    print(f"    total events=", cur.fetchone()[0])
    cur.execute(
        """
        SELECT s.value
        FROM NVTX_EVENTS e JOIN StringIds s ON e.textId = s.id
        WHERE s.value GLOB 'step *'
        ORDER BY e.start
        LIMIT 5
        """
    )
    print(f"    first 5: {cur.fetchall()}")

    # All distinct text values in NVTX_EVENTS sorted by count
    print("  --- all distinct NVTX texts (top 30 by total time) ---")
    cur.execute(
        """
        SELECT s.value, COUNT(*) AS n, SUM(e.end - e.start)/1e6 AS total_ms
        FROM NVTX_EVENTS e JOIN StringIds s ON e.textId = s.id
        WHERE e.eventType = 59
        GROUP BY s.value
        ORDER BY total_ms DESC
        LIMIT 30
        """
    )
    for v, n, t in cur.fetchall():
        print(f"    {v!r:50s} count={n:>5d} total={t:>10.2f} ms")
