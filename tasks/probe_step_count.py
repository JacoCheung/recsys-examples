import sqlite3
import sys

for path in sys.argv[1:]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute(
        "SELECT COUNT(DISTINCT text), MIN(text), MAX(text), COUNT(*) FROM NVTX_EVENTS WHERE text LIKE 'step %'"
    )
    distinct, lo, hi, total = cur.fetchone()
    print(f"{path}: distinct_steps={distinct} range=[{lo}..{hi}] total_events={total}")
