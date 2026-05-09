"""Look at all NVTX-related tables / strings in nsys SQLite."""
import sqlite3
import sys

con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
cur = con.cursor()

print("=== Tables ===")
for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%NVTX%' OR name LIKE '%nvtx%'"
):
    print(" ", r[0])

print("\n=== eventType distribution ===")
for r in cur.execute("SELECT eventType, COUNT(*) FROM NVTX_EVENTS GROUP BY eventType"):
    print(" ", r)

# Check if any string has 'step' in any case anywhere
print("\n=== StringIds with 'step' (any case) ===")
for r in cur.execute(
    "SELECT id, value FROM StringIds WHERE LOWER(value) LIKE '%step%' LIMIT 20"
):
    print(" ", r)

# All eventType=59 NVTX texts
print("\n=== eventType=59 distinct values count ===")
cur.execute(
    """
    SELECT COUNT(DISTINCT s.value)
    FROM NVTX_EVENTS e JOIN StringIds s ON e.textId = s.id
    WHERE e.eventType = 59
    """
)
print(" ", cur.fetchone())

# All eventType=75 (whatever it is)
print("\n=== eventType=75 NVTX events ===")
cur.execute(
    """
    SELECT s.value, COUNT(*)
    FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
    WHERE e.eventType = 75
    GROUP BY s.value
    LIMIT 20
    """
)
for r in cur.fetchall():
    print(" ", r)

# NVTX_EVENTS schema
print("\n=== NVTX_EVENTS columns ===")
cur.execute("PRAGMA table_info(NVTX_EVENTS)")
for r in cur.fetchall():
    print(" ", r)

# events that have NULL textId — likely the 'step N' ones
print("\n=== events with NULL textId (eventType=59) ===")
cur.execute(
    """
    SELECT COUNT(*) FROM NVTX_EVENTS WHERE textId IS NULL AND eventType = 59
    """
)
print(" total NULL-text:", cur.fetchone()[0])

# do they have 'text' field directly?
print("\n=== 'step N' rows: any column has it ===")
# search for the string 'step 150' in every text-like column
cur.execute(
    """
    SELECT * FROM NVTX_EVENTS WHERE eventType = 59 LIMIT 1
    """
)
cols = [d[0] for d in cur.description]
print(" columns:", cols)
sample = cur.fetchone()
print(" sample row:", dict(zip(cols, sample)) if sample else None)

# count by null vs non-null textId
cur.execute(
    """
    SELECT
        SUM(CASE WHEN textId IS NULL THEN 1 ELSE 0 END) AS null_text,
        SUM(CASE WHEN textId IS NOT NULL THEN 1 ELSE 0 END) AS has_text
    FROM NVTX_EVENTS WHERE eventType = 59
    """
)
print(" null_text vs has_text:", cur.fetchone())

# look at events with text column directly (not textId)
cur.execute(
    """
    SELECT text, COUNT(*) FROM NVTX_EVENTS
    WHERE text IS NOT NULL AND text LIKE 'step%'
    GROUP BY text
    LIMIT 5
    """
)
print(" via 'text' column:", cur.fetchall())
