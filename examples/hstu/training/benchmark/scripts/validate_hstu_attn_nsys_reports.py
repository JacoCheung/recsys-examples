#!/usr/bin/env python3

import argparse
import csv
import sqlite3
import sys
from pathlib import Path


def _runtime_api_count(connection: sqlite3.Connection, prefix: str) -> int:
    row = connection.execute(
        """
        SELECT COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN StringIds AS strings ON strings.id = runtime.nameId
        WHERE strings.value LIKE ?
        """,
        (f"{prefix}%",),
    ).fetchone()
    return int(row[0])


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    exists = connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()[0]
    if not exists:
        return 0
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_dir", type=Path)
    args = parser.parse_args()

    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(
        [
            "report",
            "cuda_event_create",
            "cuda_event_record",
            "kernel_rows",
            "graph_node_rows",
        ]
    )
    for database in sorted(args.reports_dir.glob("*.sqlite")):
        with sqlite3.connect(database) as connection:
            writer.writerow(
                [
                    database.with_suffix(".nsys-rep").name,
                    _runtime_api_count(connection, "cudaEventCreate"),
                    _runtime_api_count(connection, "cudaEventRecord"),
                    _table_count(connection, "CUPTI_ACTIVITY_KIND_KERNEL"),
                    _table_count(connection, "CUDA_GRAPH_NODE_EVENTS"),
                ]
            )


if __name__ == "__main__":
    main()
