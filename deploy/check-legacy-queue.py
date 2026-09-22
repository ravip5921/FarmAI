#!/usr/bin/env python3
"""Return success only when a legacy SQLite queue has no active work."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ACTIVE_STATES = ("queued", "running", "processing", "claimed")
KNOWN_JOB_TABLES = ("jobs", "analysis_jobs")


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.database.is_file():
        print("database is absent; queue is idle")
        return 0

    uri = f"file:{args.database.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=5) as connection:
            connection.execute("PRAGMA query_only = ON")
            existing_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            job_tables = [
                table for table in KNOWN_JOB_TABLES if table in existing_tables
            ]
            if not job_tables:
                print("no recognized job queue table", file=sys.stderr)
                return 2

            active_count = 0
            placeholders = ", ".join("?" for _ in ACTIVE_STATES)
            for table in job_tables:
                columns = {
                    row[1]
                    for row in connection.execute(
                        f"PRAGMA table_info({quote_identifier(table)})"
                    )
                }
                if "status" not in columns:
                    print(f"{table} has no status column", file=sys.stderr)
                    return 2
                query = (
                    f"SELECT COUNT(*) FROM {quote_identifier(table)} "
                    f"WHERE status IN ({placeholders})"
                )
                active_count += int(
                    connection.execute(query, ACTIVE_STATES).fetchone()[0]
                )
    except sqlite3.Error as error:
        print(f"could not inspect legacy queue: {error}", file=sys.stderr)
        return 2

    print(f"active legacy jobs: {active_count}")
    return 0 if active_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
