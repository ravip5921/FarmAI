#!/usr/bin/env python3
"""Create and verify a consistent SQLite backup without copying WAL files."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    partial = destination.with_name(destination.name + ".partial")

    if not source.is_file():
        print(f"source database does not exist: {source}", file=sys.stderr)
        return 2
    if destination.exists() or partial.exists():
        print("backup destination already exists", file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_uri = f"file:{source}?mode=ro"
    try:
        with sqlite3.connect(source_uri, uri=True, timeout=30) as source_db:
            with sqlite3.connect(partial) as destination_db:
                source_db.backup(destination_db)
                result = destination_db.execute("PRAGMA quick_check").fetchone()
                if result is None or result[0] != "ok":
                    raise sqlite3.DatabaseError(f"backup quick_check failed: {result}")
        os.chmod(partial, 0o600)
        with partial.open("rb") as backup_file:
            os.fsync(backup_file.fileno())
        partial.replace(destination)
    except (OSError, sqlite3.Error) as error:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"database backup failed: {error}", file=sys.stderr)
        return 1

    print(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
