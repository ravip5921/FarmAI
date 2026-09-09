from __future__ import annotations

import argparse
from pathlib import Path

from .database import LATEST_SCHEMA_VERSION, get_schema_version, migrate_database


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the FarmAI UI database schema.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upgrade = subparsers.add_parser("upgrade", help="Apply all pending migrations.")
    upgrade.add_argument("--database", required=True, type=Path)

    status = subparsers.add_parser("status", help="Print the current schema version.")
    status.add_argument("--database", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    database_path = args.database.resolve()
    if args.command == "upgrade":
        applied = migrate_database(database_path)
        if applied:
            versions = ", ".join(str(version) for version in applied)
            print(f"Applied database migration(s): {versions}")
        else:
            print("Database schema is already current.")
    version = get_schema_version(database_path)
    print(f"Schema version: {version}/{LATEST_SCHEMA_VERSION}")
    return 0 if version == LATEST_SCHEMA_VERSION else 1


if __name__ == "__main__":
    raise SystemExit(main())
