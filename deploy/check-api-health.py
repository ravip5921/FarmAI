#!/usr/bin/env python3
"""Validate API health and confirm the database schema is fully migrated."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("response", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = json.loads(args.response.read_text(encoding="utf-8"))
        status = payload["status"]
        schema_version = payload["schema_version"]
        latest_schema_version = payload["latest_schema_version"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 1

    if status != "ok":
        return 1
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        return 1
    if isinstance(latest_schema_version, bool) or not isinstance(
        latest_schema_version, int
    ):
        return 1
    if schema_version < 1 or schema_version != latest_schema_version:
        return 1

    print(f"api=ok schema={schema_version}/{latest_schema_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
