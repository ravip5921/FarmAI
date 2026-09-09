#!/usr/bin/env python3
"""Validate the worker heartbeat used by the release drain protocol."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_timestamp(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("updated_at is not a string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("updated_at must include a timezone")
    return parsed.timestamp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("status_file", type=Path)
    parser.add_argument("--states", nargs="+", required=True)
    parser.add_argument("--updated-after", type=float, required=True)
    parser.add_argument("--max-age", type=float, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = json.loads(args.status_file.read_text(encoding="utf-8"))
        state = payload["state"]
        worker_id = payload["worker_id"]
        job_id = payload["job_id"]
        updated_at = parse_timestamp(payload["updated_at"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 1

    now = datetime.now(timezone.utc).timestamp()
    if state not in args.states:
        return 1
    if not isinstance(worker_id, str) or not worker_id:
        return 1
    if job_id is not None and not isinstance(job_id, str):
        return 1
    if updated_at < args.updated_after or now - updated_at > args.max_age:
        return 1
    if updated_at > now + 5:
        return 1

    print(f"worker={worker_id} state={state} job={job_id or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
