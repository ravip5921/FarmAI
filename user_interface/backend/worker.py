from __future__ import annotations

import argparse
import time

from .config import get_config
from .database import initialize_database
from .repository import JobRepository
from .services.job_runner import run_claimed_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FarmAI UI job worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one job.")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    initialize_database(config.database_path)
    repository = JobRepository(config.database_path)
    recovered = repository.recover_interrupted_jobs()
    if recovered:
        print(f"Re-queued {recovered} interrupted job(s).")

    while True:
        job = repository.claim_next_job()
        if job is not None:
            print(f"Processing job {job['id']}: {job['original_filename']}")
            run_claimed_job(job, repository)
            print(f"Finished job {job['id']}.")
            if args.once:
                return
            continue
        if args.once:
            return
        time.sleep(max(0.1, args.poll_seconds))


if __name__ == "__main__":
    main()
