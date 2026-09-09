from __future__ import annotations

import argparse
import os
import signal
import socket
import time
from pathlib import Path
from types import FrameType
from typing import Any

from .config import get_config
from .database import initialize_database
from .repository import JobRepository
from .services.lease_heartbeat import JobLeaseHeartbeat
from .services.job_runner import run_claimed_job
from .services.worker_control import WorkerControl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FarmAI UI job worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one job.")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--lease-seconds",
        type=int,
        default=int(os.getenv("FARMAI_WORKER_LEASE_SECONDS", "300")),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=float(os.getenv("FARMAI_WORKER_HEARTBEAT_SECONDS", "30")),
    )
    return parser.parse_args()


def run_worker(
    args: argparse.Namespace,
    *,
    config: Any | None = None,
    repository: JobRepository | None = None,
    control: WorkerControl | None = None,
    worker_id: str | None = None,
) -> None:
    config = config or get_config()
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    initialize_database(config.database_path)
    repository = repository or JobRepository(config.database_path)
    worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
    control = control or WorkerControl(Path(config.runtime_dir), worker_id=worker_id)
    current_job_id: str | None = None

    def publish_active_status() -> None:
        state = "draining" if control.should_stop_claiming() else "processing"
        control.write_status(state, job_id=current_job_id)

    def request_stop(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        control.request_stop()
        control.write_status(
            "draining" if current_job_id else "drained",
            job_id=current_job_id,
        )

    previous_handlers: dict[int, Any] = {}
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signal_number] = signal.signal(signal_number, request_stop)

    control.write_status("starting")
    recovered = repository.recover_interrupted_jobs()
    if recovered:
        print(f"Re-queued {recovered} job(s) whose worker lease expired.")

    try:
        while True:
            if control.should_stop_claiming():
                control.write_status("drained")
                if control.stop_requested or args.once:
                    return
                time.sleep(max(0.1, args.poll_seconds))
                continue

            repository.recover_interrupted_jobs()
            control.write_status("idle")
            job = repository.claim_next_job(
                worker_id=worker_id,
                lease_seconds=max(1, args.lease_seconds),
            )
            if job is None:
                if args.once:
                    return
                time.sleep(max(0.1, args.poll_seconds))
                continue

            current_job_id = str(job["id"])
            publish_active_status()
            print(f"Processing job {job['id']}: {job['original_filename']}")
            with JobLeaseHeartbeat(
                repository,
                job_id=current_job_id,
                worker_id=worker_id,
                lease_seconds=max(1, args.lease_seconds),
                interval_seconds=max(0.1, args.heartbeat_seconds),
                on_heartbeat=publish_active_status,
            ):
                run_claimed_job(job, repository)
            print(f"Finished job {job['id']}.")
            current_job_id = None
            if args.once:
                return
    finally:
        control.write_status("stopped", job_id=current_job_id)
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)


def main() -> None:
    run_worker(parse_args())


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
