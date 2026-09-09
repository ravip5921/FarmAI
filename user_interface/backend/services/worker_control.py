from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class WorkerControl:
    """Filesystem-based worker drain control shared with the deploy script."""

    def __init__(self, runtime_dir: Path, *, worker_id: str):
        self.runtime_dir = runtime_dir
        self.worker_id = worker_id
        self.drain_path = runtime_dir / "worker.drain"
        self.status_path = runtime_dir / "worker-status.json"
        self.stop_requested = False

    def request_stop(self) -> None:
        self.stop_requested = True

    def should_stop_claiming(self) -> bool:
        return self.stop_requested or self.drain_path.exists()

    def write_status(self, state: str, *, job_id: str | None = None) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state": state,
            "worker_id": self.worker_id,
            "job_id": job_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = self.status_path.with_name(
            f".{self.status_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.status_path)
