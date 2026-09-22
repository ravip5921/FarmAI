from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol


class LeaseRepository(Protocol):
    def heartbeat_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> bool: ...


class JobLeaseHeartbeat:
    """Keep a claimed job lease alive while a slow OCR call is in progress."""

    def __init__(
        self,
        repository: LeaseRepository,
        *,
        job_id: str,
        worker_id: str,
        lease_seconds: int,
        interval_seconds: float,
        on_heartbeat: Callable[[], None] | None = None,
    ):
        self.repository = repository
        self.job_id = job_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.interval_seconds = max(0.01, float(interval_seconds))
        self.on_heartbeat = on_heartbeat
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> JobLeaseHeartbeat:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._beat()
        self._thread = threading.Thread(
            target=self._run,
            name=f"farmai-heartbeat-{self.job_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1.0)
            self._thread = None

    def _beat(self) -> bool:
        renewed = self.repository.heartbeat_job(
            self.job_id,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if renewed and self.on_heartbeat is not None:
            self.on_heartbeat()
        return renewed

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            if not self._beat():
                return
