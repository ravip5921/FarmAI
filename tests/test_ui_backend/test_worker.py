from __future__ import annotations

import argparse
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from user_interface.backend.services.worker_control import WorkerControl
from user_interface.backend.worker import main, parse_args, run_worker


def _args(*, once: bool) -> argparse.Namespace:
    return argparse.Namespace(
        once=once,
        poll_seconds=0.01,
        lease_seconds=60,
        heartbeat_seconds=0.01,
    )


class _Repository:
    def __init__(self, jobs=(), *, recovered: int = 0) -> None:
        self.jobs = list(jobs)
        self.recovered = recovered
        self.claims = []
        self.heartbeats = []

    def recover_interrupted_jobs(self) -> int:
        value, self.recovered = self.recovered, 0
        return value

    def claim_next_job(self, **kwargs):
        self.claims.append(kwargs)
        return self.jobs.pop(0) if self.jobs else None

    def heartbeat_job(self, job_id: str, **kwargs) -> bool:
        self.heartbeats.append((job_id, kwargs))
        return True


class TestWorker(unittest.TestCase):
    def _config(self, root: Path):
        return SimpleNamespace(
            runtime_dir=root,
            jobs_dir=root / "jobs",
            database_path=root / "worker.sqlite3",
        )

    @patch("user_interface.backend.worker.signal.signal", return_value=signal.SIG_DFL)
    def test_once_without_job_initializes_recovers_and_stops(self, signal_mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repository = _Repository(recovered=2)
            control = WorkerControl(root, worker_id="test-worker")
            with patch("builtins.print") as output:
                run_worker(
                    _args(once=True),
                    config=self._config(root),
                    repository=repository,
                    control=control,
                    worker_id="test-worker",
                )
            self.assertIn('"state": "stopped"', control.status_path.read_text())
        output.assert_called_with("Re-queued 2 job(s) whose worker lease expired.")
        self.assertEqual(signal_mock.call_count, 4)

    def test_finishes_active_job_before_sigterm_exit(self) -> None:
        job = {"id": "job-1", "original_filename": "record.pdf"}
        repository = _Repository([job])
        handlers = {}

        def install_handler(number, handler):
            previous = handlers.get(number, signal.SIG_DFL)
            handlers[number] = handler
            return previous

        def stop_during_job(active_job, active_repository):
            self.assertIs(active_job, job)
            self.assertIs(active_repository, repository)
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            control = WorkerControl(root, worker_id="test-worker")
            with (
                patch(
                    "user_interface.backend.worker.signal.signal",
                    side_effect=install_handler,
                ),
                patch(
                    "user_interface.backend.worker.run_claimed_job",
                    side_effect=stop_during_job,
                ) as runner,
                patch("builtins.print"),
            ):
                run_worker(
                    _args(once=True),
                    config=self._config(root),
                    repository=repository,
                    control=control,
                    worker_id="test-worker",
                )
            status = control.status_path.read_text(encoding="utf-8")
        runner.assert_called_once_with(job, repository)
        self.assertIn('"state": "stopped"', status)
        self.assertEqual(
            repository.claims,
            [{"worker_id": "test-worker", "lease_seconds": 60}],
        )
        self.assertTrue(repository.heartbeats)
        self.assertTrue(control.stop_requested)

    @patch("user_interface.backend.worker.signal.signal", return_value=signal.SIG_DFL)
    def test_drain_sentinel_waits_without_claiming(self, signal_mock) -> None:
        del signal_mock
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repository = _Repository()
            control = WorkerControl(root, worker_id="test-worker")
            control.drain_path.touch()

            def request_shutdown(seconds):
                self.assertGreaterEqual(seconds, 0.1)
                control.request_stop()

            with patch("user_interface.backend.worker.time.sleep", request_shutdown):
                run_worker(
                    _args(once=False),
                    config=self._config(root),
                    repository=repository,
                    control=control,
                    worker_id="test-worker",
                )
        self.assertEqual(repository.claims, [])

    @patch("user_interface.backend.worker.signal.signal", return_value=signal.SIG_DFL)
    def test_idle_worker_polls_then_honors_stop(self, signal_mock) -> None:
        del signal_mock
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repository = _Repository()
            control = WorkerControl(root, worker_id="test-worker")

            with patch(
                "user_interface.backend.worker.time.sleep",
                side_effect=lambda seconds: control.request_stop(),
            ):
                run_worker(
                    _args(once=False),
                    config=self._config(root),
                    repository=repository,
                    control=control,
                    worker_id="test-worker",
                )
        self.assertEqual(len(repository.claims), 1)

    def test_parse_args_and_main_delegate(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "worker",
                    "--once",
                    "--poll-seconds",
                    "2",
                    "--lease-seconds",
                    "40",
                    "--heartbeat-seconds",
                    "5",
                ],
            ),
        ):
            args = parse_args()
        self.assertTrue(args.once)
        self.assertEqual(args.poll_seconds, 2)
        self.assertEqual(args.lease_seconds, 40)
        self.assertEqual(args.heartbeat_seconds, 5)

        with (
            patch("user_interface.backend.worker.parse_args", return_value=args),
            patch("user_interface.backend.worker.run_worker") as run,
        ):
            main()
        run.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
