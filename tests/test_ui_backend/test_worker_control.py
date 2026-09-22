from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from user_interface.backend.services.worker_control import WorkerControl


class TestWorkerControl(unittest.TestCase):
    def test_tracks_drain_stop_and_atomic_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = Path(tmpdir)
            control = WorkerControl(runtime, worker_id="worker-1")
            self.assertFalse(control.should_stop_claiming())

            control.write_status("processing", job_id="job-1")
            payload = json.loads(control.status_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["state"], "processing")
            self.assertEqual(payload["worker_id"], "worker-1")
            self.assertEqual(payload["job_id"], "job-1")
            self.assertIn("+00:00", payload["updated_at"])

            control.drain_path.touch()
            self.assertTrue(control.should_stop_claiming())
            control.drain_path.unlink()
            self.assertFalse(control.should_stop_claiming())
            control.request_stop()
            self.assertTrue(control.should_stop_claiming())


if __name__ == "__main__":
    unittest.main()
