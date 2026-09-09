from __future__ import annotations

import time
import unittest
from unittest.mock import Mock

from user_interface.backend.services.lease_heartbeat import JobLeaseHeartbeat


class TestJobLeaseHeartbeat(unittest.TestCase):
    def test_context_renews_lease_and_notifies(self) -> None:
        repository = Mock()
        repository.heartbeat_job.return_value = True
        notices = []
        heartbeat = JobLeaseHeartbeat(
            repository,
            job_id="job-1",
            worker_id="worker-1",
            lease_seconds=90,
            interval_seconds=0.01,
            on_heartbeat=lambda: notices.append(True),
        )
        with heartbeat:
            heartbeat.start()
            deadline = time.monotonic() + 0.5
            while repository.heartbeat_job.call_count < 2:
                if time.monotonic() >= deadline:
                    self.fail("heartbeat thread did not renew the lease")
                time.sleep(0.005)
        self.assertGreaterEqual(repository.heartbeat_job.call_count, 2)
        repository.heartbeat_job.assert_any_call(
            "job-1", worker_id="worker-1", lease_seconds=90
        )
        self.assertGreaterEqual(len(notices), 2)

    def test_stops_when_lease_is_no_longer_owned(self) -> None:
        repository = Mock()
        repository.heartbeat_job.side_effect = [True, False]
        heartbeat = JobLeaseHeartbeat(
            repository,
            job_id="job-2",
            worker_id="worker-2",
            lease_seconds=30,
            interval_seconds=0,
        )
        heartbeat.start()
        deadline = time.monotonic() + 0.5
        while repository.heartbeat_job.call_count < 2:
            if time.monotonic() >= deadline:
                self.fail("heartbeat thread did not observe the lost lease")
            time.sleep(0.005)
        heartbeat.stop()
        self.assertEqual(repository.heartbeat_job.call_count, 2)
        heartbeat.stop()


if __name__ == "__main__":
    unittest.main()
