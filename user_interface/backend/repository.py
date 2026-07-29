from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import connect


class JobIsRunningError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobRepository:
    def __init__(self, database_path: Path):
        self.database_path = database_path

    def create_job(
        self,
        *,
        job_id: str,
        original_filename: str,
        template_id: str | None,
        ocr_engine: str,
        extra_filtered_columns: list[str],
        input_path: Path,
        ground_truth_path: Path | None,
        artifact_directory: Path,
    ) -> dict[str, Any]:
        now = _now()
        with connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, status, stage, original_filename, template_id,
                    ocr_engine, extra_filtered_columns_json, input_path,
                    ground_truth_path, artifact_directory, created_at, updated_at
                ) VALUES (?, 'queued', 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    original_filename,
                    template_id,
                    ocr_engine,
                    json.dumps(extra_filtered_columns),
                    str(input_path),
                    str(ground_truth_path) if ground_truth_path else None,
                    str(artifact_directory),
                    now,
                    now,
                ),
            )
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError("Could not create job")
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_jobs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def reserve_job_deletion(self, job_id: str) -> dict[str, Any] | None:
        with connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            if row["status"] == "running":
                raise JobIsRunningError(
                    "A running job cannot be deleted until processing finishes."
                )
            connection.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', stage = 'cancelled', updated_at = ?
                WHERE id = ?
                """,
                (_now(), job_id),
            )
        return dict(row)

    def delete_job_record(self, job_id: str) -> bool:
        with connect(self.database_path) as connection:
            result = connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return result.rowcount == 1

    def claim_next_job(self) -> dict[str, Any] | None:
        with connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""
                SELECT * FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at
                LIMIT 1
                """).fetchone()
            if row is None:
                connection.commit()
                return None
            now = _now()
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'running', stage = 'preparing',
                    started_at = ?, updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, row["id"]),
            )
            connection.commit()
            if updated.rowcount != 1:
                return None
        return self.get_job(str(row["id"]))

    def update_progress(
        self,
        job_id: str,
        *,
        stage: str,
        current: int,
        total: int,
    ) -> None:
        with connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE jobs
                SET stage = ?, progress_current = ?, progress_total = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (stage, current, total, _now(), job_id),
            )

    def complete_job(
        self,
        job_id: str,
        *,
        result_path: Path,
        with_warnings: bool,
    ) -> None:
        now = _now()
        status = "completed_with_warnings" if with_warnings else "completed"
        with connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, stage = 'completed', result_path = ?,
                    completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, str(result_path), now, now, job_id),
            )

    def fail_job(
        self,
        job_id: str,
        *,
        error_code: str,
        user_safe_error: str,
        technical_error: str,
    ) -> None:
        now = _now()
        with connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed', stage = 'failed', error_code = ?,
                    user_safe_error = ?, technical_error = ?,
                    completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    error_code,
                    user_safe_error,
                    technical_error,
                    now,
                    now,
                    job_id,
                ),
            )

    def set_ground_truth_path(self, job_id: str, path: Path | None) -> None:
        with connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE jobs SET ground_truth_path = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(path) if path else None, _now(), job_id),
            )

    def touch(self, job_id: str) -> None:
        with connect(self.database_path) as connection:
            connection.execute(
                "UPDATE jobs SET updated_at = ? WHERE id = ?",
                (_now(), job_id),
            )

    def recover_interrupted_jobs(self) -> int:
        with connect(self.database_path) as connection:
            result = connection.execute(
                """
                UPDATE jobs
                SET status = 'queued', stage = 'queued', started_at = NULL,
                    updated_at = ?
                WHERE status = 'running'
                """,
                (_now(),),
            )
        return result.rowcount
