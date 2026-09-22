from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .database import connect


class JobIsRunningError(RuntimeError):
    pass


class JobCannotBeCancelledError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lease_expiration(seconds: int) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=max(1, int(seconds)))
    ).isoformat()


DOCUMENT_STATUSES = (
    "pending",
    "queued",
    "running",
    "completed",
    "completed_with_warnings",
    "failed",
    "cancelled",
)


def _sync_document_status(
    connection: sqlite3.Connection, document_id: str, *, now: str
) -> None:
    latest = connection.execute(
        """
        SELECT status
        FROM jobs
        WHERE document_id = ?
        ORDER BY created_at DESC, rowid DESC
        LIMIT 1
        """,
        (document_id,),
    ).fetchone()
    status = str(latest["status"]) if latest is not None else "pending"
    connection.execute(
        "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
        (status, now, document_id),
    )


def _sync_batch_status(
    connection: sqlite3.Connection, batch_id: str, *, now: str
) -> None:
    rows = connection.execute(
        "SELECT status FROM jobs WHERE batch_id = ?", (batch_id,)
    ).fetchall()
    if not rows:
        return
    statuses = [str(row["status"]) for row in rows]
    if "running" in statuses:
        batch_status = "running"
    elif "queued" in statuses:
        batch_status = "running" if any(
            value not in {"queued"} for value in statuses
        ) else "queued"
    elif all(value in {"completed", "completed_with_warnings"} for value in statuses):
        batch_status = (
            "completed_with_warnings"
            if "completed_with_warnings" in statuses
            else "completed"
        )
    elif all(value == "cancelled" for value in statuses):
        batch_status = "cancelled"
    elif all(value == "failed" for value in statuses):
        batch_status = "failed"
    else:
        batch_status = "completed_with_errors"
    is_terminal = batch_status in {
        "completed",
        "completed_with_warnings",
        "completed_with_errors",
        "failed",
        "cancelled",
    }
    connection.execute(
        """
        UPDATE analysis_batches
        SET status = ?, updated_at = ?,
            completed_at = CASE WHEN ? THEN COALESCE(completed_at, ?) ELSE NULL END
        WHERE id = ?
        """,
        (batch_status, now, is_terminal, now, batch_id),
    )


class JobRepository:
    def __init__(self, database_path: Path):
        self.database_path = database_path

    def create_documents(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        now = _now()
        with connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                """
                INSERT INTO documents (
                    id, status, original_filename, content_type, size_bytes,
                    sha256, input_path, created_at, updated_at
                ) VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(document["id"]),
                        str(document["original_filename"]),
                        str(document["content_type"]),
                        int(document["size_bytes"]),
                        str(document["sha256"]),
                        str(document["input_path"]),
                        now,
                        now,
                    )
                    for document in documents
                ],
            )
        created = []
        for document in documents:
            stored = self.get_document(str(document["id"]))
            if stored is None:
                raise RuntimeError("Could not create uploaded document")
            created.append(stored)
        return created

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT d.*,
                       latest.id AS latest_job_id,
                       latest.batch_id AS latest_batch_id,
                       latest.user_safe_error AS latest_error
                FROM documents AS d
                LEFT JOIN jobs AS latest
                  ON latest.rowid = (
                      SELECT candidate.rowid
                      FROM jobs AS candidate
                      WHERE candidate.document_id = d.id
                      ORDER BY candidate.created_at DESC, candidate.rowid DESC
                      LIMIT 1
                  )
                WHERE d.id = ?
                """,
                (document_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_documents(
        self, *, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        with connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT d.*,
                       latest.id AS latest_job_id,
                       latest.batch_id AS latest_batch_id,
                       latest.user_safe_error AS latest_error
                FROM documents AS d
                LEFT JOIN jobs AS latest
                  ON latest.rowid = (
                      SELECT candidate.rowid
                      FROM jobs AS candidate
                      WHERE candidate.document_id = d.id
                      ORDER BY candidate.created_at DESC, candidate.rowid DESC
                      LIMIT 1
                  )
                ORDER BY d.created_at DESC, d.rowid DESC
                LIMIT ? OFFSET ?
                """,
                (max(1, min(int(limit), 500)), max(0, int(offset))),
            ).fetchall()
        return [dict(row) for row in rows]

    def document_counts(self) -> dict[str, int]:
        counts = {"total": 0, **{status: 0 for status in DOCUMENT_STATUSES}}
        with connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM documents
                GROUP BY status
                """
            ).fetchall()
        for row in rows:
            status = str(row["status"])
            value = int(row["count"])
            counts["total"] += value
            counts[status] = value
        return counts

    def create_analysis_batch(
        self,
        *,
        template_id: str | None,
        ocr_engine: str,
        extra_filtered_columns: list[str],
        jobs_dir: Path,
    ) -> dict[str, Any]:
        now = _now()
        with connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            documents = connection.execute(
                """
                SELECT * FROM documents
                WHERE status = 'pending'
                ORDER BY created_at, rowid
                """
            ).fetchall()
            if not documents:
                return {
                    "id": None,
                    "status": "empty",
                    "document_count": 0,
                    "created_at": None,
                    "jobs": [],
                }

            batch_id = str(uuid4())
            serialized_filters = json.dumps(extra_filtered_columns)
            connection.execute(
                """
                INSERT INTO analysis_batches (
                    id, status, template_id, ocr_engine,
                    extra_filtered_columns_json, document_count,
                    created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    template_id,
                    ocr_engine,
                    serialized_filters,
                    len(documents),
                    now,
                    now,
                ),
            )
            jobs: list[dict[str, Any]] = []
            for document in documents:
                job_id = str(uuid4())
                artifact_directory = jobs_dir / job_id
                connection.execute(
                    """
                    INSERT INTO jobs (
                        id, status, stage, original_filename, template_id,
                        ocr_engine, extra_filtered_columns_json, input_path,
                        ground_truth_path, artifact_directory, created_at,
                        updated_at, document_id, batch_id
                    ) VALUES (
                        ?, 'queued', 'queued', ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        job_id,
                        str(document["original_filename"]),
                        template_id,
                        ocr_engine,
                        serialized_filters,
                        str(document["input_path"]),
                        str(artifact_directory),
                        now,
                        now,
                        str(document["id"]),
                        batch_id,
                    ),
                )
                connection.execute(
                    """
                    UPDATE documents
                    SET status = 'queued', updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (now, str(document["id"])),
                )
                jobs.append(
                    {
                        "id": job_id,
                        "document_id": str(document["id"]),
                        "batch_id": batch_id,
                    }
                )
        return {
            "id": batch_id,
            "status": "queued",
            "document_count": len(documents),
            "template_id": template_id,
            "ocr_engine": ocr_engine,
            "extra_filtered_columns_json": serialized_filters,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "updated_at": now,
            "jobs": jobs,
        }

    def get_analysis_batch(self, batch_id: str) -> dict[str, Any] | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM analysis_batches WHERE id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                return None
            jobs = connection.execute(
                """
                SELECT id, document_id, status
                FROM jobs
                WHERE batch_id = ?
                ORDER BY created_at, rowid
                """,
                (batch_id,),
            ).fetchall()
        result = dict(row)
        result["jobs"] = [dict(job) for job in jobs]
        return result

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
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT document_id, batch_id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return False
            result = connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            now = _now()
            if row["document_id"]:
                _sync_document_status(
                    connection, str(row["document_id"]), now=now
                )
            if row["batch_id"]:
                _sync_batch_status(connection, str(row["batch_id"]), now=now)
        return result.rowcount == 1

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        now = _now()
        with connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            if row["status"] not in {"queued", "running"}:
                connection.commit()
                raise JobCannotBeCancelledError(
                    "Only waiting or processing jobs can be cancelled."
                )
            connection.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', stage = 'cancelled',
                    completed_at = ?, updated_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE id = ?
                """,
                (now, now, job_id),
            )
            if row["document_id"]:
                _sync_document_status(
                    connection, str(row["document_id"]), now=now
                )
            if row["batch_id"]:
                _sync_batch_status(connection, str(row["batch_id"]), now=now)
            connection.commit()
        return self.get_job(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return row is not None and row["status"] == "cancelled"

    def claim_next_job(
        self,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
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
            owner = worker_id or "legacy-worker"
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'running', stage = 'preparing',
                    started_at = COALESCE(started_at, ?), updated_at = ?,
                    attempt_count = attempt_count + 1, lease_owner = ?,
                    lease_expires_at = ?, heartbeat_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (
                    now,
                    now,
                    owner,
                    _lease_expiration(lease_seconds),
                    now,
                    row["id"],
                ),
            )
            if updated.rowcount == 1:
                if row["document_id"]:
                    connection.execute(
                        """
                        UPDATE documents
                        SET status = 'running', updated_at = ?
                        WHERE id = ?
                        """,
                        (now, str(row["document_id"])),
                    )
                if row["batch_id"]:
                    connection.execute(
                        """
                        UPDATE analysis_batches
                        SET status = 'running',
                            started_at = COALESCE(started_at, ?), updated_at = ?
                        WHERE id = ?
                        """,
                        (now, now, str(row["batch_id"])),
                    )
            connection.commit()
            if updated.rowcount != 1:
                return None
        return self.get_job(str(row["id"]))

    def heartbeat_job(
        self, job_id: str, *, worker_id: str, lease_seconds: int = 300
    ) -> bool:
        now = _now()
        with connect(self.database_path) as connection:
            result = connection.execute(
                """
                UPDATE jobs
                SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                """,
                (
                    now,
                    _lease_expiration(lease_seconds),
                    now,
                    job_id,
                    worker_id,
                ),
            )
        return result.rowcount == 1

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
                    completed_at = ?, updated_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE id = ? AND status = 'running'
                """,
                (status, str(result_path), now, now, job_id),
            )
            row = connection.execute(
                "SELECT document_id, batch_id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is not None and row["document_id"]:
                _sync_document_status(
                    connection, str(row["document_id"]), now=now
                )
            if row is not None and row["batch_id"]:
                _sync_batch_status(connection, str(row["batch_id"]), now=now)

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
                    completed_at = ?, updated_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE id = ? AND status = 'running'
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
            row = connection.execute(
                "SELECT document_id, batch_id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is not None and row["document_id"]:
                _sync_document_status(
                    connection, str(row["document_id"]), now=now
                )
            if row is not None and row["batch_id"]:
                _sync_batch_status(connection, str(row["batch_id"]), now=now)

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
        now = _now()
        with connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            interrupted = connection.execute(
                """
                SELECT id, document_id, batch_id
                FROM jobs
                WHERE status = 'running'
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                """,
                (now,),
            ).fetchall()
            if not interrupted:
                return 0
            result = connection.execute(
                """
                UPDATE jobs
                SET status = 'queued', stage = 'queued', updated_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE status = 'running'
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                """,
                (now, now),
            )
            for row in interrupted:
                if row["document_id"]:
                    _sync_document_status(
                        connection, str(row["document_id"]), now=now
                    )
                if row["batch_id"]:
                    _sync_batch_status(connection, str(row["batch_id"]), now=now)
        return result.rowcount
