from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import cv2

from src.application import ProcessingProgress, ProcessingSettings, process_document
from src.application.ground_truth import GroundTruthError, score_result

from ..repository import JobRepository
from .artifact_store import write_csv_artifact, write_result


class JobCancelled(RuntimeError):
    pass


def _user_safe_error(exc: Exception) -> tuple[str, str]:
    message = str(exc)
    lowered = message.casefold()
    if "could not find the table" in lowered:
        return "table_not_found", "FarmAI could not find the table in this record."
    if "llm ocr is not configured" in lowered:
        return (
            "llm_not_configured",
            "Handwriting recognition is not configured on this computer.",
        )
    if "403" in lowered or "forbidden" in lowered:
        return (
            "llm_authorization",
            "The handwriting service could not be accessed. Contact the project administrator.",
        )
    if isinstance(exc, GroundTruthError):
        return "ground_truth_invalid", message
    return "processing_failed", "This record could not be processed."


def run_claimed_job(
    job: dict[str, Any],
    repository: JobRepository,
    *,
    engine: Any | None = None,
) -> None:
    job_id = str(job["id"])
    artifact_dir = Path(str(job["artifact_directory"]))

    def progress(value: ProcessingProgress) -> None:
        if repository.is_cancelled(job_id):
            raise JobCancelled()
        repository.update_progress(
            job_id,
            stage=value.stage,
            current=value.completed,
            total=value.total,
        )
        if repository.is_cancelled(job_id):
            raise JobCancelled()

    try:
        import json

        extra_filters = tuple(json.loads(job["extra_filtered_columns_json"]))
        processed = process_document(
            str(job["input_path"]),
            settings=ProcessingSettings(
                template_id=job["template_id"],
                ocr_engine=str(job["ocr_engine"]),
                extra_filtered_columns=extra_filters,
            ),
            progress_callback=progress,
            engine=engine,
        )
        pages_payload: list[dict[str, Any]] = []
        for page in processed.pages:
            page_dir = artifact_dir / "pages" / str(page.page_number)
            page_dir.mkdir(parents=True, exist_ok=True)
            source_path = page_dir / "deskewed_source.png"
            overlay_path = page_dir / "overlay.png"
            if not cv2.imwrite(str(source_path), page.source_image):
                raise RuntimeError(f"Could not save source preview: {source_path}")
            if not cv2.imwrite(str(overlay_path), page.overlay_image):
                raise RuntimeError(f"Could not save overlay preview: {overlay_path}")
            pages_payload.append(
                page.to_dict(
                    source_url=(f"/api/jobs/{job_id}/pages/{page.page_number}/source"),
                    overlay_url=(
                        f"/api/jobs/{job_id}/pages/{page.page_number}/overlay"
                    ),
                )
            )

        result: dict[str, Any] = {
            "job_id": job_id,
            "filename": str(job["original_filename"]),
            "template_id": processed.template_id,
            "template_name": processed.template_name,
            "ocr_engine": processed.ocr_engine,
            "warning_count": processed.warning_count,
            "metrics": None,
            "pages": pages_payload,
        }
        ground_truth_path = job.get("ground_truth_path")
        ground_truth_error = None
        if ground_truth_path:
            try:
                result = score_result(
                    result,
                    Path(str(ground_truth_path)).read_text(encoding="utf-8-sig"),
                )
            except GroundTruthError as exc:
                ground_truth_error = str(exc)
                result["ground_truth_error"] = ground_truth_error
        result_path = artifact_dir / "result.json"
        write_result(result_path, result)
        write_csv_artifact(artifact_dir / "result.csv", result)
        repository.complete_job(
            job_id,
            result_path=result_path,
            with_warnings=processed.warning_count > 0 or ground_truth_error is not None,
        )
    except JobCancelled:
        return
    except Exception as exc:
        error_code, user_message = _user_safe_error(exc)
        repository.fail_job(
            job_id,
            error_code=error_code,
            user_safe_error=user_message,
            technical_error=traceback.format_exc(),
        )
