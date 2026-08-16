from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import ValidationError

from src.application.ground_truth import (
    GroundTruthError,
    clear_ground_truth,
    score_result,
)
from src.ocr import get_ocr_engine_names
from src.templates import get_template_ids

from ..repository import (
    JobCannotBeCancelledError,
    JobIsRunningError,
    JobRepository,
)
from ..schemas import CellEdits, JobSettings
from ..services.artifact_store import (
    read_result,
    result_to_csv,
    write_csv_artifact,
    write_result,
)

router = APIRouter()
ALLOWED_RECORD_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}


def _repository(request: Request) -> JobRepository:
    return request.app.state.repository


def _job_or_404(repository: JobRepository, job_id: str) -> dict:
    try:
        UUID(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    job = repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _public_job(job: dict) -> dict:
    return {
        "job_id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress_current": job["progress_current"],
        "progress_total": job["progress_total"],
        "filename": job["original_filename"],
        "template_id": job["template_id"],
        "ocr_engine": job["ocr_engine"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "updated_at": job["updated_at"],
        "error_code": job["error_code"],
        "error": job["user_safe_error"],
        "result_url": (f"/api/jobs/{job['id']}/result" if job["result_path"] else None),
    }


async def _read_upload(upload: UploadFile, *, max_bytes: int) -> bytes:
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The uploaded file is too large.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return data


def _parse_settings(value: str) -> JobSettings:
    try:
        settings = JobSettings.model_validate_json(value)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    if settings.template_id and settings.template_id not in get_template_ids():
        raise HTTPException(status_code=422, detail="Unknown form template.")
    if settings.ocr_engine not in get_ocr_engine_names():
        raise HTTPException(status_code=422, detail="Unknown recognition method.")
    return settings


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: Request,
    record: UploadFile = File(...),
    settings: str = Form(
        '{"template_id":null,"ocr_engine":"llm-vision",'
        '"extra_filtered_columns":[]}'
    ),
    ground_truth: UploadFile | None = File(default=None),
) -> dict:
    parsed_settings = _parse_settings(settings)
    original_name = Path(record.filename or "record").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_RECORD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="This file could not be opened. Try a PDF, JPG, or PNG.",
        )

    config = request.app.state.config
    record_data = await _read_upload(record, max_bytes=config.max_upload_bytes)
    ground_truth_data = None
    if ground_truth is not None:
        if Path(ground_truth.filename or "").suffix.lower() != ".csv":
            raise HTTPException(
                status_code=400, detail="Ground truth must be a CSV file."
            )
        ground_truth_data = await _read_upload(
            ground_truth, max_bytes=config.max_upload_bytes
        )

    job_id = str(uuid4())
    artifact_dir = config.jobs_dir / job_id
    input_dir = artifact_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    input_path = input_dir / f"record{suffix}"
    input_path.write_bytes(record_data)
    ground_truth_path = None
    if ground_truth_data is not None:
        ground_truth_dir = artifact_dir / "ground_truth"
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_path = ground_truth_dir / "ground_truth.csv"
        ground_truth_path.write_bytes(ground_truth_data)

    job = _repository(request).create_job(
        job_id=job_id,
        original_filename=original_name,
        template_id=parsed_settings.template_id,
        ocr_engine=parsed_settings.ocr_engine,
        extra_filtered_columns=parsed_settings.extra_filtered_columns,
        input_path=input_path,
        ground_truth_path=ground_truth_path,
        artifact_directory=artifact_dir,
    )
    return {
        "job_id": job_id,
        "status": job["status"],
        "status_url": f"/api/jobs/{job_id}",
    }


@router.get("/jobs")
def list_jobs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
) -> dict:
    jobs = _repository(request).list_jobs(limit=limit)
    return {"jobs": [_public_job(job) for job in jobs]}


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    return _public_job(_job_or_404(_repository(request), job_id))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict:
    repository = _repository(request)
    _job_or_404(repository, job_id)
    try:
        job = repository.cancel_job(job_id)
    except JobCannotBeCancelledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _public_job(job)


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(request: Request, job_id: str) -> Response:
    repository = _repository(request)
    _job_or_404(repository, job_id)
    try:
        job = repository.reserve_job_deletion(job_id)
    except JobIsRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    jobs_root = request.app.state.config.jobs_dir.resolve()
    artifact_dir = Path(str(job["artifact_directory"])).resolve()
    if artifact_dir.parent != jobs_root or artifact_dir.name != job_id:
        raise HTTPException(
            status_code=500,
            detail="The job storage path is invalid and was not deleted.",
        )
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    if not repository.delete_job_record(job_id):
        raise HTTPException(status_code=500, detail="The job could not be deleted.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/jobs/{job_id}/result")
def get_result(request: Request, job_id: str) -> dict:
    job = _job_or_404(_repository(request), job_id)
    if not job["result_path"]:
        if job["status"] == "failed":
            raise HTTPException(status_code=409, detail=job["user_safe_error"])
        raise HTTPException(status_code=409, detail="This job is not complete.")
    return read_result(Path(str(job["result_path"])))


def _page_artifact(
    request: Request,
    job_id: str,
    page_number: int,
    filename: str,
) -> FileResponse:
    job = _job_or_404(_repository(request), job_id)
    if page_number < 1:
        raise HTTPException(status_code=404, detail="Page not found.")
    path = Path(str(job["artifact_directory"])) / "pages" / str(page_number) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Page image not found.")
    return FileResponse(path, media_type="image/png")


@router.get("/jobs/{job_id}/pages/{page_number}/overlay")
def get_overlay(request: Request, job_id: str, page_number: int) -> FileResponse:
    return _page_artifact(request, job_id, page_number, "overlay.png")


@router.get("/jobs/{job_id}/pages/{page_number}/source")
def get_source(request: Request, job_id: str, page_number: int) -> FileResponse:
    return _page_artifact(request, job_id, page_number, "deskewed_source.png")


@router.get("/jobs/{job_id}/download.csv")
def download_csv(request: Request, job_id: str) -> PlainTextResponse:
    job = _job_or_404(_repository(request), job_id)
    if not job["result_path"]:
        raise HTTPException(status_code=409, detail="This job is not complete.")
    result = read_result(Path(str(job["result_path"])))
    filename = f"{Path(str(job['original_filename'])).stem}_reviewed.csv"
    return PlainTextResponse(
        result_to_csv(result),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/jobs/{job_id}/download.json")
def download_json(request: Request, job_id: str) -> FileResponse:
    job = _job_or_404(_repository(request), job_id)
    if not job["result_path"]:
        raise HTTPException(status_code=409, detail="This job is not complete.")
    return FileResponse(
        Path(str(job["result_path"])),
        media_type="application/json",
        filename=f"{Path(str(job['original_filename'])).stem}_reviewed.json",
    )


@router.patch("/jobs/{job_id}/cells")
def edit_cells(request: Request, job_id: str, payload: CellEdits) -> dict:
    repository = _repository(request)
    job = _job_or_404(repository, job_id)
    if not job["result_path"]:
        raise HTTPException(status_code=409, detail="This job is not complete.")
    result_path = Path(str(job["result_path"]))
    result = read_result(result_path)
    cells = {
        (
            int(page["page_number"]),
            int(cell["row"]),
            str(cell["column_key"]),
        ): cell
        for page in result.get("pages", [])
        for cell in page.get("cells", [])
    }
    for edit in payload.edits:
        key = (edit.page_number, edit.row, edit.column_key)
        if key not in cells:
            raise HTTPException(status_code=404, detail=f"Cell not found: {key}")
        cell = cells[key]
        cell["reviewed_text"] = edit.reviewed_text
        cell["was_edited"] = edit.reviewed_text != str(cell.get("ocr_text", ""))
    write_result(result_path, result)
    write_csv_artifact(Path(str(job["artifact_directory"])) / "result.csv", result)
    repository.touch(job_id)
    return result


@router.post("/jobs/{job_id}/ground-truth")
async def attach_ground_truth(
    request: Request,
    job_id: str,
    ground_truth: UploadFile = File(...),
) -> dict:
    repository = _repository(request)
    job = _job_or_404(repository, job_id)
    if not job["result_path"]:
        raise HTTPException(status_code=409, detail="This job is not complete.")
    if Path(ground_truth.filename or "").suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Ground truth must be a CSV file.")
    data = await _read_upload(
        ground_truth, max_bytes=request.app.state.config.max_upload_bytes
    )
    try:
        csv_text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Ground truth must be a UTF-8 CSV file."
        ) from exc
    result_path = Path(str(job["result_path"]))
    result = read_result(result_path)
    try:
        scored = score_result(result, csv_text)
    except GroundTruthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ground_truth_dir = Path(str(job["artifact_directory"])) / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = ground_truth_dir / "ground_truth.csv"
    ground_truth_path.write_bytes(data)
    write_result(result_path, scored)
    write_csv_artifact(Path(str(job["artifact_directory"])) / "result.csv", scored)
    repository.set_ground_truth_path(job_id, ground_truth_path)
    return scored


@router.delete("/jobs/{job_id}/ground-truth")
def remove_ground_truth(request: Request, job_id: str) -> dict:
    repository = _repository(request)
    job = _job_or_404(repository, job_id)
    if not job["result_path"]:
        raise HTTPException(status_code=409, detail="This job is not complete.")
    result_path = Path(str(job["result_path"]))
    result = clear_ground_truth(read_result(result_path))
    write_result(result_path, result)
    ground_truth_path = job.get("ground_truth_path")
    if ground_truth_path:
        Path(str(ground_truth_path)).unlink(missing_ok=True)
    repository.set_ground_truth_path(job_id, None)
    return result
