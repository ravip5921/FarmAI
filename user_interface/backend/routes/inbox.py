from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, File, HTTPException, Query, Request, UploadFile, status

from src.ocr import get_ocr_engine_names
from src.templates import get_template_ids

from ..repository import JobRepository
from ..schemas import JobSettings

router = APIRouter()
MAX_DOCUMENTS_PER_UPLOAD = 100
UPLOAD_CHUNK_BYTES = 1024 * 1024


def _repository(request: Request) -> JobRepository:
    return request.app.state.repository


def _public_document(document: dict) -> dict:
    return {
        "document_id": document["id"],
        "filename": document["original_filename"],
        "status": document["status"],
        "size_bytes": document["size_bytes"],
        "sha256": document["sha256"],
        "created_at": document["created_at"],
        "uploaded_at": document["created_at"],
        "updated_at": document["updated_at"],
        "latest_job_id": document.get("latest_job_id"),
        "batch_id": document.get("latest_batch_id"),
        "error": document.get("latest_error"),
    }


def _validate_settings(settings: JobSettings) -> None:
    if settings.template_id and settings.template_id not in get_template_ids():
        raise HTTPException(status_code=422, detail="Unknown form template.")
    if settings.ocr_engine not in get_ocr_engine_names():
        raise HTTPException(status_code=422, detail="Unknown recognition method.")


def _document_response(
    repository: JobRepository,
    documents: list[dict],
    *,
    limit: int | None = None,
    offset: int | None = None,
) -> dict:
    counts = repository.document_counts()
    response: dict = {
        "documents": [_public_document(document) for document in documents],
        "counts": counts,
    }
    if limit is not None and offset is not None:
        response.update(
            {
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(documents) < counts["total"],
            }
        )
    return response


async def _store_pdf(
    upload: UploadFile, *, destination: Path, max_bytes: int
) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    header = bytearray()
    with destination.open("xb") as output:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            if size + len(chunk) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="The uploaded file is too large.",
                )
            size += len(chunk)
            digest.update(chunk)
            if len(header) < 1024:
                header.extend(chunk[: 1024 - len(header)])
            output.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if b"%PDF-" not in header:
        raise HTTPException(
            status_code=400,
            detail="Only valid PDF documents can be added to the inbox.",
        )
    return size, digest.hexdigest()


@router.get("/documents")
def list_documents(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    repository = _repository(request)
    documents = repository.list_documents(limit=limit, offset=offset)
    return _document_response(
        repository, documents, limit=limit, offset=offset
    )


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_documents(
    request: Request,
    documents: list[UploadFile] = File(...),
) -> dict:
    if not documents:
        raise HTTPException(status_code=400, detail="Select at least one PDF.")
    if len(documents) > MAX_DOCUMENTS_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Upload at most {MAX_DOCUMENTS_PER_UPLOAD} PDFs at a time.",
        )

    config = request.app.state.config
    prepared: list[dict] = []
    for upload in documents:
        original_name = Path((upload.filename or "document.pdf").replace("\\", "/")).name
        if Path(original_name).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF documents can be added to the inbox.",
            )
        document_id = str(uuid4())
        prepared.append(
            {
                "id": document_id,
                "original_filename": original_name,
                "content_type": "application/pdf",
                "input_path": config.runtime_dir
                / "documents"
                / document_id
                / "document.pdf",
                "upload": upload,
            }
        )

    created_directories: list[Path] = []
    try:
        for document in prepared:
            input_path = Path(document["input_path"])
            input_path.parent.mkdir(parents=True, exist_ok=False)
            created_directories.append(input_path.parent)
            size_bytes, sha256 = await _store_pdf(
                document["upload"],
                destination=input_path,
                max_bytes=config.max_upload_bytes,
            )
            document["size_bytes"] = size_bytes
            document["sha256"] = sha256
        stored = _repository(request).create_documents(prepared)
    except Exception:
        for directory in reversed(created_directories):
            input_path = directory / "document.pdf"
            input_path.unlink(missing_ok=True)
            try:
                directory.rmdir()
            except OSError:
                pass
        raise

    return _document_response(_repository(request), stored)


def _public_batch(batch: dict) -> dict:
    if batch["id"] is None:
        return {
            "batch_id": None,
            "status": "empty",
            "document_count": 0,
            "accepted_count": 0,
            "job_ids": [],
            "created_at": None,
        }
    jobs = list(batch.get("jobs", []))
    settings = {
        "template_id": batch.get("template_id"),
        "ocr_engine": batch.get("ocr_engine"),
        "extra_filtered_columns": json.loads(
            batch.get("extra_filtered_columns_json", "[]")
        ),
    }
    return {
        "batch_id": batch["id"],
        "status": batch["status"],
        "document_count": batch["document_count"],
        "accepted_count": batch["document_count"],
        "job_ids": [job["id"] for job in jobs],
        "settings": settings,
        "created_at": batch["created_at"],
        "started_at": batch.get("started_at"),
        "completed_at": batch.get("completed_at"),
        "updated_at": batch.get("updated_at"),
    }


@router.post("/analysis-batches", status_code=status.HTTP_202_ACCEPTED)
def create_analysis_batch(
    request: Request,
    settings: JobSettings | None = Body(default=None),
) -> dict:
    selected_settings = settings or JobSettings()
    _validate_settings(selected_settings)
    repository = _repository(request)
    batch = repository.create_analysis_batch(
        template_id=selected_settings.template_id,
        ocr_engine=selected_settings.ocr_engine,
        extra_filtered_columns=selected_settings.extra_filtered_columns,
        jobs_dir=request.app.state.config.jobs_dir,
    )
    return {**_public_batch(batch), "counts": repository.document_counts()}


@router.get("/analysis-batches/{batch_id}")
def get_analysis_batch(request: Request, batch_id: str) -> dict:
    try:
        UUID(batch_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Analysis batch not found.") from exc
    batch = _repository(request).get_analysis_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Analysis batch not found.")
    return _public_batch(batch)
