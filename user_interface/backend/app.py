from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import PROJECT_ROOT, get_config
from .database import initialize_database
from .repository import JobRepository
from .routes.jobs import router as jobs_router
from .routes.settings import router as settings_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    initialize_database(config.database_path)
    app.state.config = config
    app.state.repository = JobRepository(config.database_path)
    yield


app = FastAPI(
    title="FarmAI User Interface API",
    version="0.1.0",
    lifespan=lifespan,
)
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)
app.include_router(settings_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")

FRONTEND_DIST = PROJECT_ROOT / "user_interface" / "frontend" / "dist"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _frontend_index() -> FileResponse:
    index_path = FRONTEND_DIST / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="The frontend has not been built. Run npm run build first.",
        )
    return FileResponse(index_path)


@app.get("/", include_in_schema=False)
def frontend_root() -> FileResponse:
    return _frontend_index()


@app.get("/{path:path}", include_in_schema=False)
def frontend_fallback(path: str) -> FileResponse:
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    requested_path = (FRONTEND_DIST / path).resolve()
    try:
        requested_path.relative_to(FRONTEND_DIST.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc
    if requested_path.is_file():
        return FileResponse(requested_path)
    return _frontend_index()
