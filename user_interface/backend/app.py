from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_config
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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
