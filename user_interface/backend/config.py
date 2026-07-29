from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class UiConfig:
    runtime_dir: Path
    database_path: Path
    jobs_dir: Path
    max_upload_bytes: int
    allowed_origins: tuple[str, ...]


def get_config() -> UiConfig:
    runtime_dir = Path(
        os.getenv(
            "FARMAI_UI_RUNTIME_DIR",
            str(PROJECT_ROOT / "user_interface" / "runtime"),
        )
    ).resolve()
    origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "FARMAI_UI_ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )
    return UiConfig(
        runtime_dir=runtime_dir,
        database_path=runtime_dir / "farmai_ui.sqlite3",
        jobs_dir=runtime_dir / "jobs",
        max_upload_bytes=int(
            os.getenv("FARMAI_UI_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))
        ),
        allowed_origins=origins,
    )
