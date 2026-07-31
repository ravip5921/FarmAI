from __future__ import annotations

from pydantic import BaseModel, Field


class JobSettings(BaseModel):
    template_id: str | None = None
    ocr_engine: str = "llm-vision"
    extra_filtered_columns: list[str] = Field(default_factory=list)


class CellEdit(BaseModel):
    page_number: int = Field(ge=1)
    row: int = Field(ge=1)
    column_key: str
    reviewed_text: str


class CellEdits(BaseModel):
    edits: list[CellEdit]
