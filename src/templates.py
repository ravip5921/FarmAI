from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


@dataclass(frozen=True)
class TemplateColumn:
    index: int
    key: str
    name: str
    value_type: str = "text"
    filter_out: bool = False
    format: str | None = None
    range_min: float | None = None
    range_max: float | None = None
    common_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class FormTemplate:
    id: str
    name: str
    description: str
    column_widths: tuple[float, ...]
    uniform_row_height: bool
    columns: tuple[TemplateColumn, ...]

    @property
    def filtered_column_indices(self) -> set[int]:
        return {column.index for column in self.columns if column.filter_out}

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in sorted(self.columns, key=lambda item: item.index)]

    def indices_for_column_names(self, names: set[str]) -> set[int]:
        normalized = {_normalize_name(name) for name in names if _normalize_name(name)}
        return {
            column.index
            for column in self.columns
            if _normalize_name(column.name) in normalized
            or _normalize_name(column.key) in normalized
        }


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def _column_from_dict(data: dict[str, Any]) -> TemplateColumn:
    value_range = data.get("range") or {}
    return TemplateColumn(
        index=int(data["index"]),
        key=str(data.get("key") or data["name"]),
        name=str(data["name"]),
        value_type=str(data.get("value_type", "text")),
        filter_out=bool(data.get("filter_out", False)),
        format=data.get("format"),
        range_min=(
            float(value_range["min"]) if "min" in value_range else None
        ),
        range_max=(
            float(value_range["max"]) if "max" in value_range else None
        ),
        common_values=tuple(str(value) for value in data.get("common_values", [])),
    )


def form_template_from_dict(data: dict[str, Any]) -> FormTemplate:
    table = data.get("table") or {}
    columns = tuple(
        sorted(
            (_column_from_dict(column) for column in data.get("columns", [])),
            key=lambda column: column.index,
        )
    )
    column_widths = tuple(float(width) for width in table.get("column_widths", []))
    if len(column_widths) != len(columns):
        raise ValueError(
            "Template column_widths length must match the number of columns"
        )
    if any(width <= 0 for width in column_widths):
        raise ValueError("Template column_widths must all be positive")

    expected_indices = list(range(len(columns)))
    actual_indices = [column.index for column in columns]
    if actual_indices != expected_indices:
        raise ValueError("Template column indices must be contiguous from 0")

    return FormTemplate(
        id=str(data["id"]),
        name=str(data.get("name", data["id"])),
        description=str(data.get("description", "")),
        column_widths=column_widths,
        uniform_row_height=bool(table.get("uniform_row_height", False)),
        columns=columns,
    )


def load_template(
    template_id: str,
    *,
    template_dir: Path = DEFAULT_TEMPLATE_DIR,
) -> FormTemplate:
    path = template_dir / f"{template_id}.json"
    if not path.exists():
        choices = ", ".join(get_template_ids(template_dir=template_dir))
        suffix = f" Available templates: {choices}" if choices else ""
        raise ValueError(f"Unknown template '{template_id}'.{suffix}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return form_template_from_dict(data)


def get_template_ids(*, template_dir: Path = DEFAULT_TEMPLATE_DIR) -> list[str]:
    if not template_dir.exists():
        return []
    return sorted(path.stem for path in template_dir.glob("*.json"))
