from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from src.templates import TemplateColumn, load_template


CELL_NAME_RE = re.compile(
    r"^(?P<column_key>.+)_row_(?P<row>\d+)"
    r"_coords_x(?P<x>\d+)_y(?P<y>\d+)_w(?P<width>\d+)_h(?P<height>\d+)\.png$"
)


@dataclass(frozen=True)
class CellImageRecord:
    path: Path
    column_key: str
    row: int
    x: int | None
    y: int | None
    width: int | None
    height: int | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a manual labeling CSV manifest for FarmAI cropped cell images."
        )
    )
    parser.add_argument(
        "cells_dir",
        type=Path,
        help="Directory containing cropped cell PNG files from --save-cells",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path to write. Defaults to <cells_dir parent>/<cells_dir name>_labels.csv",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Optional template ID, e.g. boar_room, used to add column metadata",
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        default=None,
        help="Optional original source image/PDF path for traceability",
    )
    parser.add_argument(
        "--skip-header-row",
        action="store_true",
        help="Skip row 0 cells, useful when row 0 contains printed headers",
    )
    return parser.parse_args(argv)


def parse_cell_image(path: Path) -> CellImageRecord:
    match = CELL_NAME_RE.match(path.name)
    if match is None:
        return CellImageRecord(
            path=path,
            column_key="",
            row=-1,
            x=None,
            y=None,
            width=None,
            height=None,
        )
    groups = match.groupdict()
    return CellImageRecord(
        path=path,
        column_key=groups["column_key"],
        row=int(groups["row"]),
        x=int(groups["x"]),
        y=int(groups["y"]),
        width=int(groups["width"]),
        height=int(groups["height"]),
    )


def template_columns_by_key(template_id: str | None) -> dict[str, TemplateColumn]:
    if template_id is None:
        return {}
    template = load_template(template_id)
    return {column.key: column for column in template.columns}


def output_path_for(cells_dir: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    return cells_dir.parent / f"{cells_dir.name}_labels.csv"


def write_manifest(
    records: list[CellImageRecord],
    output_path: Path,
    *,
    source_image: Path | None,
    template_id: str | None,
    columns_by_key: dict[str, TemplateColumn],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell_image",
        "label",
        "label_status",
        "source_image",
        "template_id",
        "column_key",
        "column_name",
        "value_type",
        "format",
        "range_min",
        "range_max",
        "row",
        "x",
        "y",
        "width",
        "height",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            column = columns_by_key.get(record.column_key)
            writer.writerow(
                {
                    "cell_image": record.path.name,
                    "label": "",
                    "label_status": "",
                    "source_image": "" if source_image is None else str(source_image),
                    "template_id": template_id or "",
                    "column_key": record.column_key,
                    "column_name": "" if column is None else column.name,
                    "value_type": "" if column is None else column.value_type,
                    "format": "" if column is None else (column.format or ""),
                    "range_min": "" if column is None else _optional_number(column.range_min),
                    "range_max": "" if column is None else _optional_number(column.range_max),
                    "row": "" if record.row < 0 else record.row,
                    "x": _optional_number(record.x),
                    "y": _optional_number(record.y),
                    "width": _optional_number(record.width),
                    "height": _optional_number(record.height),
                    "notes": "",
                }
            )


def _optional_number(value: int | float | None) -> str | int | float:
    return "" if value is None else value


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cells_dir = args.cells_dir
    if not cells_dir.is_dir():
        raise NotADirectoryError(f"Cell image directory does not exist: {cells_dir}")

    records = [parse_cell_image(path) for path in sorted(cells_dir.glob("*.png"))]
    if args.skip_header_row:
        records = [record for record in records if record.row != 0]

    output_path = output_path_for(cells_dir, args.output)
    write_manifest(
        records,
        output_path,
        source_image=args.source_image,
        template_id=args.template,
        columns_by_key=template_columns_by_key(args.template),
    )
    print(f"Wrote {len(records)} rows to: {output_path}")


if __name__ == "__main__":
    main()
