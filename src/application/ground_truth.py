from __future__ import annotations

import csv
import io
from collections import defaultdict
from typing import Any


class GroundTruthError(ValueError):
    pass


def _normalize_header(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def _exact_value(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalized_value(value: str, value_type: str) -> str:
    stripped = _exact_value(value)
    if value_type == "temperature":
        try:
            return str(float(stripped))
        except ValueError:
            return stripped
    return " ".join(stripped.casefold().split())


def _visible_columns(result: dict[str, Any]) -> list[dict[str, Any]]:
    pages = result.get("pages") or []
    if not pages:
        raise GroundTruthError("The OCR result does not contain any pages.")
    return list(pages[0].get("columns") or [])


def parse_ground_truth_csv(
    csv_text: str,
    *,
    columns: list[dict[str, Any]],
    expected_rows: int,
) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
        headers = reader.fieldnames
        if not headers:
            raise GroundTruthError("The ground-truth CSV has no header row.")
        rows = list(reader)
    except csv.Error as exc:
        raise GroundTruthError(f"The ground-truth CSV is invalid: {exc}") from exc

    header_lookup: dict[str, str] = {}
    for header in headers:
        normalized = _normalize_header(header or "")
        if not normalized:
            raise GroundTruthError("The ground-truth CSV contains a blank header.")
        if normalized in header_lookup:
            raise GroundTruthError(
                f"The ground-truth CSV contains duplicate header '{header}'."
            )
        header_lookup[normalized] = header

    column_by_alias: dict[str, dict[str, Any]] = {}
    for column in columns:
        for alias in (column.get("key", ""), column.get("name", "")):
            normalized = _normalize_header(str(alias))
            if normalized:
                column_by_alias[normalized] = column

    unknown = [
        original
        for normalized, original in header_lookup.items()
        if normalized not in column_by_alias
    ]
    if unknown:
        raise GroundTruthError("Unknown ground-truth column(s): " + ", ".join(unknown))

    mapped_headers: dict[str, str] = {}
    for normalized, original in header_lookup.items():
        column_key = str(column_by_alias[normalized]["key"])
        if column_key in mapped_headers:
            raise GroundTruthError(f"More than one CSV header maps to '{column_key}'.")
        mapped_headers[column_key] = original

    missing = [
        str(column["name"])
        for column in columns
        if str(column["key"]) not in mapped_headers
    ]
    if missing:
        raise GroundTruthError(
            "Missing required ground-truth column(s): " + ", ".join(missing)
        )
    if len(rows) != expected_rows:
        raise GroundTruthError(
            f"Ground truth has {len(rows)} data rows; the OCR result has "
            f"{expected_rows}."
        )

    mapped_rows: list[dict[str, str]] = []
    for row in rows:
        mapped_rows.append(
            {
                str(column["key"]): str(
                    row.get(mapped_headers[str(column["key"])], "") or ""
                )
                for column in columns
            }
        )
    return mapped_rows


def score_result(
    result: dict[str, Any],
    csv_text: str,
) -> dict[str, Any]:
    columns = _visible_columns(result)
    expected_rows = sum(
        int(page.get("data_row_count", 0)) for page in result.get("pages", [])
    )
    truth_rows = parse_ground_truth_csv(
        csv_text,
        columns=columns,
        expected_rows=expected_rows,
    )
    column_lookup = {str(column["key"]): column for column in columns}
    correct = normalized_correct = scored = 0
    correct_rows = 0
    incorrect = 0
    column_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "scored": 0}
    )
    truth_offset = 0

    for page in result.get("pages", []):
        page_rows = int(page.get("data_row_count", 0))
        page_truth = truth_rows[truth_offset : truth_offset + page_rows]
        truth_offset += page_rows
        matches_by_row: dict[int, list[bool]] = defaultdict(list)

        for cell in page.get("cells", []):
            row_index = int(cell["row"]) - 1
            if not (0 <= row_index < len(page_truth)):
                continue
            key = str(cell["column_key"])
            expected = page_truth[row_index][key]
            actual = str(cell.get("ocr_text", ""))
            column = column_lookup[key]
            is_match = _exact_value(actual) == _exact_value(expected)
            is_normalized_match = _normalized_value(
                actual, str(column.get("value_type", "text"))
            ) == _normalized_value(expected, str(column.get("value_type", "text")))
            cell["ground_truth_text"] = expected
            cell["ground_truth_match"] = is_match
            validation_warning = bool(cell.get("validation_error"))
            if is_match:
                cell["state"] = (
                    "validation_warning" if validation_warning else "correct"
                )
            else:
                cell["state"] = (
                    "mismatch_and_warning"
                    if validation_warning
                    else "ground_truth_mismatch"
                )
            scored += 1
            correct += int(is_match)
            normalized_correct += int(is_normalized_match)
            incorrect += int(not is_match)
            column_counts[key]["scored"] += 1
            column_counts[key]["correct"] += int(is_match)
            matches_by_row[int(cell["row"])].append(is_match)

        correct_rows += sum(
            1 for matches in matches_by_row.values() if matches and all(matches)
        )
        page["metrics"] = _metrics_for_cells(page.get("cells", []), columns)

    result["metrics"] = {
        "correct_cells": correct,
        "incorrect_cells": incorrect,
        "scored_cells": scored,
        "exact_accuracy": (correct / scored if scored else None),
        "normalized_accuracy": (normalized_correct / scored if scored else None),
        "correct_rows": correct_rows,
        "scored_rows": expected_rows,
        "validation_warning_count": sum(
            1
            for page in result.get("pages", [])
            for cell in page.get("cells", [])
            if cell.get("validation_error")
        ),
        "by_column": [
            {
                "column_key": str(column["key"]),
                "column_name": str(column["name"]),
                "correct": column_counts[str(column["key"])]["correct"],
                "scored": column_counts[str(column["key"])]["scored"],
                "accuracy": (
                    column_counts[str(column["key"])]["correct"]
                    / column_counts[str(column["key"])]["scored"]
                    if column_counts[str(column["key"])]["scored"]
                    else None
                ),
            }
            for column in columns
        ],
    }
    return result


def clear_ground_truth(result: dict[str, Any]) -> dict[str, Any]:
    result["metrics"] = None
    for page in result.get("pages", []):
        page["metrics"] = None
        for cell in page.get("cells", []):
            cell["ground_truth_text"] = None
            cell["ground_truth_match"] = None
            cell["state"] = (
                "validation_warning" if cell.get("validation_error") else "unscored"
            )
    return result


def _metrics_for_cells(
    cells: list[dict[str, Any]],
    columns: list[dict[str, Any]],
) -> dict[str, Any]:
    scored = [cell for cell in cells if cell.get("ground_truth_match") is not None]
    correct = sum(1 for cell in scored if cell["ground_truth_match"])
    return {
        "correct_cells": correct,
        "incorrect_cells": len(scored) - correct,
        "scored_cells": len(scored),
        "exact_accuracy": correct / len(scored) if scored else None,
        "column_count": len(columns),
    }
