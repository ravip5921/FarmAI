from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from src.templates import FormTemplate, TemplateColumn, get_template_ids, load_template


@dataclass(frozen=True)
class ScorePair:
    result_csv: Path
    ground_truth_csv: Path
    template_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare redacted llm-vision CSV outputs against ground-truth CSVs "
            "and write accuracy metrics."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("out") / "redacted_llm_results",
        help="Directory containing *_llm.csv files produced by the batch runner.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("ground_truth") / "csv",
        help="Directory containing template ground-truth CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out") / "redacted_llm_metrics",
        help="Directory where metric CSVs will be written.",
    )
    parser.add_argument(
        "--include-cell-details",
        action="store_true",
        help="Also write one row per compared cell.",
    )
    return parser.parse_args()


def normalize_header(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def exact_value(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalized_value(value: str, value_type: str) -> str:
    stripped = exact_value(value)
    if value_type == "temperature":
        try:
            return str(float(stripped))
        except ValueError:
            return stripped
    if value_type == "date_dd_mon":
        normalized = re.sub(r"[^a-z0-9]", "", stripped.casefold())
        match = re.fullmatch(r"(\d{1,2})([a-z]{3,})", normalized)
        if match:
            day, month = match.groups()
            return f"{int(day):02d}{month[:3]}"
        return normalized
    if value_type == "english_text":
        return re.sub(r"[^a-z0-9]", "", stripped.casefold())
    return " ".join(stripped.casefold().split())


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        headers = list(reader.fieldnames or [])
        rows = [
            {str(key): str(value or "") for key, value in row.items()}
            for row in reader
        ]
    if not headers:
        raise ValueError(f"{path} has no header row.")
    return headers, rows


def visible_template_columns(template: FormTemplate) -> list[TemplateColumn]:
    filtered = template.filtered_column_indices
    return [
        column
        for column in sorted(template.columns, key=lambda item: item.index)
        if column.index not in filtered
    ]


def map_headers(
    headers: list[str],
    columns: list[TemplateColumn],
) -> dict[str, str]:
    header_lookup = {normalize_header(header): header for header in headers}
    mapped: dict[str, str] = {}
    for column in columns:
        for alias in (column.key, column.name):
            header = header_lookup.get(normalize_header(alias))
            if header is not None:
                mapped[column.key] = header
                break
    return mapped


def infer_template_id(path: Path) -> str | None:
    name = path.stem
    available = sorted(get_template_ids(), key=len, reverse=True)
    for template_id in available:
        if name.endswith(f"_{template_id}_llm"):
            return template_id
    return None


def find_pairs(results_dir: Path, ground_truth_dir: Path) -> list[ScorePair]:
    pairs: list[ScorePair] = []
    for result_csv in sorted(results_dir.glob("*_llm.csv")):
        template_id = infer_template_id(result_csv)
        if template_id is None:
            continue
        ground_truth_csv = ground_truth_dir / f"{template_id}.csv"
        if not ground_truth_csv.exists():
            continue
        pairs.append(
            ScorePair(
                result_csv=result_csv,
                ground_truth_csv=ground_truth_csv,
                template_id=template_id,
            )
        )
    return pairs


def percent(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_pair(pair: ScorePair) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    template = load_template(pair.template_id)
    columns = visible_template_columns(template)
    result_headers, result_rows = read_csv(pair.result_csv)
    truth_headers, truth_rows = read_csv(pair.ground_truth_csv)
    result_header_map = map_headers(result_headers, columns)
    truth_header_map = map_headers(truth_headers, columns)
    compared_columns = [
        column
        for column in columns
        if column.key in result_header_map and column.key in truth_header_map
    ]
    missing_result_columns = [
        column.name for column in columns if column.key not in result_header_map
    ]
    missing_truth_columns = [
        column.name for column in columns if column.key not in truth_header_map
    ]

    rows_compared = min(len(result_rows), len(truth_rows))
    exact_correct = 0
    normalized_correct = 0
    scored_cells = 0
    fully_exact_rows = 0
    fully_normalized_rows = 0
    by_column: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []

    for column in compared_columns:
        column_exact = 0
        column_normalized = 0
        for row_index in range(rows_compared):
            actual = result_rows[row_index].get(result_header_map[column.key], "")
            expected = truth_rows[row_index].get(truth_header_map[column.key], "")
            exact_match = exact_value(actual) == exact_value(expected)
            normalized_match = normalized_value(
                actual,
                column.value_type,
            ) == normalized_value(expected, column.value_type)
            column_exact += int(exact_match)
            column_normalized += int(normalized_match)
            exact_correct += int(exact_match)
            normalized_correct += int(normalized_match)
            scored_cells += 1
            cell_rows.append(
                {
                    "result_file": pair.result_csv.name,
                    "template_id": pair.template_id,
                    "row_number": row_index + 1,
                    "column_key": column.key,
                    "column_name": column.name,
                    "value_type": column.value_type,
                    "ocr_value": actual,
                    "ground_truth_value": expected,
                    "exact_match": exact_match,
                    "normalized_match": normalized_match,
                }
            )
        by_column.append(
            {
                "result_file": pair.result_csv.name,
                "template_id": pair.template_id,
                "column_key": column.key,
                "column_name": column.name,
                "value_type": column.value_type,
                "scored_cells": rows_compared,
                "exact_correct": column_exact,
                "exact_accuracy": percent(column_exact, rows_compared),
                "normalized_correct": column_normalized,
                "normalized_accuracy": percent(column_normalized, rows_compared),
            }
        )

    for row_index in range(rows_compared):
        row_exact = []
        row_normalized = []
        for column in compared_columns:
            actual = result_rows[row_index].get(result_header_map[column.key], "")
            expected = truth_rows[row_index].get(truth_header_map[column.key], "")
            row_exact.append(exact_value(actual) == exact_value(expected))
            row_normalized.append(
                normalized_value(actual, column.value_type)
                == normalized_value(expected, column.value_type)
            )
        fully_exact_rows += int(bool(row_exact) and all(row_exact))
        fully_normalized_rows += int(bool(row_normalized) and all(row_normalized))

    summary = {
        "result_file": pair.result_csv.name,
        "ground_truth_file": pair.ground_truth_csv.name,
        "template_id": pair.template_id,
        "ocr_rows": len(result_rows),
        "ground_truth_rows": len(truth_rows),
        "rows_compared": rows_compared,
        "row_count_difference": len(result_rows) - len(truth_rows),
        "columns_compared": len(compared_columns),
        "missing_result_columns": "; ".join(missing_result_columns),
        "missing_ground_truth_columns": "; ".join(missing_truth_columns),
        "scored_cells": scored_cells,
        "exact_correct": exact_correct,
        "exact_accuracy": percent(exact_correct, scored_cells),
        "normalized_correct": normalized_correct,
        "normalized_accuracy": percent(normalized_correct, scored_cells),
        "fully_exact_rows": fully_exact_rows,
        "fully_normalized_rows": fully_normalized_rows,
    }
    return summary, by_column, cell_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pairs = find_pairs(args.results_dir, args.ground_truth_dir)
    if not pairs:
        raise SystemExit(f"No result/ground-truth pairs found in {args.results_dir}.")

    summaries: list[dict[str, object]] = []
    columns: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for pair in pairs:
        try:
            summary, by_column, cell_rows = score_pair(pair)
            summaries.append(summary)
            columns.extend(by_column)
            cells.extend(cell_rows)
            print(
                f"Scored {pair.result_csv.name}: "
                f"{summary['normalized_accuracy']:.3f} normalized accuracy"
            )
        except Exception as exc:
            failures.append(
                {
                    "result_file": pair.result_csv.name,
                    "template_id": pair.template_id,
                    "error": str(exc),
                }
            )
            print(f"Failed {pair.result_csv.name}: {exc}")

    write_csv(args.output_dir / "metrics_summary.csv", summaries)
    write_csv(args.output_dir / "metrics_by_column.csv", columns)
    if args.include_cell_details:
        write_csv(args.output_dir / "metrics_by_cell.csv", cells)
    if failures:
        write_csv(args.output_dir / "metrics_failures.csv", failures)
    print(f"\nWrote metrics to {args.output_dir}.")


if __name__ == "__main__":
    main()
