from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.templates import (
    FormTemplate,
    TemplateColumn,
    form_template_from_dict,
    get_template_ids,
    load_template,
)


def _template_data(**overrides):
    data = {
        "id": "sample",
        "table": {"column_widths": [1, 1]},
        "columns": [{"index": 0, "name": "A"}, {"index": 1, "name": "B"}],
    }
    data.update(overrides)
    return data


class TestTemplatesCoverage(unittest.TestCase):
    def test_column_names_are_sorted_by_index(self) -> None:
        template = FormTemplate(
            id="t",
            name="Template",
            description="",
            column_widths=(1.0, 1.0),
            uniform_row_height=False,
            columns=(
                TemplateColumn(index=1, key="b", name="B"),
                TemplateColumn(index=0, key="a", name="A"),
            ),
        )

        self.assertEqual(template.column_names, ["A", "B"])

    def test_template_validation_errors_are_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "length"):
            form_template_from_dict(_template_data(table={"column_widths": [1]}))
        with self.assertRaisesRegex(ValueError, "positive"):
            form_template_from_dict(_template_data(table={"column_widths": [1, 0]}))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            form_template_from_dict(
                _template_data(
                    table={"column_widths": [1, 1]},
                    columns=[{"index": 0, "name": "A"}, {"index": 2, "name": "B"}],
                )
            )

    def test_unknown_template_error_lists_available_choices_when_present(self) -> None:
        with TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            (template_dir / "known.json").write_text(
                json.dumps(_template_data(id="known")),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Available templates: known"):
                load_template("missing", template_dir=template_dir)

        with TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "Unknown template 'missing'."):
                load_template("missing", template_dir=Path(tmpdir))

    def test_get_template_ids_returns_empty_for_missing_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing"

            self.assertEqual(get_template_ids(template_dir=missing), [])


if __name__ == "__main__":
    unittest.main()
