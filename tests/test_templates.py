from __future__ import annotations

import unittest

from src.templates import get_template_ids, load_template


class TestTemplates(unittest.TestCase):
    def test_load_boar_room_template(self) -> None:
        template = load_template("boar_room")

        self.assertEqual(template.id, "boar_room")
        self.assertEqual(
            template.column_widths,
            (1.5, 1.5, 1.6, 1.6, 0.4, 1.6, 20.0, 3.0, 0.4, 2.7),
        )
        self.assertEqual(
            template.filtered_column_indices,
            {4, 5, 7, 8, 9},
        )
        self.assertEqual(template.column_keys[0], "date")
        self.assertEqual(template.indices_for_column_names({"HI"}), {2})
        self.assertIn("boar_room", get_template_ids())


if __name__ == "__main__":
    unittest.main()
