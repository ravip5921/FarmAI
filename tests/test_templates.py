from __future__ import annotations

import unittest

from src.templates import get_template_ids, load_template


class TestTemplates(unittest.TestCase):
    def test_load_boar_room_template(self) -> None:
        template = load_template("boar_room")

        self.assertEqual(template.id, "boar_room")
        self.assertEqual(template.column_widths, (2, 2, 2, 2, 2, 20, 3, 3))
        self.assertEqual(
            template.filtered_column_indices,
            {4, 6, 7},
        )
        self.assertEqual(template.indices_for_column_names({"HI"}), {2})
        self.assertIn("boar_room", get_template_ids())


if __name__ == "__main__":
    unittest.main()
