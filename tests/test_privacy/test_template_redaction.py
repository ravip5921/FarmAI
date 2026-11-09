from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np

from src.privacy import (
    redact_filtered_template_columns,
    redacted_template_column_indices,
)
from src.table.grid_reconstruction import GridStructure
from src.templates import load_template


class TestTemplateRedaction(unittest.TestCase):
    def test_border_masks_accept_both_hough_output_shapes(self) -> None:
        template = load_template("nursery_room_4")
        image = np.full((100, 130, 3), 255, np.uint8)
        grid = GridStructure([10, 30, 50, 70, 90], list(range(0, 121, 10)), [])
        # Hough coordinates are relative to the cropped table. Both borders
        # slope left, so a rectangular fallback would expose initials at the
        # bottom and obscure comments at the top.
        segments = np.array([[92, 0, 88, 80], [122, 0, 118, 80]], dtype=np.int32)
        for shape in ((2, 1, 4), (2, 4)):
            with (
                self.subTest(shape=shape),
                patch(
                    "src.privacy.template_redaction.cv2.HoughLinesP",
                    return_value=segments.reshape(shape),
                ),
            ):
                redacted = redact_filtered_template_columns(image, grid, template)
                self.assertTrue(np.all(redacted[15, 93:122] == 0))
                self.assertTrue(np.all(redacted[85, 89:118] == 0))
                self.assertTrue(np.all(redacted[15, 90] == 255))
                self.assertTrue(np.all(redacted[85, 122] == 255))
                np.testing.assert_array_equal(redacted[:, :85], image[:, :85])
                np.testing.assert_array_equal(redacted[:10], image[:10])
                np.testing.assert_array_equal(redacted[90:], image[90:])

    def test_finisher_masks_follow_bowed_rules_and_preserve_comments(self) -> None:
        template = load_template("finisher_room_3")
        image = np.full((800, 1250, 3), 255, np.uint8)
        columns = [
            51,
            107,
            166,
            222,
            281,
            290,
            348,
            427,
            437,
            504,
            574,
            582,
            1024,
            1105,
            1118,
            1172,
        ]
        grid = GridStructure(list(range(100, 701, 30)), columns, [])
        ys = np.arange(100, 700)
        shift = np.rint(12 * np.cos((ys - 100) * 2 * np.pi / 600)).astype(int)
        for x in columns:
            points = np.column_stack((x + shift, ys)).astype(np.int32)
            cv2.polylines(image, [points], False, (0, 0, 0), 2)
        redacted = redact_filtered_template_columns(image, grid, template)
        for y in range(110, 690, 20):
            offset = shift[y - 100]
            self.assertTrue(np.all(redacted[y, 1028 + offset : 1168 + offset] == 0))
            self.assertTrue(
                np.array_equal(redacted[y, : 1018 + offset], image[y, : 1018 + offset])
            )

    def test_nursery_masks_follow_slanted_rules_without_covering_comments(self) -> None:
        template = load_template("nursery_room_4")
        image = np.full((800, 1200, 3), 255, np.uint8)
        columns = [60, 120, 183, 243, 303, 315, 375, 462, 474, 990, 1080, 1092, 1152]
        grid = GridStructure(
            row_coords=list(range(100, 701, 30)), col_coords=columns, cells=[]
        )
        for x in columns:
            cv2.line(image, (x + 8, 100), (x - 8, 700), (0, 0, 0), 2)
        for y in grid.row_coords:
            shift = round(8 - 16 * (y - 100) / 600)
            cv2.line(image, (60 + shift, y), (1152 + shift, y), (0, 0, 0), 1)

        redacted = redact_filtered_template_columns(image, grid, template)
        for y in range(110, 691, 20):
            shift = round(8 - 16 * (y - 100) / 600)
            self.assertTrue(np.all(redacted[y, 993 + shift : 1149 + shift] == 0))
            self.assertTrue(
                np.array_equal(redacted[y, : 985 + shift], image[y, : 985 + shift])
            )
        self.assertTrue(np.array_equal(redacted[:100], image[:100]))

        # Turning off border following retains the existing rectangular policy,
        # including leaving structural dividers alone for other templates.
        disabled = replace(template, redaction_follow_column_borders=False)
        rectangular = redact_filtered_template_columns(image, grid, disabled)
        self.assertTrue(
            np.array_equal(rectangular[115, 1080:1092], image[115, 1080:1092])
        )

    def test_nursery_without_visible_borders_uses_grid_bounds(self) -> None:
        template = load_template("nursery_room_4")
        image = np.full((40, 130), 255, np.uint8)
        grid = GridStructure([5, 20, 35], list(range(0, 121, 10)), [])
        redacted = redact_filtered_template_columns(image, grid, template)
        self.assertTrue(np.all(redacted[5:35, 90:120] == 0))
        self.assertTrue(np.all(redacted[5:35, :90] == 255))

    def test_finisher_unreliable_borders_fall_back_to_grid_bounds(self) -> None:
        template = load_template("finisher_room_3")
        # A short table cannot support tracing; a taller blank table has no
        # reliable rule to follow. Both must still mask the sensitive fields.
        for rows in ([5, 8, 11], [5, 20, 35]):
            with self.subTest(rows=rows):
                image = np.full((40, 160, 3), 255, np.uint8)
                grid = GridStructure(rows, list(range(0, 151, 10)), [])
                redacted = redact_filtered_template_columns(image, grid, template)
                self.assertTrue(np.all(redacted[rows[0] : rows[-1], 120:150] == 0))
                np.testing.assert_array_equal(redacted[:, :120], image[:, :120])
                np.testing.assert_array_equal(redacted[:, 150:], image[:, 150:])
                np.testing.assert_array_equal(redacted[: rows[0]], image[: rows[0]])
                np.testing.assert_array_equal(redacted[rows[-1] :], image[rows[-1] :])

    def test_empty_and_outside_grid_are_unchanged(self) -> None:
        template = load_template("boar_room")
        image = np.full((10, 10, 3), 255, dtype=np.uint8)
        grid = GridStructure(row_coords=[0, 10], col_coords=[0, 2], cells=[])
        self.assertEqual(
            redacted_template_column_indices(
                template, extra_filtered_columns={"comments"}
            ),
            {5, 6, 7, 9},
        )
        redacted = redact_filtered_template_columns(image, grid, template)
        self.assertTrue(np.array_equal(redacted, image))
        grid = GridStructure(row_coords=[0], col_coords=[0, 10], cells=[])
        self.assertTrue(
            np.array_equal(
                redact_filtered_template_columns(image, grid, template), image
            )
        )
        grid = GridStructure(
            row_coords=[0, 10], col_coords=[0, 2, 4, 6, 8, 10, 12], cells=[]
        )
        redacted = redact_filtered_template_columns(image, grid, template)
        self.assertTrue(np.array_equal(redacted, image))
        grid = GridStructure(row_coords=[10, 12], col_coords=list(range(11)), cells=[])
        redacted = redact_filtered_template_columns(image, grid, template)
        self.assertTrue(np.array_equal(redacted, image))

    def test_redacted_indices_skip_only_dividers(self) -> None:
        template = load_template("boar_room")

        self.assertEqual(redacted_template_column_indices(template), {5, 7, 9})

    def test_redacts_entire_sensitive_column_span(self) -> None:
        template = load_template("boar_room")
        image = np.full((20, 40, 3), 255, dtype=np.uint8)
        grid = GridStructure(
            row_coords=[2, 10, 18],
            col_coords=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
            cells=[],
        )

        redacted = redact_filtered_template_columns(image, grid, template)

        self.assertTrue(np.all(redacted[2:18, 20:24] == 0))
        self.assertTrue(np.all(redacted[2:18, 28:32] == 0))
        self.assertTrue(np.all(redacted[2:18, 36:40] == 0))
        self.assertTrue(np.all(redacted[2:18, 16:20] == 255))


if __name__ == "__main__":
    unittest.main()
