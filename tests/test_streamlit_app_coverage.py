from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

import streamlit_app
from src.templates import FormTemplate, TemplateColumn
from tests.test_streamlit_app import _app_result, _FakeStreamlit, _SessionState


class TestStreamlitAppCoverage(unittest.TestCase):
    def test_resolve_template_filter_indices_merges_template_and_ui_columns(
        self,
    ) -> None:
        template = FormTemplate(
            id="room",
            name="Room",
            description="",
            column_widths=(1.0, 1.0),
            uniform_row_height=False,
            columns=(
                TemplateColumn(index=0, key="pen", name="Pen", filter_out=True),
                TemplateColumn(index=1, key="date", name="Date"),
            ),
        )

        self.assertEqual(
            streamlit_app.resolve_template_filter_indices(template, {"Date"}),
            {0, 1},
        )

    def test_main_renders_template_caption_for_template_results(self) -> None:
        result = _app_result()
        result.template_id = "boar_room"
        fake_st = _FakeStreamlit(
            session_state=_SessionState(
                result=result,
                edited_df=pd.DataFrame([["A1"]], columns=["Pen"]),
            ),
            buttons=[False, False],
        )

        with patch("streamlit_app.st", fake_st):
            streamlit_app.main()

        self.assertIn(("caption", "Template: boar_room"), fake_st.calls)


if __name__ == "__main__":
    unittest.main()
