from __future__ import annotations

from fastapi import APIRouter

from src.ocr import get_ocr_engine_specs
from src.templates import get_template_ids, load_template

router = APIRouter()


@router.get("/settings")
def get_settings() -> dict:
    templates = []
    for template_id in get_template_ids():
        template = load_template(template_id)
        templates.append(
            {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "columns": [
                    {
                        "key": column.key,
                        "name": column.name,
                        "filter_out": column.filter_out,
                    }
                    for column in template.columns
                ],
            }
        )
    engines = [
        {
            "name": spec.name,
            "label": (
                "Best handwriting recognition"
                if spec.name == "llm-vision"
                else spec.label
            ),
            "description": spec.description,
        }
        for spec in get_ocr_engine_specs()
    ]
    return {
        "defaults": {
            "template_id": None,
            "ocr_engine": "llm-vision",
            "extra_filtered_columns": [],
        },
        "templates": templates,
        "ocr_engines": engines,
    }
