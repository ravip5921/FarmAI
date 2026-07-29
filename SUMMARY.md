# FarmAI Code Summary

FarmAI is a Python 3.11 project for layout-aware OCR of bordered farm record tables. The project converts scanned images or PDFs into structured table data by detecting the table grid first, cropping individual cells, running OCR per cell, and exporting CSV/JSON results.

## Start Here

- `README.md` explains the project goal, install steps, CLI usage, Streamlit usage, and OCR engine choices.
- `main.py` is the command-line entry point used by the installed `farm-ai` console script.
- `streamlit_app.py` is the local review UI for uploading a record, previewing table detection, editing OCR output, and downloading CSV.
- `templates/` contains known farm form layouts, including the Boar Room log template.
- `src/templates.py` loads and validates template metadata used by table repair, column filtering, and column-aware OCR.
- `src/` contains the reusable pipeline code.
- `tests/` mirrors the source packages and is the best place to see small examples of expected behavior.

## Runtime Flow

The main pipeline is:

1. Load an image or PDF with `src/core/io.py`.
2. Wrap pages in `DocumentImage` objects from `src/core/image.py`.
3. Run preprocessing stages: grayscale, Sauvola binarization, denoise, deskew.
4. Detect horizontal and vertical table lines.
5. Detect line intersections and reconstruct a grid.
6. Refine grid row/column axes with projection profiles and support checks.
7. Optionally repair the grid with a known form template.
8. Optionally run perspective correction from detected table corners and re-detect the table.
9. Crop cells from the detected or template-repaired grid.
10. Run OCR on each cropped cell, optionally using template-driven column rules.
11. Export or display the result as CSV/JSON/editable table data.

## Top-Level Files

- `main.py`: CLI orchestration. Parses command-line options, builds the preprocessing pipeline, loads images/PDFs, runs table detection, applies optional templates/perspective correction, runs OCR, prints CSV, and optionally saves debug images, overlays, cropped cells, CSV, and JSON files.
- `streamlit_app.py`: Streamlit app for manual review. Handles file upload, OCR engine selection, optional template selection, optional column filtering, document/grid/line previews, editable dataframe output, confidence summaries, reset, and CSV download.
- `pyproject.toml`: Package metadata, dependencies, optional extras, setuptools package discovery, and the `farm-ai = main:main` console script.
- `requirements.txt`: Runtime and developer dependencies used by local setup and CI.
- `noxfile.py`: Automation sessions for type checking, tests with coverage, and formatting. Uses external tools from the active environment.
- `mypy.ini`: Mypy configuration for Python 3.11, with missing imports ignored and skipped import following.
- `.coveragerc`: Coverage settings. Omits tests and `__init__.py` files from coverage reports.
- `README.md`: Human-facing project description, motivation, pipeline outline, install steps, CLI examples, Streamlit instructions, and OCR backend notes.
- `templates/boar_room.json`: Known Boar Room table layout. Stores proportional column widths, uniform-row hint, column keys/names, template-level filter flags, value types, allowed ranges, and common values.

## Source Package

### `src/core`

Foundation types and utilities shared by the rest of the project.

- `src/core/image.py`: Defines `DocumentImage`, a small wrapper for an image array plus metadata. Most pipeline stages accept and return this type.
- `src/core/stage.py`: Defines the abstract `PipelineStage` interface. New image-processing stages implement `process(doc)`.
- `src/core/pipeline.py`: Defines `Pipeline`, a sequential runner for `PipelineStage` objects. It supports adding/extending stages and can be called directly.
- `src/core/io.py`: Loads raster images with OpenCV and PDFs with `pypdfium2`. Returns either a single `DocumentImage` or a `LoadedDocument` with page-level metadata.
- `src/core/visualization.py`: Debug/display helpers. `show()` uses matplotlib for visual inspection, and `save_debug()` writes image-like data to disk.
- `src/core/__init__.py`: Package marker.

### `src/templates.py`

Template metadata for known farm forms.

- Defines `TemplateColumn` and `FormTemplate`.
- Loads templates from `templates/*.json`.
- Validates that `column_widths` match contiguous column definitions.
- Exposes template-derived column names, column keys, filtered column indices, and lookup helpers for UI/default column filtering.
- The Boar Room template currently drives proportional column reconstruction, template-level filtering, known header names, and column-specific OCR rules.

### `src/preprocessing`

Image cleanup stages that prepare a page for table detection.

- `src/preprocessing/grayscale.py`: `GrayscaleStage` converts BGR/color input to grayscale and marks metadata with `grayscale=True`.
- `src/preprocessing/sauvola.py`: `SauvolaBinarizationStage` applies Sauvola adaptive thresholding and outputs a 0/255 binary image.
- `src/preprocessing/denoise.py`: `MorphologicalDenoiseStage` removes small binary foreground components or applies median blur to non-binary images.
- `src/preprocessing/skew.py`: `SkewCorrectionStage` estimates page skew from Hough line angles and rotates the image when the angle is meaningful.
- `src/preprocessing/perspective.py`: Reserved preprocessing placeholder. The active table-corner perspective correction currently lives under `src/table/perspective_correction.py` because it depends on detected table masks/corners.
- `src/preprocessing/__init__.py`: Package marker.

### `src/analysis`

Reusable image-analysis helpers used by table detection and refinement.

- `src/analysis/connected_components.py`: Wraps OpenCV connected-component analysis, filters components by area, and estimates median character height.
- `src/analysis/character_size.py`: Builds a `CharacterSizeReport` from connected-component heights. Table line detection uses this to choose morphology kernel sizes.
- `src/analysis/projection_profiles.py`: Counts foreground pixels by row or column, smooths profiles, clusters high-support bins into peaks, and returns peak positions.
- `src/analysis/__init__.py`: Package marker.

### `src/table`

Table structure detection, grid reconstruction, cell extraction, and visual previews.

- `src/table/__init__.py`: Public table API. Defines `TablePipelineResult`, runs the full table-detection sequence in `process_table_image()`, applies optional template guidance, renders grid-only images, renders overlays, and re-exports key table helpers.
- `src/table/line_detection.py`: Detects horizontal and vertical table-rule masks from a binary image. Uses morphology, character-size-aware kernel selection, gap closing, connected-component filtering, Hough fallback for weak verticals, and crossing-based cleanup.
- `src/table/intersections.py`: Finds intersections by combining horizontal and vertical masks, filters intersection components, and returns centroid coordinates.
- `src/table/grid_reconstruction.py`: Converts intersection centroids into clustered row/column coordinates and `GridCell` bounding boxes.
- `src/table/line_refinement.py`: Refines the raw grid using projection-profile peaks, crossing support, segment support, header-row evidence, endpoint candidates, missing-row filling, duplicate-column pruning, and wide-field cleanup.
- `src/table/cell_extraction.py`: Safely clips and crops each detected `GridCell` into `ExtractedCell` images for OCR.
- `src/table/template_guidance.py`: Repairs a detected grid using known template column proportions. It infers internal column boundaries from the detected outer span and can regularize row coordinates when a template declares uniform row height.
- `src/table/perspective_correction.py`: Detects the table contour from horizontal/vertical line masks, expands corners by optional padding, warps the table to a rectangle, and renders corner overlays for debugging.

### `src/ocr`

OCR abstractions, engines, table recognition, and export orchestration.

- `src/ocr/base.py`: Defines `OcrText` and the `CellOcrEngine` protocol. `OcrText` carries final text, confidence, optional raw rejected text, and optional validation errors. Any OCR backend only needs a `recognize(image)` method.
- `src/ocr/registry.py`: Lists supported OCR engines and creates engine instances. Default engine is `tesseract`; optional handwritten engine is `trocr-handwritten`.
- `src/ocr/tesseract_engine.py`: Tesseract backend. Prepares cell images, checks that the Tesseract executable is on `PATH`, runs `pytesseract`, and computes mean word confidence.
- `src/ocr/trocr_engine.py`: Optional Microsoft TrOCR handwritten backend. Loads Hugging Face processor/model, chooses CPU or CUDA, prepares images as RGB PIL images, and decodes generated text.
- `src/ocr/llm_client.py`: Vision-LLM OCR backend and verifier. Uses the configured chat/image endpoint, encodes cropped cell images as base64, prompts with template context when available, parses JSON responses, and can verify/correct baseline OCR output.
- `src/ocr/cell_ocr.py`: Defines `OcrCell` and `OcrTable`, recognizes cropped cells, builds row/column matrices, saves optional cropped-cell debug images, applies template/header-based column filtering before compacting output columns, and applies column-specific OCR rules to data cells.
- `src/ocr/column_rules.py`: Template-driven OCR rule layer. Builds rules from template columns, currently for `temperature` columns. Temperature values accept `N`, `NN`, `NN.N`, `NN.NN`, `NNN`, `NNN.N`, and `NNN.NN`; Tesseract receives a numeric whitelist (`0123456789.`) and numeric mode; invalid values are blanked while raw rejected text and validation errors are retained.
- `src/ocr/table_ocr.py`: Higher-level table OCR helpers. Runs cell OCR for a `GridStructure`, passes template column rules through to the cell OCR layer, and optionally writes CSV/JSON exports.
- `src/ocr/column_filter.py`: Holds default header names to filter from OCR output. It is currently an empty set and is overridden by the Streamlit UI when the user enters filters.
- `src/ocr/__init__.py`: Re-exports the public OCR API.

### `src/export`

Serialization helpers for OCR tables.

- `src/export/csv_export.py`: Converts an `OcrTable` to rows or CSV text and writes CSV files.
- `src/export/json_export.py`: Converts an `OcrTable` to a JSON-compatible dictionary/string and writes JSON files with cell positions, text, confidence, and optional validation metadata such as `raw_text` and `validation_error`.
- `src/export/__init__.py`: Re-exports CSV and JSON export helpers.

## Tests

Tests use `unittest` and mostly small synthetic images/mocks. The suite is organized by source package:

- `tests/test_main_pipeline.py`: CLI pipeline construction, image processing, per-file debug output directories, OCR export paths, page handling, and module entrypoint behavior.
- `tests/test_main_pipeline_coverage.py`: Additional coverage for perspective debug saves and template-driven OCR wiring.
- `tests/test_cli_output.py`: CSV print block formatting for CLI inspection.
- `tests/test_streamlit_app.py`: Streamlit helper functions and UI flow using fake Streamlit objects.
- `tests/test_core/`: `DocumentImage`, document loading, PDF handling, and pipeline/stage behavior.
- `tests/test_preprocessing/`: grayscale conversion, Sauvola binarization, denoise behavior, and skew correction.
- `tests/test_analysis/`: connected components, character-size estimates, and projection-profile utilities.
- `tests/test_table/`: line detection, helper fallbacks, intersections, grid reconstruction, grid refinement, template guidance, perspective correction, cell extraction, and full table pipeline behavior.
- `tests/test_ocr/`: OCR registry, Tesseract wrapper behavior, TrOCR wrapper behavior with fake modules, column OCR rules, and cell/table OCR logic.
- `tests/test_export/`: CSV and JSON serialization, validation metadata, and file writing.
- `tests/test_templates.py` and `tests/test_templates_coverage.py`: Template loading, validation, column keys/names, template IDs, and error handling.

Run tests through:

```bash
nox -s tests
```

or directly:

```bash
python -m unittest discover -s tests
```

## Data, Demos, and Generated Artifacts

- `examples/`: Sample input images/PDFs used for manual runs and experimentation.
- `demo/`: Demo presentation/materials and example outputs such as overlays, line-detection images, and OCR JSON.
- `references/`: Research PDFs and bibliography entries that informed the table-recognition approach.
- `notebooks/`: Reserved for notebooks; currently only contains `.gitkeep`.
- `debug_outputs/`: Generated debug images and OCR exports from previous runs. CLI output is organized by input name, for example `debug_outputs/<image_name>/`. Cropped cells are saved under `debug_outputs/<image_name>/<image_name>_cells/` when `--save-cells` or `--save-all` is used, and filenames include the template column key, row number, and crop coordinates.
- `farm_ai.egg-info/`: Generated packaging metadata from editable/install builds.
- `.github/workflows/ci.yml`: CI workflow. Runs on Ubuntu, Windows, and macOS, creates a Python 3.11 conda environment, installs dependencies and the editable package, then runs `nox`.

## Common Commands

Install locally:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the CLI:

```bash
farm-ai ./examples/sample_01.jpg
farm-ai ./examples/sample_01.jpg --save-all
farm-ai ./examples/sample_01.jpg --ocr-engine tesseract
farm-ai ./examples/sample_01.jpg --ocr-engine trocr-handwritten
farm-ai ./examples/sample_01.jpg --template boar_room --ocr-engine llm-vision
farm-ai ./examples/sample_01.jpg --template boar_room --ocr-engine tesseract --llm-verify invalid
farm-ai ./examples/sample_01.jpg --template boar_room --save-cells --ocr-context-padding 8
farm-ai ./examples/sample_01.jpg --template boar_room --save-all --perspective-correct
```

Run the review UI:

```bash
streamlit run streamlit_app.py
```

Run automation:

```bash
nox -s typecheck
nox -s tests
nox -s format
```

## Where To Make Changes

- Add or adjust known form layouts in `templates/*.json`; update `src/templates.py` if the template schema changes.
- Add or tune preprocessing behavior in `src/preprocessing/`.
- Change grid/table detection in `src/table/line_detection.py`, `src/table/intersections.py`, `src/table/grid_reconstruction.py`, or `src/table/line_refinement.py`.
- Change template-guided grid repair in `src/table/template_guidance.py`.
- Change optional table-corner perspective correction in `src/table/perspective_correction.py`.
- Add a new OCR backend by implementing `CellOcrEngine` from `src/ocr/base.py` and registering it in `src/ocr/registry.py`.
- Change the vision LLM endpoint/model defaults or prompt handling in `src/ocr/llm_client.py`.
- Change template-driven OCR validation, Tesseract whitelists, regexes, or retry behavior in `src/ocr/column_rules.py`.
- Change CSV/JSON output shape in `src/export/`.
- Change CLI options or saved artifacts in `main.py`.
- Change the review workflow, editable table UI, or upload behavior in `streamlit_app.py`.
