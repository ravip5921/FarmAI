# FarmAI Cell OCR Dataset Plan

This document is a prompt/playbook for an agent building a labeled OCR dataset from FarmAI table-cell crops.

## Current Project Integration

The persistent web interface under `user_interface/` now accepts an optional
ground-truth CSV before processing or after a job completes. That CSV is for
evaluation of a complete extracted table and has visible template columns as
headers, for example:

```csv
Date,Current Temperature,HI,LO,Comments
01-May,67.8,95,,All good
```

This table-level CSV is different from the cell-image training manifest
described below:

- UI ground truth scores a completed OCR job by row and visible column.
- The dataset manifest links one cropped image to one training label and keeps
  crop/template metadata.

Attaching UI ground truth after OCR does not rerun the LLM. The result page
colors exact mismatches and reports cell accuracy. Reviewed UI edits are stored
separately from immutable `ocr_text`, so corrected exports can later provide
feedback data without overwriting the original model output.

For future dataset collection, add an explicit export step that converts
reviewed job cells and their saved crops into the manifest schema below. That
export is not implemented yet.

## Current Project Integration

The persistent web interface under `user_interface/` now accepts an optional
ground-truth CSV before processing or after a job completes. That CSV is for
evaluation of a complete extracted table and has visible template columns as
headers, for example:

```csv
Date,Current Temperature,HI,LO,Comments
01-May,67.8,95,,All good
```

This table-level CSV is different from the cell-image training manifest
described below:

- UI ground truth scores a completed OCR job by row and visible column.
- The dataset manifest links one cropped image to one training label and keeps
  crop/template metadata.

Attaching UI ground truth after OCR does not rerun the LLM. The result page
colors exact mismatches and reports cell accuracy. Reviewed UI edits are stored
separately from immutable `ocr_text`, so corrected exports can later provide
feedback data without overwriting the original model output.

For future dataset collection, add an explicit export step that converts
reviewed job cells and their saved crops into the manifest schema below. That
export is not implemented yet.

## Goal

Build a ground-truth dataset of individual cropped farm-record cell images. Each dataset row should connect one saved cell image to one manually entered label so the OCR engine can later be fine-tuned or evaluated on the same kind of crops FarmAI sees at runtime.

FarmAI already detects the table, applies optional template guidance, crops cells, and saves cropped cell images. Use that existing pipeline to create the image set. Then create a CSV manifest with empty labels and manually fill the labels.

## Why Cell-Level Ground Truth

The OCR task in this project is not full-page text recognition. The production pipeline gives the OCR engine a single cropped table cell. The training/evaluation dataset should match that:

- one sample = one cropped cell image,
- one label = the exact text visible in that cell,
- metadata = source record, template, column key, row number, value type, and crop coordinates.

This keeps the future OCR training aligned with FarmAI's real inference path.

## Recommended Dataset Layout

Use a dataset folder separate from generated debug output:

```text
datasets/
  cell_ocr/
    raw_manifests/
      record_001_labels.csv
      record_002_labels.csv
    cells/
      record_001/
        date_row_001_coords_x0000_y0100_w0054_h0043.png
        current_temperature_row_001_coords_x0098_y0100_w0054_h0043.png
      record_002/
        ...
    master_labels.csv
    splits/
      train.csv
      val.csv
      test.csv
```

During early prototyping, it is fine to keep the files under `debug_outputs/<image_name>/`. Before training, copy or consolidate the accepted crops into `datasets/cell_ocr/`.

## Step 1: Generate Cell Crops From One Source Image

Activate the conda environment first:

```powershell
conda activate farm-ai
```

Run FarmAI on a source record image or PDF and save cropped cells:

```powershell
python main.py .\examples\sample_01.jpg --template boar_room --save-cells --save-json --save-overlay --ocr-padding 0 --ocr-context-padding 8
```

For a more complete debug run:

```powershell
python main.py .\examples\sample_01.jpg --template boar_room --save-all --ocr-padding 0 --ocr-context-padding 8
```

Output will be grouped by input name:

```text
debug_outputs/<image_name>/
  <image_name>_cells/
    date_row_000_coords_x....
    current_temperature_row_001_coords_x....
    hi_row_001_coords_x....
    lo_row_001_coords_x....
    comments_row_001_coords_x....
  table_ocr_<image_name>.json
  table_overlay_<image_name>.png
  final_table_<image_name>.png
```

Use `--ocr-context-padding` to expand the saved crop outward. This is important because handwriting is sometimes cut off by tight table-cell boundaries.

Suggested sweep for a hard image:

```powershell
python main.py .\examples\sample_01.jpg --template boar_room --save-cells --ocr-padding 0 --ocr-context-padding 4
python main.py .\examples\sample_01.jpg --template boar_room --save-cells --ocr-padding 0 --ocr-context-padding 8
python main.py .\examples\sample_01.jpg --template boar_room --save-cells --ocr-padding 0 --ocr-context-padding 12
```

Choose one padding setting for a dataset batch so samples remain consistent.

## Step 2: Create an Empty Label CSV

After cells are saved, create a manual labeling manifest:

```powershell
python scripts\create_label_manifest.py .\debug_outputs\sample_01\sample_01_cells --template boar_room --source-image .\examples\sample_01.jpg --output .\debug_outputs\sample_01\sample_01_labels.csv
```

If row 0 is a printed header row and should not be part of handwritten OCR training:

```powershell
python scripts\create_label_manifest.py .\debug_outputs\sample_01\sample_01_cells --template boar_room --source-image .\examples\sample_01.jpg --output .\debug_outputs\sample_01\sample_01_labels.csv --skip-header-row
```

The CSV contains:

```text
cell_image,label,label_status,source_image,template_id,column_key,column_name,value_type,format,range_min,range_max,row,x,y,width,height,notes
```

The important manual field is `label`. Leave it empty at generation time, then fill it manually.

The metadata columns are useful later for:

- training/evaluating temperature-only OCR,
- grouping comments separately from numeric cells,
- excluding bad crops,
- finding repeated failure modes by column,
- rebuilding the source context for a questionable crop.

## Step 3: Manual Labeling Rules

Open the cell images and the generated CSV side by side.

Fill `label` using these rules:

- Transcribe exactly what is visible in the crop.
- Do not correct the value using row context unless the writing is visually clear.
- Preserve decimals in temperature fields, for example `88.1`.
- For truly blank cells, leave `label` empty and set `label_status` to `blank`.
- For illegible cells, leave `label` empty and set `label_status` to `illegible`.
- For crops that cut off handwriting, leave `label` empty and set `label_status` to `bad_crop`.
- For uncertain but usable labels, enter the best transcription and set `label_status` to `uncertain`.
- For good labels, set `label_status` to `ok`.

Suggested `label_status` values:

```text
ok
blank
uncertain
illegible
bad_crop
not_handwritten
```

Only `ok` rows should be used for normal training at first. `uncertain`, `illegible`, and `bad_crop` rows are still valuable for error analysis.

## Step 4: Use Template Metadata

The Boar Room template provides column-level metadata:

- `date`: `date_dd_mon`
- `current_temperature`: `temperature`
- `hi`: `temperature`
- `lo`: `temperature`
- `comments`: `english_text`

Filtered template columns such as divider gaps, Boar Pac, empty columns, and manager initials are not saved by `--save-cells`, because FarmAI now saves only unfiltered columns.

This means the generated label CSV is already focused on columns that matter for output.

Use `value_type` to create task-specific subsets:

```text
temperature dataset: current_temperature, hi, lo
date dataset: date
comment dataset: comments
all handwritten cells: all non-header rows with label_status=ok
```

## Step 5: Quality Control Before Training

Before using labels for training:

1. Check that every `ok` row has a non-empty label unless the true class is intentionally blank.
2. Check that every listed `cell_image` exists.
3. Review `bad_crop` rows to decide whether cell padding or grid detection should be adjusted.
4. Keep train/validation/test splits by source image, not by random cell, so the same physical form does not leak across splits.
5. Keep the original source image and debug overlay for each labeled batch.

Recommended split strategy:

```text
train: 70 percent of source images
val:   15 percent of source images
test:  15 percent of source images
```

Split by `source_image`, not by individual crop.

## Step 6: Batch Workflow For Many Images

For each source image:

```powershell
python main.py .\examples\record_001.jpg --template boar_room --save-cells --save-json --save-overlay --ocr-padding 0 --ocr-context-padding 8
python scripts\create_label_manifest.py .\debug_outputs\record_001\record_001_cells --template boar_room --source-image .\examples\record_001.jpg --output .\debug_outputs\record_001\record_001_labels.csv --skip-header-row
```

Repeat for each record:

```powershell
python main.py .\examples\record_002.jpg --template boar_room --save-cells --save-json --save-overlay --ocr-padding 0 --ocr-context-padding 8
python scripts\create_label_manifest.py .\debug_outputs\record_002\record_002_cells --template boar_room --source-image .\examples\record_002.jpg --output .\debug_outputs\record_002\record_002_labels.csv --skip-header-row
```

After manual labeling, copy the final labeled CSVs into:

```text
datasets/cell_ocr/raw_manifests/
```

and copy or reference the corresponding cell image folders.

## Agent Task Prompt

Use this prompt for an implementation agent:

```text
You are working in the FarmAI repository. Read SUMMARY.md and DATASET.md first.

Goal:
Create a cell-level OCR labeling dataset from farm record images.

Current tools:
- main.py / farm-ai can process a source image or PDF.
- --template boar_room applies template-guided table reconstruction.
- --save-cells saves cropped cell images under debug_outputs/<image_name>/<image_name>_cells/.
- --ocr-context-padding expands cell crops outward.
- scripts/create_label_manifest.py creates an empty-label CSV from a saved cell directory.

Workflow for each source image:
1. Activate the farm-ai conda environment.
2. Run FarmAI with --template boar_room, --save-cells, and a chosen --ocr-context-padding value.
3. Inspect the overlay and a few crops to confirm the grid/crop quality is acceptable.
4. Run scripts/create_label_manifest.py on the saved cells directory.
5. Use --skip-header-row unless printed headers should be labeled.
6. Manually fill the label column.
7. Mark label_status as ok, blank, uncertain, illegible, bad_crop, or not_handwritten.
8. Keep source_image, template_id, column_key, value_type, row, and coordinate metadata intact.

Do not train on uncertain, illegible, or bad_crop rows initially.
Do not randomly split cells from the same source image across train and test.
Split by source_image.
```

## Notes For Future Training

Start with evaluation before fine-tuning:

- Use the labeled CSV to run the current OCR engines on the same cell images.
- Report accuracy by `value_type` and `column_key`.
- Track exact-match accuracy for temperature columns.
- Track character error rate for comments.

Then fine-tune or adapt OCR with the same crop format FarmAI uses in production.
