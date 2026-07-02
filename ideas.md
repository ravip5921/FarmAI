# FarmAI OCR Improvement Ideas

## Current Diagnosis

The OCR problem is probably not only an OCR-engine problem. On real farm logs,
the cropped cell images are often too small, tightly cropped, ambiguous, or
missing enough visual context that even a human has trouble reading them.

This suggests the next phase should reduce uncertainty before OCR rather than
only swapping OCR engines. The most promising direction is to use the known form
layout and known column semantics as constraints.

## Key Idea

Farm record sheets are not arbitrary documents. They usually come from a small
set of stable templates. If FarmAI knows the form template, it can:

- identify expected columns without reading the header row,
- crop only fillable regions instead of whole bordered cells,
- apply column-specific OCR settings,
- validate OCR output using expected field types,
- flag impossible values for review,
- compare detected table geometry against expected geometry.

This changes the project from generic table OCR into template-aware form
understanding.

## Does This Need an AI Element?

Not immediately.

The useful "intelligence" at this stage can be mostly deterministic:

- template matching,
- predefined boundaries,
- column metadata,
- validation rules,
- confidence scoring,
- manual correction loops.

AI can be added later where it helps most:

- fine-tuned TrOCR for handwritten cells,
- prompt-based correction using column context,
- learned template classification,
- learned crop refinement.

For a prototype, a semi-automatic workflow is probably stronger and easier to
evaluate than a fully automatic AI-heavy workflow.

## Proposed Prototype Direction

Start with a fixed known layout selected manually by the user.

Example:

```txt
farm-ai record.jpg --template boar-room-v1 --save-cells
```

The template tells FarmAI what the table should look like. Current table
detection is still useful, but it becomes a way to align the image to the known
template instead of discovering everything from scratch.

## Template-Aware Pipeline

1. Load image or PDF page.
2. Run current preprocessing and table-line detection.
3. Select template manually or detect from a small template set.
4. Align detected table to the template.
5. Use template column/row definitions to produce expected cell boxes.
6. Crop cells or fillable regions using template-aware padding.
7. Run OCR with column-specific settings.
8. Apply column-specific validation and correction.
9. Export CSV/JSON with confidence and review flags.

## Template Definition

A template could be stored as JSON or YAML.

Example fields:

```yaml
id: boar-room-v1
name: Boar Room Daily Log
page_size: letter
table:
  anchor: outer_border
  columns:
    - name: temperature
      index: 0
      type: temperature
      expected_format: decimal_or_integer
      crop:
        left_padding: 8
        right_padding: 8
        top_padding: 2
        bottom_padding: 2
      ocr_engine: trocr-handwritten
    - name: pen
      index: 1
      type: short_text
    - name: notes
      index: 5
      type: free_text
rows:
  header_rows: 1
  data_rows: dynamic
```

The first version does not need to be perfect. It can start with:

- template ID,
- expected column count,
- column names,
- column types,
- per-column crop padding,
- per-column OCR engine,
- per-column validation rules.

## Layout Alignment Options

### Option 1: Use Current Detected Grid, Then Validate Against Template

Use current table detection to produce row and column coordinates. Compare:

- detected column count vs expected column count,
- relative column widths vs template widths,
- row count vs expected range,
- outer table bounds vs expected page region.

If detection is close, accept it. If not, flag for review.

This is the easiest first step.

### Option 2: Template-Guided Grid Reconstruction

Instead of trusting every detected line, use the template to repair the grid:

- if a column line is missing, infer it from template proportions,
- if an extra line appears due to handwriting, remove it,
- if row spacing is regular, snap rows to expected intervals,
- if the outer border is detected, derive internal columns from known ratios.

This could improve table accuracy without needing a new OCR engine.

### Option 3: Manual Anchor Selection

For early experiments, allow the user to click or input:

- top-left table corner,
- bottom-right table corner,
- template name.

Then FarmAI maps the known template grid into that rectangle. This avoids
spending too much time on fully automatic layout detection before proving the OCR
benefit.

## Column-Aware OCR Advantages

Column knowledge is a major advantage.

For each column, FarmAI can choose:

- OCR engine,
- crop strategy,
- allowed character set,
- expected value format,
- correction rules,
- confidence thresholds.

Examples:

### Temperature Column

Expected values:

- `98`
- `99.5`
- `101`
- maybe range `90-110`

Helpful constraints:

- digits only plus decimal point,
- reject alphabetic output,
- correct common OCR confusions:
  - `l` or `I` -> `1`
  - `O` -> `0`
  - `S` -> `5`
  - comma -> decimal point
- flag impossible values such as `18`, `190`, `abc`.

### Date Column

Expected values:

- date-like strings,
- often repeated or sequential.

Helpful constraints:

- prefer valid dates,
- infer missing year/month from surrounding rows,
- correct separators.

### Notes Column

Expected values:

- free handwriting,
- longer text,
- lower confidence expected.

Helpful strategy:

- wider context padding,
- TrOCR or future fine-tuned HTR,
- human review priority.

## Predefined Boundary / Distinct Chunk Idea

Prof2's suggestion can be interpreted as testing OCR on controlled crop regions.

For each known column:

1. Define a fixed crop region inside the cell.
2. Save the crop.
3. Run OCR.
4. Compare to manually labeled ground truth.
5. Try variants:
   - whole cell crop,
   - fillable-area crop,
   - expanded crop,
   - binarized crop,
   - grayscale crop,
   - prepared OCR crop.

This creates a direct evaluation framework instead of judging results only from
the final CSV.

## Evaluation Plan

Create a small labeled dataset from real forms.

Suggested structure:

```txt
datasets/
  cell_eval/
    images/
      boar-room-v1_row001_col_temperature.png
    labels.csv
```

`labels.csv`:

```csv
image,template,column,row,truth
boar-room-v1_row001_col_temperature.png,boar-room-v1,temperature,1,101.4
```

Then run experiments:

- Tesseract vs TrOCR,
- binarized crop vs grayscale crop,
- context padding levels,
- template crop vs detected cell crop,
- column-specific postprocessing on/off.

Metrics:

- exact match accuracy,
- normalized numeric accuracy for temperatures,
- character error rate,
- number of cells requiring manual review.

## Near-Term Implementation Ideas

### 1. Add Template Metadata

Create a `templates/` directory with one JSON/YAML file per form.

Each template defines:

- column names,
- expected column count,
- column types,
- crop padding,
- validation rules.

### 2. Add CLI Template Flag

Add:

```txt
--template boar-room-v1
```

At first, this only skips header OCR and assigns known column names.

### 3. Add Column-Specific OCR Config

Let each column choose:

- Tesseract or TrOCR,
- PSM mode,
- crop padding,
- context padding,
- allowed output type.

### 4. Add Template-Guided Grid Repair

Use template expected column count and relative widths to repair missing or extra
vertical lines.

### 5. Add Field Validators

Implement validators for:

- temperature,
- dates,
- numeric counts,
- IDs,
- short categorical fields.

Validators should not silently overwrite uncertain values. They should produce:

- raw OCR text,
- corrected text,
- confidence/reason,
- review flag.

### 6. Add Cell Evaluation Script

Create a script like:

```txt
python eval_cells.py datasets/cell_eval/labels.csv --ocr-engine trocr-handwritten
```

Output:

- accuracy by column,
- errors by column,
- before/after postprocessing results,
- hardest cells.

## Practical First Experiment

For the next prototype, do this:

1. Pick one farm form layout.
2. Save cropped cells with several context padding values.
3. Manually label only column 1 temperature cells.
4. Add a template file for that form with column 1 marked as `temperature`.
5. Run OCR on only that column.
6. Apply temperature-specific cleanup and validation.
7. Report accuracy before and after template/column logic.

This directly tests the professors' ideas without needing a large AI training
effort yet.

## Recommended Research Direction

The most defensible direction is:

```txt
generic table OCR
-> template-aware form OCR
-> column-aware OCR and validation
-> small labeled evaluation set
-> optional fine-tuned handwritten model
```

Fine-tuning TrOCR may still be useful, but it should come after creating the
labeled cell dataset and column-level evaluation framework. Otherwise it will be
hard to know whether model training actually improved the system.

## Summary

The next big improvement should be template awareness, not just a stronger OCR
engine. A fixed form template can make table detection more reliable, remove the
need to OCR headers, improve cropping, constrain OCR outputs, and enable better
validation. This fits the project well because farm forms are stable and the
current fully generic pipeline is being asked to solve too much at once.
