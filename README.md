# FarmAI

FarmAI converts scanned or photographed farm record tables into structured,
reviewable CSV/JSON data. It detects the table first, crops individual cells,
uses a selected OCR engine to read each cell, and applies form-template
knowledge where available.

The project has three user surfaces:

- the `farm-ai` CLI for experiments and debug artifacts
- `streamlit_app.py` for developer-oriented grid and preprocessing review
- `user_interface/` for the persistent, worker-facing web application

## Current Pipeline

```text
Image or PDF
-> grayscale, adaptive binarization, denoise, and deskew
-> table-line detection and grid reconstruction
-> optional template-guided column reconstruction
-> coordinate-aligned deskewed source crops
-> cell-wise OCR
-> template validation
-> CSV/JSON export and manual review
```

FarmAI currently supports:

- bordered farm-record tables
- raster images and multi-page PDFs
- template-guided and detected-grid-only processing
- Tesseract, TrOCR handwritten, and vision-LLM OCR engines
- template-level and user-added column filtering
- temperature format/range validation
- saved cell crops, overlays, and debug artifacts
- persistent background jobs in the web interface
- optional ground-truth CSV scoring
- editable reviewed output

## Installation

```powershell
conda create -n farm-ai python=3.11 -y
conda activate farm-ai
python -m pip install -r requirements.txt
python -m pip install -e .
```

TrOCR additionally requires PyTorch and the optional handwritten dependencies:

```powershell
conda activate farm-ai
python -m pip install torch
python -m pip install -e .[htr]
```

## CLI

```powershell
farm-ai .\examples\sample_01.jpg
farm-ai .\examples\sample_01.jpg --save-all
farm-ai .\examples\sample_01.jpg --ocr-engine tesseract
farm-ai .\examples\sample_01.jpg --ocr-engine trocr-handwritten
farm-ai '.\examples\Boar room.pdf' --template boar_room --ocr-engine llm-vision
farm-ai .\examples\sample_01.jpg --template boar_room --save-cells --ocr-context-padding 8
```

The vision LLM reads its endpoint and model from the untracked root `.env`:

```dotenv
FARMAI_LLM_API_URL=http://example/api/chat
FARMAI_LLM_MODEL=your-vision-model
FARMAI_LLM_TIMEOUT_SECONDS=120
```

## Worker-Facing Web Interface

The React/FastAPI interface is the recommended pilot UI for farm users. It
provides upload, background processing, persistent job links, deskewed overlays,
editable results, validation flags, optional ground-truth metrics, and downloads.

Start three terminals from the repository root:

```powershell
conda activate farm-ai
python -m uvicorn user_interface.backend.app:app --reload --port 8000
```

```powershell
conda activate farm-ai
python -m user_interface.backend.worker
```

```powershell
cd user_interface\frontend
npm run dev
```

Open `http://localhost:5173`.

Jobs are stored in an ignored SQLite database under `user_interface/runtime/`.
The main screen links to queued, running, and previous jobs. Deleting an
eligible job removes both its database row and saved artifacts; running jobs
are protected until they finish.

See `user_interface/README.md` and
`user_interface/IMPLEMENTATION_PLAN.md` for implementation details.

## Developer Streamlit UI

Use Streamlit when inspecting source, overlay, grid, and line-detection views:

```powershell
conda activate farm-ai
streamlit run streamlit_app.py
```

## Templates

`templates/boar_room.json` describes the current Boar Room form:

- proportional column widths
- stable column keys and names
- template-level filtered columns
- expected value types and formats
- numeric ranges
- common values such as `All good`

The web UI defaults to this template but also provides
`Detected table (no template)` to use the reconstructed grid as-is.

## Dataset Preparation

Use `--save-cells` to create cell-level OCR samples, then generate an empty
label manifest:

```powershell
farm-ai .\examples\sample_01.jpg --template boar_room --save-cells --save-json
python scripts\create_label_manifest.py `
  .\debug_outputs\sample_01\sample_01_cells `
  --template boar_room `
  --source-image .\examples\sample_01.jpg `
  --output .\debug_outputs\sample_01\sample_01_labels.csv
```

See `DATASET.md` for the full labeling workflow.

## Verification

```powershell
conda activate farm-ai
python -m unittest discover -s tests
mypy src --config-file mypy.ini --no-sqlite-cache
```

```powershell
cd user_interface\frontend
npm run lint
npm run build
```

## Documentation

- `SUMMARY.md`: current code and architecture handoff
- `user_interface/IMPLEMENTATION_PLAN.md`: UI architecture and roadmap
- `user_interface/README.md`: UI startup and operation
- `DATASET.md`: cell-labeling dataset workflow
- `ideas.md`: OCR and template experimentation history
