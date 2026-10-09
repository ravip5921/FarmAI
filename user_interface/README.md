# FarmAI User Interface

This directory contains the worker-facing FarmAI web application:

- `frontend/`: React and TypeScript interface
- `backend/`: FastAPI service and persistent job worker
- `runtime/`: generated job database and artifacts, ignored by Git

The existing root-level `streamlit_app.py` remains the developer/debugging UI.

## Start The Application

Open three PowerShell terminals from the repository root.

### 1. API

```powershell
conda activate farm-ai
python -m uvicorn user_interface.backend.app:app --reload --port 8000
```

### 2. Job Worker

```powershell
conda activate farm-ai
python -m user_interface.backend.worker
```

The worker processes one queued job at a time. It must remain running while
records are being processed.

### 3. Frontend

```powershell
cd user_interface\frontend
npm run dev
```

Open `http://localhost:5173`.

## Default Workflow

The upload screen defaults to:

- Detected table (no template)
- LLM vision handwriting recognition
- no additional filtered columns
- no ground-truth CSV

The settings button exposes template and recognition options when they need to
be changed. Choose a named template to apply its known column proportions.

Before uploading, use **Settings** beside a selected PDF, **Settings for selected
PDFs**, or **Settings for all PDFs**. Apply the settings, then upload. Each file's
settings are stored with it before analysis starts.

After uploading, use **Edit** beside a PDF to customize its template, recognition
method, hidden columns, reference ID, and comments. Settings appear directly in
the Edit dialog. Each PDF's saved settings take precedence over global defaults.

Select multiple inbox rows and choose **Delete selected** to permanently remove
those PDFs, their jobs, and saved results after confirmation. Processing PDFs
cannot be selected. Selection applies to loaded rows; use **Load older PDFs** to
include older records. Individual Edit buttons remain available.

Use **Edit job** on a queued job's page, or **Edit** in the list, to change its
settings before its first processing attempt. Running jobs and jobs resumed after
an interruption keep their processing settings. Reference IDs and comments remain
editable at every status. Reference IDs are optional, non-unique user labels;
the immutable system ID is available in the ID column tooltip.

Schema migration 4 adds saved document settings, reference IDs, and comments.
The API and worker apply pending migrations automatically on startup, preserving
existing records and artifacts. Restart both services after updating.

Jobs and generated artifacts persist under `user_interface/runtime/`, so
refreshing or closing the browser does not stop processing. The main screen
lists queued, running, and previous jobs with links back to each job. The
table also provides confirmed deletion. Deletion removes the SQLite record and
all saved artifacts; a running job must finish before it can be deleted. The
configured LLM URL and model remain in the backend environment and are never
sent to the browser.

## Verification

```powershell
conda activate farm-ai
python -m unittest discover -s tests
```

```powershell
cd user_interface\frontend
npm run lint
npm run build
```
