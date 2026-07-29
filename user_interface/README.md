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

- Boar Room Log template
- LLM vision handwriting recognition
- no additional filtered columns
- no ground-truth CSV

The settings button exposes those options when they need to be changed.
Choose "Detected table (no template)" to use FarmAI's detected grid as-is.

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
