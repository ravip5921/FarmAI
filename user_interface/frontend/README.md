# FarmAI Frontend

React and TypeScript frontend for the persistent FarmAI worker interface.

## Responsibilities

- record upload and default/advanced settings
- recent SQLite-backed job list and direct job links
- confirmed job deletion
- queued/running progress polling
- deskewed source and detected-grid overlay review
- editable AG Grid OCR table
- validation and ground-truth mismatch states
- ground-truth CSV attachment
- reviewed CSV download

The frontend never receives the configured LLM URL, model, or credentials.
It communicates only with the FastAPI routes under `/api`.

## Development

Start the FastAPI server and Python worker as described in
`user_interface/README.md`, then run:

```powershell
cd user_interface\frontend
npm install
npm run dev
```

Vite proxies `/api` requests to `http://127.0.0.1:8000`.

## Verification

```powershell
npm run lint
npx tsc --noEmit -p tsconfig.app.json
npm run build
```

## Main Modules

- `src/pages/UploadPage.tsx`: upload, settings, and recent jobs
- `src/pages/JobPage.tsx`: progress and completed review workflow
- `src/components/RecentJobsTable.tsx`: links, statuses, and deletion
- `src/components/OverlayViewer.tsx`: source/overlay view and cell selection
- `src/components/OcrResultGrid.tsx`: editable result table and cell states
- `src/api/jobs.ts`: typed API calls
- `src/types/api.ts`: frontend data contracts
