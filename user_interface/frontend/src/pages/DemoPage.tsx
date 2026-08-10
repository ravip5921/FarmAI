import {
  Alert,
  Button,
  Checkbox,
  FormControlLabel,
  IconButton,
  LinearProgress,
  Tooltip,
} from '@mui/material'
import { Download, FileText, Home, RotateCcw, Upload, X } from 'lucide-react'
import {
  useMemo,
  useRef,
  useState,
  type DragEvent,
} from 'react'
import { Link } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { OcrResultGrid } from '../components/OcrResultGrid'
import { OverlayViewer } from '../components/OverlayViewer'
import { createDemoResult } from '../demo/demoResult'
import type { JobResult, ResultCell } from '../types/api'

type DemoStep = 'upload' | 'processing' | 'review'

function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function percent(value: number | null | undefined) {
  return value == null ? 'Not scored' : `${(value * 100).toFixed(1)}%`
}

function updateCell(
  data: JobResult,
  pageNumber: number,
  target: ResultCell,
  value: string,
): JobResult {
  return {
    ...data,
    pages: data.pages.map((page) =>
      page.page_number === pageNumber
        ? {
            ...page,
            cells: page.cells.map((cell) =>
              cell.row === target.row && cell.column_key === target.column_key
                ? {
                    ...cell,
                    reviewed_text: value,
                    was_edited: value !== cell.ocr_text,
                  }
                : cell,
            ),
          }
        : page,
    ),
  }
}

function csvValue(value: string) {
  return /[",\n\r]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value
}

function downloadCsv(data: JobResult) {
  const page = data.pages[0]
  const lines = [page.columns.map((column) => csvValue(column.name)).join(',')]
  for (let row = 1; row <= page.data_row_count; row += 1) {
    lines.push(
      page.columns
        .map((column) => {
          const cell = page.cells.find(
            (item) => item.row === row && item.column_key === column.key,
          )
          return csvValue(cell?.reviewed_text ?? '')
        })
        .join(','),
    )
  }
  const blob = new Blob([`${lines.join('\n')}\n`], {
    type: 'text/csv;charset=utf-8',
  })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'farmai-demo-reviewed.csv'
  link.click()
  URL.revokeObjectURL(url)
}

export function DemoPage() {
  const fileInput = useRef<HTMLInputElement>(null)
  const [record, setRecord] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [step, setStep] = useState<DemoStep>('upload')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<JobResult | null>(null)
  const [selectedCell, setSelectedCell] = useState<ResultCell | null>(null)
  const [needsReviewOnly, setNeedsReviewOnly] = useState(false)

  const acceptFile = (file?: File) => {
    if (file) setRecord(file)
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    acceptFile(event.dataTransfer.files[0])
  }

  const startDemo = () => {
    if (!record) return
    setStep('processing')
    setProgress(0)
    const started = Date.now()
    const id = window.setInterval(() => {
      const next = Math.min(100, Math.round(((Date.now() - started) / 2000) * 100))
      setProgress(next)
      if (next >= 100) {
        window.clearInterval(id)
        setResult(createDemoResult(record.name || 'boar-room-demo.jpg'))
        setStep('review')
      }
    }, 120)
  }

  const page = result?.pages[0]
  const reviewCount = useMemo(
    () =>
      page?.cells.filter(
        (cell) =>
          Boolean(cell.validation_error) || cell.ground_truth_match === false,
      ).length ?? 0,
    [page],
  )

  if (step === 'processing') {
    return (
      <div className="app-shell">
        <AppHeader backTo="/" />
        <main className="page">
          <div className="progress-layout">
            <div className="page-heading">
              <h1>Reading demo record</h1>
              <p>FarmAI is simulating the same review flow with saved results.</p>
            </div>
            <section className="progress-panel" aria-live="polite">
              <p className="progress-file">{record?.name ?? 'Demo record'}</p>
              <p className="progress-stage">
                {progress < 35
                  ? 'Finding the table'
                  : progress < 82
                    ? `Reading cells (${Math.round((progress / 100) * 40)} of 40)`
                    : 'Preparing review'}
              </p>
              <LinearProgress
                variant="determinate"
                value={progress}
                aria-label="Demo progress"
                sx={{ height: 9, borderRadius: 1 }}
              />
              <div className="progress-meta">
                <span>{progress}% of demo</span>
                <span>About 2 seconds</span>
              </div>
              <div className="return-note">
                This demo does not contact the handwriting service.
              </div>
            </section>
          </div>
        </main>
      </div>
    )
  }

  if (step === 'review' && result && page) {
    return (
      <div className="app-shell">
        <AppHeader backTo="/" />
        <main className="page review-page">
          <Alert severity="info" sx={{ mb: 2 }}>
            Demo mode: these are saved sample results for the Boar Room template.
          </Alert>
          <div className="review-topbar">
            <div className="review-title">
              <h1>{result.filename}</h1>
              <p>Boar Room | Best handwriting recognition</p>
            </div>
            <div className="review-actions">
              <Button
                component={Link}
                to="/"
                variant="outlined"
                startIcon={<Home size={17} />}
              >
                Home
              </Button>
              <Tooltip title="Start demo again">
                <IconButton
                  aria-label="Start demo again"
                  onClick={() => {
                    setStep('upload')
                    setProgress(0)
                    setResult(null)
                    setSelectedCell(null)
                  }}
                >
                  <RotateCcw size={19} />
                </IconButton>
              </Tooltip>
              <Button
                variant="contained"
                startIcon={<Download size={17} />}
                onClick={() => downloadCsv(result)}
              >
                Download CSV
              </Button>
            </div>
          </div>

          <section className="metrics-band" aria-label="Result summary">
            <div className="metric">
              <span className="metric__label">Rows</span>
              <span className="metric__value">{page.data_row_count}</span>
            </div>
            <div className="metric">
              <span className="metric__label">Cells read</span>
              <span className="metric__value">{page.cells.length}</span>
            </div>
            <div className="metric">
              <span className="metric__label">Needs review</span>
              <span className="metric__value">{reviewCount}</span>
            </div>
            <div className="metric">
              <span className="metric__label">Exact accuracy</span>
              <span className="metric__value">
                {percent(result.metrics?.exact_accuracy)}
              </span>
            </div>
          </section>

          <div className="review-workspace">
            <OverlayViewer
              page={page}
              selectedCell={selectedCell}
              onSelectCell={setSelectedCell}
            />
            <section className="review-pane">
              <div className="pane-toolbar">
                <span className="pane-title">Extracted table</span>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={needsReviewOnly}
                      onChange={(event) =>
                        setNeedsReviewOnly(event.target.checked)
                      }
                    />
                  }
                  label="Needs review only"
                />
              </div>
              <OcrResultGrid
                page={page}
                needsReviewOnly={needsReviewOnly}
                selectedCell={selectedCell}
                onSelectCell={setSelectedCell}
                onEdit={(cell, value) => {
                  setResult((current) =>
                    current ? updateCell(current, page.page_number, cell, value) : current,
                  )
                }}
              />
            </section>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <AppHeader backTo="/" />
      <main className="page upload-page">
        <div className="page-heading">
          <h1>Demo a farm record</h1>
          <p>
            Choose any record image or PDF. Demo mode shows saved Boar Room
            results without contacting the handwriting service.
          </p>
        </div>

        <div
          className={[
            'dropzone',
            dragActive ? 'dropzone--active' : '',
            record ? 'dropzone--selected' : '',
          ].join(' ')}
          role="button"
          tabIndex={0}
          onClick={() => fileInput.current?.click()}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              fileInput.current?.click()
            }
          }}
          onDragEnter={(event) => {
            event.preventDefault()
            setDragActive(true)
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
        >
          <input
            ref={fileInput}
            hidden
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.pdf"
            onChange={(event) => acceptFile(event.target.files?.[0])}
          />
          {record ? (
            <div className="selected-file">
              <span className="selected-file__icon">
                <FileText size={22} aria-hidden="true" />
              </span>
              <span>
                <span className="selected-file__name">{record.name}</span>
                <span className="selected-file__size">
                  {formatSize(record.size)}
                </span>
              </span>
              <Tooltip title="Remove file">
                <IconButton
                  aria-label="Remove selected file"
                  onClick={(event) => {
                    event.stopPropagation()
                    setRecord(null)
                  }}
                >
                  <X size={20} />
                </IconButton>
              </Tooltip>
            </div>
          ) : (
            <div className="dropzone__content">
              <span className="dropzone__icon">
                <Upload size={24} aria-hidden="true" />
              </span>
              <p className="dropzone__title">Drop a record here</p>
              <p className="dropzone__hint">
                or click to choose a PDF, JPG, PNG, or TIFF
              </p>
            </div>
          )}
        </div>

        <div className="settings-summary">
          Demo mode | Boar Room | Best handwriting recognition
        </div>

        <div className="upload-actions">
          <Button
            component={Link}
            to="/"
            variant="outlined"
            size="large"
            startIcon={<Home size={19} />}
            sx={{ minHeight: 48 }}
          >
            Back home
          </Button>
          <Button
            variant="contained"
            size="large"
            disabled={!record}
            onClick={startDemo}
            sx={{ minWidth: 160, minHeight: 48 }}
          >
            Read demo record
          </Button>
        </div>
      </main>
    </div>
  )
}
