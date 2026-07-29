import {
  Alert,
  Button,
  Checkbox,
  FormControlLabel,
  IconButton,
  MenuItem,
  Select,
  Tooltip,
} from '@mui/material'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Download,
  FileCheck2,
  RefreshCw,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  attachGroundTruth,
  editCell,
  getJob,
  getResult,
} from '../api/jobs'
import { AppHeader } from '../components/AppHeader'
import { JobProgress } from '../components/JobProgress'
import { OcrResultGrid } from '../components/OcrResultGrid'
import { OverlayViewer } from '../components/OverlayViewer'
import type { JobResult, ResultCell } from '../types/api'

const terminalStatuses = new Set([
  'completed',
  'completed_with_warnings',
  'failed',
  'cancelled',
])

function percent(value: number | null | undefined) {
  return value == null ? 'Not scored' : `${(value * 100).toFixed(1)}%`
}

export function JobPage() {
  const { jobId = '' } = useParams()
  const queryClient = useQueryClient()
  const [pageNumber, setPageNumber] = useState(1)
  const [selectedCell, setSelectedCell] = useState<ResultCell | null>(null)
  const [needsReviewOnly, setNeedsReviewOnly] = useState(false)

  const job = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId),
    refetchInterval: (query) =>
      terminalStatuses.has(query.state.data?.status ?? '') ? false : 2500,
  })
  const complete =
    job.data?.status === 'completed' ||
    job.data?.status === 'completed_with_warnings'
  const result = useQuery({
    queryKey: ['job-result', jobId],
    queryFn: () => getResult(jobId),
    enabled: complete,
  })

  const edit = useMutation({
    mutationFn: ({
      cell,
      value,
    }: {
      cell: ResultCell
      value: string
    }) =>
      editCell(jobId, {
        page_number: pageNumber,
        row: cell.row,
        column_key: cell.column_key,
        reviewed_text: value,
      }),
    onSuccess: (updated) => {
      queryClient.setQueryData(['job-result', jobId], updated)
    },
  })

  const groundTruth = useMutation({
    mutationFn: (file: File) => attachGroundTruth(jobId, file),
    onSuccess: (updated) => {
      queryClient.setQueryData(['job-result', jobId], updated)
    },
  })

  const page = result.data?.pages.find(
    (item) => item.page_number === pageNumber,
  )
  const reviewCount = useMemo(
    () =>
      page?.cells.filter(
        (cell) =>
          Boolean(cell.validation_error) || cell.ground_truth_match === false,
      ).length ?? 0,
    [page],
  )

  if (job.isLoading) {
    return (
      <div className="app-shell">
        <AppHeader />
        <main className="page">
          <div className="empty-message">Loading job...</div>
        </main>
      </div>
    )
  }

  if (job.error || !job.data) {
    return (
      <div className="app-shell">
        <AppHeader />
        <main className="page">
          <div className="error-panel">
            <h2>Job unavailable</h2>
            <p>{job.error?.message ?? 'This job could not be found.'}</p>
            <Button component={Link} to="/" variant="contained">
              Upload another record
            </Button>
          </div>
        </main>
      </div>
    )
  }

  if (job.data.status === 'failed') {
    return (
      <div className="app-shell">
        <AppHeader />
        <main className="page">
          <div className="error-panel">
            <h2>Record could not be read</h2>
            <p>{job.data.error ?? 'This record could not be processed.'}</p>
            <Button component={Link} to="/" variant="contained">
              Try another record
            </Button>
          </div>
        </main>
      </div>
    )
  }

  if (!complete) {
    return (
      <div className="app-shell">
        <AppHeader />
        <main className="page">
          <JobProgress job={job.data} />
        </main>
      </div>
    )
  }

  if (result.isLoading || !result.data || !page) {
    return (
      <div className="app-shell">
        <AppHeader />
        <main className="page">
          <div className="empty-message">Preparing the review...</div>
        </main>
      </div>
    )
  }

  const data: JobResult = result.data
  return (
    <div className="app-shell">
      <AppHeader />
      <main className="page review-page">
        <div className="review-topbar">
          <div className="review-title">
            <h1>{data.filename}</h1>
            <p>
              {data.template_name ?? 'Detected table'} |{' '}
              {data.ocr_engine === 'llm-vision'
                ? 'Best handwriting recognition'
                : data.ocr_engine}
            </p>
          </div>
          <div className="review-actions">
            {data.pages.length > 1 && (
              <Select
                size="small"
                value={pageNumber}
                onChange={(event) => {
                  setPageNumber(Number(event.target.value))
                  setSelectedCell(null)
                }}
                aria-label="Page"
              >
                {data.pages.map((item) => (
                  <MenuItem key={item.page_number} value={item.page_number}>
                    Page {item.page_number}
                  </MenuItem>
                ))}
              </Select>
            )}
            <Button
              component="label"
              variant="outlined"
              startIcon={<FileCheck2 size={17} />}
            >
              {data.metrics ? 'Replace accuracy CSV' : 'Add accuracy CSV'}
              <input
                hidden
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => {
                  const file = event.target.files?.[0]
                  if (file) groundTruth.mutate(file)
                  event.target.value = ''
                }}
              />
            </Button>
            <Button
              component="a"
              href={`/api/jobs/${jobId}/download.csv`}
              variant="contained"
              startIcon={<Download size={17} />}
            >
              Download CSV
            </Button>
            <Tooltip title="Refresh results">
              <IconButton
                aria-label="Refresh results"
                onClick={() =>
                  queryClient.invalidateQueries({
                    queryKey: ['job-result', jobId],
                  })
                }
              >
                <RefreshCw size={19} />
              </IconButton>
            </Tooltip>
          </div>
        </div>

        {groundTruth.error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {groundTruth.error.message}
          </Alert>
        )}
        {data.ground_truth_error && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            The record was read, but the accuracy CSV could not be used:{' '}
            {data.ground_truth_error}
          </Alert>
        )}
        {edit.error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            Your last edit could not be saved. {edit.error.message}
          </Alert>
        )}

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
              {percent(data.metrics?.exact_accuracy)}
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
              onEdit={(cell, value) => edit.mutate({ cell, value })}
            />
          </section>
        </div>
      </main>
    </div>
  )
}
