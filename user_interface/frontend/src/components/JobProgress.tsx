import { LinearProgress } from '@mui/material'
import { useEffect, useState } from 'react'
import type { JobSummary } from '../types/api'

const STAGE_LABELS: Record<string, string> = {
  queued: 'Waiting to start',
  preparing: 'Preparing image',
  detecting_table: 'Finding the table',
  recognizing_cells: 'Reading cells',
  preparing_review: 'Preparing review',
  completed: 'Complete',
}

function elapsedSince(value: string | null) {
  if (!value) return 'Not started'
  const seconds = Math.max(
    0,
    Math.floor((Date.now() - new Date(value).getTime()) / 1000),
  )
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return minutes ? `${minutes}m ${remainder}s elapsed` : `${remainder}s elapsed`
}

export function JobProgress({ job }: { job: JobSummary }) {
  const [, tick] = useState(0)
  useEffect(() => {
    const id = window.setInterval(() => tick((value) => value + 1), 1000)
    return () => window.clearInterval(id)
  }, [])

  const determinate = job.progress_total > 0
  const progress = determinate
    ? Math.min(100, (job.progress_current / job.progress_total) * 100)
    : undefined
  const stage = STAGE_LABELS[job.stage] ?? 'Processing record'

  return (
    <div className="progress-layout">
      <div className="page-heading">
        <h1>Reading your record</h1>
        <p>FarmAI is finding the table and reading each handwritten cell.</p>
      </div>
      <section className="progress-panel" aria-live="polite">
        <p className="progress-file">{job.filename}</p>
        <p className="progress-stage">
          {stage}
          {job.stage === 'recognizing_cells' && determinate
            ? ` (${job.progress_current} of ${job.progress_total})`
            : ''}
        </p>
        <LinearProgress
          variant={determinate ? 'determinate' : 'indeterminate'}
          value={progress}
          aria-label={stage}
          sx={{ height: 9, borderRadius: 1 }}
        />
        <div className="progress-meta">
          <span>
            {determinate
              ? `${Math.round(progress ?? 0)}% of current step`
              : 'Starting'}
          </span>
          <span>{elapsedSince(job.started_at ?? job.created_at)}</span>
        </div>
        <div className="return-note">
          You can close this page. Processing will continue, and this link will
          return to the job.
        </div>
      </section>
    </div>
  )
}
