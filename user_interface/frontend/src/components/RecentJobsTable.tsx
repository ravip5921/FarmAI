import {
  Alert,
  Button,
  Checkbox,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
} from '@mui/material'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, Clock3, Square, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { cancelJob, deleteJob } from '../api/jobs'
import type { JobStatus, JobSummary } from '../types/api'

const STATUS_LABELS: Record<JobStatus, string> = {
  queued: 'Waiting',
  running: 'Processing',
  completed: 'Complete',
  completed_with_warnings: 'Needs review',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

function statusColor(status: JobStatus) {
  if (status === 'completed') return 'success'
  if (status === 'completed_with_warnings') return 'warning'
  if (status === 'failed' || status === 'cancelled') return 'error'
  if (status === 'running') return 'primary'
  return 'default'
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

export function RecentJobsTable({ jobs }: { jobs: JobSummary[] }) {
  const queryClient = useQueryClient()
  const [checkedJobIds, setCheckedJobIds] = useState<Set<string>>(new Set())
  const [jobsToDelete, setJobsToDelete] = useState<JobSummary[]>([])
  const deletableJobs = useMemo(
    () => jobs.filter((job) => job.status !== 'running'),
    [jobs],
  )
  const selectedJobs = deletableJobs.filter((job) =>
    checkedJobIds.has(job.job_id),
  )
  const allSelected =
    deletableJobs.length > 0 &&
    deletableJobs.every((job) => checkedJobIds.has(job.job_id))
  const someSelected = selectedJobs.length > 0 && !allSelected

  const remove = useMutation({
    mutationFn: (targets: JobSummary[]) =>
      Promise.all(targets.map((job) => deleteJob(job.job_id))),
    onSuccess: async () => {
      setCheckedJobIds(new Set())
      setJobsToDelete([])
      await queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })

  const cancel = useMutation({
    mutationFn: (jobId: string) => cancelJob(jobId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })

  const toggleJob = (jobId: string) => {
    setCheckedJobIds((current) => {
      const next = new Set(current)
      if (next.has(jobId)) next.delete(jobId)
      else next.add(jobId)
      return next
    })
  }

  if (!jobs.length) return null

  return (
    <>
      <section className="recent-jobs" aria-labelledby="recent-jobs-heading">
        <div className="recent-jobs__heading">
          <div>
            <h2 id="recent-jobs-heading">Recent jobs</h2>
            <p>Open a running job or return to a previous result.</p>
          </div>
          <span className="recent-jobs__heading-actions">
            <Button
              color="error"
              variant="outlined"
              size="small"
              startIcon={<Trash2 size={17} />}
              disabled={!selectedJobs.length || remove.isPending}
              onClick={() => {
                remove.reset()
                setJobsToDelete(selectedJobs)
              }}
            >
              Delete selected
            </Button>
            <Clock3 size={19} aria-hidden="true" />
          </span>
        </div>
        {cancel.error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {cancel.error.message}
          </Alert>
        )}
        <TableContainer
          component={Paper}
          variant="outlined"
          sx={{ borderRadius: 1, boxShadow: 'none' }}
        >
          <Table size="small" aria-label="Running and previous FarmAI jobs">
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">
                  <Checkbox
                    size="small"
                    checked={allSelected}
                    indeterminate={someSelected}
                    disabled={!deletableJobs.length || remove.isPending}
                    slotProps={{
                      input: { 'aria-label': 'Select all deletable jobs' },
                    }}
                    onChange={() =>
                      setCheckedJobIds(
                        allSelected
                          ? new Set()
                          : new Set(deletableJobs.map((job) => job.job_id)),
                      )
                    }
                  />
                </TableCell>
                <TableCell>Record</TableCell>
                <TableCell>Status</TableCell>
                <TableCell className="recent-jobs__date">Started</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {jobs.map((job) => (
                <TableRow hover key={job.job_id}>
                  <TableCell padding="checkbox">
                    <Tooltip
                      title={
                        job.status === 'running'
                          ? 'Running jobs cannot be deleted'
                          : 'Select job'
                      }
                    >
                      <span>
                        <Checkbox
                          size="small"
                          checked={checkedJobIds.has(job.job_id)}
                          disabled={job.status === 'running' || remove.isPending}
                          slotProps={{
                            input: {
                              'aria-label': `Select ${job.filename}`,
                            },
                          }}
                          onChange={() => toggleJob(job.job_id)}
                        />
                      </span>
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <Link className="job-link" to={`/jobs/${job.job_id}`}>
                      {job.filename}
                    </Link>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={STATUS_LABELS[job.status]}
                      color={statusColor(job.status)}
                      size="small"
                      variant={
                        job.status === 'queued' ? 'outlined' : 'filled'
                      }
                    />
                  </TableCell>
                  <TableCell className="recent-jobs__date">
                    {formatDate(job.started_at ?? job.created_at)}
                  </TableCell>
                  <TableCell align="right">
                    <span className="job-actions">
                      <Tooltip title="Open job">
                        <Link
                          className="job-open-link"
                          to={`/jobs/${job.job_id}`}
                          aria-label={`Open ${job.filename}`}
                        >
                          <ArrowRight size={18} aria-hidden="true" />
                        </Link>
                      </Tooltip>
                      {(job.status === 'queued' || job.status === 'running') && (
                        <Tooltip title="Cancel job">
                          <span>
                            <IconButton
                              aria-label={`Cancel ${job.filename}`}
                              color="warning"
                              disabled={cancel.isPending}
                              onClick={() => cancel.mutate(job.job_id)}
                              size="small"
                              sx={{ width: 36, height: 36 }}
                            >
                              <Square size={16} />
                            </IconButton>
                          </span>
                        </Tooltip>
                      )}
                      <Tooltip
                        title={
                          job.status === 'running'
                            ? 'Wait until processing finishes'
                            : 'Delete job'
                        }
                      >
                        <span>
                          <IconButton
                            aria-label={`Delete ${job.filename}`}
                            color="error"
                            disabled={job.status === 'running'}
                            onClick={() => {
                              remove.reset()
                              setJobsToDelete([job])
                            }}
                            size="small"
                            sx={{ width: 36, height: 36 }}
                          >
                            <Trash2 size={17} />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </span>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </section>

      <Dialog
        open={jobsToDelete.length > 0}
        onClose={() => !remove.isPending && setJobsToDelete([])}
        aria-labelledby="delete-job-title"
      >
        <DialogTitle id="delete-job-title">
          {jobsToDelete.length === 1
            ? 'Delete this job?'
            : `Delete ${jobsToDelete.length} jobs?`}
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            {jobsToDelete.length === 1
              ? `${jobsToDelete[0]?.filename} and all of its saved results will be permanently removed.`
              : `The selected ${jobsToDelete.length} jobs and all of their saved results will be permanently removed.`}
          </DialogContentText>
          {remove.error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {remove.error.message}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            disabled={remove.isPending}
            onClick={() => setJobsToDelete([])}
          >
            Keep job
          </Button>
          <Button
            color="error"
            variant="contained"
            disabled={!jobsToDelete.length || remove.isPending}
            onClick={() => remove.mutate(jobsToDelete)}
          >
            {remove.isPending
              ? 'Deleting...'
              : jobsToDelete.length === 1
                ? 'Delete job'
                : 'Delete jobs'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  )
}
