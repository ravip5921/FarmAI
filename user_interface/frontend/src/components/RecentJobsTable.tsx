import {
  Alert,
  Button,
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
import { ArrowRight, Clock3, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { deleteJob } from '../api/jobs'
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
  const [selectedJob, setSelectedJob] = useState<JobSummary | null>(null)
  const remove = useMutation({
    mutationFn: (jobId: string) => deleteJob(jobId),
    onSuccess: async () => {
      setSelectedJob(null)
      await queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })

  if (!jobs.length) return null

  return (
    <>
      <section className="recent-jobs" aria-labelledby="recent-jobs-heading">
        <div className="recent-jobs__heading">
          <div>
            <h2 id="recent-jobs-heading">Recent jobs</h2>
            <p>Open a running job or return to a previous result.</p>
          </div>
          <Clock3 size={19} aria-hidden="true" />
        </div>
        <TableContainer
          component={Paper}
          variant="outlined"
          sx={{ borderRadius: 1, boxShadow: 'none' }}
        >
          <Table size="small" aria-label="Running and previous FarmAI jobs">
            <TableHead>
              <TableRow>
                <TableCell>Record</TableCell>
                <TableCell>Status</TableCell>
                <TableCell className="recent-jobs__date">Started</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {jobs.map((job) => (
                <TableRow hover key={job.job_id}>
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
                              setSelectedJob(job)
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
        open={selectedJob !== null}
        onClose={() => !remove.isPending && setSelectedJob(null)}
        aria-labelledby="delete-job-title"
      >
        <DialogTitle id="delete-job-title">Delete this job?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {selectedJob?.filename} and all of its saved results will be
            permanently removed.
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
            onClick={() => setSelectedJob(null)}
          >
            Keep job
          </Button>
          <Button
            color="error"
            variant="contained"
            disabled={!selectedJob || remove.isPending}
            onClick={() => selectedJob && remove.mutate(selectedJob.job_id)}
          >
            {remove.isPending ? 'Deleting...' : 'Delete job'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  )
}
