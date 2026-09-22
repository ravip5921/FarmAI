import {
  Button,
  Chip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material'
import { ArrowRight, Files } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { DocumentStatus, DocumentSummary } from '../types/api'

const STATUS_LABELS: Record<DocumentStatus, string> = {
  pending: 'Awaiting analysis',
  queued: 'Waiting',
  running: 'Processing',
  completed: 'Complete',
  completed_with_warnings: 'Needs review',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

function statusColor(status: DocumentStatus) {
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

function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

interface DocumentInboxTableProps {
  documents: DocumentSummary[]
  hasMore?: boolean
  isLoadingMore?: boolean
  onLoadMore?: () => void
}

export function DocumentInboxTable({
  documents,
  hasMore = false,
  isLoadingMore = false,
  onLoadMore,
}: DocumentInboxTableProps) {
  return (
    <section className="document-list" aria-labelledby="document-list-heading">
      <div className="document-list__heading">
        <div>
          <h2 id="document-list-heading">Uploaded PDFs</h2>
          <p>
            Each PDF stays available here while it waits, processes, or is
            reviewed.
          </p>
        </div>
        <Files size={21} aria-hidden="true" />
      </div>

      {documents.length ? (
        <TableContainer
          component={Paper}
          variant="outlined"
          sx={{ borderRadius: 1, boxShadow: 'none' }}
        >
          <Table size="small" aria-label="Uploaded PDF inbox">
            <TableHead>
              <TableRow>
                <TableCell>PDF</TableCell>
                <TableCell>Status</TableCell>
                <TableCell className="document-list__size">Size</TableCell>
                <TableCell className="document-list__date">Uploaded</TableCell>
                <TableCell align="right">Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {documents.map((document) => (
                <TableRow hover key={document.document_id}>
                  <TableCell>
                    {document.latest_job_id ? (
                      <Link
                        className="job-link"
                        to={`/jobs/${document.latest_job_id}`}
                      >
                        {document.filename}
                      </Link>
                    ) : (
                      <span className="document-name">{document.filename}</span>
                    )}
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={STATUS_LABELS[document.status]}
                      color={statusColor(document.status)}
                      size="small"
                      variant={document.status === 'pending' ? 'outlined' : 'filled'}
                    />
                  </TableCell>
                  <TableCell className="document-list__size">
                    {formatSize(document.size_bytes)}
                  </TableCell>
                  <TableCell className="document-list__date">
                    {formatDate(document.created_at)}
                  </TableCell>
                  <TableCell align="right">
                    {document.latest_job_id ? (
                      <Button
                        component={Link}
                        to={`/jobs/${document.latest_job_id}`}
                        size="small"
                        endIcon={<ArrowRight size={16} />}
                      >
                        {document.status === 'completed' ||
                        document.status === 'completed_with_warnings'
                          ? 'Review'
                          : 'Open job'}
                      </Button>
                    ) : (
                      <span className="document-list__waiting">Stored</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      ) : (
        <div className="document-list__empty">
          <FileEmptyIcon />
          <div>
            <strong>No PDFs uploaded yet</strong>
            <span>Uploaded records will appear here.</span>
          </div>
        </div>
      )}

      {hasMore && onLoadMore && (
        <div className="document-list__more">
          <Button
            variant="outlined"
            disabled={isLoadingMore}
            onClick={onLoadMore}
          >
            {isLoadingMore ? 'Loading…' : 'Load older PDFs'}
          </Button>
        </div>
      )}
    </section>
  )
}

function FileEmptyIcon() {
  return (
    <span className="document-list__empty-icon" aria-hidden="true">
      <Files size={22} />
    </span>
  )
}
