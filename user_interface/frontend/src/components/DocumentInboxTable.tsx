import { EditableMetadataCell } from './EditableMetadataCell'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiRequest } from '../api/client'
import { useState } from 'react'
import { RecordEditor } from './RecordEditor'
import {
  Button,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Checkbox,
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
import type { DocumentStatus, DocumentSummary, JobSettings } from '../types/api'

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
  defaultSettings?: JobSettings
  editable?: boolean
  documents: DocumentSummary[]
  hasMore?: boolean
  isLoadingMore?: boolean
  onLoadMore?: () => void
}

export function DocumentInboxTable({
  documents,
  editable = true,
  defaultSettings,
  hasMore = false,
  isLoadingMore = false,
  onLoadMore,
}: DocumentInboxTableProps) {
  const [selected, setSelected] = useState<string[]>([])
  const [editing, setEditing] = useState<DocumentSummary[]>([])
  const client = useQueryClient()
  const [deleting, setDeleting] = useState<DocumentSummary[]>([])
  const [cleanupWarning, setCleanupWarning] = useState<string | null>(null)
  const deletableDocuments = documents.filter(item => item.status !== 'running')
  const selectedDocuments = deletableDocuments.filter(item => selected.includes(item.document_id))
  const remove = useMutation({
    mutationFn: () => apiRequest<{ deleted: number; cleanup_warning: string | null }>('/api/documents/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document_ids: deleting.map(item => item.document_id) }),
    }),
    onSuccess: async result => {
      setDeleting([])
      setSelected([])
      setCleanupWarning(result.cleanup_warning)
      await Promise.all(['documents', 'jobs', 'job'].map(key => client.invalidateQueries({ queryKey: [key] })))
    },
  })
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

      <Button color="error" disabled={!editable || remove.isPending || !selectedDocuments.length} onClick={() => { remove.reset(); setDeleting(selectedDocuments) }}>Delete selected ({selectedDocuments.length})</Button>
      {cleanupWarning && <Alert severity="warning">{cleanupWarning}</Alert>}
      <Dialog open={deleting.length > 0} onClose={() => !remove.isPending && setDeleting([])} fullWidth maxWidth="sm">
        <DialogTitle>Delete {deleting.length} selected PDFs?</DialogTitle>
        <DialogContent>
          <p>This permanently removes these PDFs, their queued jobs, and saved results. Processing PDFs cannot be selected.</p>
          <ul>{deleting.map(item => <li key={item.document_id}>{item.filename}{item.reference_id ? ` (${item.reference_id})` : ''}</li>)}</ul>
          {remove.error && <Alert severity="error">{remove.error.message}</Alert>}
        </DialogContent>
        <DialogActions>
          <Button disabled={remove.isPending} onClick={() => setDeleting([])}>Cancel</Button>
          <Button color="error" variant="contained" disabled={remove.isPending} onClick={() => remove.mutate()}>{remove.isPending ? 'Deleting...' : 'Delete selected'}</Button>
        </DialogActions>
      </Dialog>
      {editing.length > 0 && <RecordEditor
        ids={editing.map(item => item.document_id)} documents
        title={editing.length === 1 ? `Edit ${editing[0].filename}` : `Edit ${editing.length} PDFs`}
        initialSettings={editing[0].settings ?? defaultSettings} referenceId={editing[0].reference_id} comments={editing[0].comments}
        canEditSettings={editing.every(item => item.status === 'pending' || item.status === 'queued')}
        onClose={() => setEditing([])}
      />}
      {documents.length ? (
        <TableContainer
          component={Paper}
          variant="outlined"
          sx={{ borderRadius: 1, boxShadow: 'none' }}
        >
          <Table size="small" aria-label="Uploaded PDF inbox">
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox"><Checkbox disabled={!editable || remove.isPending || !deletableDocuments.length} aria-label="Select all deletable loaded PDFs" checked={deletableDocuments.length > 0 && selectedDocuments.length === deletableDocuments.length} indeterminate={selectedDocuments.length > 0 && selectedDocuments.length < deletableDocuments.length} onChange={(_, checked) => setSelected(checked ? deletableDocuments.map(item => item.document_id) : [])} /></TableCell>
                <TableCell>PDF</TableCell>
                <TableCell>ID</TableCell>
                <TableCell>Comments / label</TableCell>
                <TableCell>Settings</TableCell>
                <TableCell>Status</TableCell>
                <TableCell className="document-list__size">Size</TableCell>
                <TableCell className="document-list__date">Uploaded</TableCell>
                <TableCell align="right">Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {documents.map((document) => (
                <TableRow hover key={document.document_id}>
                  <TableCell padding="checkbox"><Checkbox disabled={!editable || remove.isPending || document.status === 'running'} aria-label={`Select ${document.filename}`} checked={document.status !== 'running' && selected.includes(document.document_id)} onChange={(_, checked) => setSelected(current => checked ? [...current, document.document_id] : current.filter(id => id !== document.document_id))} /></TableCell>
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
                  <EditableMetadataCell id={document.document_id} documents disabled={!editable || remove.isPending} field="reference_id" value={document.reference_id} fallback={document.document_id.slice(0, 8)} />
                  <EditableMetadataCell id={document.document_id} documents disabled={!editable || remove.isPending} field="comments" value={document.comments} />
                  <TableCell tabIndex={editable ? 0 : undefined} title="Double-click or press Enter to edit settings"
                    onDoubleClick={() => editable && setEditing([document])}
                    onKeyDown={event => { if (editable && (event.key === 'Enter' || event.key === 'F2')) { event.preventDefault(); setEditing([document]) } }}
                    sx={{ cursor: editable ? 'pointer' : undefined }}
                  >{document.settings ? `${document.settings.template_id ?? 'No template'} ? ${document.settings.ocr_engine}` : 'Global defaults'}</TableCell>
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
                    <Button disabled={!editable} size="small" onClick={() => setEditing([document])}>Edit</Button>
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
