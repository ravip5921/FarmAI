import {
  Alert,
  Button,
  CircularProgress,
  IconButton,
  Tooltip,
} from '@mui/material'
import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import {
  FileText,
  PlayCircle,
  RefreshCw,
  Settings,
  Sparkles,
  Upload,
  X,
} from 'lucide-react'
import {
  useMemo,
  useRef,
  useState,
  type DragEvent,
} from 'react'
import { Link } from 'react-router-dom'
import {
  analyzePendingDocuments,
  getDocuments,
  uploadDocuments,
} from '../api/documents'
import { getJobs, getSettings } from '../api/jobs'
import { AppHeader } from '../components/AppHeader'
import { DocumentInboxTable } from '../components/DocumentInboxTable'
import { RecentJobsTable } from '../components/RecentJobsTable'
import { SettingsDrawer } from '../components/SettingsDrawer'
import type { DocumentCounts, JobSettings } from '../types/api'

const defaultSettings: JobSettings = {
  template_id: null,
  ocr_engine: 'llm-vision',
  extra_filtered_columns: [],
}

function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function plural(count: number, singular: string, pluralForm = `${singular}s`) {
  return count === 1 ? singular : pluralForm
}

function isPdf(file: File) {
  return (
    file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
  )
}

function CountCard({
  label,
  value,
  note,
}: {
  label: string
  value?: number
  note: string
}) {
  return (
    <div className="inbox-count">
      <span className="inbox-count__label">{label}</span>
      <span className="inbox-count__value">{value ?? '—'}</span>
      <span className="inbox-count__note">{note}</span>
    </div>
  )
}

function CountsBand({ counts }: { counts?: DocumentCounts }) {
  const processing = counts ? counts.queued + counts.running : undefined
  const completed = counts
    ? counts.completed + counts.completed_with_warnings
    : undefined

  return (
    <section className="inbox-counts" aria-label="PDF inbox summary">
      <CountCard
        label="Uploaded total"
        value={counts?.total}
        note="Stored PDFs"
      />
      <CountCard
        label="Awaiting analysis"
        value={counts?.pending}
        note="Ready when you are"
      />
      <CountCard
        label="Queued / running"
        value={processing}
        note={
          counts
            ? `${counts.queued} waiting · ${counts.running} active`
            : 'Analysis jobs'
        }
      />
      <CountCard
        label="Completed"
        value={completed}
        note={
          counts?.completed_with_warnings
            ? `${counts.completed_with_warnings} ${
                counts.completed_with_warnings === 1 ? 'needs' : 'need'
              } review`
            : 'Ready to review'
        }
      />
      <CountCard
        label="Failed"
        value={counts?.failed}
        note={
          counts?.cancelled ? `${counts.cancelled} cancelled` : 'Needs attention'
        }
      />
    </section>
  )
}

export function UploadPage() {
  const queryClient = useQueryClient()
  const fileInput = useRef<HTMLInputElement>(null)
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [selectionError, setSelectionError] = useState<string | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [settingsOverride, setSettingsOverride] =
    useState<JobSettings | null>(null)

  const options = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })
  const previousJobs = useQuery({
    queryKey: ['jobs'],
    queryFn: () => getJobs(100),
    refetchInterval: (query) =>
      query.state.data?.jobs.some(
        (job) =>
          job.document_id === null &&
          (job.status === 'queued' || job.status === 'running'),
      )
        ? 2500
        : 10000,
  })
  const inbox = useInfiniteQuery({
    queryKey: ['documents'],
    queryFn: ({ pageParam }) => getDocuments(pageParam),
    initialPageParam: 0,
    getNextPageParam: (lastPage) =>
      lastPage.has_more
        ? (lastPage.offset ?? 0) + (lastPage.limit ?? 50)
        : undefined,
    refetchInterval: (query) => {
      const currentCounts = query.state.data?.pages[0]?.counts
      return currentCounts &&
        (currentCounts.queued > 0 || currentCounts.running > 0)
        ? 2500
        : 10000
    },
  })
  const settings = settingsOverride ?? options.data?.defaults ?? defaultSettings
  const counts = inbox.data?.pages[0]?.counts
  const documents = useMemo(() => {
    const seen = new Set<string>()
    return (inbox.data?.pages ?? []).flatMap((page) =>
      page.documents.filter((document) => {
        if (seen.has(document.document_id)) return false
        seen.add(document.document_id)
        return true
      }),
    )
  }, [inbox.data?.pages])
  const legacyJobs = (previousJobs.data?.jobs ?? []).filter(
    (job) => job.document_id === null,
  )

  const upload = useMutation({
    mutationFn: uploadDocuments,
    onSuccess: async () => {
      setSelectedFiles([])
      if (fileInput.current) fileInput.current.value = ''
      await queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })
  const analyze = useMutation({
    mutationFn: () => analyzePendingDocuments(settings),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['documents'] }),
        queryClient.invalidateQueries({ queryKey: ['jobs'] }),
      ])
    },
  })

  const acceptFiles = (files?: FileList | File[]) => {
    if (!files || upload.isPending) return
    const candidates = Array.from(files)
    const rejected = candidates.filter((file) => !isPdf(file))
    const accepted = candidates.filter(isPdf)

    upload.reset()
    if (!analyze.isPending) analyze.reset()
    setSelectionError(
      rejected.length
        ? `${rejected.length} ${plural(rejected.length, 'file')} ${
            rejected.length === 1 ? 'was' : 'were'
          } skipped. Only PDF files can be uploaded.`
        : null,
    )
    setSelectedFiles((current) => {
      const existing = new Set(
        current.map((file) => `${file.name}:${file.size}:${file.lastModified}`),
      )
      return [
        ...current,
        ...accepted.filter(
          (file) =>
            !existing.has(`${file.name}:${file.size}:${file.lastModified}`),
        ),
      ]
    })
    if (fileInput.current) fileInput.current.value = ''
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    acceptFiles(event.dataTransfer.files)
  }

  const templateName = settings.template_id
    ? options.data?.templates.find((item) => item.id === settings.template_id)
        ?.name ?? settings.template_id
    : 'Detected table (no template)'
  const engineName =
    options.data?.ocr_engines.find(
      (item) => item.name === settings.ocr_engine,
    )?.label ?? 'Best handwriting recognition'

  return (
    <div className="app-shell">
      <AppHeader />
      <main className="page inbox-page">
        <div className="inbox-heading">
          <div className="page-heading">
            <h1>PDF inbox</h1>
            <p>
              Store scanned farm records first, then start analysis when the
              full upload is ready. Uploading never runs OCR or AI.
            </p>
          </div>
          <Button
            component={Link}
            to="/demo"
            variant="outlined"
            startIcon={<PlayCircle size={18} />}
          >
            Try demo
          </Button>
        </div>

        {inbox.error && (
          <Alert
            severity="error"
            sx={{ mb: 2 }}
            action={
              <Button
                color="inherit"
                size="small"
                startIcon={<RefreshCw size={15} />}
                onClick={() => inbox.refetch()}
              >
                Retry
              </Button>
            }
          >
            The PDF inbox could not be loaded. Your stored files have not been
            changed.
          </Alert>
        )}
        {options.error && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Analysis options could not be loaded. The standard settings are
            selected for now.
          </Alert>
        )}

        <CountsBand counts={counts} />

        <section className="inbox-panel" aria-labelledby="upload-pdfs-heading">
          <div className="inbox-panel__heading">
            <span className="step-number" aria-hidden="true">1</span>
            <div>
              <h2 id="upload-pdfs-heading">Upload and store PDFs</h2>
              <p>Select several scanned records at once or add more later.</p>
            </div>
          </div>

          {selectionError && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              {selectionError}
            </Alert>
          )}
          {upload.error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {upload.error.message}
            </Alert>
          )}
          {upload.data && (
            <Alert severity="success" sx={{ mb: 2 }}>
              {upload.data.documents.length}{' '}
              {plural(upload.data.documents.length, 'PDF')} safely uploaded.
              No analysis has run yet.
            </Alert>
          )}

          <div
            className={[
              'dropzone',
              'dropzone--compact',
              dragActive ? 'dropzone--active' : '',
            ].join(' ')}
            role="button"
            tabIndex={upload.isPending ? -1 : 0}
            aria-disabled={upload.isPending}
            onClick={() => !upload.isPending && fileInput.current?.click()}
            onKeyDown={(event) => {
              if (
                !upload.isPending &&
                (event.key === 'Enter' || event.key === ' ')
              ) {
                fileInput.current?.click()
              }
            }}
            onDragEnter={(event) => {
              event.preventDefault()
              if (!upload.isPending) setDragActive(true)
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
          >
            <input
              ref={fileInput}
              hidden
              multiple
              disabled={upload.isPending}
              type="file"
              accept=".pdf,application/pdf"
              onChange={(event) => acceptFiles(event.target.files ?? undefined)}
            />
            <div className="dropzone__content">
              <span className="dropzone__icon">
                <Upload size={24} aria-hidden="true" />
              </span>
              <p className="dropzone__title">Drop PDF records here</p>
              <p className="dropzone__hint">
                or click to choose one or more PDFs
              </p>
            </div>
          </div>

          {selectedFiles.length > 0 && (
            <div className="staged-files" aria-live="polite">
              <div className="staged-files__heading">
                <strong>
                  {selectedFiles.length}{' '}
                  {plural(selectedFiles.length, 'PDF')} ready to upload
                </strong>
                <Button
                  size="small"
                  disabled={upload.isPending}
                  onClick={() => setSelectedFiles([])}
                >
                  Clear all
                </Button>
              </div>
              <ul>
                {selectedFiles.map((file) => {
                  const key = `${file.name}:${file.size}:${file.lastModified}`
                  return (
                    <li key={key}>
                      <span className="selected-file__icon">
                        <FileText size={20} aria-hidden="true" />
                      </span>
                      <span className="staged-file__details">
                        <span className="selected-file__name">{file.name}</span>
                        <span className="selected-file__size">
                          {formatSize(file.size)}
                        </span>
                      </span>
                      <Tooltip title="Remove PDF">
                        <span>
                          <IconButton
                            aria-label={`Remove ${file.name}`}
                            disabled={upload.isPending}
                            onClick={() =>
                              setSelectedFiles((current) =>
                                current.filter(
                                  (candidate) =>
                                    `${candidate.name}:${candidate.size}:${candidate.lastModified}` !==
                                    key,
                                ),
                              )
                            }
                          >
                            <X size={18} />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </li>
                  )
                })}
              </ul>
            </div>
          )}

          <div className="inbox-panel__actions">
            <span className="storage-note">
              PDFs remain awaiting analysis after upload.
            </span>
            <Button
              variant="contained"
              size="large"
              startIcon={
                upload.isPending ? (
                  <CircularProgress size={17} color="inherit" />
                ) : (
                  <Upload size={18} />
                )
              }
              disabled={!selectedFiles.length || upload.isPending}
              onClick={() => upload.mutate(selectedFiles)}
            >
              {upload.isPending
                ? `Uploading ${selectedFiles.length}…`
                : selectedFiles.length
                  ? `Upload ${selectedFiles.length} ${plural(
                      selectedFiles.length,
                      'PDF',
                    )}`
                  : 'Upload PDFs'}
            </Button>
          </div>
        </section>

        <section
          className="inbox-panel analysis-control"
          aria-labelledby="analyze-heading"
        >
          <div className="inbox-panel__heading">
            <span className="step-number" aria-hidden="true">2</span>
            <div>
              <h2 id="analyze-heading">Analyze awaiting PDFs</h2>
              <p>
                This creates one independent job for every PDF currently
                awaiting analysis.
              </p>
            </div>
          </div>

          {analyze.error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {analyze.error.message}
            </Alert>
          )}
          {analyze.data && (
            <Alert
              severity={analyze.data.document_count ? 'success' : 'info'}
              sx={{ mb: 2 }}
            >
              {analyze.data.document_count
                ? `${analyze.data.document_count} ${plural(
                    analyze.data.document_count,
                    'PDF',
                  )} queued for analysis.`
                : 'There were no awaiting PDFs to analyze.'}
            </Alert>
          )}

          <div className="analysis-control__body">
            <div>
              <span className="analysis-control__label">
                Settings for this batch
              </span>
              <strong>{templateName}</strong>
              <span>
                {engineName}
                {settings.extra_filtered_columns.length
                  ? ` · ${settings.extra_filtered_columns.length} additional ${plural(
                      settings.extra_filtered_columns.length,
                      'column',
                    )} hidden`
                  : ''}
              </span>
            </div>
            <div className="analysis-control__actions">
              <Button
                variant="outlined"
                startIcon={<Settings size={18} />}
                onClick={() => setSettingsOpen(true)}
              >
                Settings
              </Button>
              <Button
                variant="contained"
                size="large"
                startIcon={
                  analyze.isPending ? (
                    <CircularProgress size={17} color="inherit" />
                  ) : (
                    <Sparkles size={18} />
                  )
                }
                disabled={
                  !counts?.pending ||
                  analyze.isPending ||
                  upload.isPending ||
                  inbox.isLoading
                }
                onClick={() => {
                  upload.reset()
                  analyze.mutate()
                }}
              >
                {analyze.isPending
                  ? 'Starting analysis…'
                  : `Analyze ${counts?.pending ?? 0} awaiting ${plural(
                      counts?.pending ?? 0,
                      'PDF',
                    )}`}
              </Button>
            </div>
          </div>
        </section>

        {inbox.isLoading && !inbox.data ? (
          <div className="inbox-loading" role="status">
            <CircularProgress size={24} />
            Loading PDF inbox…
          </div>
        ) : !inbox.error ? (
          <DocumentInboxTable
            documents={documents}
            hasMore={inbox.hasNextPage}
            isLoadingMore={inbox.isFetchingNextPage}
            onLoadMore={() => inbox.fetchNextPage()}
          />
        ) : null}

        {previousJobs.error && (
          <Alert
            severity="warning"
            sx={{ mt: 4 }}
            action={
              <Button
                color="inherit"
                size="small"
                startIcon={<RefreshCw size={15} />}
                onClick={() => previousJobs.refetch()}
              >
                Retry
              </Button>
            }
          >
            Earlier job history could not be loaded. Stored PDFs and current
            inbox jobs are unaffected.
          </Alert>
        )}
        <RecentJobsTable
          jobs={legacyJobs}
          title="Earlier jobs"
          description="Jobs created before the PDF inbox remain available for review."
        />
      </main>

      <SettingsDrawer
        open={settingsOpen}
        options={options.data}
        value={settings}
        showGroundTruth={false}
        onChange={setSettingsOverride}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  )
}
