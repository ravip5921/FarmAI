import {
  Alert,
  Button,
  IconButton,
  Tooltip,
} from '@mui/material'
import { useMutation, useQuery } from '@tanstack/react-query'
import { FileText, Settings, Upload, X } from 'lucide-react'
import {
  useRef,
  useState,
  type DragEvent,
} from 'react'
import { useNavigate } from 'react-router-dom'
import { createJob, getJobs, getSettings } from '../api/jobs'
import { AppHeader } from '../components/AppHeader'
import { RecentJobsTable } from '../components/RecentJobsTable'
import { SettingsDrawer } from '../components/SettingsDrawer'
import type { JobSettings } from '../types/api'

const defaultSettings: JobSettings = {
  template_id: 'boar_room',
  ocr_engine: 'llm-vision',
  extra_filtered_columns: [],
}

function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function UploadPage() {
  const navigate = useNavigate()
  const fileInput = useRef<HTMLInputElement>(null)
  const [record, setRecord] = useState<File | null>(null)
  const [groundTruth, setGroundTruth] = useState<File | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [settingsOverride, setSettingsOverride] =
    useState<JobSettings | null>(null)
  const options = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })
  const recentJobs = useQuery({
    queryKey: ['jobs'],
    queryFn: getJobs,
    refetchInterval: (query) =>
      query.state.data?.jobs.some(
        (job) => job.status === 'queued' || job.status === 'running',
      )
        ? 2500
        : 10000,
  })
  const settings = settingsOverride ?? options.data?.defaults ?? defaultSettings

  const create = useMutation({
    mutationFn: () => {
      if (!record) throw new Error('Choose a record first.')
      return createJob(record, settings, groundTruth)
    },
    onSuccess: (job) => navigate(`/jobs/${job.job_id}`),
  })

  const acceptFile = (file?: File) => {
    if (file) setRecord(file)
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    acceptFile(event.dataTransfer.files[0])
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
      <AppHeader
        action={
          <Tooltip title="Advanced settings">
            <IconButton
              aria-label="Advanced settings"
              onClick={() => setSettingsOpen(true)}
              sx={{ width: 44, height: 44 }}
            >
              <Settings size={21} />
            </IconButton>
          </Tooltip>
        }
      />
      <main className="page upload-page">
        <div className="page-heading">
          <h1>Read a farm record</h1>
          <p>
            Choose a scanned record or PDF. FarmAI will find the table and
            prepare it for review.
          </p>
        </div>

        {create.error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {create.error.message}
          </Alert>
        )}

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
          {templateName} | {engineName}
          {groundTruth ? ` | Accuracy CSV: ${groundTruth.name}` : ''}
        </div>

        <div className="upload-actions">
          <Button
            variant="contained"
            size="large"
            disabled={!record || create.isPending}
            onClick={() => create.mutate()}
            sx={{ minWidth: 160, minHeight: 48 }}
          >
            {create.isPending ? 'Starting...' : 'Read record'}
          </Button>
        </div>

        <RecentJobsTable jobs={recentJobs.data?.jobs ?? []} />
      </main>

      <SettingsDrawer
        open={settingsOpen}
        options={options.data}
        value={settings}
        groundTruth={groundTruth}
        onChange={setSettingsOverride}
        onGroundTruthChange={setGroundTruth}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  )
}
