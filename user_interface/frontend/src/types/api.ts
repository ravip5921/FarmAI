export type JobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'completed_with_warnings'
  | 'failed'
  | 'cancelled'

export interface TemplateColumnOption {
  key: string
  name: string
  filter_out: boolean
}

export interface TemplateOption {
  id: string
  name: string
  description: string
  columns: TemplateColumnOption[]
}

export interface OcrEngineOption {
  name: string
  label: string
  description: string
}

export interface AppSettingsResponse {
  defaults: JobSettings
  templates: TemplateOption[]
  ocr_engines: OcrEngineOption[]
}

export interface JobSettings {
  template_id: string | null
  ocr_engine: string
  extra_filtered_columns: string[]
}

export interface CreatedJob {
  job_id: string
  status: JobStatus
  status_url: string
}

export interface JobSummary {
  job_id: string
  status: JobStatus
  stage: string
  progress_current: number
  progress_total: number
  filename: string
  template_id: string | null
  ocr_engine: string
  created_at: string
  started_at: string | null
  completed_at: string | null
  updated_at: string
  error_code: string | null
  error: string | null
  result_url: string | null
}

export interface JobsResponse {
  jobs: JobSummary[]
}

export interface ResultColumn {
  index: number
  source_index: number
  key: string
  name: string
  value_type: string
  format: string | null
  range_min: number | null
  range_max: number | null
}

export type CellState =
  | 'ok'
  | 'validation_warning'
  | 'ground_truth_mismatch'
  | 'mismatch_and_warning'
  | 'correct'
  | 'unscored'

export interface ResultCell {
  row: number
  column_index: number
  source_column_index: number
  column_key: string
  column_name: string
  bbox: [number, number, number, number]
  ocr_text: string
  reviewed_text: string
  was_edited: boolean
  confidence: number | null
  raw_text: string | null
  validation_error: string | null
  ground_truth_text: string | null
  ground_truth_match: boolean | null
  state: CellState
}

export interface AccuracyMetrics {
  correct_cells: number
  incorrect_cells: number
  scored_cells: number
  exact_accuracy: number | null
  normalized_accuracy?: number | null
  correct_rows?: number
  scored_rows?: number
  validation_warning_count?: number
}

export interface ResultPage {
  page_number: number
  source_url: string
  overlay_url: string
  image_width: number
  image_height: number
  skew_angle: number
  columns: ResultColumn[]
  cells: ResultCell[]
  data_row_count: number
  warning_count: number
  metrics: AccuracyMetrics | null
}

export interface JobResult {
  job_id: string
  filename: string
  template_id: string | null
  template_name: string | null
  ocr_engine: string
  warning_count: number
  ground_truth_error?: string | null
  metrics: AccuracyMetrics | null
  pages: ResultPage[]
}
