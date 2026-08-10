import type {
  AccuracyMetrics,
  CellState,
  JobResult,
  ResultCell,
  ResultColumn,
} from '../types/api'

const columns: ResultColumn[] = [
  {
    index: 0,
    source_index: 0,
    key: 'date',
    name: 'Date',
    value_type: 'date_dd_mon',
    format: 'DD-Mon',
    range_min: null,
    range_max: null,
  },
  {
    index: 1,
    source_index: 1,
    key: 'current_temperature',
    name: 'Current Temperature',
    value_type: 'temperature',
    format: 'TT.T',
    range_min: 50,
    range_max: 110,
  },
  {
    index: 2,
    source_index: 2,
    key: 'hi',
    name: 'HI',
    value_type: 'temperature',
    format: 'TT.T',
    range_min: 50,
    range_max: 110,
  },
  {
    index: 3,
    source_index: 3,
    key: 'lo',
    name: 'LO',
    value_type: 'temperature',
    format: 'TT.T',
    range_min: 50,
    range_max: 110,
  },
  {
    index: 4,
    source_index: 6,
    key: 'comments',
    name: 'Comments',
    value_type: 'english_text',
    format: null,
    range_min: null,
    range_max: null,
  },
]

const rows = [
  ['01-May', '67.8', '95', '61.5', 'All good'],
  ['02-May', '68.4', '96', '62', 'All good'],
  ['03-May', '69.1', '190', '63.2', 'Fan checked'],
  ['04-May', '70', '97.5', '64', 'Water line fixed'],
  ['05-May', '71.2', '98', '65', 'All good'],
  ['06-May', '72.0', '99', '66.1', 'OK'],
  ['07-May', '72.5', '99.4', '66.8', 'Feed adjusted'],
  ['08-May', '73.1', '100', '67.5', 'All good'],
]

const truthRows = [
  ['01-May', '67.8', '95', '61.5', 'All good'],
  ['02-May', '68.4', '96', '62', 'All good'],
  ['03-May', '69.1', '100', '63.2', 'Fan checked'],
  ['04-May', '70', '97.5', '64', 'Water line fixed'],
  ['05-May', '71.2', '98', '65', 'All good'],
  ['06-May', '72.0', '99', '66.1', 'OK'],
  ['07-May', '72.5', '99.4', '66.8', 'Feed adjusted'],
  ['08-May', '73.1', '100', '67.5', 'All good'],
]

const bboxByColumn = [
  [87, 162, 24, 19],
  [115, 162, 24, 19],
  [143, 162, 26, 19],
  [173, 162, 26, 19],
  [240, 162, 368, 19],
] as const

function stateFor(
  value: string,
  truth: string,
  validationError: string | null,
): CellState {
  if (validationError && value !== truth) return 'mismatch_and_warning'
  if (validationError) return 'validation_warning'
  if (value !== truth) return 'ground_truth_mismatch'
  return 'correct'
}

function makeCells(): ResultCell[] {
  return rows.flatMap((row, rowIndex) =>
    columns.map((column, columnIndex) => {
      const value = row[columnIndex]
      const truth = truthRows[rowIndex][columnIndex]
      const validationError =
        column.key === 'hi' && value === '190'
          ? 'Temperature must be between 50 and 110.'
          : null
      const [x, baseY, width, height] = bboxByColumn[columnIndex]
      return {
        row: rowIndex + 1,
        column_index: column.index,
        source_column_index: column.source_index,
        column_key: column.key,
        column_name: column.name,
        bbox: [x, baseY + rowIndex * 20, width, height],
        ocr_text: value,
        reviewed_text: value,
        was_edited: false,
        confidence: column.key === 'comments' ? 0.88 : 0.94,
        raw_text: validationError ? '19O' : null,
        validation_error: validationError,
        ground_truth_text: truth,
        ground_truth_match: value === truth,
        state: stateFor(value, truth, validationError),
      }
    }),
  )
}

export const demoMetrics: AccuracyMetrics = {
  correct_cells: 39,
  incorrect_cells: 1,
  scored_cells: 40,
  exact_accuracy: 0.975,
  normalized_accuracy: 0.975,
  correct_rows: 7,
  scored_rows: 8,
  validation_warning_count: 1,
}

export function createDemoResult(filename: string): JobResult {
  return {
    job_id: 'demo-boar-room',
    filename,
    template_id: 'boar_room',
    template_name: 'Boar Room',
    ocr_engine: 'llm-vision',
    warning_count: 1,
    metrics: demoMetrics,
    pages: [
      {
        page_number: 1,
        source_url: '/demo/boar-room-source.png',
        overlay_url: '/demo/boar-room-overlay.png',
        image_width: 820,
        image_height: 420,
        skew_angle: 2.09,
        columns,
        cells: makeCells(),
        data_row_count: rows.length,
        warning_count: 1,
        metrics: demoMetrics,
      },
    ],
  }
}
