import type {
  AccuracyMetrics,
  CellState,
  JobResult,
  ResultCell,
  ResultColumn,
} from '../types/api'

interface DebugOcrCell {
  row: number
  col: number
  bbox: [number, number, number, number]
  text: string
  confidence: number | null
  raw_text?: string
  validation_error?: string
}

interface DebugOcrTable {
  row_count: number
  col_count: number
  cells: DebugOcrCell[]
}

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

function parseCsv(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let value = ''
  let quoted = false

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]
    const next = text[index + 1]
    if (quoted) {
      if (char === '"' && next === '"') {
        value += '"'
        index += 1
      } else if (char === '"') {
        quoted = false
      } else {
        value += char
      }
    } else if (char === '"') {
      quoted = true
    } else if (char === ',') {
      row.push(value)
      value = ''
    } else if (char === '\n') {
      row.push(value)
      rows.push(row)
      row = []
      value = ''
    } else if (char !== '\r') {
      value += char
    }
  }

  if (value || row.length) {
    row.push(value)
    rows.push(row)
  }
  return rows
}

function exact(value: string) {
  return value.trim().replace(/\r\n/g, '\n').replace(/\r/g, '\n')
}

function normalized(value: string) {
  const trimmed = exact(value)
  const number = Number(trimmed)
  if (trimmed !== '' && Number.isFinite(number)) {
    return String(number)
  }
  return trimmed.replace(/\s+/g, ' ').toLocaleLowerCase()
}

function cellState(
  ocrText: string,
  truthText: string | null,
  validationError: string | null,
): CellState {
  const mismatch = truthText != null && exact(ocrText) !== exact(truthText)
  if (mismatch && validationError) return 'mismatch_and_warning'
  if (mismatch) return 'ground_truth_mismatch'
  if (validationError) return 'validation_warning'
  if (truthText != null) return 'correct'
  return 'unscored'
}

function score(cells: ResultCell[], rowCount: number): AccuracyMetrics {
  const scored = cells.filter((cell) => cell.ground_truth_text != null)
  const correct = scored.filter(
    (cell) => exact(cell.ocr_text) === exact(cell.ground_truth_text ?? ''),
  )
  const normalizedCorrect = scored.filter(
    (cell) =>
      normalized(cell.ocr_text) === normalized(cell.ground_truth_text ?? ''),
  )
  let correctRows = 0
  for (let row = 1; row <= rowCount; row += 1) {
    const rowCells = scored.filter((cell) => cell.row === row)
    if (
      rowCells.length > 0 &&
      rowCells.every(
        (cell) => exact(cell.ocr_text) === exact(cell.ground_truth_text ?? ''),
      )
    ) {
      correctRows += 1
    }
  }
  return {
    correct_cells: correct.length,
    incorrect_cells: scored.length - correct.length,
    scored_cells: scored.length,
    exact_accuracy: scored.length ? correct.length / scored.length : null,
    normalized_accuracy: scored.length
      ? normalizedCorrect.length / scored.length
      : null,
    correct_rows: correctRows,
    scored_rows: rowCount,
    validation_warning_count: cells.filter((cell) =>
      Boolean(cell.validation_error),
    ).length,
  }
}

export async function createDemoResult(filename: string): Promise<JobResult> {
  const [ocrResponse, truthResponse] = await Promise.all([
    fetch('/demo/boar-room-ocr.json'),
    fetch('/demo/boar-room-ground-truth.csv'),
  ])
  if (!ocrResponse.ok || !truthResponse.ok) {
    throw new Error('Demo files could not be loaded.')
  }

  const ocr = (await ocrResponse.json()) as DebugOcrTable
  const truthRows = parseCsv(await truthResponse.text()).slice(1)
  const dataRowCount = ocr.row_count - 1
  const cells: ResultCell[] = ocr.cells
    .filter((cell) => cell.row > 0)
    .map((cell) => {
      const column = columns[cell.col]
      const truthText = truthRows[cell.row - 1]?.[cell.col] ?? null
      const validationError = cell.validation_error ?? null
      return {
        row: cell.row,
        column_index: cell.col,
        source_column_index: column.source_index,
        column_key: column.key,
        column_name: column.name,
        bbox: cell.bbox,
        ocr_text: cell.text,
        reviewed_text: cell.text,
        was_edited: false,
        confidence: cell.confidence,
        raw_text: cell.raw_text ?? null,
        validation_error: validationError,
        ground_truth_text: truthText,
        ground_truth_match:
          truthText == null ? null : exact(cell.text) === exact(truthText),
        state: cellState(cell.text, truthText, validationError),
      }
    })

  const metrics = score(cells, dataRowCount)
  return {
    job_id: 'demo-boar-room-p1-good',
    filename,
    template_id: 'boar_room',
    template_name: 'Boar Room',
    ocr_engine: 'llm-vision',
    warning_count: metrics.validation_warning_count ?? 0,
    metrics,
    pages: [
      {
        page_number: 1,
        source_url: '/demo/boar-room-source.png',
        overlay_url: '/demo/boar-room-overlay.png',
        image_width: 1190,
        image_height: 1684,
        skew_angle: 0,
        columns,
        cells,
        data_row_count: dataRowCount,
        warning_count: metrics.validation_warning_count ?? 0,
        metrics,
      },
    ],
  }
}
