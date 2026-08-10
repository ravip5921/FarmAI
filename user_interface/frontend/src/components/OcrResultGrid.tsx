import { AlertTriangle } from 'lucide-react'
import {
  AllCommunityModule,
  ModuleRegistry,
  themeQuartz,
  type CellClickedEvent,
  type CellValueChangedEvent,
  type ColDef,
  type ICellRendererParams,
} from 'ag-grid-community'
import { AgGridReact } from 'ag-grid-react'
import { useMemo } from 'react'
import type { ResultCell, ResultColumn, ResultPage } from '../types/api'

ModuleRegistry.registerModules([AllCommunityModule])

interface ReviewRow {
  _row: number
  _cells: Record<string, ResultCell>
  [key: string]: unknown
}

interface OcrResultGridProps {
  page: ResultPage
  needsReviewOnly: boolean
  selectedCell: ResultCell | null
  onSelectCell: (cell: ResultCell) => void
  onEdit: (cell: ResultCell, value: string) => void
}

const gridTheme = themeQuartz.withParams({
  accentColor: '#216449',
  borderColor: '#dbe1dc',
  browserColorScheme: 'light',
  cellHorizontalPaddingScale: 0.85,
  fontFamily: 'Inter, "Segoe UI", sans-serif',
  fontSize: 14,
  headerBackgroundColor: '#f0f4f1',
  headerTextColor: '#25342c',
  oddRowBackgroundColor: '#fafbfa',
  rowBorder: { color: '#e2e7e3' },
  spacing: 7,
})

function cellTooltip(
  cell: ResultCell | undefined,
  column: ResultColumn,
) {
  if (!cell) return ''
  const details: string[] = []
  if (cell.validation_error) {
    details.push(`Needs review: ${friendlyValidation(cell.validation_error)}`)
  }
  if (isDisplayMismatch(cell, column)) {
    details.push(`Expected value: ${displayValue(cell.ground_truth_text)}`)
    details.push(`FarmAI read: ${displayValue(cell.ocr_text)}`)
  }
  const rawNote = friendlyRawNote(cell.raw_text)
  if (rawNote) {
    details.push(rawNote)
  }
  if (cell.was_edited) {
    details.push(`Original reading: ${displayValue(cell.ocr_text)}`)
  }
  return details.join('\n')
}

function displayValue(value: string | null | undefined) {
  return value ? value : 'blank'
}

function friendlyValidation(value: string) {
  if (value.includes('expected temperature pattern')) {
    return 'This should look like a temperature.'
  }
  return value
}

function friendlyRawNote(value: string | null) {
  if (!value) return ''
  const jsonText = value
    .replace(/^```json\s*/i, '')
    .replace(/```$/i, '')
    .trim()
  try {
    const parsed = JSON.parse(jsonText) as {
      status?: string
      text?: string
      reason?: string
    }
    if (parsed.status && parsed.status !== 'ok') {
      return `Recognizer marked this as ${parsed.status}.`
    }
    if (parsed.reason) {
      return `Recognizer note: ${parsed.reason}`
    }
  } catch {
    if (/timed out/i.test(value)) return 'Recognizer timed out on this cell.'
  }
  return ''
}

function exact(value: string | null | undefined) {
  return (value ?? '').trim().replace(/\r\n/g, '\n').replace(/\r/g, '\n')
}

function normalized(value: string | null | undefined, valueType: string) {
  const trimmed = exact(value)
  if (valueType === 'temperature') {
    const number = Number(trimmed)
    return trimmed !== '' && Number.isFinite(number) ? String(number) : trimmed
  }
  if (valueType === 'date_dd_mon') {
    const compact = trimmed.toLocaleLowerCase().replace(/[^a-z0-9]/g, '')
    const match = compact.match(/^(\d{1,2})([a-z]{3,})$/)
    return match ? `${match[1].padStart(2, '0')}${match[2].slice(0, 3)}` : compact
  }
  if (valueType === 'english_text') {
    return trimmed.toLocaleLowerCase().replace(/[^a-z0-9]/g, '')
  }
  return trimmed.replace(/\s+/g, ' ').toLocaleLowerCase()
}

function isDisplayMismatch(cell: ResultCell | undefined, column: ResultColumn) {
  if (!cell || cell.ground_truth_text == null) return false
  return (
    cell.ground_truth_match === false &&
    normalized(cell.ocr_text, column.value_type) !==
      normalized(cell.ground_truth_text, column.value_type)
  )
}

function isFlagged(cell: ResultCell | undefined, column: ResultColumn) {
  return Boolean(cell?.validation_error) || isDisplayMismatch(cell, column)
}

function ReviewCellRenderer(
  params: ICellRendererParams<ReviewRow> & { columnDef?: ResultColumn },
) {
  const field = params.colDef?.field
  const cell = field ? params.data?._cells[field] : undefined
  const flagged = params.columnDef ? isFlagged(cell, params.columnDef) : false
  return (
    <span className="review-cell">
      <span className="review-cell__text">{String(params.value ?? '')}</span>
      {flagged && (
        <AlertTriangle
          className="review-cell__flag"
          size={15}
          aria-label="Needs review"
        />
      )}
    </span>
  )
}

export function OcrResultGrid({
  page,
  needsReviewOnly,
  selectedCell,
  onSelectCell,
  onEdit,
}: OcrResultGridProps) {
  const allRows = useMemo<ReviewRow[]>(() => {
    const rows = new Map<number, ReviewRow>()
    for (let row = 1; row <= page.data_row_count; row += 1) {
      rows.set(row, { _row: row, _cells: {} })
    }
    for (const cell of page.cells) {
      const row = rows.get(cell.row)
      if (!row) continue
      row[cell.column_key] = cell.reviewed_text
      row._cells[cell.column_key] = cell
    }
    return [...rows.values()]
  }, [page])

  const rowData = useMemo(
    () =>
      needsReviewOnly
        ? allRows.filter((row) =>
            page.columns.some(
              (column) => isFlagged(row._cells[column.key], column),
            ),
          )
        : allRows,
    [allRows, needsReviewOnly, page.columns],
  )

  const columnDefs = useMemo<ColDef<ReviewRow>[]>(
    () =>
      page.columns.map((column) => ({
        field: column.key,
        headerName: column.name,
        headerTooltip: column.name,
        editable: true,
        flex: column.key === 'comments' ? 3 : 1,
        minWidth: column.key === 'comments' ? 240 : 105,
        cellRenderer: (params: ICellRendererParams<ReviewRow>) => (
          <ReviewCellRenderer {...params} columnDef={column} />
        ),
        tooltipValueGetter: (params) =>
          cellTooltip(params.data?._cells[column.key], column),
        cellClassRules: {
          'cell-warning': (params) => {
            const cell = params.data?._cells[column.key]
            return Boolean(cell?.validation_error) && !isDisplayMismatch(cell, column)
          },
          'cell-mismatch': (params) => {
            return isDisplayMismatch(params.data?._cells[column.key], column)
          },
          'cell-correct': (params) =>
            !isFlagged(params.data?._cells[column.key], column) &&
            params.data?._cells[column.key]?.ground_truth_text != null,
          'cell-edited': (params) =>
            Boolean(params.data?._cells[column.key]?.was_edited),
        },
      })),
    [page.columns],
  )

  const handleCellClick = (event: CellClickedEvent<ReviewRow>) => {
    const field = event.colDef.field
    const cell = field ? event.data?._cells[field] : undefined
    if (cell) onSelectCell(cell)
  }

  const handleValueChange = (event: CellValueChangedEvent<ReviewRow>) => {
    const field = event.colDef.field
    const cell = field ? event.data?._cells[field] : undefined
    if (cell) onEdit(cell, String(event.newValue ?? ''))
  }

  return (
    <div className="grid-wrap">
      <AgGridReact<ReviewRow>
        theme={gridTheme}
        rowData={rowData}
        columnDefs={columnDefs}
        getRowId={(params) => String(params.data._row)}
        rowHeight={42}
        headerHeight={48}
        tooltipShowDelay={250}
        stopEditingWhenCellsLoseFocus
        onCellClicked={handleCellClick}
        onCellValueChanged={handleValueChange}
        rowClassRules={{
          'selected-result-row': (params) =>
            params.data?._row === selectedCell?.row,
        }}
      />
    </div>
  )
}
