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
import type { ResultCell, ResultPage } from '../types/api'

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

function cellTooltip(cell: ResultCell | undefined) {
  if (!cell) return ''
  const details: string[] = []
  if (cell.validation_error) details.push(cell.validation_error)
  if (cell.ground_truth_match === false) {
    details.push(`Expected: ${cell.ground_truth_text ?? 'blank'}`)
    details.push(`OCR: ${cell.ocr_text || 'blank'}`)
  }
  if (cell.raw_text && cell.raw_text !== cell.ocr_text) {
    details.push(`Raw response: ${cell.raw_text}`)
  }
  if (cell.was_edited) details.push(`Original OCR: ${cell.ocr_text || 'blank'}`)
  return details.join('\n')
}

function ReviewCellRenderer(params: ICellRendererParams<ReviewRow>) {
  const field = params.colDef?.field
  const cell = field ? params.data?._cells[field] : undefined
  const flagged =
    cell?.state === 'validation_warning' ||
    cell?.state === 'ground_truth_mismatch' ||
    cell?.state === 'mismatch_and_warning'
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
            Object.values(row._cells).some(
              (cell) =>
                cell.validation_error || cell.ground_truth_match === false,
            ),
          )
        : allRows,
    [allRows, needsReviewOnly],
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
        cellRenderer: ReviewCellRenderer,
        tooltipValueGetter: (params) =>
          cellTooltip(params.data?._cells[column.key]),
        cellClassRules: {
          'cell-warning': (params) => {
            const state = params.data?._cells[column.key]?.state
            return state === 'validation_warning'
          },
          'cell-mismatch': (params) => {
            const state = params.data?._cells[column.key]?.state
            return (
              state === 'ground_truth_mismatch' ||
              state === 'mismatch_and_warning'
            )
          },
          'cell-correct': (params) =>
            params.data?._cells[column.key]?.state === 'correct',
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
