import type { JobResult, ResultCell } from '../types/api'

export interface ColumnAnalysis {
  key: string
  name: string
  scored: number
  correct: number
  incorrect: number
  accuracy: number | null
  flagged: number
  missedErrors: number
}

export interface AnalysisMetrics {
  hasGroundTruth: boolean
  scoredCells: number
  correctCells: number
  incorrectCells: number
  normalizedAccuracy: number | null
  exactAccuracy: number | null
  reviewPrecision: number | null
  reviewRecall: number | null
  reviewF1: number | null
  flaggedCells: number
  falseAlarms: number
  missedErrors: number
  warningCells: number
  columns: ColumnAnalysis[]
}

function exact(value: string | null | undefined) {
  return (value ?? '').trim().replace(/\r\n/g, '\n').replace(/\r/g, '\n')
}

function isFlagged(cell: ResultCell) {
  return Boolean(cell.validation_error) || cell.ground_truth_match === false
}

function percent(numerator: number, denominator: number) {
  return denominator ? numerator / denominator : null
}

export function calculateAnalysisMetrics(result: JobResult): AnalysisMetrics {
  const cells = result.pages.flatMap((page) => page.cells)
  const scored = cells.filter((cell) => cell.ground_truth_text != null)
  const correct = scored.filter((cell) => cell.ground_truth_match === true)
  const exactCorrect = scored.filter(
    (cell) => exact(cell.ocr_text) === exact(cell.ground_truth_text),
  )
  const incorrect = scored.filter((cell) => cell.ground_truth_match === false)
  const flagged = scored.filter(isFlagged)
  const trueFlags = flagged.filter((cell) => cell.ground_truth_match === false)
  const falseAlarms = flagged.filter((cell) => cell.ground_truth_match === true)
  const missedErrors = scored.filter(
    (cell) => cell.ground_truth_match === false && !isFlagged(cell),
  )
  const precision = percent(trueFlags.length, flagged.length)
  const recall = percent(trueFlags.length, incorrect.length)
  const f1 =
    precision != null && recall != null && precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : null

  const columnMap = new Map<string, ColumnAnalysis>()
  for (const page of result.pages) {
    for (const column of page.columns) {
      if (!columnMap.has(column.key)) {
        columnMap.set(column.key, {
          key: column.key,
          name: column.name,
          scored: 0,
          correct: 0,
          incorrect: 0,
          accuracy: null,
          flagged: 0,
          missedErrors: 0,
        })
      }
    }
  }

  for (const cell of scored) {
    const column = columnMap.get(cell.column_key)
    if (!column) continue
    column.scored += 1
    column.correct += cell.ground_truth_match === true ? 1 : 0
    column.incorrect += cell.ground_truth_match === false ? 1 : 0
    column.flagged += isFlagged(cell) ? 1 : 0
    column.missedErrors +=
      cell.ground_truth_match === false && !isFlagged(cell) ? 1 : 0
  }

  const columns = [...columnMap.values()].map((column) => ({
    ...column,
    accuracy: percent(column.correct, column.scored),
  }))

  return {
    hasGroundTruth: scored.length > 0,
    scoredCells: scored.length,
    correctCells: correct.length,
    incorrectCells: incorrect.length,
    normalizedAccuracy: percent(correct.length, scored.length),
    exactAccuracy: percent(exactCorrect.length, scored.length),
    reviewPrecision: precision,
    reviewRecall: recall,
    reviewF1: f1,
    flaggedCells: flagged.length,
    falseAlarms: falseAlarms.length,
    missedErrors: missedErrors.length,
    warningCells: cells.filter((cell) => Boolean(cell.validation_error)).length,
    columns,
  }
}
