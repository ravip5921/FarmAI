import type { JobResult } from '../types/api'
import { calculateAnalysisMetrics } from '../analysis/analysisMetrics'

interface AnalysisPanelProps {
  result: JobResult
}

function formatPercent(value: number | null) {
  return value == null ? 'Not available' : `${(value * 100).toFixed(1)}%`
}

export function AnalysisPanel({ result }: AnalysisPanelProps) {
  const metrics = calculateAnalysisMetrics(result)

  if (!metrics.hasGroundTruth) {
    return (
      <section className="analysis-panel" aria-label="Analysis results">
        <div className="analysis-heading">
          <h2>Analysis</h2>
          <p>Add an accuracy CSV to see recall, precision, and column results.</p>
        </div>
      </section>
    )
  }

  return (
    <section className="analysis-panel" aria-label="Analysis results">
      <div className="analysis-heading">
        <h2>Analysis</h2>
        <p>
          These checks compare FarmAI readings with the attached accuracy CSV
          after project-specific normalization.
        </p>
      </div>

      <div className="analysis-grid">
        <div className="analysis-card">
          <span className="analysis-card__label">Normalized accuracy</span>
          <span className="analysis-card__value">
            {formatPercent(metrics.normalizedAccuracy)}
          </span>
          <span className="analysis-card__note">
            {metrics.correctCells} of {metrics.scoredCells} cells accepted
          </span>
        </div>
        <div className="analysis-card">
          <span className="analysis-card__label">Strict exact match</span>
          <span className="analysis-card__value">
            {formatPercent(metrics.exactAccuracy)}
          </span>
          <span className="analysis-card__note">
            Before ignoring case, spaces, punctuation, and date separators
          </span>
        </div>
        <div className="analysis-card">
          <span className="analysis-card__label">Review recall</span>
          <span className="analysis-card__value">
            {formatPercent(metrics.reviewRecall)}
          </span>
          <span className="analysis-card__note">
            Incorrect cells that were flagged for review
          </span>
        </div>
        <div className="analysis-card">
          <span className="analysis-card__label">Review precision</span>
          <span className="analysis-card__value">
            {formatPercent(metrics.reviewPrecision)}
          </span>
          <span className="analysis-card__note">
            Flagged cells that were actually different from ground truth
          </span>
        </div>
        <div className="analysis-card">
          <span className="analysis-card__label">Review F1</span>
          <span className="analysis-card__value">
            {formatPercent(metrics.reviewF1)}
          </span>
          <span className="analysis-card__note">
            Balance of review precision and recall
          </span>
        </div>
        <div className="analysis-card">
          <span className="analysis-card__label">Missed errors</span>
          <span className="analysis-card__value">{metrics.missedErrors}</span>
          <span className="analysis-card__note">
            Wrong cells that were not flagged
          </span>
        </div>
        <div className="analysis-card">
          <span className="analysis-card__label">False alarms</span>
          <span className="analysis-card__value">{metrics.falseAlarms}</span>
          <span className="analysis-card__note">
            Flagged cells that matched ground truth
          </span>
        </div>
        <div className="analysis-card">
          <span className="analysis-card__label">Template warnings</span>
          <span className="analysis-card__value">{metrics.warningCells}</span>
          <span className="analysis-card__note">
            Cells with format or range concerns
          </span>
        </div>
      </div>

      <div className="analysis-table-wrap">
        <table className="analysis-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Accuracy</th>
              <th>Correct</th>
              <th>Incorrect</th>
              <th>Flagged</th>
              <th>Missed</th>
            </tr>
          </thead>
          <tbody>
            {metrics.columns.map((column) => (
              <tr key={column.key}>
                <td>{column.name}</td>
                <td>{formatPercent(column.accuracy)}</td>
                <td>{column.correct}</td>
                <td>{column.incorrect}</td>
                <td>{column.flagged}</td>
                <td>{column.missedErrors}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
