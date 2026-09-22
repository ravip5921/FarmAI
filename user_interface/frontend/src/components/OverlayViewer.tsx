import {
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
} from '@mui/material'
import { Maximize2, Minus, Plus, RotateCcw } from 'lucide-react'
import { useState } from 'react'
import type { ResultCell, ResultPage } from '../types/api'

interface OverlayViewerProps {
  page: ResultPage
  selectedCell: ResultCell | null
  onSelectCell: (cell: ResultCell) => void
}

export function OverlayViewer({
  page,
  selectedCell,
  onSelectCell,
}: OverlayViewerProps) {
  const [zoom, setZoom] = useState(1)
  const [view, setView] = useState<'overlay' | 'source'>('overlay')

  return (
    <section className="review-pane">
      <div className="pane-toolbar">
        <ToggleButtonGroup
          exclusive
          size="small"
          value={view}
          onChange={(_, next) => next && setView(next)}
          aria-label="Document view"
        >
          <ToggleButton value="overlay">Detected cells</ToggleButton>
          <ToggleButton value="source">Record</ToggleButton>
        </ToggleButtonGroup>
        <div>
          <Tooltip title="Zoom out">
            <span>
              <IconButton
                aria-label="Zoom out"
                disabled={zoom <= 0.6}
                onClick={() => setZoom((value) => Math.max(0.6, value - 0.2))}
              >
                <Minus size={18} />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Zoom in">
            <span>
              <IconButton
                aria-label="Zoom in"
                disabled={zoom >= 2.4}
                onClick={() => setZoom((value) => Math.min(2.4, value + 0.2))}
              >
                <Plus size={18} />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Fit to width">
            <IconButton aria-label="Fit to width" onClick={() => setZoom(1)}>
              <Maximize2 size={18} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Reset view">
            <IconButton
              aria-label="Reset view"
              onClick={() => {
                setZoom(1)
                setView('overlay')
              }}
            >
              <RotateCcw size={18} />
            </IconButton>
          </Tooltip>
        </div>
      </div>
      <div className="image-viewport">
        <div
          className="image-stage"
          style={{ width: `${zoom * 100}%` }}
        >
          <img
            src={view === 'overlay' ? page.overlay_url : page.source_url}
            alt={
              view === 'overlay'
                ? 'Deskewed farm record with detected table cells'
                : 'Deskewed farm record'
            }
          />
          {page.cells.map((cell) => {
            const [x, y, width, height] = cell.bbox
            const selected =
              selectedCell?.row === cell.row &&
              selectedCell.column_key === cell.column_key
            return (
              <button
                type="button"
                key={`${cell.row}-${cell.column_key}`}
                className={`cell-hotspot ${
                  selected ? 'cell-hotspot--selected' : ''
                }`}
                aria-label={`${cell.column_name}, row ${cell.row}`}
                title={`${cell.column_name}, row ${cell.row}: ${
                  cell.reviewed_text || 'blank'
                }`}
                onClick={() => onSelectCell(cell)}
                style={{
                  left: `${(x / page.image_width) * 100}%`,
                  top: `${(y / page.image_height) * 100}%`,
                  width: `${(width / page.image_width) * 100}%`,
                  height: `${(height / page.image_height) * 100}%`,
                }}
              />
            )
          })}
        </div>
      </div>
    </section>
  )
}
