import { useId } from 'react'
import { Checkbox, FormControl, InputLabel, ListItemText, MenuItem, Select, Stack } from '@mui/material'
import type { AppSettingsResponse, JobSettings } from '../types/api'

interface Props {
  options?: AppSettingsResponse
  value: JobSettings
  onChange: (settings: JobSettings) => void
  disabled?: boolean
}

export function SettingsFields({ options, value, onChange, disabled = false }: Props) {
  const id = useId()
  const template = options?.templates.find(
    (item) => item.id === value.template_id,
  )
  const visibleColumns =
    template?.columns.filter((column) => !column.filter_out) ?? []

  return <Stack spacing={2.5}>
        <FormControl fullWidth disabled={disabled}>
          <InputLabel id={`${id}-template`}>Record type</InputLabel>
          <Select
            labelId={`${id}-template`}
            label="Record type"
            value={value.template_id ?? ''}
            onChange={(event) =>
              onChange({
                ...value,
                template_id: event.target.value || null,
                extra_filtered_columns: [],
              })
            }
          >
            <MenuItem value="">
              Detected table (no template)
            </MenuItem>
            {options?.templates.map((item) => (
              <MenuItem key={item.id} value={item.id}>
                {item.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl fullWidth disabled={disabled}>
          <InputLabel id={`${id}-engine`}>Recognition method</InputLabel>
          <Select
            labelId={`${id}-engine`}
            label="Recognition method"
            value={value.ocr_engine}
            onChange={(event) =>
              onChange({ ...value, ocr_engine: event.target.value })
            }
          >
            {options?.ocr_engines.map((engine) => (
              <MenuItem key={engine.name} value={engine.name}>
                {engine.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl fullWidth disabled={disabled}>
          <InputLabel id={`${id}-filter`}>Hide additional columns</InputLabel>
          <Select
            multiple
            labelId={`${id}-filter`}
            label="Hide additional columns"
            value={value.extra_filtered_columns}
            renderValue={(selected) =>
              selected.length ? selected.join(', ') : 'None'
            }
            onChange={(event) =>
              onChange({
                ...value,
                extra_filtered_columns:
                  typeof event.target.value === 'string'
                    ? event.target.value.split(',')
                    : event.target.value,
              })
            }
          >
            {visibleColumns.map((column) => (
              <MenuItem key={column.key} value={column.key}>
                <Checkbox
                  checked={value.extra_filtered_columns.includes(column.key)}
                />
                <ListItemText primary={column.name} />
              </MenuItem>
            ))}
          </Select>
        </FormControl>

  </Stack>
}
