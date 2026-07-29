import {
  Button,
  Checkbox,
  Divider,
  Drawer,
  FormControl,
  InputLabel,
  ListItemText,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material'
import { FileCheck2, X } from 'lucide-react'
import type {
  AppSettingsResponse,
  JobSettings,
} from '../types/api'

interface SettingsDrawerProps {
  open: boolean
  options?: AppSettingsResponse
  value: JobSettings
  groundTruth: File | null
  onChange: (settings: JobSettings) => void
  onGroundTruthChange: (file: File | null) => void
  onClose: () => void
}

export function SettingsDrawer({
  open,
  options,
  value,
  groundTruth,
  onChange,
  onGroundTruthChange,
  onClose,
}: SettingsDrawerProps) {
  const template = options?.templates.find(
    (item) => item.id === value.template_id,
  )
  const visibleColumns =
    template?.columns.filter((column) => !column.filter_out) ?? []

  return (
    <Drawer anchor="right" open={open} onClose={onClose}>
      <Stack sx={{ width: { xs: 320, sm: 400 }, p: 3, gap: 2.5 }}>
        <Stack
          direction="row"
          sx={{ alignItems: 'center', justifyContent: 'space-between' }}
        >
          <div>
            <Typography variant="h6" sx={{ fontWeight: 750 }}>
              Advanced settings
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Defaults work for most Boar Room records.
            </Typography>
          </div>
          <Button
            aria-label="Close settings"
            onClick={onClose}
            sx={{ minWidth: 44, width: 44, height: 44 }}
          >
            <X size={20} aria-hidden="true" />
          </Button>
        </Stack>

        <FormControl fullWidth>
          <InputLabel id="template-label">Record type</InputLabel>
          <Select
            labelId="template-label"
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

        <FormControl fullWidth>
          <InputLabel id="engine-label">Recognition method</InputLabel>
          <Select
            labelId="engine-label"
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

        <FormControl fullWidth>
          <InputLabel id="filter-label">Hide additional columns</InputLabel>
          <Select
            multiple
            labelId="filter-label"
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

        <Divider />

        <div>
          <Typography sx={{ fontWeight: 700 }}>Ground-truth CSV</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
            Optional. Add known answers to measure recognition accuracy.
          </Typography>
          <Button
            component="label"
            variant="outlined"
            startIcon={<FileCheck2 size={18} />}
          >
            {groundTruth ? 'Replace CSV' : 'Choose CSV'}
            <input
              hidden
              type="file"
              accept=".csv,text/csv"
              onChange={(event) =>
                onGroundTruthChange(event.target.files?.[0] ?? null)
              }
            />
          </Button>
          {groundTruth && (
            <Stack
              direction="row"
              sx={{
                mt: 1.25,
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <Typography
                variant="body2"
                sx={{
                  maxWidth: 270,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {groundTruth.name}
              </Typography>
              <Button size="small" onClick={() => onGroundTruthChange(null)}>
                Remove
              </Button>
            </Stack>
          )}
        </div>

        <Button variant="contained" size="large" onClick={onClose}>
          Done
        </Button>
      </Stack>
    </Drawer>
  )
}
