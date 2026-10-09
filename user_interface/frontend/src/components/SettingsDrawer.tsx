import { SettingsFields } from './SettingsFields'
import {
  Button,
  Divider,
  Drawer,
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
  groundTruth?: File | null
  showGroundTruth?: boolean
  onChange: (settings: JobSettings) => void
  onGroundTruthChange?: (file: File | null) => void
  onClose: () => void
}

export function SettingsDrawer({
  open,
  options,
  value,
  groundTruth,
  showGroundTruth = true,
  onChange,
  onGroundTruthChange,
  onClose,
}: SettingsDrawerProps) {
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
              These settings are applied when analysis starts.
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

        <SettingsFields options={options} value={value} onChange={onChange} />

        {showGroundTruth && onGroundTruthChange && (
          <>
            <Divider />

            <div>
              <Typography sx={{ fontWeight: 700 }}>Ground-truth CSV</Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mb: 1.5 }}
              >
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
          </>
        )}

        <Button variant="contained" size="large" onClick={onClose}>
          Done
        </Button>
      </Stack>
    </Drawer>
  )
}
