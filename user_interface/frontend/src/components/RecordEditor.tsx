import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Checkbox, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, Stack, TextField } from '@mui/material'
import { apiRequest } from '../api/client'
import { getSettings } from '../api/jobs'
import { SettingsFields } from './SettingsFields'
import type { JobSettings } from '../types/api'

interface Props {
  ids: string[]
  documents?: boolean
  title: string
  initialSettings?: JobSettings | null
  referenceId?: string
  comments?: string
  canEditSettings: boolean
  onClose: () => void
}

export function RecordEditor({ ids, documents = false, title, initialSettings, referenceId = '', comments = '', canEditSettings, onClose }: Props) {
  const client = useQueryClient()
  const options = useQuery({ queryKey: ['settings'], queryFn: getSettings })
  const [settings, setSettings] = useState<JobSettings | null>(initialSettings ?? null)
  const [changeSettings, setChangeSettings] = useState(false)
  const [reference, setReference] = useState(referenceId)
  const [notes, setNotes] = useState(comments)
  const [changeMetadata, setChangeMetadata] = useState(ids.length === 1)
  const value = settings ?? options.data?.defaults ?? { template_id: null, ocr_engine: 'llm-vision', extra_filtered_columns: [] }
  const save = useMutation({
    mutationFn: () => apiRequest(documents ? '/api/documents' : `/api/jobs/${ids[0]}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...(documents ? { document_ids: ids } : {}),
        ...(changeSettings ? { settings: value } : {}),
        ...(changeMetadata ? { reference_id: reference, comments: notes } : {}),
      }),
    }),
    onSuccess: async () => {
      await Promise.all(['documents', 'jobs', 'job'].map(key => client.invalidateQueries({ queryKey: [key] })))
      onClose()
    },
  })
  return <>
    <Dialog open onClose={() => !save.isPending && onClose()} fullWidth maxWidth="sm">
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ pt: 1 }}>
          {save.error && <Alert severity="error">{save.error.message}</Alert>}
          {ids.length > 1 && <FormControlLabel control={<Checkbox checked={changeMetadata} onChange={(_, checked) => setChangeMetadata(checked)} />} label="Replace ID and comments on all selected files" />}
          <TextField label="Reference ID" value={reference} disabled={!changeMetadata || save.isPending} onChange={event => setReference(event.target.value)} slotProps={{ htmlInput: { maxLength: 200 } }} helperText="Optional identifier for your records; the system ID stays the same." />
          <TextField label="Comments / label" multiline minRows={3} value={notes} disabled={!changeMetadata || save.isPending} onChange={event => setNotes(event.target.value)} slotProps={{ htmlInput: { maxLength: 5000 } }} />
          {canEditSettings ? <>
            {ids.length > 1 && <FormControlLabel
              control={<Checkbox disabled={save.isPending} checked={changeSettings} onChange={(_, checked) => setChangeSettings(checked)} />}
              label="Apply settings to all selected files"
            />}
            <SettingsFields options={options.data} value={value}
              disabled={save.isPending || !options.data || (ids.length > 1 && !changeSettings)}
              onChange={next => { setSettings(next); setChangeSettings(true) }}
            />
            {options.error && <Alert severity="error">Settings could not be loaded.</Alert>}
          </> : <Alert severity="info">Processing settings can only change before a job first starts.</Alert>}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button disabled={save.isPending} onClick={onClose}>Cancel</Button>
        <Button variant="contained" disabled={save.isPending || (!changeMetadata && !changeSettings)} onClick={() => save.mutate()}>{save.isPending ? 'Saving...' : 'Save'}</Button>
      </DialogActions>
    </Dialog>
  </>
}
