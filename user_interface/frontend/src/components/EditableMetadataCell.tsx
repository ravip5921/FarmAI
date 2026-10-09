import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Stack, TableCell, TextField } from '@mui/material'
import { apiRequest } from '../api/client'

interface Props {
  id: string
  documents?: boolean
  field: 'comments' | 'reference_id'
  value: string
  fallback?: string
  disabled?: boolean
}

export function EditableMetadataCell({ id, documents = false, field, value, fallback = '-', disabled = false }: Props) {
  const client = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value)
  const label = field === 'comments' ? 'Comments / label' : 'Reference ID'
  const save = useMutation({
    mutationFn: () => apiRequest(documents ? '/api/documents' : `/api/jobs/${id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...(documents ? { document_ids: [id] } : {}), [field]: draft }),
    }),
    onSuccess: async () => {
      await Promise.all(['documents', 'jobs', 'job'].map(key => client.invalidateQueries({ queryKey: [key] })))
      setEditing(false)
    },
  })
  const begin = () => {
    if (disabled || editing) return
    setDraft(value)
    save.reset()
    setEditing(true)
  }
  return <TableCell
    tabIndex={disabled || editing ? undefined : 0}
    title={disabled ? undefined : `${field === 'reference_id' ? `System ID: ${id}. ` : ''}Double-click or press Enter to edit ${label.toLowerCase()}`}
    onDoubleClick={begin}
    onKeyDown={event => {
      if (!editing && (event.key === 'Enter' || event.key === 'F2')) {
        event.preventDefault()
        begin()
      }
    }}
    sx={{ whiteSpace: 'pre-wrap', minWidth: field === 'comments' ? 180 : 120, maxWidth: 320, overflowWrap: 'anywhere', cursor: disabled ? undefined : 'text' }}
  >
    {editing ? <Stack spacing={1}>
      <TextField autoFocus fullWidth size="small" label={label} value={draft}
        multiline={field === 'comments'} minRows={field === 'comments' ? 2 : undefined}
        disabled={save.isPending}
        slotProps={{ htmlInput: { maxLength: field === 'comments' ? 5000 : 200 } }}
        onChange={event => setDraft(event.target.value)}
        onKeyDown={event => {
          event.stopPropagation()
          if (event.nativeEvent.isComposing || save.isPending) return
          if (event.key === 'Escape') { event.preventDefault(); setEditing(false) }
          if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); save.mutate() }
        }}
        helperText={field === 'comments' ? 'Enter to save; Shift+Enter for a new line; Esc to cancel.' : 'Enter to save; Esc to cancel.'}
      />
      {save.error && <Alert severity="error">{save.error.message}</Alert>}
      <Stack direction="row" spacing={1}>
        <Button size="small" disabled={save.isPending} onClick={() => save.mutate()}>{save.isPending ? 'Saving...' : 'Save'}</Button>
        <Button size="small" disabled={save.isPending} onClick={() => setEditing(false)}>Cancel</Button>
      </Stack>
    </Stack> : value || fallback}
  </TableCell>
}
