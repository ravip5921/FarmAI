import { apiRequest } from './client'
import type {
  AppSettingsResponse,
  CreatedJob,
  JobResult,
  JobSettings,
  JobSummary,
  JobsResponse,
} from '../types/api'

export function getSettings() {
  return apiRequest<AppSettingsResponse>('/api/settings')
}

export function createJob(
  record: File,
  settings: JobSettings,
  groundTruth?: File | null,
) {
  const body = new FormData()
  body.append('record', record)
  body.append('settings', JSON.stringify(settings))
  if (groundTruth) {
    body.append('ground_truth', groundTruth)
  }
  return apiRequest<CreatedJob>('/api/jobs', {
    method: 'POST',
    body,
  })
}

export function getJob(jobId: string) {
  return apiRequest<JobSummary>(`/api/jobs/${jobId}`)
}

export function getJobs() {
  return apiRequest<JobsResponse>('/api/jobs')
}

export async function deleteJob(jobId: string) {
  const response = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' })
  if (!response.ok) {
    let message = 'The job could not be deleted.'
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') message = body.detail
    } catch {
      // Keep the user-safe fallback.
    }
    throw new Error(message)
  }
}

export function cancelJob(jobId: string) {
  return apiRequest<JobSummary>(`/api/jobs/${jobId}/cancel`, {
    method: 'POST',
  })
}

export function getResult(jobId: string) {
  return apiRequest<JobResult>(`/api/jobs/${jobId}/result`)
}

export function editCell(
  jobId: string,
  edit: {
    page_number: number
    row: number
    column_key: string
    reviewed_text: string
  },
) {
  return apiRequest<JobResult>(`/api/jobs/${jobId}/cells`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ edits: [edit] }),
  })
}

export function attachGroundTruth(jobId: string, file: File) {
  const body = new FormData()
  body.append('ground_truth', file)
  return apiRequest<JobResult>(`/api/jobs/${jobId}/ground-truth`, {
    method: 'POST',
    body,
  })
}
