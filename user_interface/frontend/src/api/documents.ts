import { apiRequest } from './client'
import type {
  AnalysisBatchResponse,
  DocumentsResponse,
  JobSettings,
} from '../types/api'

const DOCUMENT_PAGE_SIZE = 50

export function getDocuments(offset = 0) {
  const query = new URLSearchParams({
    limit: String(DOCUMENT_PAGE_SIZE),
    offset: String(offset),
  })
  return apiRequest<DocumentsResponse>(`/api/documents?${query}`)
}

export function uploadDocuments(documents: File[]) {
  const body = new FormData()
  for (const document of documents) {
    body.append('documents', document)
  }
  return apiRequest<DocumentsResponse>('/api/documents', {
    method: 'POST',
    body,
  })
}

export function analyzePendingDocuments(settings: JobSettings) {
  return apiRequest<AnalysisBatchResponse>('/api/analysis-batches', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  })
}
