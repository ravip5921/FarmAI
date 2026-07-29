import { Navigate, Route, Routes } from 'react-router-dom'
import { JobPage } from './pages/JobPage'
import { UploadPage } from './pages/UploadPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/jobs/:jobId" element={<JobPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
