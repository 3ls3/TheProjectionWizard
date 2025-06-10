import { useState } from 'react'
import axios from 'axios'
import './App.css'

// Get API base URL from environment or default to localhost
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [file, setFile] = useState(null)
  const [runId, setRunId] = useState('')
  const [uploadResult, setUploadResult] = useState(null)
  const [targetSuggestion, setTargetSuggestion] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleFileChange = (event) => {
    setFile(event.target.files[0])
    setError('')
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a CSV file')
      return
    }

    setLoading(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(`${API_BASE}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setUploadResult(response.data)
      setRunId(response.data.run_id)
      setError('')
    } catch (err) {
      setError(err.response?.data?.detail?.detail || 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  const getTargetSuggestion = async () => {
    if (!runId) return

    setLoading(true)
    try {
      const response = await axios.get(`${API_BASE}/api/target-suggestion`, {
        params: { run_id: runId }
      })
      setTargetSuggestion(response.data)
    } catch (err) {
      setError(err.response?.data?.detail?.detail || 'Failed to get target suggestion')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔮 The Projection Wizard</h1>
        <p>Upload your CSV and let the magic begin!</p>
      </header>

      <main className="App-main">
        <div className="upload-section">
          <h2>Step 1: Upload CSV</h2>
          <input 
            type="file" 
            accept=".csv" 
            onChange={handleFileChange}
            disabled={loading}
          />
          <button 
            onClick={handleUpload} 
            disabled={!file || loading}
            className="upload-btn"
          >
            {loading ? 'Uploading...' : 'Upload File'}
          </button>
        </div>

        {error && (
          <div className="error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {uploadResult && (
          <div className="result-section">
            <h3>Upload Successful!</h3>
            <p><strong>Run ID:</strong> {uploadResult.run_id}</p>
            <p><strong>Shape:</strong> {uploadResult.shape[0]} rows × {uploadResult.shape[1]} columns</p>
            
            <div className="preview">
              <h4>Data Preview:</h4>
              <table>
                <thead>
                  <tr>
                    {uploadResult.preview[0]?.map((header, i) => (
                      <th key={i}>{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {uploadResult.preview.slice(1).map((row, i) => (
                    <tr key={i}>
                      {row.map((cell, j) => (
                        <td key={j}>{cell}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <button 
              onClick={getTargetSuggestion}
              disabled={loading}
              className="suggestion-btn"
            >
              {loading ? 'Getting suggestion...' : 'Get Target Suggestion'}
            </button>
          </div>
        )}

        {targetSuggestion && (
          <div className="suggestion-section">
            <h3>Target Suggestion</h3>
            <p><strong>Suggested Column:</strong> {targetSuggestion.suggested_column}</p>
            <p><strong>Task Type:</strong> {targetSuggestion.task_type}</p>
            <p><strong>Confidence:</strong> {(targetSuggestion.confidence * 100).toFixed(1)}%</p>
          </div>
        )}

        <div className="api-info">
          <p><strong>API Base URL:</strong> {API_BASE}</p>
          <p><em>Backend and frontend are communicating successfully! 🎉</em></p>
        </div>
      </main>
    </div>
  )
}

export default App 