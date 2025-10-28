import React from 'react'
import CoreSettings from './pages/CoreSettings'
import ChatPanel from './components/ChatPanel'
import UploadDropzone from './components/UploadDropzone'

function App() {
  return (
    <div style={{ 
      fontFamily: 'system-ui, -apple-system, sans-serif',
      padding: '20px',
      maxWidth: '1200px',
      margin: '0 auto'
    }}>
      <header style={{ 
        marginBottom: '2rem',
        borderBottom: '2px solid #eee',
        paddingBottom: '1rem'
      }}>
        <h1 style={{ margin: 0, color: '#333' }}>
          🚀 Rebecca Platform Dashboard
        </h1>
        <p style={{ color: '#666', margin: '0.5rem 0 0 0' }}>
          Multi-Agent Automation Platform for Development
        </p>
      </header>

      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr', 
        gap: '2rem',
        marginBottom: '2rem'
      }}>
        <div style={{ 
          background: '#f8f9fa', 
          padding: '1.5rem', 
          borderRadius: '8px',
          border: '1px solid #e9ecef'
        }}>
          <h2 style={{ marginTop: 0, color: '#495057' }}>
            ⚙️ Core Settings
          </h2>
          <CoreSettings />
        </div>

        <div style={{ 
          background: '#f8f9fa', 
          padding: '1.5rem', 
          borderRadius: '8px',
          border: '1px solid #e9ecef'
        }}>
          <h2 style={{ marginTop: 0, color: '#495057' }}>
            📤 Document Upload
          </h2>
          <UploadDropzone />
        </div>
      </div>

      <div style={{ 
        background: '#f8f9fa', 
        padding: '1.5rem', 
        borderRadius: '8px',
        border: '1px solid #e9ecef'
      }}>
        <h2 style={{ marginTop: 0, color: '#495057' }}>
          💬 Chat Interface
        </h2>
        <ChatPanel />
      </div>

      <footer style={{ 
        marginTop: '2rem',
        paddingTop: '1rem',
        borderTop: '1px solid #e9ecef',
        color: '#6c757d',
        textAlign: 'center'
      }}>
        <p>
          Rebecca Platform v0.1.0 | Multi-Agent Automation System
        </p>
      </footer>
    </div>
  )
}

export default App
