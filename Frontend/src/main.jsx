import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ReposProvider } from './hooks/useRepos.jsx'
import { ThemeProvider } from './hooks/useTheme.jsx'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <ReposProvider>
          <App />
        </ReposProvider>
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>,
)
