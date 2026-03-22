import React from 'react'
import ReactDOM from 'react-dom/client'
import Dashboard from './pages/Dashboard'
import './styles/globals.css'

// Add animation styles for the map marker
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse-ring {
    0% { transform: scale(0.8); opacity: 0.5; }
    50% { transform: scale(1.5); opacity: 0.2; }
    100% { transform: scale(0.8); opacity: 0.5; }
  }
`;
document.head.appendChild(style);

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Dashboard />
  </React.StrictMode>
)