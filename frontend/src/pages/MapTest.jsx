import { useState, useEffect } from 'react';
import SimpleMap from '../components/SimpleMap';

export default function MapTest() {
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    // Test backend connection
    fetch('http://localhost:8000/health')
      .then(res => res.json())
      .then(data => {
        console.log('✅ Backend connected:', data);
        setApiStatus('connected');
      })
      .catch(err => {
        console.error('❌ Backend connection failed:', err);
        setApiStatus('disconnected');
      });
  }, []);

  return (
    <div style={{ 
      padding: '20px', 
      background: '#0B0F19', 
      color: 'white', 
      minHeight: '100vh',
      fontFamily: 'Space Grotesk, sans-serif'
    }}>
      <h1 style={{ color: '#00D4FF' }}>🗺️ Map Test Page</h1>
      
      <div style={{ 
        display: 'flex', 
        gap: '20px', 
        marginBottom: '20px',
        padding: '10px',
        background: '#1A2540',
        borderRadius: '8px'
      }}>
        <div>
          <strong>Backend:</strong> {apiStatus === 'connected' ? 
            <span style={{ color: '#00E57A' }}>✅ Connected</span> : 
            <span style={{ color: '#FF3B5C' }}>❌ {apiStatus}</span>}
        </div>
        <div>
          <strong>MapLibre Version:</strong> {maplibregl.version || 'unknown'}
        </div>
      </div>

      <div style={{ 
        height: '600px', 
        border: '2px solid #00D4FF',
        borderRadius: '12px',
        overflow: 'hidden'
      }}>
        <SimpleMap />
      </div>

      <div style={{ marginTop: '20px', color: '#94A3B8' }}>
        <p>If map doesn't appear, check browser console (F12) for errors.</p>
      </div>
    </div>
  );
}