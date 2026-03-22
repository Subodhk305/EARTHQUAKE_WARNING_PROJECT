import { useRef, useEffect, useState } from 'react';
import maplibregl from 'maplibre-gl';

export default function SimpleMap() {
  const containerRef = useRef(null);
  const [mapStatus, setMapStatus] = useState('initializing');

  useEffect(() => {
    if (!containerRef.current) return;

    console.log('🗺️ Creating map...');
    setMapStatus('creating');

    try {
      const map = new maplibregl.Map({
        container: containerRef.current,
        style: 'https://demotiles.maplibre.org/style.json', // Free style, no key needed
        center: [78.9629, 20.5937], // India
        zoom: 3
      });

      map.on('load', () => {
        console.log('✅ Map loaded successfully');
        setMapStatus('loaded');
        
        // Add a navigation control
        map.addControl(new maplibregl.NavigationControl(), 'top-right');
        
        // Add a marker at center
        new maplibregl.Marker()
          .setLngLat([78.9629, 20.5937])
          .addTo(map);
      });

      map.on('error', (e) => {
        console.error('❌ Map error:', e);
        setMapStatus('error: ' + (e.error?.message || 'Unknown error'));
      });

      return () => map.remove();
    } catch (error) {
      console.error('❌ Map creation error:', error);
      setMapStatus('error: ' + error.message);
    }
  }, []);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div 
        ref={containerRef} 
        style={{ 
          width: '100%', 
          height: '100%',
          minHeight: '500px',
          background: '#0B0F19'
        }} 
      />
      <div style={{
        position: 'absolute',
        top: 10,
        left: 10,
        background: 'rgba(0,0,0,0.8)',
        color: mapStatus === 'loaded' ? '#00E57A' : '#FF3B5C',
        padding: '8px 12px',
        borderRadius: '4px',
        fontSize: '12px',
        zIndex: 1000,
        fontFamily: 'monospace'
      }}>
        Map Status: {mapStatus}
      </div>
    </div>
  );
}