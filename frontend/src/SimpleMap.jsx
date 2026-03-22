import { useRef, useEffect } from 'react';
import maplibregl from 'maplibre-gl';

export default function SimpleMap() {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    console.log('🗺️ Creating simple map...');

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: 'https://demotiles.maplibre.org/style.json',
      center: [0, 0],
      zoom: 1
    });

    map.on('load', () => {
      console.log('✅ Simple map loaded');
    });

    map.on('error', (e) => {
      console.error('❌ Simple map error:', e);
    });

    return () => map.remove();
  }, []);

  return (
    <div 
      ref={containerRef} 
      style={{ 
        width: '100%', 
        height: '100%',
        minHeight: '500px',
        background: '#0B0F19'
      }} 
    />
  );
}