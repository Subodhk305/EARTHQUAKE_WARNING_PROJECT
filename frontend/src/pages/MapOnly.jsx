import React, { useRef, useEffect, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

export default function MapOnly() {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    console.log('🗺️ Creating standalone map...');

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: 'https://demotiles.maplibre.org/style.json',
      center: [0, 0],
      zoom: 1
    });

    map.on('load', () => {
      console.log('✅ Standalone map loaded');
    });

    return () => map.remove();
  }, []);

  return (
    <div style={{ height: '100vh', width: '100vw' }}>
      <div ref={containerRef} style={{ height: '100%', width: '100%' }} />
    </div>
  );
}