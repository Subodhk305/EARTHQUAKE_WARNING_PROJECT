// frontend/src/components/EarthquakeMap.jsx
import { useEffect, useRef, useState } from 'react';
import { Loader2, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

export default function EarthquakeMap({ onLocationSelect, selectedLocation, historicalEvents = [] }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [zoom, setZoom] = useState(4);
  const [center, setCenter] = useState({ lat: 20.5937, lng: 78.9629 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0, lat: 0, lng: 0 });
  const tileCache = useRef({});
  const [mapReady, setMapReady] = useState(false);

  const MIN_ZOOM = 2;
  const MAX_ZOOM = 10;
  const TILE_SIZE = 256;

  // Convert lat/lon to pixel coordinates
  const latLonToPixels = (lat, lng, zoom, width, height) => {
    const x = (lng + 180) / 360 * Math.pow(2, zoom);
    const y = (1 - Math.log(Math.tan(lat * Math.PI / 180) + 1 / Math.cos(lat * Math.PI / 180)) / Math.PI) / 2 * Math.pow(2, zoom);
    
    const tileSize = TILE_SIZE;
    const pixelX = x * tileSize;
    const pixelY = y * tileSize;
    
    return { pixelX, pixelY };
  };

  // Load a single tile
  const loadTile = (x, y, zoom) => {
    return new Promise((resolve, reject) => {
      const cacheKey = `${zoom}/${x}/${y}`;
      if (tileCache.current[cacheKey]) {
        resolve(tileCache.current[cacheKey]);
        return;
      }

      const img = new Image();
      img.crossOrigin = "Anonymous";
      img.src = `https://tile.openstreetmap.org/${zoom}/${x}/${y}.png`;
      
      img.onload = () => {
        tileCache.current[cacheKey] = img;
        resolve(img);
      };
      img.onerror = reject;
    });
  };

  // Draw the map
  const drawMap = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    if (width === 0 || height === 0) return;
    
    setLoading(true);
    
    // Clear canvas
    ctx.fillStyle = '#0B0F19';
    ctx.fillRect(0, 0, width, height);
    
    // Calculate tile range to cover the canvas
    const centerPixel = latLonToPixels(center.lat, center.lng, zoom, width, height);
    const startX = Math.floor((centerPixel.pixelX - width / 2) / TILE_SIZE);
    const endX = Math.ceil((centerPixel.pixelX + width / 2) / TILE_SIZE);
    const startY = Math.floor((centerPixel.pixelY - height / 2) / TILE_SIZE);
    const endY = Math.ceil((centerPixel.pixelY + height / 2) / TILE_SIZE);
    
    const maxTile = Math.pow(2, zoom) - 1;
    const tilePromises = [];
    
    for (let tx = Math.max(0, startX); tx <= Math.min(maxTile, endX); tx++) {
      for (let ty = Math.max(0, startY); ty <= Math.min(maxTile, endY); ty++) {
        tilePromises.push(loadTile(tx, ty, zoom).catch(() => null));
      }
    }
    
    const tiles = await Promise.all(tilePromises);
    setLoading(false);
    
    let idx = 0;
    for (let tx = Math.max(0, startX); tx <= Math.min(maxTile, endX); tx++) {
      for (let ty = Math.max(0, startY); ty <= Math.min(maxTile, endY); ty++) {
        const tile = tiles[idx++];
        if (!tile) continue;
        
        const tilePixelX = tx * TILE_SIZE;
        const tilePixelY = ty * TILE_SIZE;
        
        const screenX = (tilePixelX - centerPixel.pixelX) + width / 2;
        const screenY = (tilePixelY - centerPixel.pixelY) + height / 2;
        
        ctx.drawImage(tile, screenX, screenY, TILE_SIZE, TILE_SIZE);
      }
    }
    
    // Draw historical events
    historicalEvents?.slice(0, 100).forEach(event => {
      if (event.latitude && event.longitude) {
        const { pixelX, pixelY } = latLonToPixels(event.latitude, event.longitude, zoom, width, height);
        const screenX = (pixelX - centerPixel.pixelX) + width / 2;
        const screenY = (pixelY - centerPixel.pixelY) + height / 2;
        
        if (screenX >= -20 && screenX <= width + 20 && screenY >= -20 && screenY <= height + 20) {
          const magnitude = event.magnitude || 0;
          let color, size;
          if (magnitude >= 6) { color = '#FF3B5C'; size = 12; }
          else if (magnitude >= 5) { color = '#FF6B3C'; size = 10; }
          else if (magnitude >= 4) { color = '#FFB020'; size = 8; }
          else if (magnitude >= 3) { color = '#7B2FFF'; size = 6; }
          else { color = '#00D4FF'; size = 5; }
          
          ctx.shadowBlur = 8;
          ctx.shadowColor = color;
          ctx.beginPath();
          ctx.arc(screenX, screenY, size, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = 'white';
          ctx.lineWidth = 1.5;
          ctx.stroke();
          
          if (magnitude >= 5) {
            ctx.fillStyle = 'white';
            ctx.font = 'bold 10px monospace';
            ctx.shadowBlur = 0;
            ctx.fillText(`M${magnitude.toFixed(1)}`, screenX + size + 3, screenY - 3);
          }
        }
      }
    });
    
    ctx.shadowBlur = 0;
    
    // Draw selected location
    if (selectedLocation) {
      const { pixelX, pixelY } = latLonToPixels(selectedLocation.latitude, selectedLocation.longitude, zoom, width, height);
      const screenX = (pixelX - centerPixel.pixelX) + width / 2;
      const screenY = (pixelY - centerPixel.pixelY) + height / 2;
      
      if (screenX >= -50 && screenX <= width + 50 && screenY >= -50 && screenY <= height + 50) {
        const time = Date.now() / 400;
        const pulseSize = 16 + Math.sin(time) * 4;
        
        ctx.beginPath();
        ctx.arc(screenX, screenY, pulseSize, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(255, 59, 92, 0.3)';
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(screenX, screenY, 12, 0, 2 * Math.PI);
        ctx.fillStyle = '#FF3B5C';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    
    setMapReady(true);
  };

  // Convert screen coordinates to lat/lon
  const screenToLatLon = (screenX, screenY, width, height, zoom, center) => {
    const centerPixel = latLonToPixels(center.lat, center.lng, zoom, width, height);
    const pixelX = centerPixel.pixelX + (screenX - width / 2);
    const pixelY = centerPixel.pixelY + (screenY - height / 2);
    
    const x = pixelX / TILE_SIZE;
    const y = pixelY / TILE_SIZE;
    
    const lng = (x / Math.pow(2, zoom)) * 360 - 180;
    const lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / Math.pow(2, zoom)))) * 180 / Math.PI;
    
    return { lat, lng };
  };

  // Handle canvas click
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const screenX = (e.clientX - rect.left) * scaleX;
    const screenY = (e.clientY - rect.top) * scaleY;
    
    const { lat, lng } = screenToLatLon(screenX, screenY, canvas.width, canvas.height, zoom, center);
    
    if (onLocationSelect && !isNaN(lat) && !isNaN(lng)) {
      onLocationSelect({
        latitude: Math.max(-85, Math.min(85, lat)),
        longitude: Math.max(-180, Math.min(180, lng)),
        locationName: `${lat.toFixed(2)}°, ${lng.toFixed(2)}°`
      });
    }
  };

  // Handle wheel zoom
  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -1 : 1;
    const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom + delta));
    if (newZoom !== zoom) {
      setZoom(newZoom);
    }
  };

  // Handle drag
  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
    const rect = canvasRef.current.getBoundingClientRect();
    setDragStart({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      lat: center.lat,
      lng: center.lng
    });
  };
  
  const handleMouseMove = (e) => {
    if (!isDragging || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const dx = e.clientX - rect.left - dragStart.x;
    const dy = e.clientY - rect.top - dragStart.y;
    
    const width = canvasRef.current.width;
    const height = canvasRef.current.height;
    
    const lngDelta = dx * 360 / (width * Math.pow(2, zoom - 2));
    const latDelta = dy * 180 / (height * Math.pow(2, zoom - 2));
    
    setCenter({
      lng: dragStart.lng - lngDelta,
      lat: Math.max(-85, Math.min(85, dragStart.lat + latDelta))
    });
  };
  
  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Setup canvas and resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const container = containerRef.current;
    if (!container) return;
    
    const resizeCanvas = () => {
      const rect = container.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      drawMap();
    };
    
    const observer = new ResizeObserver(resizeCanvas);
    observer.observe(container);
    resizeCanvas();
    
    return () => observer.disconnect();
  }, []);
  
  // Redraw when data changes
  useEffect(() => {
    if (canvasRef.current) {
      drawMap();
    }
  }, [zoom, center, historicalEvents, selectedLocation]);

  return (
    <div ref={containerRef} className="w-full h-full rounded-xl overflow-hidden relative">
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="w-full h-full block cursor-grab active:cursor-grabbing"
        style={{ background: '#0B0F19' }}
      />
      
      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10 pointer-events-none">
          <Loader2 size={32} className="text-[#00D4FF] animate-spin" />
        </div>
      )}
      
      {/* Zoom controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 z-10">
        <button
          onClick={() => setZoom(Math.min(MAX_ZOOM, zoom + 1))}
          className="w-8 h-8 bg-black/70 hover:bg-black/90 rounded-lg flex items-center justify-center transition-colors backdrop-blur-sm"
        >
          <ZoomIn size={16} className="text-white" />
        </button>
        <button
          onClick={() => setZoom(Math.max(MIN_ZOOM, zoom - 1))}
          className="w-8 h-8 bg-black/70 hover:bg-black/90 rounded-lg flex items-center justify-center transition-colors backdrop-blur-sm"
        >
          <ZoomOut size={16} className="text-white" />
        </button>
        <button
          onClick={() => {
            setZoom(4);
            setCenter({ lat: 20.5937, lng: 78.9629 });
          }}
          className="w-8 h-8 bg-black/70 hover:bg-black/90 rounded-lg flex items-center justify-center transition-colors backdrop-blur-sm"
        >
          <Maximize2 size={14} className="text-white" />
        </button>
      </div>
      
      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg p-3 text-xs font-mono border border-[#00D4FF]/30 z-10">
        <div className="text-[#00D4FF] font-bold text-xs mb-2">📊 Magnitude</div>
        <div className="space-y-1">
          <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#FF3B5C]" /><span className="text-white">M6+</span></div>
          <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#FFB020]" /><span className="text-white">M4-5</span></div>
          <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#7B2FFF]" /><span className="text-white">M3-4</span></div>
          <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#00D4FF]" /><span className="text-white">M{`<`}3</span></div>
        </div>
      </div>
      
      {/* Instructions */}
      <div className="absolute bottom-4 right-4 bg-black/50 rounded-lg px-2 py-1 text-[10px] text-gray-400 z-10 backdrop-blur-sm">
        🖱️ Drag to pan • Scroll to zoom • Click to select
      </div>
    </div>
  );
}