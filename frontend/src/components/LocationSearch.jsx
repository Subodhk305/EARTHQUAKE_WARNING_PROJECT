import { useState, useEffect, useCallback } from 'react';
import { Search, Loader2, X, WifiOff } from 'lucide-react';

export default function LocationSearch({ onSelect }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [offline, setOffline] = useState(!navigator.onLine);
  const [searchError, setSearchError] = useState(null);

  // Monitor online/offline status
  useEffect(() => {
    const handleOnline = () => {
      setOffline(false);
      setSearchError(null);
    };
    const handleOffline = () => setOffline(true);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const search = useCallback(async (searchQuery) => {
    if (!searchQuery.trim() || searchQuery.length < 3 || offline) {
      setResults([]);
      return;
    }

    setLoading(true);
    setSearchError(null);
    
    try {
      // Use OpenStreetMap Nominatim API (free, no key required)
      const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=5&addressdetails=1`;
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const response = await fetch(url, {
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'SeismoAI-Earthquake-Prediction-App'
        },
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      const locations = data.map(item => ({
        id: item.place_id,
        name: item.display_name,
        latitude: parseFloat(item.lat),
        longitude: parseFloat(item.lon),
      }));
      
      setResults(locations);
      setSearchError(null);
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Search request timed out');
        setSearchError('Search timed out. Please try again.');
      } else {
        console.error('Search error:', error);
        setSearchError('Unable to search. Please try again.');
      }
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [offline]);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (query.length >= 3) {
        search(query);
      } else {
        setResults([]);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [query, search]);

  const handleSelect = (location) => {
    onSelect({
      latitude: location.latitude,
      longitude: location.longitude,
      locationName: location.name,
    });
    setQuery(location.name.split(',')[0]); // Show just the city name
    setResults([]);
    setShowResults(false);
    setSearchError(null);
  };

  return (
    <div className="relative w-full">
      <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl glass"
        style={{ border: '1px solid rgba(0,212,255,0.2)' }}>
        {offline ? (
          <WifiOff size={14} className="text-yellow-500 shrink-0" />
        ) : loading ? (
          <Loader2 size={14} className="text-[#00D4FF] animate-spin shrink-0" />
        ) : (
          <Search size={14} className="text-[#00D4FF] shrink-0" />
        )}
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setShowResults(true);
            setSearchError(null);
          }}
          onFocus={() => setShowResults(true)}
          placeholder={offline ? "Search unavailable (offline)" : "Search city or region..."}
          disabled={offline}
          className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-600 outline-none font-mono disabled:opacity-50"
        />
        {query && (
          <button 
            onClick={() => {
              setQuery('');
              setResults([]);
              setShowResults(false);
              setSearchError(null);
            }}
            className="text-slate-500 hover:text-slate-300 transition-colors"
          >
            <X size={12} />
          </button>
        )}
      </div>

      {/* Error message */}
      {searchError && !offline && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 glass rounded-xl p-3">
          <p className="text-xs text-yellow-500">{searchError}</p>
        </div>
      )}

      {/* Offline message */}
      {offline && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 glass rounded-xl p-3">
          <p className="text-xs text-yellow-500">You are offline.</p>
          <p className="text-xs text-slate-400 mt-1">Click on the map to select locations.</p>
        </div>
      )}

      {/* Search results */}
      {!offline && showResults && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 glass rounded-xl overflow-hidden max-h-60 overflow-y-auto"
          style={{ border: '1px solid rgba(26,37,64,0.8)' }}>
          {results.map((location) => (
            <button
              key={location.id}
              onClick={() => handleSelect(location)}
              className="w-full text-left px-3 py-2.5 text-xs font-mono text-slate-300 hover:text-[#00D4FF] hover:bg-[rgba(0,212,255,0.05)] transition-colors border-b border-[rgba(26,37,64,0.5)] last:border-0"
            >
              <span className="text-[#00D4FF] mr-2">→</span>
              {location.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}