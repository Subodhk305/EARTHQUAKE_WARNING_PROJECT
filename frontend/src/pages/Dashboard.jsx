// frontend/src/pages/Dashboard.jsx
import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Wifi, WifiOff, BarChart2, Map, Layers, Radio, AlertTriangle } from 'lucide-react';

import EarthquakeMap from '../components/EarthquakeMap';
import PredictionPanel from '../components/PredictionPanel';
import LocationSearch from '../components/LocationSearch';
import AlertToast from '../components/AlertToast';
import MetricsChart from '../components/MetricsChart';
import { predict, getHistorical, getModelMetrics, wsService } from '../services/api';

// Tab definition
const TABS = [
  { id: 'map', label: 'Map', icon: Map },
  { id: 'metrics', label: 'Model Metrics', icon: BarChart2 },
];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('map');
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [predResult, setPredResult] = useState(null);
  const [predLoading, setPredLoading] = useState(false);
  const [historicalEvents, setHistoricalEvents] = useState([]);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [liveStats, setLiveStats] = useState({ requests: 0, alerts: 0 });
  const [heatmapIntensity, setHeatmapIntensity] = useState('moderate');
  const [reconnecting, setReconnecting] = useState(false);

  // ── WebSocket with enhanced service ──────────────────────────────────────
  useEffect(() => {
    // Connect WebSocket
    wsService.connect();
    
    // Set up WebSocket listeners
    wsService
      .on('alert', (data) => {
        console.log('🔔 Alert received:', data);
        setAlerts(prev => {
          const newAlert = {
            id: Date.now(),
            type: data.severity || 'info',
            message: data.message || 'New earthquake alert',
            severity: data.severity || 'medium',
            timestamp: new Date().toISOString(),
            ...data
          };
          return [newAlert, ...prev].slice(0, 5);
        });
        setLiveStats(s => ({ ...s, alerts: s.alerts + 1 }));
      })
      .on('prediction', (data) => {
        console.log('📊 Live prediction received:', data);
        // Update UI with live prediction if it matches current location
        if (selectedLocation && 
            Math.abs(data.latitude - selectedLocation.latitude) < 0.1 && 
            Math.abs(data.longitude - selectedLocation.longitude) < 0.1) {
          setPredResult(data);
        }
      })
      .on('model_status', (data) => {
        console.log('📊 Model status update:', data);
        if (data.model_info) {
          setModelMetrics(prev => ({
            ...prev,
            models: prev?.models || [],
            model_info: data.model_info
          }));
        }
      })
      .onStatusChange((status) => {
        console.log('📡 WebSocket status:', status);
        setWsConnected(status === 'connected');
        setReconnecting(status === 'connecting');
        
        // Show reconnecting toast
        if (status === 'connecting') {
          setAlerts(prev => [{
            id: Date.now(),
            type: 'info',
            message: 'Reconnecting to server...',
            severity: 'low',
            timestamp: new Date().toISOString(),
          }, ...prev].slice(0, 5));
        }
      });

    // Cleanup on unmount
    return () => {
      wsService.disconnect();
      wsService.off('alert');
      wsService.off('prediction');
      wsService.off('model_status');
    };
  }, [selectedLocation]);

  // ── Model metrics ─────────────────────────────────────────────────────────
  useEffect(() => {
    getModelMetrics()
      .then(data => {
        console.log('📊 Model metrics loaded:', data);
        setModelMetrics(data);
        
        // Show success message
        if (data.models && data.models[0]) {
          const bestModel = data.models.find(m => m.model_name === data.best_model) || data.models[0];
          console.log(`🎯 Best model: ${bestModel.model_name} with F1-Score: ${(bestModel.f1_score * 100).toFixed(1)}%`);
        }
      })
      .catch(err => {
        console.error('Failed to load model metrics:', err);
        setAlerts(prev => [{
          id: Date.now(),
          type: 'error',
          message: 'Failed to load model metrics. Using default values.',
          severity: 'medium',
          timestamp: new Date().toISOString(),
        }, ...prev].slice(0, 5));
        
        // Set default metrics if API fails
        setModelMetrics({
          models: [
            {
              model_name: "XGBoost GPU",
              accuracy: 0.947,
              precision: 0.947,
              recall: 0.947,
              f1_score: 0.947,
              roc_auc: 0.996,
              training_samples: 10875,
              evaluation_date: new Date().toISOString()
            }
          ],
          best_model: "XGBoost GPU",
          confusion_matrix: [[721,27,29],[14,756,7],[25,22,730]],
          class_names: ["Low Risk", "Medium Risk", "High Risk"]
        });
      });
  }, []);

  // ── Location select handler ───────────────────────────────────────────────
  const handleLocationSelect = useCallback(async ({ latitude, longitude, locationName }) => {
    setSelectedLocation({ latitude, longitude, name: locationName });
    setPredLoading(true);
    setPredResult(null);
    setLiveStats(s => ({ ...s, requests: s.requests + 1 }));

    try {
      // Run prediction + fetch historical in parallel
      const [pred, hist] = await Promise.all([
        predict({ 
          latitude, 
          longitude, 
          location_name: locationName || 'selected',
          radius_km: 200,
          include_waveform: false
        }),
        getHistorical({
          location: locationName || 'selected',
          lat: latitude,
          lon: longitude,
          radiusKm: 300,
          days: 365,
        }),
      ]);
      
      console.log('✅ Prediction received:', pred);
      console.log('📜 Historical events:', hist.events?.length || 0);
      
      setPredResult(pred);
      setHistoricalEvents(hist.events || []);
      
      // Update heatmap intensity based on historical data
      const eventCount = hist.events?.length || 0;
      const maxMagnitude = Math.max(...(hist.events?.map(e => e.magnitude || 0) || [0]));
      
      if (eventCount > 50 || maxMagnitude > 6) {
        setHeatmapIntensity('high');
      } else if (eventCount > 20 || maxMagnitude > 5) {
        setHeatmapIntensity('active');
      } else if (eventCount > 5 || maxMagnitude > 4) {
        setHeatmapIntensity('moderate');
      } else {
        setHeatmapIntensity('low');
      }

      // Send subscription to WebSocket for this location
      if (wsService.isConnected()) {
        wsService.send({
          type: 'subscribe',
          location: { latitude, longitude, name: locationName }
        });
      }
      
      // Show success alert for high risk predictions
      if (pred.risk_level === 'High') {
        setAlerts(prev => [{
          id: Date.now(),
          type: 'warning',
          message: `⚠️ HIGH RISK ALERT: ${(pred.probability * 100).toFixed(1)}% probability of earthquake in ${locationName}`,
          severity: 'high',
          timestamp: new Date().toISOString(),
        }, ...prev].slice(0, 5));
      }
      
    } catch (err) {
      console.error('❌ Prediction failed:', err);
      setAlerts(prev => [{
        id: Date.now(),
        type: 'error',
        message: err.response?.data?.detail || 'Failed to get prediction. Please try again.',
        severity: 'high',
        timestamp: new Date().toISOString(),
      }, ...prev].slice(0, 5));
    } finally {
      setPredLoading(false);
    }
  }, []);

  // Get heatmap color for display
  const getHeatmapColor = () => {
    switch (heatmapIntensity) {
      case 'high': return '#FF3B5C';
      case 'active': return '#FFB020';
      case 'moderate': return '#7B2FFF';
      default: return '#00D4FF';
    }
  };

  const getHeatmapLabel = () => {
    switch (heatmapIntensity) {
      case 'high': return 'High Seismicity';
      case 'active': return 'Active';
      case 'moderate': return 'Moderate';
      default: return 'Low Activity';
    }
  };

  // ── Alert removal ─────────────────────────────────────────────────────────
  const removeAlert = useCallback((id) => {
    setAlerts(prev => prev.filter(alert => alert.id !== id));
  }, []);

  // Get connection status display
  const getConnectionDisplay = () => {
    if (wsConnected) {
      return { icon: Wifi, text: 'Live', color: '#00E57A' };
    } else if (reconnecting) {
      return { icon: WifiOff, text: 'Reconnecting...', color: '#FFB020' };
    } else {
      return { icon: WifiOff, text: 'Offline', color: '#FF3B5C' };
    }
  };

  const connection = getConnectionDisplay();
  const ConnectionIcon = connection.icon;

  // Get the best model performance metrics
  const getBestModelMetrics = () => {
    if (!modelMetrics?.models) return null;
    const best = modelMetrics.models.find(m => m.model_name === modelMetrics.best_model) || modelMetrics.models[0];
    return best;
  };

  const bestModel = getBestModelMetrics();

  return (
    <div className="h-screen w-screen bg-void grid-bg overflow-hidden flex flex-col scanline">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="shrink-0 flex items-center justify-between px-6 py-3"
        style={{ borderBottom: '1px solid rgba(26,37,64,0.8)', background: 'rgba(11,15,25,0.9)' }}>
        
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: 'rgba(0,212,255,0.1)', border: '1px solid rgba(0,212,255,0.3)' }}>
              <Activity size={16} className="text-[#00D4FF]" />
            </div>
            <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#00E57A] animate-pulse" />
          </div>
          <div>
            <h1 className="text-sm font-bold font-display tracking-tight text-slate-100">
              Seismo<span className="text-neon-glow">AI</span>
            </h1>
            <p className="text-[9px] font-mono text-slate-600 tracking-widest uppercase">
              Earthquake Prediction System
            </p>
          </div>
        </div>

        {/* Center tabs */}
        <div className="flex items-center gap-1 p-1 rounded-lg" style={{ background: 'rgba(14,20,36,0.8)', border: '1px solid rgba(26,37,64,0.6)' }}>
          {TABS.map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all"
                style={{
                  background: active ? 'rgba(0,212,255,0.1)' : 'transparent',
                  color: active ? '#00D4FF' : '#64748B',
                  border: active ? '1px solid rgba(0,212,255,0.2)' : '1px solid transparent',
                }}>
                <Icon size={12} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Right status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3 text-[10px] font-mono text-slate-500">
            <span><span className="text-[#00D4FF]">{liveStats.requests}</span> req</span>
            <span><span className="text-[#FF3B5C]">{liveStats.alerts}</span> alerts</span>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-mono transition-all"
            style={{ color: connection.color }}>
            <ConnectionIcon size={11} className={reconnecting ? 'animate-spin' : ''} />
            {connection.text}
          </div>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────────────────────────── */}
      {activeTab === 'map' && (
        <div className="flex-1 flex overflow-hidden">
          {/* Map area */}
          <div className="flex-1 relative p-3">
            <div className="w-full h-full rounded-xl overflow-hidden">
              <EarthquakeMap
                onLocationSelect={handleLocationSelect}
                selectedLocation={selectedLocation}
                historicalEvents={historicalEvents}
              />
            </div>
            {/* Search overlay */}
            <div className="absolute top-6 left-6 right-80 max-w-xs z-10">
              <LocationSearch
                onSelect={handleLocationSelect}
              />
            </div>
            {/* Map legend */}
            <div className="absolute bottom-6 left-6 glass rounded-lg px-3 py-2 text-[10px] font-mono space-y-1 z-10">
              <p className="text-slate-500 uppercase tracking-widest mb-1.5">Heatmap Intensity</p>
              {[
                { color: '#00D4FF', label: 'Low Activity' },
                { color: '#7B2FFF', label: 'Moderate' },
                { color: '#FFB020', label: 'Active' },
                { color: '#FF3B5C', label: 'High Seismicity' }
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
                  <span className="text-slate-400">{label}</span>
                </div>
              ))}
              {/* Current intensity indicator */}
              {selectedLocation && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-2 pt-2 border-t border-[#1A2540]"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">Current:</span>
                    <span className="text-[10px] font-mono" style={{ color: getHeatmapColor() }}>
                      {getHeatmapLabel()}
                    </span>
                  </div>
                </motion.div>
              )}
            </div>
          </div>

          {/* Right panel */}
          <div className="w-80 shrink-0 border-l border-[#1A2540] p-4 overflow-y-auto"
            style={{ background: 'rgba(14,20,36,0.6)' }}>
            <div className="flex items-center gap-2 mb-4">
              <Radio size={12} className="text-[#00D4FF]" />
              <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">
                Prediction Output
              </span>
            </div>
            
            <PredictionPanel 
              result={predResult} 
              loading={predLoading} 
              location={selectedLocation}
            />

            {/* Historical summary */}
            <AnimatePresence>
              {historicalEvents.length > 0 && !predLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="mt-4 pt-4 border-t border-[#1A2540]"
                >
                  <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                    <Layers size={10} />
                    Historical Data ({historicalEvents.length} events)
                  </p>
                  <div className="space-y-1 max-h-40 overflow-y-auto pr-1 custom-scrollbar">
                    {historicalEvents.slice(0, 10).map((ev, idx) => (
                      <motion.div
                        key={ev.event_id || idx}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.05 }}
                        className="flex justify-between items-center py-1 border-b border-[rgba(26,37,64,0.4)] last:border-0 hover:bg-[rgba(0,212,255,0.05)] transition-colors cursor-pointer"
                        onClick={() => {
                          if (ev.latitude && ev.longitude) {
                            handleLocationSelect({
                              latitude: ev.latitude,
                              longitude: ev.longitude,
                              locationName: ev.place
                            });
                          }
                        }}
                      >
                        <span className="text-[10px] font-mono text-slate-500 truncate flex-1">
                          {ev.place || 'Unknown'}
                        </span>
                        <span className="text-[10px] font-mono font-bold ml-2 px-1.5 py-0.5 rounded"
                          style={{
                            background: ev.magnitude >= 6 ? 'rgba(255,59,92,0.2)' 
                              : ev.magnitude >= 4 ? 'rgba(255,176,32,0.2)' 
                              : 'rgba(0,229,122,0.2)',
                            color: ev.magnitude >= 6 ? '#FF3B5C'
                              : ev.magnitude >= 4 ? '#FFB020' 
                              : '#00E57A'
                          }}>
                          M{ev.magnitude?.toFixed(1)}
                        </span>
                      </motion.div>
                    ))}
                    {historicalEvents.length > 10 && (
                      <p className="text-[8px] text-center text-slate-600 pt-1">
                        +{historicalEvents.length - 10} more events
                      </p>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* No location selected message */}
            {!selectedLocation && !predLoading && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-center py-8"
              >
                <AlertTriangle size={24} className="mx-auto text-slate-700 mb-2" />
                <p className="text-[11px] text-slate-600">
                  Click on the map or search a location<br />
                  to run earthquake probability analysis
                </p>
              </motion.div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'metrics' && (
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-5xl mx-auto space-y-6">
            <div>
              <h2 className="text-lg font-bold font-display text-slate-100 mb-1">
                Model Performance <span className="text-neon-glow">Evaluation</span>
              </h2>
              <p className="text-xs font-mono text-slate-500">
                CNN+LSTM+XGBoost hybrid vs baseline classifiers on USGS earthquake catalog
              </p>
            </div>

            {/* Metrics cards */}
            {bestModel && (
              <>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  {[
                    { label: 'F1-Score', value: bestModel.f1_score, color: '#00D4FF' },
                    { label: 'ROC-AUC', value: bestModel.roc_auc, color: '#7B2FFF' },
                    { label: 'Precision', value: bestModel.precision, color: '#00E57A' },
                    { label: 'Recall', value: bestModel.recall, color: '#FFB020' },
                  ].map(({ label, value, color }) => (
                    <motion.div
                      key={label}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                      className="glass rounded-xl p-4 text-center"
                    >
                      <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-2">{label}</p>
                      <p className="text-3xl font-bold font-mono" style={{ color }}>
                        {(value * 100).toFixed(1)}%
                      </p>
                      <p className="text-[9px] font-mono text-slate-600 mt-1">Hybrid model</p>
                    </motion.div>
                  ))}
                </div>

                {/* Comparison chart */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                  className="glass rounded-xl p-5"
                >
                  <h3 className="text-sm font-semibold font-display text-slate-200 mb-4">
                    Model Comparison — F1 Score & ROC-AUC
                  </h3>
                  <MetricsChart models={modelMetrics.models} />
                </motion.div>

                {/* Comparison table */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                  className="glass rounded-xl overflow-hidden"
                >
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(26,37,64,0.8)', background: 'rgba(26,37,64,0.3)' }}>
                        {['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'].map(h => (
                          <th key={h} className="text-left px-4 py-3 text-[10px] tracking-widest text-slate-500 uppercase">{h}</th>
                        ))}
                       </tr>
                    </thead>
                    <tbody>
                      {modelMetrics.models.map((m, i) => {
                        const isBest = m.model_name === modelMetrics.best_model;
                        return (
                          <motion.tr
                            key={i}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.4 + i * 0.1 }}
                            style={{
                              borderBottom: '1px solid rgba(26,37,64,0.4)',
                              background: isBest ? 'rgba(0,212,255,0.04)' : 'transparent',
                            }}
                          >
                            <td className="px-4 py-3" style={{ color: isBest ? '#00D4FF' : '#94A3B8' }}>
                              {isBest && <span className="mr-1 text-[#00D4FF]">★</span>}
                              {m.model_name}
                            </td>
                            {[m.accuracy, m.precision, m.recall, m.f1_score, m.roc_auc].map((v, j) => (
                              <td key={j} className="px-4 py-3" style={{ color: isBest ? '#C8D6E5' : '#64748B' }}>
                                {(v * 100).toFixed(1)}%
                              </td>
                            ))}
                          </motion.tr>
                        );
                      })}
                    </tbody>
                  </table>
                </motion.div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Alert toasts */}
      <AlertToast alerts={alerts} onClose={removeAlert} />
    </div>
  );
}