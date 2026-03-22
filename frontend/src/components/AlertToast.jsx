import { useEffect, useState } from 'react';
import { AlertTriangle, X, MapPin, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function AlertToast({ alerts, onClose }) {
  const [visible, setVisible] = useState([]);

  useEffect(() => {
    if (!alerts.length) return;
    
    // Add unique IDs to alerts
    const alertsWithIds = alerts.map(alert => ({
      ...alert,
      id: alert.id || Date.now() + Math.random()
    }));
    
    setVisible(prev => {
      const combined = [...alertsWithIds, ...prev];
      // Keep only last 5 unique alerts
      const unique = Array.from(new Map(combined.map(a => [a.id, a])).values());
      return unique.slice(0, 5);
    });

    // Auto-remove alerts after 8 seconds
    alertsWithIds.forEach(alert => {
      const timer = setTimeout(() => {
        setVisible(prev => prev.filter(a => a.id !== alert.id));
        if (onClose) onClose(alert.id);
      }, 8000);
      return () => clearTimeout(timer);
    });
  }, [alerts, onClose]);

  const handleClose = (id) => {
    setVisible(prev => prev.filter(a => a.id !== id));
    if (onClose) onClose(id);
  };

  const riskColors = {
    High: { border: '#FF3B5C', bg: 'rgba(255,59,92,0.08)', icon: '#FF3B5C' },
    Medium: { border: '#FFB020', bg: 'rgba(255,176,32,0.08)', icon: '#FFB020' },
    Low: { border: '#00E57A', bg: 'rgba(0,229,122,0.08)', icon: '#00E57A' },
  };

  const defaultColors = { border: '#00D4FF', bg: 'rgba(0,212,255,0.08)', icon: '#00D4FF' };

  return (
    <div className="fixed top-6 right-6 z-50 space-y-3 pointer-events-none">
      <AnimatePresence>
        {visible.map(alert => {
          const colors = riskColors[alert.risk_level] || 
                        (alert.severity === 'high' ? riskColors.High :
                         alert.severity === 'medium' ? riskColors.Medium :
                         alert.severity === 'low' ? riskColors.Low : defaultColors);
          
          const probability = alert.probability || 0;
          const probabilityPercent = (probability * 100).toFixed(1);
          
          return (
            <motion.div
              key={alert.id}
              initial={{ x: 400, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 400, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              className="pointer-events-auto w-80 rounded-xl p-4 relative overflow-hidden"
              style={{
                background: colors.bg,
                border: `1px solid ${colors.border}`,
                boxShadow: `0 0 30px ${colors.border}30`,
              }}>
              {/* Animated scan line */}
              <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <motion.div
                  className="absolute left-0 right-0 h-px opacity-30"
                  style={{ background: colors.border }}
                  animate={{ y: ['0%', '100%'] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                />
              </div>

              <div className="flex items-start gap-3">
                <div className="relative mt-0.5">
                  <AlertTriangle size={18} style={{ color: colors.icon }} />
                  {alert.risk_level === 'High' && (
                    <span
                      className="absolute inset-0 rounded-full animate-ping"
                      style={{ background: colors.icon, opacity: 0.3 }}
                    />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-mono font-bold tracking-widest uppercase"
                      style={{ color: colors.icon }}>
                      {alert.risk_level || alert.type || 'Alert'}
                    </span>
                    <button
                      onClick={() => handleClose(alert.id)}
                      className="text-slate-500 hover:text-slate-300 transition-colors">
                      <X size={14} />
                    </button>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-slate-400 mb-2">
                    <MapPin size={10} />
                    <span className="truncate">{alert.location || 'Unknown location'}</span>
                  </div>
                  <div className="text-sm text-slate-200 leading-relaxed">
                    {!isNaN(probability) ? `${probabilityPercent}% probability — ` : ''}
                    <span className="capitalize font-semibold">
                      {alert.magnitude_class || alert.message || 'Event'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="mt-3 flex items-center gap-2">
                <Zap size={10} style={{ color: colors.icon }} />
                <span className="text-[10px] font-mono text-slate-500">
                  {alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString()}
                </span>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}