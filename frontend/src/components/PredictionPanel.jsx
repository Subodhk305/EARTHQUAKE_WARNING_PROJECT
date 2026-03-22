import { motion, AnimatePresence } from 'framer-motion';
import { Activity, MapPin, Cpu, Clock, AlertTriangle, Shield, TrendingUp, Layers } from 'lucide-react';
import ProbabilityGauge from './ProbabilityGauge';
import ConfidenceMeter from './ConfidenceMeter';

const MAG_INFO = {
  micro:    { label: 'Micro',    color: '#4A5568', desc: '< 2.0 — Not felt' },
  minor:    { label: 'Minor',    color: '#4299E1', desc: '2.0–3.9 — Rarely felt' },
  moderate: { label: 'Moderate', color: '#68D391', desc: '4.0–4.9 — Felt widely' },
  strong:   { label: 'Strong',   color: '#FFB020', desc: '5.0–5.9 — Damage possible' },
  major:    { label: 'Major',    color: '#FC8181', desc: '6.0–6.9 — Serious damage' },
  great:    { label: 'Great',    color: '#FF3B5C', desc: '≥ 7.0 — Devastating' },
};

const RISK_ICON = {
  High:   <AlertTriangle size={14} className="text-[#FF3B5C]" />,
  Medium: <Activity size={14} className="text-[#FFB020]" />,
  Low:    <Shield size={14} className="text-[#00E57A]" />,
};

// Helper function to safely format magnitude values
const formatMagnitude = (mag) => {
  if (mag === null || mag === undefined) return 'N/A';
  if (typeof mag === 'number') return mag.toFixed(1);
  if (typeof mag === 'string') return mag;
  return 'N/A';
};

// Helper function to safely format numbers with toFixed
const safeToFixed = (value, digits = 0, suffix = '') => {
  if (value === null || value === undefined) return 'N/A';
  if (typeof value === 'number') return value.toFixed(digits) + suffix;
  if (typeof value === 'string') return value + suffix;
  return 'N/A';
};

function StatBadge({ icon, label, value, color }) {
  return (
    <div className="flex items-center gap-2 p-3 rounded-lg" style={{ background: 'rgba(26,37,64,0.5)' }}>
      <div style={{ color: color || '#00D4FF' }}>{icon}</div>
      <div>
        <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">{label}</p>
        <p className="text-sm font-mono font-semibold" style={{ color: color || '#C8D6E5' }}>{value}</p>
      </div>
    </div>
  );
}

export default function PredictionPanel({ result, loading, location }) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 py-16">
        <div className="relative w-12 h-12">
          <div className="absolute inset-0 rounded-full border-2 border-[#00D4FF] opacity-20 animate-ping" />
          <div className="absolute inset-2 rounded-full border-2 border-[#7B2FFF] animate-spin" />
          <Cpu size={16} className="absolute inset-0 m-auto text-[#00D4FF]" />
        </div>
        <div className="text-center">
          <p className="text-sm font-mono text-[#00D4FF] text-neon-glow">Analyzing seismic data</p>
          <p className="text-xs font-mono text-slate-500 mt-1">Fetching IRIS waveforms...</p>
        </div>
      </div>
    );
  }

  if (!result && !location) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 py-16 text-center px-4">
        <div className="w-16 h-16 rounded-full border border-[#1A2540] flex items-center justify-center">
          <MapPin size={24} className="text-slate-600" />
        </div>
        <p className="text-sm font-mono text-slate-400">Click on the map or search a location</p>
        <p className="text-xs font-mono text-slate-600">to run earthquake probability analysis</p>
      </div>
    );
  }

  if (!result && location) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 py-16 text-center px-4">
        <div className="relative w-12 h-12">
          <div className="absolute inset-0 rounded-full border-2 border-[#00D4FF] opacity-20 animate-pulse" />
          <MapPin size={20} className="absolute inset-0 m-auto text-[#00D4FF]" />
        </div>
        <p className="text-sm font-mono text-slate-400">Fetching prediction for</p>
        <p className="text-xs font-mono text-[#00D4FF]">
          {location.name || `${location.latitude?.toFixed(4) || '0'}°, ${location.longitude?.toFixed(4) || '0'}°`}
        </p>
      </div>
    );
  }

  // Check if result has valid data
  if (!result || typeof result.probability !== 'number' || isNaN(result.probability)) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 py-16 text-center px-4">
        <AlertTriangle size={24} className="text-red-500" />
        <p className="text-sm font-mono text-red-400">Invalid prediction data</p>
        <p className="text-xs font-mono text-slate-500">Please try again or select another location</p>
      </div>
    );
  }

  // Get magnitude class from result
  const magnitudeClass = result.predicted_magnitude_class?.toLowerCase() || 'minor';
  const mag = MAG_INFO[magnitudeClass] || MAG_INFO.minor;

  // Calculate risk level if not provided
  const riskLevel = result.risk_level || (
    result.probability >= 0.7 ? 'High' :
    result.probability >= 0.4 ? 'Medium' : 'Low'
  );

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={result.request_id || Date.now()}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="space-y-4">

        {/* Location header */}
        <div className="flex items-start gap-2">
          <MapPin size={14} className="text-[#00D4FF] mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-semibold text-slate-200 leading-tight">
              {result.location || location?.name || 'Selected Location'}
            </p>
            <p className="text-[10px] font-mono text-slate-500">
              {result.latitude?.toFixed(4) || location?.latitude?.toFixed(4) || '0'}°, {result.longitude?.toFixed(4) || location?.longitude?.toFixed(4) || '0'}°
            </p>
          </div>
        </div>

        {/* Gauge */}
        <div className="flex justify-center py-2">
          <ProbabilityGauge
            probability={result.probability}
            riskLevel={riskLevel}
          />
        </div>

        {/* Magnitude class */}
        <div
          className="rounded-xl p-4 text-center"
          style={{
            background: `${mag.color}10`,
            border: `1px solid ${mag.color}40`,
          }}>
          <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-1">
            Predicted Magnitude Class
          </p>
          <p className="text-2xl font-bold font-display" style={{ color: mag.color }}>
            {mag.label}
          </p>
          <p className="text-xs font-mono text-slate-400 mt-1">{mag.desc}</p>
          {result.magnitude_estimate && (
            <p className="text-xs font-mono mt-1" style={{ color: mag.color }}>
              M {formatMagnitude(result.magnitude_estimate)}
            </p>
          )}
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-2 gap-2">
          <StatBadge
            icon={RISK_ICON[riskLevel]}
            label="Risk Level"
            value={riskLevel}
            color={riskLevel === 'High' ? '#FF3B5C' : riskLevel === 'Medium' ? '#FFB020' : '#00E57A'}
          />
          <StatBadge
            icon={<TrendingUp size={14} />}
            label="Seismicity"
            value={result.recent_seismicity_score ? `${(result.recent_seismicity_score * 100).toFixed(0)}%` : 'N/A'}
            color="#7B2FFF"
          />
          <StatBadge
            icon={<Layers size={14} />}
            label="Nearby M3+ Events"
            value={safeToFixed(result.nearby_active_faults, 0)}
          />
          <StatBadge
            icon={<Clock size={14} />}
            label="Processing"
            value={safeToFixed(result.processing_time_ms, 0, 'ms')}
          />
        </div>

        {/* Confidence meter */}
        <ConfidenceMeter confidence={result.confidence || 0.85} label="Model Confidence" />

        {/* Timestamp + model version */}
        <div className="flex justify-between items-center pt-1">
          <span className="text-[9px] font-mono text-slate-600">
            v{result.model_version || '1.0.0'} • CNN+LSTM+XGBoost
          </span>
          <span className="text-[9px] font-mono text-slate-600">
            {result.timestamp ? new Date(result.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString()}
          </span>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}