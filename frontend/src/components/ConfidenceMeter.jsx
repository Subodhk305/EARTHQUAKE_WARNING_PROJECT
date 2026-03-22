import { useEffect, useRef } from 'react';

export default function ConfidenceMeter({ confidence = 0, label = 'Model Confidence' }) {
  const barRef = useRef(null);
  const pct = Math.round(confidence * 100);

  useEffect(() => {
    if (!barRef.current) return;
    barRef.current.style.width = '0%';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        barRef.current.style.width = `${pct}%`;
      });
    });
  }, [pct]);

  const color = pct >= 80 ? '#00D4FF' : pct >= 60 ? '#7B2FFF' : '#FFB020';

  return (
    <div className="w-full space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-xs font-mono text-slate-400 tracking-widest uppercase">{label}</span>
        <span className="text-sm font-mono font-bold" style={{ color }}>{pct}%</span>
      </div>
      <div className="h-2 rounded-full bg-[#1A2540] overflow-hidden relative">
        <div
          ref={barRef}
          className="h-full rounded-full transition-all duration-1000 ease-out"
          style={{
            background: `linear-gradient(90deg, ${color}80, ${color})`,
            boxShadow: `0 0 10px ${color}60`,
            width: 0,
          }}
        />
        {/* Tick marks */}
        {[25, 50, 75].map(tick => (
          <div
            key={tick}
            className="absolute top-0 bottom-0 w-px bg-[#0B0F19]"
            style={{ left: `${tick}%` }}
          />
        ))}
      </div>
      <div className="flex justify-between">
        {[0, 25, 50, 75, 100].map(v => (
          <span key={v} className="text-[9px] font-mono text-slate-600">{v}</span>
        ))}
      </div>
    </div>
  );
}
