import { useEffect, useRef } from 'react';

const SIZE = 200;
const STROKE = 14;
const R = (SIZE - STROKE) / 2;
const CIRCUMFERENCE = 2 * Math.PI * R;
// Arc is 240 degrees (2/3 of circle), starting from 150°
const ARC = (CIRCUMFERENCE * 240) / 360;

function getRiskColor(riskLevel) {
  switch (riskLevel) {
    case 'High':   return '#FF3B5C';
    case 'Medium': return '#FFB020';
    default:       return '#00E57A';
  }
}

export default function ProbabilityGauge({ probability = 0, riskLevel = 'Low', animated = true }) {
  const arcRef = useRef(null);
  const color = getRiskColor(riskLevel);
  const offset = ARC - ARC * Math.min(1, Math.max(0, probability));

  useEffect(() => {
    if (!arcRef.current || !animated) return;
    arcRef.current.style.setProperty('--target-offset', offset);
    arcRef.current.style.strokeDashoffset = ARC;
    // Force reflow
    void arcRef.current.offsetWidth;
    arcRef.current.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1)';
    arcRef.current.style.strokeDashoffset = offset;
  }, [probability, offset, animated]);

  const pct = Math.round(probability * 100);

  return (
    <div className="relative flex flex-col items-center gap-2">
      <svg width={SIZE} height={SIZE + 10} viewBox={`0 0 ${SIZE} ${SIZE + 10}`}>
        {/* Shadow filter */}
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background arc track */}
        <circle
          cx={SIZE / 2} cy={SIZE / 2}
          r={R}
          fill="none"
          stroke="rgba(26,37,64,0.9)"
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={`${ARC} ${CIRCUMFERENCE}`}
          strokeDashoffset={0}
          transform={`rotate(150 ${SIZE / 2} ${SIZE / 2})`}
        />

        {/* Value arc */}
        <circle
          ref={arcRef}
          cx={SIZE / 2} cy={SIZE / 2}
          r={R}
          fill="none"
          stroke={color}
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={`${ARC} ${CIRCUMFERENCE}`}
          strokeDashoffset={offset}
          transform={`rotate(150 ${SIZE / 2} ${SIZE / 2})`}
          filter="url(#glow)"
          style={{ transition: 'stroke 0.5s' }}
        />

        {/* Center text */}
        <text x={SIZE / 2} y={SIZE / 2 - 6}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={color}
          fontSize="36"
          fontWeight="700"
          fontFamily="'JetBrains Mono', monospace"
          filter="url(#glow)">
          {pct}%
        </text>
        <text x={SIZE / 2} y={SIZE / 2 + 22}
          textAnchor="middle"
          fill="rgba(200,214,229,0.5)"
          fontSize="10"
          fontFamily="'JetBrains Mono', monospace"
          letterSpacing="2">
          PROBABILITY
        </text>
      </svg>

      {/* Risk badge */}
      <div
        className="px-4 py-1 rounded-full text-xs font-mono font-semibold tracking-widest uppercase"
        style={{
          border: `1px solid ${color}`,
          color,
          boxShadow: `0 0 12px ${color}40`,
          background: `${color}15`,
        }}>
        {riskLevel} Risk
      </div>
    </div>
  );
}
