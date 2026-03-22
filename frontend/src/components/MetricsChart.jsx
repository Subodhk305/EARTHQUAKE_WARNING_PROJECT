import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

const COLORS = {
  'Logistic Regression': '#4A5568',
  'Random Forest': '#4299E1',
  'LSTM Only': '#9F7AEA',
  'CNN+LSTM+XGBoost (Ours)': '#00D4FF',
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass rounded-lg p-3 text-xs font-mono border border-[#1A2540]">
      <p className="text-neon mb-2 font-bold">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {(p.value * 100).toFixed(1)}%
        </p>
      ))}
    </div>
  );
};

export default function MetricsChart({ models = [] }) {
  const chartData = models.map(m => ({
    name: m.model_name.replace('CNN+LSTM+XGBoost (Ours)', 'Hybrid (Ours)'),
    F1: m.f1_score,
    AUC: m.roc_auc,
    Precision: m.precision,
    Recall: m.recall,
    fullName: m.model_name,
  }));

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,37,64,0.6)" vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fill: '#64748B', fontSize: 9, fontFamily: 'JetBrains Mono' }}
            axisLine={{ stroke: '#1A2540' }}
            tickLine={false}
          />
          <YAxis
            domain={[0.6, 1.0]}
            tick={{ fill: '#64748B', fontSize: 9, fontFamily: 'JetBrains Mono' }}
            axisLine={false}
            tickLine={false}
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono', color: '#64748B' }}
          />
          <Bar dataKey="F1" fill="#00D4FF" radius={[3, 3, 0, 0]} maxBarSize={24}>
            {chartData.map((entry) => (
              <Cell
                key={entry.name}
                fill={entry.fullName === 'CNN+LSTM+XGBoost (Ours)' ? '#00D4FF' : '#1A3A5C'}
                opacity={entry.fullName === 'CNN+LSTM+XGBoost (Ours)' ? 1 : 0.7}
              />
            ))}
          </Bar>
          <Bar dataKey="AUC" fill="#7B2FFF" radius={[3, 3, 0, 0]} maxBarSize={24} opacity={0.8} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
