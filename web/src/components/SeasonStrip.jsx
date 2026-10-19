import { useMemo } from "react";

// One bar per day of the selected date's year: fires discovered, orange when the top 3 caught one.
export default function SeasonStrip({ timeline: t, date, onPick }) {
  const year = date.slice(0, 4);
  const idx = useMemo(() => t.dates.map((d, i) => (d.startsWith(year) ? i : -1)).filter((i) => i >= 0), [t, year]);
  const max = Math.max(...idx.map((i) => t.fires[i]), 1);
  const cursor = t.dates.indexOf(date) - idx[0];

  const pick = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const i = Math.floor(((e.clientX - r.left) / r.width) * idx.length);
    const d = t.dates[idx[0] + Math.max(0, i)];
    if (d) onPick(d);
  };

  return (
    <div className="card strip-card">
      <div className="strip-head">
        <span className="small"><b>{year}</b> · fires discovered per day</span>
        <span className="small muted"><i className="sw sw-hit"></i>top 3 caught a fire <i className="sw sw-miss"></i>missed</span>
      </div>
      <svg className="strip" viewBox={`0 0 ${idx.length} 100`} preserveAspectRatio="none" role="img" aria-label="Fires per day this year; click to jump" onClick={pick}>
        {idx.map((i, x) => {
          const h = t.fires[i] ? Math.max(6, (t.fires[i] / max) * 92) : 0;
          return h ? (
            <rect key={i} x={x + 0.12} y={100 - h} width="0.76" height={h} fill={t.caught[i] > 0 ? "var(--phoenix)" : "var(--stone)"}>
              <title>{`${t.dates[i]}: ${t.fires[i]} fires`}</title>
            </rect>
          ) : null;
        })}
        <rect x={cursor} y="0" width="1" height="100" fill="rgba(17,17,17,0.18)" />
      </svg>
    </div>
  );
}
