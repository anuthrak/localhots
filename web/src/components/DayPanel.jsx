import { RISK, fires, fmtDay, name, pct, riskColor } from "../lib.js";

export default function DayPanel({ ov, day, selected, onSelect }) {
  if (!day) {
    return (
      <aside className="panel">
        <div className="card day-card"><p className="eyebrow">Issued —</p><h3>Loading…</h3></div>
      </aside>
    );
  }
  const k = ov.metrics.review_k;
  const s = day.summary;
  const verdict = s.fire_cells === 0 ? "" : s.caught > 0 ? "good" : "bad";
  const cell = day.cells.find((c) => c.id === selected);

  return (
    <aside className="panel" aria-live="polite">
      <div className="card day-card">
        <p className="eyebrow">Outlook issued evening of {fmtDay(day.issued, { weekday: "short", month: "short", day: "numeric" })}</p>
        <h3>{fmtDay(day.date, { weekday: "long", month: "long", day: "numeric", year: "numeric" })}</h3>
        <div className="verdict">
          <div><b>{s.fires}</b><span>fires discovered</span></div>
          <div><b>{s.fire_cells}</b><span>areas with fire</span></div>
          <div className={verdict}><b>{s.caught}/{k}</b><span>top picks with fire</span></div>
        </div>
        <p className="tiny muted">
          {ov.runtime.snapshot
            ? `Scored by the model on IBM LinuxONE (${ov.runtime.machine}); this site shows the saved scores.`
            : `Scored live on ${day.machine} in ${day.live_ms.toFixed(1)} ms.`}
        </p>
      </div>

      <div className="watch">
        {day.cells.slice(0, k).map((c) => (
          <button key={c.id} className={`watch-item${selected === c.id ? " active" : ""}`} onClick={() => onSelect(c.id)}>
            <span className="rank-badge">{c.rank}</span>
            <span className="name">{name(c.id)}</span>
            <span className="score" style={{ background: riskColor(c.score), color: c.score >= RISK[2] ? "#fff" : "#111" }}>{c.score.toFixed(2)}</span>
            <ul>{(c.reasons.length ? c.reasons : ["Ranked high mostly on its fire history for this month"]).map((r) => <li key={r}>{r}</li>)}</ul>
            <span className={`result ${c.fires ? "hit" : "miss"}`}>{c.fires ? `${fires(c.fires)} discovered here` : "No fire recorded here"}</span>
          </button>
        ))}
      </div>

      {cell && <CellCard c={cell} />}
    </aside>
  );
}

function Anomaly({ z, invert = false }) {
  if (z == null || Math.abs(z) < 1) return null;
  const up = z > 0;
  const bad = invert ? !up : up;
  return <em className={bad ? "up" : "down"} style={{ display: "block", fontStyle: "normal", fontSize: 11 }}>{Math.abs(z).toFixed(1)}σ {up ? "above" : "below"} normal</em>;
}

function CellCard({ c }) {
  const w = c.weather, a = c.anomaly;
  return (
    <div className="card cell-card">
      <p className="eyebrow">Area {c.id.toUpperCase()}</p>
      <h4>{name(c.id)}</h4>
      <p className="small muted">Rank {c.rank} of 24 · score {c.score.toFixed(3)}{c.alert ? " · above alert line" : ""}</p>
      <div className="wx">
        <div><span>Yesterday's high</span><b>{w.tmax.toFixed(1)} °C</b><Anomaly z={a.heat} /></div>
        <div><span>Lowest humidity</span><b>{w.rh.toFixed(0)}%</b><Anomaly z={a.humidity} invert /></div>
        <div><span>Max wind</span><b>{w.wind.toFixed(1)} m/s</b><Anomaly z={a.wind} /></div>
        <div><span>Rain, last 30 days</span><b>{w.rain30.toFixed(1)} mm</b><Anomaly z={a.rain30} invert /></div>
        <div><span>Dry days in a row</span><b>{w.dry_days >= 30 ? "30+" : w.dry_days}</b></div>
        <div><span>Past fire rate (this month)</span><b>{pct(c.past_rate)}</b></div>
      </div>
      <p className={`small ${c.fires ? "hit" : "miss"}`} style={{ marginTop: 12 }}>{c.fires ? `${fires(c.fires)} discovered here that day` : "No fire recorded here that day"}</p>
    </div>
  );
}
