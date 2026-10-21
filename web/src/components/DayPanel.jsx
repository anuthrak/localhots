import { RISK, fires, fmtDay, name, pct, riskColor } from "../lib.js";

// forecast: the live forecast's metadata; without it the panel describes a replay day and how it turned out.
export default function DayPanel({ ov, day, selected, onSelect, forecast }) {
  if (!day) {
    return (
      <aside className="panel">
        <div className="card day-card"><p className="eyebrow">Issued —</p><h3>Loading…</h3></div>
      </aside>
    );
  }
  const k = ov.metrics.review_k;
  const shown = ov.metrics.watch_k ?? k; // long crew lists show only their top few here
  const scale = ov.metrics.heat_scale ?? 1;
  const s = day.summary;
  const verdict = s.fire_cells === 0 ? "" : s.caught > 0 ? "good" : "bad";
  const cell = day.cells.find((c) => c.id === selected);

  return (
    <aside className="panel" aria-live="polite">
      <div className="card day-card">
        <p className="eyebrow">{forecast ? "Forecast" : "Outlook"} issued evening of {fmtDay(day.issued, { weekday: "short", month: "short", day: "numeric" })}</p>
        <h3>{fmtDay(day.date, { weekday: "long", month: "long", day: "numeric", year: "numeric" })}</h3>
        {forecast ? (
          <p className="small muted">Weather: gridMET through {fmtDay(forecast.gridmet_through, { month: "short", day: "numeric" })}, then the Open-Meteo forecast. Scored by the lightning model each evening.</p>
        ) : (
          <>
            <div className="verdict">
              <div><b>{s.fires}</b><span>fires discovered</span></div>
              <div><b>{s.fire_cells}</b><span>areas with fire</span></div>
              <div className={verdict}><b>{s.caught}/{k}</b><span>top picks with fire</span></div>
            </div>
            <p className="tiny muted">
              {ov.region
                ? `Scored ahead of time by the ${ov.region.step}° model; this page shows the saved scores.`
                : ov.runtime.snapshot
                  ? `Scored by the model on IBM LinuxONE (${ov.runtime.machine}); this site shows the saved scores.`
                  : `Scored live on ${day.machine} in ${day.live_ms.toFixed(1)} ms.`}
            </p>
          </>
        )}
      </div>

      {shown < k && <p className="small muted">Top {shown} of {forecast ? "tomorrow" : "today"}'s {k}-area crew list:</p>}
      <div className="watch">
        {day.cells.slice(0, shown).map((c) => (
          <button key={c.id} className={`watch-item${selected === c.id ? " active" : ""}`} onClick={() => onSelect(c.id)}>
            <span className="rank-badge">{c.rank}</span>
            <span className="name">{name(c.id)}</span>
            <span className="score" style={{ background: riskColor(c.score * scale), color: c.score * scale >= RISK[2] ? "#fff" : "#111" }}>{c.score.toFixed(2)}</span>
            <ul>{(c.reasons.length ? c.reasons : ["Ranked high mostly on its fire history for this month"]).map((r) => <li key={r}>{r}</li>)}</ul>
            {!forecast && <span className={`result ${c.fires ? "hit" : "miss"}`}>{c.fires ? `${fires(c.fires)} discovered here` : "No fire recorded here"}</span>}
          </button>
        ))}
      </div>

      {cell && <CellCard c={cell} ov={ov} forecast={forecast} />}
    </aside>
  );
}

function Anomaly({ z, invert = false }) {
  if (z == null || Math.abs(z) < 1) return null;
  const up = z > 0;
  const bad = invert ? !up : up;
  return <em className={bad ? "up" : "down"} style={{ display: "block", fontStyle: "normal", fontSize: 11 }}>{Math.abs(z).toFixed(1)}σ {up ? "above" : "below"} normal</em>;
}

function CellCard({ c, ov, forecast }) {
  const w = c.weather, a = c.anomaly;
  // The finer-grid page uses gridMET, whose wind is the day's average rather than its peak.
  const alert = ov.region ? " · on the crew list" : " · above alert line";
  return (
    <div className="card cell-card">
      <p className="eyebrow">Area {c.id.toUpperCase()}</p>
      <h4>{name(c.id)}</h4>
      <p className="small muted">Rank {c.rank} of {ov.cells.length} · score {c.score.toFixed(3)}{c.alert ? alert : ""}</p>
      <div className="wx">
        <div><span>Yesterday's high</span><b>{w.tmax.toFixed(1)} °C</b><Anomaly z={a.heat} /></div>
        <div><span>Lowest humidity</span><b>{w.rh.toFixed(0)}%</b><Anomaly z={a.humidity} invert /></div>
        <div><span>{ov.region ? "Average wind" : "Max wind"}</span><b>{w.wind.toFixed(1)} m/s</b><Anomaly z={a.wind} /></div>
        <div><span>Rain, last 30 days</span><b>{w.rain30.toFixed(1)} mm</b><Anomaly z={a.rain30} invert /></div>
        <div><span>Dry days in a row</span><b>{w.dry_days >= 30 ? "30+" : w.dry_days}</b></div>
        <div><span>Past fire rate (this month)</span><b>{pct(c.past_rate)}</b></div>
      </div>
      {!forecast && <p className={`small ${c.fires ? "hit" : "miss"}`} style={{ marginTop: 12 }}>{c.fires ? `${fires(c.fires)} discovered here that day` : "No fire recorded here that day"}</p>}
    </div>
  );
}
