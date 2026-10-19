import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReplayMap from "./ReplayMap.jsx";
import SeasonStrip from "./SeasonStrip.jsx";
import DayPanel from "./DayPanel.jsx";
import { fmtDay, getJSON } from "../lib.js";

// api: folder of day files; expand: turns a day file into per-cell objects (the main API already sends them).
export default function Replay({ ov, api = "api", expand = (d) => d }) {
  const dates = ov.timeline.dates;
  // Busiest days (by fires discovered) as quick jumps.
  const busiest = useMemo(() => dates.map((d, i) => ({ d, f: ov.timeline.fires[i] }))
    .sort((a, b) => b.f - a.f).slice(0, 6).sort((a, b) => (a.d < b.d ? -1 : 1)), [ov]);

  const [date, setDate] = useState(() => {
    const fromHash = location.hash.match(/d=(\d{4}-\d{2}-\d{2})/)?.[1];
    return fromHash && dates.includes(fromHash) ? fromHash : busiest[busiest.length - 1].d;
  });
  const [day, setDay] = useState(null);
  const [selected, setSelected] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(300);
  const cache = useRef(new Map());

  useEffect(() => {
    let current = true;
    const cached = cache.current.get(date);
    (cached ? Promise.resolve(cached) : getJSON(`${api}/day/${date}.json`).then((raw) => expand(raw))).then((d) => {
      cache.current.set(date, d);
      if (cache.current.size > 400) cache.current.delete(cache.current.keys().next().value);
      if (current) setDay(d);
    });
    history.replaceState(null, "", `#d=${date}`);
    return () => { current = false; };
  }, [date]);

  const step = useCallback((delta) => setDate((d) => {
    const i = dates.indexOf(d) + delta;
    if (i < 0 || i >= dates.length) { setPlaying(false); return d; }
    return dates[i];
  }), [dates]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => step(1), speed);
    return () => clearInterval(timer);
  }, [playing, speed, step]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.matches("input, select")) return;
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === "ArrowRight") step(1);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [step]);

  const select = useCallback((id) => setSelected((s) => (s === id ? null : id)), []);
  const jump = (d) => { setPlaying(false); setDate(d); };
  const k = ov.metrics.review_k, badges = ov.metrics.badge_k ?? k;

  return (
    <section id="replay" className="section">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Replay map</p>
          <h2>Pick a day. See where it said to look.</h2>
          <p className="muted">The glow shows the model's risk, issued the evening before. {badges === k
            ? `Numbered badges are the top ${k} areas a crew would check.`
            : `Numbered badges are the top ${badges} of the ${k} cells a crew would check.`} Dots are the fires actually discovered that day.</p>
        </div>

        <div className="controls card">
          <div className="control-row">
            <button className="btn btn-icon" aria-label="Previous day" onClick={() => step(-1)}>&lsaquo;</button>
            <input type="date" aria-label="Date" min={ov.dates.first} max={ov.dates.last} value={date}
              onChange={(e) => dates.includes(e.target.value) && setDate(e.target.value)} />
            <button className="btn btn-icon" aria-label="Next day" onClick={() => step(1)}>&rsaquo;</button>
            <button className="btn btn-primary" aria-pressed={playing} onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play season"}</button>
            <label className="speed">Speed
              <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
                <option value="700">Slow</option>
                <option value="300">Normal</option>
                <option value="120">Fast</option>
              </select>
            </label>
          </div>
          <div className="jump">
            <span className="muted small">Jump to a busy day:</span>{" "}
            <span id="jumps">
              {busiest.map((b) => (
                <button key={b.d} onClick={() => jump(b.d)}>{fmtDay(b.d, { month: "short", day: "numeric", year: "numeric" })} · {b.f} fires</button>
              ))}
            </span>
          </div>
        </div>

        <div className="replay-grid">
          <div className="map-col">
            <ReplayMap ov={ov} day={day} selected={selected} onSelect={select} duration={playing ? Math.min(220, speed * 0.7) : 380} />
            <div className="legend">
              <span className="ramp-wrap">Lower risk <i className="ramp"></i> Higher risk</span>
              <span><i className="fire-dot"></i>Actual fire</span>
              <span><i className="rank-mini">1</i>Top {badges} picks</span>
              <span><i className="fade-swatch"></i>Faded = outside model area</span>
            </div>
            <SeasonStrip timeline={ov.timeline} date={date} onPick={jump} k={k} />
          </div>
          <DayPanel ov={ov} day={day} selected={selected} onSelect={select} />
        </div>
      </div>
    </section>
  );
}
