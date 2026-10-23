import { useCallback, useEffect, useMemo, useState } from "react";
import ChartBox, { grid } from "./ChartBox.jsx";
import DayPanel from "./DayPanel.jsx";
import ReplayMap from "./ReplayMap.jsx";
import { FORECAST_URL, RISK, TRACK_URL, expandDay, fmtDay, name, nf, pct } from "../lib.js";

const LINKS = [
  ["#replay", "Map"], ["#findings", "What we found"], ["#how", "How it works"], ["#linuxone", "LinuxONE"], ["#forecast", "Live forecast"],
];
const CTA = ["https://github.com/anuthrak/localhots", "Code on GitHub"];

const km = (deg, lat) => Math.round(deg * 111.3 * (lat === undefined ? 1 : Math.cos((lat * Math.PI) / 180)));

// base: the datathon's 24-area overview, which carries the LinuxONE runtime numbers.
export function Nav({ base, error }) {
  const r = base?.runtime;
  return (
    <header className="nav">
      <div className="wrap nav-inner">
        <a href="#top" className="logo"><span className="flame" aria-hidden="true"></span>localhots</a>
        <nav aria-label="Sections">
          {LINKS.map(([href, label]) => <a key={href} href={href}>{label}</a>)}
        </nav>
        <span className={`chip chip-live${r ? " ok" : ""}`}>
          <span className="dot"></span>
          {error ? "Data unavailable" : !r ? "Connecting…" : r.snapshot ? `Built on IBM LinuxONE · ${r.machine}` : `Live server · ${r.machine}`}
        </span>
        <a href={CTA[0]} className="btn btn-primary nav-cta">{CTA[1]}</a>
      </div>
    </header>
  );
}

export function Hero({ all, lightning }) {
  const m = lightning?.metrics;
  return (
    <section className="hero">
      <div className="wrap">
        <p className="eyebrow">Started at IBM Z Datathon 2026 · Northern California</p>
        <h1 className="display">Where will the next fire start?</h1>
        <p className="lede">We trained a model on 13 years of wildfires and weather to rank where a new fire is most likely <em>tomorrow</em>. It turns out each place's fire history explains most fires, because people start them where people are. Weather adds real value for one kind: fires started by lightning.</p>
        <div className="hero-actions">
          <a href="#replay" className="btn btn-primary">Replay a fire season</a>
          <a href="#findings" className="btn btn-light">What we found</a>
        </div>
        <dl className="hero-stats">
          <div><dt>Fires in the record, 2012–2024</dt><dd>{all ? nf.format(all.data.fire_records_total) : "—"}</dd></div>
          <div><dt>Areas ranked each day</dt><dd>{all?.cells.length ?? "—"}</dd></div>
          <div><dt>Lightning fire days caught checking 1 in 8 areas</dt><dd>{m ? pct(m.review_recall) : "—"}</dd></div>
          <div><dt>With fire history alone · random picks</dt><dd>{m ? `${pct(m.baseline_review_recall)} · ${pct(m.review_random)}` : "—"}</dd></div>
        </dl>
      </div>
    </section>
  );
}

const TYPES = [["human", "Human-caused"], ["unknown", "Cause unknown"], ["large", ["Large fires", "(10+ acres)"]], ["lightning", "Lightning"]];

// ov.experiment rows -> { type: { score: row } }
function byType(rows = []) {
  const out = {};
  for (const r of rows) (out[r.type] ??= {})[r.score] = r;
  return out;
}

// Writes each bar's value just past its end, so the chart reads without hovering.
const barLabels = {
  id: "barLabels",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    ctx.save();
    ctx.font = "500 12px Inter, system-ui, sans-serif";
    ctx.fillStyle = "#111";
    ctx.textBaseline = "middle";
    chart.data.datasets.forEach((ds, i) => chart.getDatasetMeta(i).data.forEach((bar, j) => ctx.fillText(pct(ds.data[j]), bar.x + 6, bar.y)));
    ctx.restore();
  },
};

export function Findings({ lightning }) {
  const t = byType(lightning.experiment);
  const m = lightning.metrics;
  const types = TYPES.filter(([k]) => t[k]);
  const chart = useMemo(() => ({
    type: "bar",
    data: {
      labels: types.map(([, label]) => label),
      datasets: [
        { label: "Fire history only", data: types.map(([k]) => t[k].past_fire_rate.top_k_recall), backgroundColor: "#8a8987", borderRadius: 4, barPercentage: 0.8 },
        { label: "History + weather (model)", data: types.map(([k]) => t[k].model.top_k_recall), backgroundColor: "#e8400d", borderRadius: 4, barPercentage: 0.8 },
      ],
    },
    options: {
      indexAxis: "y",
      plugins: {
        legend: { display: true, position: "top", align: "start", labels: { boxWidth: 12, color: "#111" } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${pct(c.raw)} of fire days caught` } },
      },
      scales: {
        x: { min: 0, max: 0.6, grid, ticks: { callback: (v) => pct(v) }, title: { display: true, text: `Share of fire days caught, checking the riskiest ${m.review_k} of ${lightning.cells.length} areas` } },
        y: { grid: { display: false }, ticks: { color: "#111" } },
      },
    },
    plugins: [barLabels],
  }), [lightning]);
  const L = t.lightning;

  return (
    <section id="findings" className="section band">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">What we found</p>
          <h2>Weather matters for some fires, not others</h2>
          <p className="muted">We trained a separate model for each kind of fire and compared it with the simplest possible guess: rank areas by how often that kind of fire happened there before. If weather helps, the model should beat that guess.</p>
        </div>
        <div className="card">
          <ChartBox config={chart} />
          <p className="tiny muted table-note">Tested on 2022–2024, years no model saw in training. Random picks catch {pct(m.review_random)}. Every fire type uses the same model settings so they compare fairly; the lightning model on the map was tuned on its own and catches {pct(m.review_recall)}.</p>
        </div>
        <div className="limits findings">
          <div className="card"><h3 className="h-sm">Most fires: history wins</h3><p className="muted">People start fires where people are: roads, towns, power lines. Fire history already knows those places, so for human-caused and unknown-cause fires, weather adds nothing.</p></div>
          <div className="card"><h3 className="h-sm">Lightning: weather is the signal</h3><p className="muted">Lightning fires come from dry thunderstorms that roll through on particular days. Fire history can't see those coming, but weather can: {pct(L.past_fire_rate.top_k_recall)} caught with history, {pct(L.model.top_k_recall)} with the model.</p></div>
          <div className="card"><h3 className="h-sm">The official index tells you when, not where</h3><p className="muted">The US fire danger index (ERC) measures how dry fuels are. That tracks the season, but for picking which areas to watch it catches only {pct(L.erc.top_k_recall)} of lightning fire days: no better than random.</p></div>
        </div>
      </div>
    </section>
  );
}

export function HowItWorks({ all }) {
  const step = all.region.step, midLat = (all.region.bounds[1] + all.region.bounds[3]) / 2;
  return (
    <section id="how" className="section">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">How it works</p>
          <h2>From 13 years of fires and weather to a daily watch list</h2>
        </div>
        <div className="steps">
          <article className="tile tone-petal">
            <span className="step-n">01</span>
            <h3>Collect</h3>
            <p><b>{nf.format(all.data.fire_records_total)}</b> fires from the US Forest Service, plus daily weather for {all.cells.length} areas about {km(step)} × {km(step, midLat)} km, from the Bay Area to the Oregon border.</p>
          </article>
          <article className="tile tone-canary">
            <span className="step-n">02</span>
            <h3>Describe each day</h3>
            <p>For every area and day: recent heat, humidity, wind, rain over 1–30 days, dry streaks, and how unusual each is <em>for that place and month</em>.</p>
          </article>
          <article className="tile tone-mint">
            <span className="step-n">03</span>
            <h3>Learn</h3>
            <p>Gradient-boosted trees, trained on 2013–2021: one model for all fires and one for lightning fires only, each blended with the area's past fire rate.</p>
          </article>
          <article className="tile tone-violet">
            <span className="step-n">04</span>
            <h3>Rank</h3>
            <p>Every evening, score all {all.cells.length} areas for tomorrow. A crew checks the riskiest {all.metrics.review_k}, one in eight. Tested on 2022–2024.</p>
          </article>
        </div>
        <div className="split">
          <div className="card">
            <h3 className="h-sm">No peeking at the future</h3>
            <p className="muted">Every input for a target day uses data up to the day before. Training stops at 31 Dec 2021 and the models are frozen after that, so they never trained on any day shown on the map.</p>
          </div>
          <div className="card">
            <h3 className="h-sm">Always checked against a simple guess</h3>
            <p className="muted">Every model is compared with ranking areas by past fire rate alone. Where weather doesn't beat that guess, we say so, which is how we found the lightning result.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export function LinuxOne({ ov }) {
  const r = ov.runtime, mo = ov.model, m = ov.metrics;
  const runs = [
    ["Architecture", r.machine, r.system],
    ["CPUs", r.cpus, r.snapshot ? "LinuxONE Community Cloud" : "this machine"],
    ["Python / scikit-learn", `${r.python} / ${r.sklearn}`, `numpy ${r.numpy} · pandas ${r.pandas}`],
    ["Model size", `${(mo.size_bytes / 1024).toFixed(0)} KB`, `${mo.trees} trees · ${mo.features} inputs`],
    ["Load model + data", `${nf.format(Math.round(r.load_ms))} ms`, "at server start"],
    ["Score all 3 years", `${nf.format(Math.round(r.score_all_ms))} ms`, `${nf.format(m.rows)} area-days`],
    ["Score one day", `${r.score_day_ms.toFixed(1)} ms`, "24 areas, median of 20 runs"],
    ["Model fingerprint", (mo.sha256 ?? "").slice(0, 12), "SHA-256, matches manifest"],
  ];
  return (
    <section id="linuxone" className="section dark">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow on-dark">Built on IBM LinuxONE</p>
          <h2>The whole pipeline ran on one IBM Z machine</h2>
          <p className="on-dark-muted">The project started at the IBM Z Datathon as a 24-area model, and data, training, model selection, inference and the website all ran on a LinuxONE Community Cloud instance. {r.snapshot ? "The numbers below were measured there." : "The numbers below are reported live by the server you're running."}</p>
        </div>
        <div className="run-grid">
          {runs.map(([k, v, s]) => <div key={k}><span>{k}</span><b>{v}</b><small>{s}</small></div>)}
        </div>
        <div className="flow">
          <div><b>USFS fires + ERA5 weather</b><span>public data</span></div>
          <div><b>Feature builder</b><span>past-only, per area/day</span></div>
          <div><b>HistGradientBoosting</b><span>trained on s390x</span></div>
          <div><b>Docker container</b><span>read-only, 3 CPU / 2 GB</span></div>
          <div><b>The website</b><span>served from LinuxONE at the datathon</span></div>
        </div>
      </div>
    </section>
  );
}

// Tomorrow's lightning outlook from the nightly job; until one loads (or if none can), the plan for it.
export function Forecast({ all, lightning }) {
  const [raw, setRaw] = useState(null);
  const [selected, setSelected] = useState(null);
  useEffect(() => {
    if (!lightning) return;
    fetch(FORECAST_URL, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then(setRaw, () => {});
  }, [lightning]);
  const day = useMemo(() => raw && expandDay(lightning, raw), [raw, lightning]);
  const select = useCallback((id) => setSelected((x) => (x === id ? null : id)), []);
  if (!day) return <ForecastPlan all={all} />;

  const f = raw.forecast;
  const today = new Date().toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });
  const stale = day.date < today;
  const m = lightning.metrics;
  // Below the map's first colour step everywhere: the badges still mark the top areas, but none really stands out.
  const quiet = Math.max(...raw.score) * (m.heat_scale ?? 1) < RISK[0];
  return (
    <section id="forecast" className="section band">
      <div className="wrap">
        <div className="section-head">
          <span className={`chip ${stale ? "chip-soon" : "chip-live ok"}`}><span className="dot"></span>{stale ? `Last forecast: ${fmtDay(day.date, { month: "short", day: "numeric" })}` : "Live · updated every evening"}</span>
          <h2>Tomorrow's lightning outlook</h2>
          <p className="muted">Every evening the lightning model scores all {lightning.cells.length} areas for the next day from the latest weather. Numbered badges are the top {m.badge_k} of the {m.review_k} areas a crew would check.{stale ? " Tonight's forecast hasn't been published yet." : ""}</p>
          {quiet && <p className="quiet-note"><b>A quiet night.</b> Lightning risk is low everywhere, so the badges mark the least-unlikely areas rather than real hot spots.</p>}
        </div>
        <div className="replay-grid">
          <div className="map-col">
            <ReplayMap ov={lightning} day={day} selected={selected} onSelect={select} duration={380} />
            <div className="legend">
              <span className="ramp-wrap">Lower risk <i className="ramp"></i> Higher risk</span>
              <span><i className="rank-mini">1</i>Top {m.badge_k} picks</span>
              <span><i className="fade-swatch"></i>Faded = outside model area</span>
            </div>
          </div>
          <DayPanel ov={lightning} day={day} selected={selected} onSelect={select} forecast={f} />
        </div>
        <div className="split">
          <div className="card">
            <h3 className="h-sm">How it's made</h3>
            <p className="muted">The model trained on gridMET weather, which runs about two days behind. So the last {f.open_meteo_days.length} day{f.open_meteo_days.length === 1 ? "" : "s"} before the forecast come from the Open-Meteo forecast instead, with rain averaged over five points in each area: storm rain is patchy, and one point mostly misses it.{f.storms ? " The model knows how dry each area is but not where storms will be, so each score is then adjusted by NOAA's HRRR storm forecast for the day: lightning, storm energy (CAPE) and rain." : " Tonight's HRRR storm forecast wasn't available, so this one goes out without it."} Issued {new Date(f.issued_at).toLocaleString("en-US", { timeZone: "America/Los_Angeles", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })} Pacific.</p>
          </div>
          <div className="card">
            <h3 className="h-sm">How much to trust it</h3>
            <p className="muted">We replayed summer 2024 with the weather forecasts actually issued at the time. On their own, the model's scores caught 42% of lightning fire area-days in the crew list, about as many as fire history alone (38%): it was good at knowing <i>when</i> lightning fires come, not <i>where</i>. Adding HRRR's storm forecast, fitted on 2022–23, raised that to 53%. That's from only 116 fire area-days in one summer, so the true gain could be anywhere from none to about 20 points. Outside summer, lightning fires are rare and the map stays quiet.</p>
          </div>
        </div>
        <TrackRecord cells={lightning.cells} />
      </div>
    </section>
  );
}

// Each forecast's crew list scored against the lightning fires later reported (ml/track.py).
// cells: the site's cell list, which the record's cell indexes refer to.
function TrackRecord({ cells }) {
  const [t, setT] = useState(null);
  useEffect(() => {
    fetch(TRACK_URL, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then(setT, () => {});
  }, []);
  if (!t) return null;
  const { live, reconstructed } = t.summary;
  const fireDays = t.days.filter((d) => d.over && d.fire_cells).reverse();
  const firstLive = t.days.find((d) => !d.reconstructed)?.date;
  const row = (label, s) => (
    <tr>
      <td>{label}</td>
      <td>{s.days}</td>
      <td>{s.fire_cells}</td>
      <td><b>{s.recall == null ? "—" : `${pct(s.recall)} (${s.caught})`}</b></td>
      <td>{s.recall_history == null ? "—" : `${pct(s.recall_history)} (${s.caught_history})`}</td>
      <td>{pct(t.random)}</td>
    </tr>
  );
  return (
    <div className="track">
      <div className="section-head">
        <h3 className="h-sm">Track record</h3>
        <p className="muted">Every night, each past forecast is checked against the lightning fires reported since. A fire counts as caught when it starts in one of that day's {t.review_k} crew-list areas.</p>
      </div>
      <div className="card table-card">
        <table className="track-table">
          <thead>
            <tr><th></th><th>Days</th><th>Lightning fire area-days</th><th>Caught by the model</th><th>By fire history alone</th><th>Random</th></tr>
          </thead>
          <tbody>
            {row(firstLive ? `Live, since ${fmtDay(firstLive, { month: "short", day: "numeric" })}` : "Live", live)}
            {reconstructed.days > 0 && row("2026 season, reconstructed", reconstructed)}
          </tbody>
        </table>
        <p className="tiny muted table-note">
          {live.fire_cells === 0 && "No lightning fires have been reported in the forecast area since it went live; lightning season returns in summer. "}
          Reconstructed days use the forecasts Open-Meteo issued at the time and gridMET with its usual two-day delay: what this pipeline would have published. Fires come from the national incident feed (NIFC WFIGS), which labels only about 1 in 4 lightning fires as lightning, so these counts are a sample.
        </p>
      </div>
      {fireDays.length > 0 && (
        <div className="card table-card">
          <table className="track-table">
            <thead><tr><th>Day</th><th>Lightning fires</th><th className="l">Where</th><th>Model's list</th><th>History's list</th></tr></thead>
            <tbody>
              {fireDays.slice(0, 12).map((d) => (
                <tr key={d.date}>
                  <td>{fmtDay(d.date, { month: "short", day: "numeric" })}{d.reconstructed ? "" : " · live"}</td>
                  <td>{d.fires.length}</td>
                  <td className="l">{[...new Set(d.fires.map((f) => name(cells[f[2]].id)))].slice(0, 3).join(", ")}</td>
                  <td className={d.caught ? "up" : ""}>{d.caught} of {d.fire_cells}</td>
                  <td className={d.caught_history ? "up" : ""}>{d.caught_history} of {d.fire_cells}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {fireDays.length > 12 && <p className="tiny muted table-note">Latest 12 of {fireDays.length} days with lightning fires.</p>}
        </div>
      )}
    </div>
  );
}

function ForecastPlan({ all }) {
  const n = all?.cells.length ?? 299;
  return (
    <section id="forecast" className="section">
      <div className="wrap">
        <div className="forecast card-dark">
          <div>
            <span className="chip chip-soon">Coming soon</span>
            <h2>Live forecast</h2>
            <p className="on-dark-muted">Today the models replay history. Next, we'll pull tomorrow's weather forecast every evening, build the same inputs, and publish a real outlook for the next day, starting with lightning, where weather matters most.</p>
            <ol className="forecast-steps">
              <li>Fetch the daily forecast for all {n} areas (Open-Meteo)</li>
              <li>Add thunderstorm signals the current model can't see yet</li>
              <li>Score every evening and publish tomorrow's watch list</li>
              <li>Track how it did the next day</li>
            </ol>
          </div>
          <div className="forecast-preview" aria-hidden="true">
            <div className="ghost-grid"></div>
            <span>Tomorrow's map will appear here</span>
          </div>
        </div>
      </div>
    </section>
  );
}

export function Limits({ lightning }) {
  return (
    <section className="section band">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Honest limits</p>
          <h2>What this is, and isn't</h2>
        </div>
        <div className="limits">
          <div className="card"><h3 className="h-sm">A ranking, not a probability</h3><p className="muted">Scores say which areas are riskier than others. They are not calibrated chances of fire.</p></div>
          <div className="card"><h3 className="h-sm">Hindsight weather</h3><p className="muted">The weather data is compiled after the fact. Real forecasts are noisier, so a live version would likely do worse.</p></div>
          <div className="card"><h3 className="h-sm">Few lightning fires</h3><p className="muted">Only {nf.format(lightning.data.replay_fires)} lightning fires in the test years, many on a handful of storm days. The direction is solid; exact numbers are less certain.</p></div>
          <div className="card"><h3 className="h-sm">Cause records have gaps</h3><p className="muted">About 4 in 10 fires have no recorded cause. Some were surely lightning, so the lightning label misses some real ones.</p></div>
        </div>
      </div>
    </section>
  );
}

// Everything a technical reader might want, collapsed so it stays out of everyone else's way.
// Charts only mount once it's opened, so they size against a visible box.
export function Details({ all, lightning, base }) {
  const [open, setOpen] = useState(false);
  return (
    <section id="details" className="section">
      <div className="wrap">
        <details className="details" onToggle={(e) => setOpen(e.currentTarget.open)}>
          <summary>
            <span className="eyebrow">Technical details</span>
            <span className="h-sm">Full metrics, charts and the original 24-area model</span>
          </summary>
          {open && <DetailsBody all={all} lightning={lightning} base={base} />}
        </details>
      </div>
    </section>
  );
}

function DetailsBody({ all, lightning, base }) {
  const m = lightning.metrics, c = m.confusion, n = lightning.cells.length;
  const tiles = [
    { tone: "tone-violet", label: "Average precision", value: m.average_precision.toFixed(3), vs: `Random guessing: ${m.prevalence.toFixed(4)} · fire history alone: ${m.baseline_average_precision.toFixed(3)}` },
    { tone: "tone-canary", label: "ROC AUC", value: m.roc_auc.toFixed(3), vs: "Random guessing: 0.500" },
    { tone: "tone-mint", label: `Fire days caught checking ${m.review_k} of ${n} areas`, value: pct(m.review_recall), vs: `Random picks: ${pct(m.review_random, 1)} · fire history alone: ${pct(m.baseline_review_recall)}` },
    { tone: "tone-petal", label: "Lightning fires in the test period", value: nf.format(lightning.data.replay_fires), vs: `${nf.format(m.positives)} area-days with a lightning fire` },
  ];
  const charts = useMemo(() => ({
    pr: {
      type: "line",
      data: {
        datasets: [
          { label: "Model", data: m.pr_curve.map(([x, y]) => ({ x, y })), borderColor: "#e8400d", backgroundColor: "rgba(232,64,13,0.12)", fill: true, pointRadius: 0, tension: 0.15 },
          { label: "Random", data: [{ x: 0, y: m.prevalence }, { x: 1, y: m.prevalence }], borderColor: "#b1b1af", borderDash: [6, 6], pointRadius: 0 },
        ],
      },
      options: {
        plugins: { legend: { display: true, labels: { boxWidth: 12 } } },
        scales: { x: { type: "linear", min: 0, max: 1, grid, title: { display: true, text: "Recall (share of fire area-days caught)" } }, y: { min: 0, max: 1, grid, title: { display: true, text: "Precision" } } },
      },
    },
    importance: {
      type: "bar",
      data: { labels: lightning.importance.slice(0, 8).map((d) => d.feature), datasets: [{ data: lightning.importance.slice(0, 8).map((d) => d.importance), backgroundColor: "#e8400d", borderRadius: 4 }] },
      options: { indexAxis: "y", scales: { x: { grid, title: { display: true, text: "Drop in average precision" } }, y: { grid: { display: false }, ticks: { color: "#111" } } } },
    },
    years: {
      type: "bar",
      data: { labels: m.years.map((y) => y.year), datasets: [{ data: m.years.map((y) => y.top3_recall), backgroundColor: ["#e2ddfd", "#ffef99", "#b7efb2"], borderRadius: 6 }] },
      options: { scales: { y: { min: 0, max: 0.8, grid, ticks: { callback: (v) => pct(v) } }, x: { grid: { display: false }, ticks: { color: "#111" } } } },
    },
  }), [lightning]);

  // Fire rates differ between grids, so each model is measured against random guessing on its own grid.
  const lift = (x) => `${(x.average_precision / x.prevalence).toFixed(1)}×`;
  const am = all.metrics, bm = base.metrics;

  return (
    <div className="details-body">
      <h3 className="h-sm">The lightning model · 2022–2024 test period</h3>
      <div className="metric-grid">
        {tiles.map((x) => (
          <div key={x.label} className={`metric ${x.tone}`}><span className="label">{x.label}</span><span className="value">{x.value}</span><span className="vs">{x.vs}</span></div>
        ))}
      </div>
      <div className="split">
        <div className="card">
          <h3 className="h-sm">Precision vs. recall</h3>
          <p className="muted small">Above the dashed line means better than random guessing at every setting.</p>
          <ChartBox config={charts.pr} />
        </div>
        <div className="card">
          <h3 className="h-sm">The accuracy trap</h3>
          <p className="muted">Only <b>{pct(m.prevalence, 2)}</b> of area-days have a lightning fire. A "model" that always says <em>no fire</em> is <b>{pct(m.always_no_accuracy, 1)}</b> accurate and useless. Ours is <b>{pct(m.accuracy)}</b> accurate on its crew list, because it deliberately flags areas to catch more fires. So we judge it on how well it <em>ranks</em> risky places.</p>
          <div className="confusion">
            <div className="hd"></div><div className="hd">Fire happened</div><div className="hd">No fire</div>
            <div className="hd">On crew list</div><div className="good"><b>{nf.format(c.tp)}</b>caught</div><div className="bad"><b>{nf.format(c.fp)}</b>false alarms</div>
            <div className="hd">Not on list</div><div className="bad"><b>{nf.format(c.fn)}</b>missed</div><div className="good"><b>{nf.format(c.tn)}</b>correctly quiet</div>
          </div>
        </div>
      </div>
      <div className="split">
        <div className="card">
          <h3 className="h-sm">Fire days caught, by year</h3>
          <p className="muted small">Share of lightning-fire area-days inside the top {m.review_k} (random = {pct(m.review_random, 1)}).</p>
          <ChartBox config={charts.years} short />
        </div>
        <div className="card">
          <h3 className="h-sm">What the model leans on</h3>
          <p className="muted small">How much the score quality drops when each input is scrambled (permutation importance on a sample of the test years).</p>
          <ChartBox config={charts.importance} short />
        </div>
      </div>

      <h3 className="h-sm details-sub">The all-fire model and the original 24-area model</h3>
      <div className="split">
        <div className="card">
          <h3 className="h-sm">Three models, each against random guessing</h3>
          <p className="muted small">Smaller areas catch fire less often, so raw scores aren't comparable across grids. Each column is measured on its own grid, checking one area in eight.</p>
          <div className="confusion compare three">
            <div className="hd"></div><div className="hd">Datathon, 24 areas</div><div className="hd">All fires, {n} areas</div><div className="hd">Lightning, {n} areas</div>
            <div className="hd">Average precision vs random</div>
            <div><b>{lift(bm)}</b>{bm.average_precision.toFixed(3)}</div>
            <div><b>{lift(am)}</b>{am.average_precision.toFixed(3)}</div>
            <div className="good"><b>{lift(m)}</b>{m.average_precision.toFixed(3)}</div>
            <div className="hd">Fire days caught</div>
            <div><b>{pct(bm.review_recall)}</b>top 3 of 24</div>
            <div><b>{pct(am.review_recall)}</b>top {am.review_k} of {n}</div>
            <div className="good"><b>{pct(m.review_recall)}</b>top {m.review_k} of {n}</div>
            <div className="hd">Average precision, fire history alone</div>
            <div><b>{bm.baseline_average_precision.toFixed(3)}</b>model: {bm.average_precision.toFixed(3)}</div>
            <div><b>{am.baseline_average_precision.toFixed(3)}</b>model: {am.average_precision.toFixed(3)}</div>
            <div className="good"><b>{m.baseline_average_precision.toFixed(3)}</b>model: {m.average_precision.toFixed(3)}</div>
          </div>
        </div>
        <div className="card">
          <h3 className="h-sm">What changed after the datathon</h3>
          <p className="muted">The datathon model ranked 24 areas half a degree wide, using one ERA5 weather point each. The current models cover about four times the land in {n} quarter-degree areas, with gridMET weather averaged over each area.</p>
          <p className="muted">Before touching new data, the new dataset builder rebuilt the datathon's table from its raw downloads and matched all 113,976 rows exactly.</p>
          <p className="muted">Why not even smaller areas? At 0.1° (about 10 km), the typical area has only about a dozen fires in 13 years: too few to learn its own fire rate, the models' strongest input.</p>
        </div>
      </div>
    </div>
  );
}

const SOURCES = "Fire data: US Forest Service, Fire Program Analysis Fire-Occurrence Database (FPA FOD), 7th edition. Weather: gridMET (University of California Merced / Climatology Lab); the datathon model used Open-Meteo / Copernicus ERA5, CC BY 4.0. Place names: US Census Bureau Gazetteer. Map tiles © OpenStreetMap contributors. Research model, not an official warning service.";

export function Footer() {
  return (
    <footer className="footer">
      <div className="wrap">
        <p><b>localhots</b> · started at IBM Z Datathon 2026</p>
        <p className="tiny">{SOURCES}</p>
      </div>
    </footer>
  );
}
