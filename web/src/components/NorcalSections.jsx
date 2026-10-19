import { useEffect, useMemo, useState } from "react";
import ChartBox, { grid } from "./ChartBox.jsx";
import { getJSON, nf, pct } from "../lib.js";

const km = (deg, lat) => Math.round(deg * 111.3 * (lat === undefined ? 1 : Math.cos((lat * Math.PI) / 180)));

export function NorcalHero({ ov }) {
  const n = ov?.cells.length, step = ov?.region.step ?? 0.25;
  const midLat = ov ? (ov.region.bounds[1] + ov.region.bounds[3]) / 2 : 39.5;
  return (
    <section className="hero">
      <div className="wrap">
        <p className="eyebrow">Detailed map · Northern California</p>
        <h1 className="display">Same question, sharper map.</h1>
        <p className="lede">The datathon model ranked 24 big areas. This version covers about four times the land, from the Bay Area to the Oregon border, in {n ?? "~300"} cells a quarter-degree wide. Same recipe, new weather data, retrained and tested on 2022–2024.</p>
        <div className="hero-actions">
          <a href="#replay" className="btn btn-primary">Replay a fire season</a>
          <a href={import.meta.env.BASE_URL} className="btn btn-light">See the original map</a>
        </div>
        <dl className="hero-stats">
          <div><dt>Cells ranked each day</dt><dd>{n ?? "—"}</dd></div>
          <div><dt>Cell size</dt><dd>~{km(step)} × {km(step, midLat)} km</dd></div>
          <div><dt>Fires in the record</dt><dd>{ov ? nf.format(ov.data.fire_records_total) : "—"}</dd></div>
          <div><dt>Fire days caught checking 12.5% of cells</dt><dd>{ov ? pct(ov.metrics.review_recall) : "—"}</dd></div>
        </dl>
      </div>
    </section>
  );
}

export function NorcalChanges({ ov }) {
  const mo = ov.model;
  return (
    <section id="how" className="section band">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">What changed</p>
          <h2>Four changes from the datathon model</h2>
        </div>
        <div className="steps">
          <article className="tile tone-petal">
            <span className="step-n">01</span>
            <h3>Bigger area</h3>
            <p>Bay Area to the Oregon border, coast to the Sierra. <b>{nf.format(ov.data.fire_records_total)}</b> fires from 2012 to 2024, about three times the original record.</p>
          </article>
          <article className="tile tone-canary">
            <span className="step-n">02</span>
            <h3>Finer cells</h3>
            <p>{ov.cells.length} land cells of {ov.region.step}°, each a quarter the size of an original area. Cells that are mostly ocean are left out.</p>
          </article>
          <article className="tile tone-mint">
            <span className="step-n">03</span>
            <h3>Weather per cell</h3>
            <p>Daily heat, humidity, wind and rain from gridMET's 4 km grid, averaged over each cell, instead of one weather point per area.</p>
          </article>
          <article className="tile tone-violet">
            <span className="step-n">04</span>
            <h3>Retrained</h3>
            <p>Same gradient-boosted trees ({mo.trees} trees, {mo.features} inputs), trained on 2013–2021, settings re-picked on 2015–2021 and blended {Math.round(mo.weather_weight * 100)}/{Math.round((1 - mo.weather_weight) * 100)} with each cell's past fire rate.</p>
          </article>
        </div>
        <div className="split">
          <div className="card">
            <h3 className="h-sm">Checked against the original first</h3>
            <p className="muted">Before touching new data, the new dataset builder rebuilt the datathon's table from its raw downloads and matched all 113,976 rows exactly. Then the same code ran on the bigger, finer grid.</p>
          </div>
          <div className="card">
            <h3 className="h-sm">Why not even smaller cells?</h3>
            <p className="muted">At 0.1° (about 10 km), the typical cell has only about a dozen fires in 13 years: too few to learn its own fire rate, which is the model's strongest input. That needs inputs that carry across cells, like terrain, fuels and roads.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export function NorcalResults({ ov }) {
  const m = ov.metrics, n = ov.cells.length;
  const [base, setBase] = useState(null);
  useEffect(() => { getJSON("api/overview.json").then(setBase, () => {}); }, []);

  const tiles = [
    { tone: "tone-petal", label: "Average precision", value: m.average_precision.toFixed(3), vs: `Random guessing: ${m.prevalence.toFixed(3)} · past fire rate alone: ${m.baseline_average_precision.toFixed(3)}` },
    { tone: "tone-canary", label: "ROC AUC", value: m.roc_auc.toFixed(3), vs: "Random guessing: 0.500" },
    { tone: "tone-mint", label: `Fire days caught checking ${m.review_k} of ${n} cells`, value: pct(m.review_recall), vs: `Random picks: ${pct(m.review_random, 1)} · past fire rate alone: ${pct(m.baseline_review_recall)}` },
    { tone: "tone-violet", label: "Fires in the test period", value: nf.format(ov.data.replay_fires), vs: `${nf.format(m.positives)} cell-days with fire, none seen in training` },
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
        scales: { x: { type: "linear", min: 0, max: 1, grid, title: { display: true, text: "Recall (share of fire cell-days caught)" } }, y: { min: 0, max: 1, grid, title: { display: true, text: "Precision" } } },
      },
    },
    years: {
      type: "bar",
      data: { labels: m.years.map((y) => y.year), datasets: [{ data: m.years.map((y) => y.top3_recall), backgroundColor: ["#ffd7f0", "#ffef99", "#b7efb2"], borderRadius: 6 }] },
      options: { scales: { y: { min: 0, max: 0.6, grid, ticks: { callback: (v) => pct(v) } }, x: { grid: { display: false }, ticks: { color: "#111" } } } },
    },
    importance: {
      type: "bar",
      data: { labels: ov.importance.slice(0, 8).map((d) => d.feature), datasets: [{ data: ov.importance.slice(0, 8).map((d) => d.importance), backgroundColor: "#e8400d", borderRadius: 4 }] },
      options: { indexAxis: "y", scales: { x: { grid, title: { display: true, text: "Drop in average precision" } }, y: { grid: { display: false }, ticks: { color: "#111" } } } },
    },
  }), [m, ov.importance]);

  // Scores and fire rates differ between grids, so compare each model against random guessing on its own grid.
  const lift = (ap, prev) => `${(ap / prev).toFixed(1)}×`;
  const bm = base?.metrics;

  return (
    <section id="results" className="section">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Results · 2022–2024 test period</p>
          <h2>Ranking ~300 cells instead of 24</h2>
          <p className="muted">{nf.format(m.rows)} cell-days ({n} cells × 3 years), {nf.format(m.positives)} of them with at least one fire ({pct(m.prevalence, 1)}). Crews check the riskiest {m.review_k} cells a day: 12.5% of the land, the same share as 3 of the original 24 areas.</p>
        </div>

        <div className="metric-grid">
          {tiles.map((t) => (
            <div key={t.label} className={`metric ${t.tone}`}><span className="label">{t.label}</span><span className="value">{t.value}</span><span className="vs">{t.vs}</span></div>
          ))}
        </div>

        <div className="split">
          <div className="card">
            <h3 className="h-sm">Compared with the 24-area map</h3>
            <p className="muted small">Smaller cells catch fire less often, so raw scores aren't comparable. Instead, each model is measured against random guessing on its own grid.</p>
            <div className="confusion compare">
              <div className="hd"></div><div className="hd">24 areas (0.5°)</div><div className="hd">{n} cells ({ov.region.step}°)</div>
              <div className="hd">Average precision vs random</div>
              <div><b>{bm ? lift(bm.average_precision, bm.prevalence) : "—"}</b>{bm ? `${bm.average_precision.toFixed(3)} vs ${bm.prevalence.toFixed(3)}` : ""}</div>
              <div className="good"><b>{lift(m.average_precision, m.prevalence)}</b>{`${m.average_precision.toFixed(3)} vs ${m.prevalence.toFixed(3)}`}</div>
              <div className="hd">Fire days caught, checking 12.5%</div>
              <div><b>{bm ? pct(bm.review_recall) : "—"}</b>top 3 of 24</div>
              <div className="good"><b>{pct(m.review_recall)}</b>top {m.review_k} of {n}</div>
            </div>
            <p className="muted small" style={{ marginTop: 12 }}>A crew list on the finer grid covers the same share of land, but in smaller, more precise pieces.</p>
          </div>
          <div className="card">
            <h3 className="h-sm">Precision vs. recall</h3>
            <p className="muted small">Above the dashed line means better than random guessing at every setting.</p>
            <ChartBox config={charts.pr} />
          </div>
        </div>

        <div className="split">
          <div className="card">
            <h3 className="h-sm">Fire days caught, by year</h3>
            <p className="muted small">Share of fire cell-days inside the top {m.review_k} (random = {pct(m.review_random, 1)}).</p>
            <ChartBox config={charts.years} short />
          </div>
          <div className="card">
            <h3 className="h-sm">What the model leans on</h3>
            <p className="muted small">How much the score quality drops when each input is scrambled (permutation importance on a sample of the test years).</p>
            <ChartBox config={charts.importance} short />
          </div>
        </div>
      </div>
    </section>
  );
}

export function NorcalLimits({ ov }) {
  const totals = ov.cells.map((c) => c.fires_total).sort((a, b) => a - b);
  const median = totals[Math.floor(totals.length / 2)];
  return (
    <section className="section band">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Honest limits</p>
          <h2>What this is, and isn't</h2>
        </div>
        <div className="limits">
          <div className="card"><h3 className="h-sm">Weather adds little so far</h3><p className="muted">Each cell's past fire rate alone catches {pct(ov.metrics.baseline_review_recall)} of fire days, versus {pct(ov.metrics.review_recall)} for the full model. Most of the gain comes from finer cells, not from the weather.</p></div>
          <div className="card"><h3 className="h-sm">Fewer fires per cell</h3><p className="muted">The typical cell had {median} fires in the three test years, so day-to-day results are noisier than on the 24-area map.</p></div>
          <div className="card"><h3 className="h-sm">Different weather data</h3><p className="muted">gridMET is finer than ERA5 but reports average wind, not the day's peak, so wind inputs aren't the same as the original model's.</p></div>
          <div className="card"><h3 className="h-sm">A ranking, from hindsight</h3><p className="muted">Scores rank cells; they aren't chances of fire and don't compare across pages. gridMET is built after the fact, so live forecasts would likely do worse.</p></div>
        </div>
      </div>
    </section>
  );
}
