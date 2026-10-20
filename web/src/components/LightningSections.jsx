import { useMemo } from "react";
import ChartBox, { grid } from "./ChartBox.jsx";
import { nf, pct } from "../lib.js";

const TYPE_LABELS = { lightning: "Lightning", large: "Large fires (10+ acres)", human: "Human-caused", unknown: "Cause unknown" };

// ov.experiment rows -> { type: { score: row } }
function byType(rows = []) {
  const out = {};
  for (const r of rows) (out[r.type] ??= {})[r.score] = r;
  return out;
}

export function LightningHero({ ov }) {
  const m = ov?.metrics;
  const sharper = m ? (m.average_precision / m.baseline_average_precision).toFixed(1) : null;
  return (
    <section className="hero">
      <div className="wrap">
        <p className="eyebrow">Lightning fires · Northern California</p>
        <h1 className="display">Where will lightning start a fire?</h1>
        <p className="lede">Most wildfires are started by people, and weather barely helps predict those. Lightning fires are different: they follow the weather. A model trained only on lightning fires ranks them {sharper ? `${sharper}×` : "far"} more sharply than fire history alone, while the official fire danger index does little better than chance.</p>
        <div className="hero-actions">
          <a href="#replay" className="btn btn-primary">Replay a fire season</a>
          <a href="#experiment" className="btn btn-light">Why lightning?</a>
        </div>
        <dl className="hero-stats">
          <div><dt>Lightning fires, 2012–2024</dt><dd>{ov ? nf.format(ov.data.fire_records_total) : "—"}</dd></div>
          <div><dt>Cells ranked each day</dt><dd>{ov?.cells.length ?? "—"}</dd></div>
          <div><dt>Fire days caught checking 12.5% of cells</dt><dd>{m ? pct(m.review_recall) : "—"}</dd></div>
          <div><dt>With fire history alone</dt><dd>{m ? pct(m.baseline_review_recall) : "—"}</dd></div>
        </dl>
      </div>
    </section>
  );
}

export function LightningExperiment({ ov }) {
  const t = byType(ov.experiment);
  const m = ov.metrics;
  return (
    <section id="experiment" className="section band">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">The experiment</p>
          <h2>Weather matters for some fires, not others</h2>
          <p className="muted">We trained a separate model for each kind of fire and checked how much weather adds on top of knowing where that kind of fire usually happens. Each row: if a crew checks the riskiest {m.review_k} of {ov.cells.length} cells every day, what share of fire days do they catch?</p>
        </div>
        <div className="card table-card">
          <table className="result-table">
            <thead>
              <tr><th>Fire type</th><th>Fire history only</th><th>History + weather (model)</th><th>Official index (ERC)</th><th>Random</th></tr>
            </thead>
            <tbody>
              {Object.keys(TYPE_LABELS).filter((k) => t[k]).map((k) => {
                const gain = t[k].model.top_k_recall - t[k].past_fire_rate.top_k_recall;
                return (
                  <tr key={k} className={k === "lightning" ? "highlight" : ""}>
                    <td>{TYPE_LABELS[k]}</td>
                    <td>{pct(t[k].past_fire_rate.top_k_recall)}</td>
                    <td className={gain > 0.02 ? "up" : gain < -0.02 ? "down" : ""}><b>{pct(t[k].model.top_k_recall)}</b></td>
                    <td>{pct(t[k].erc.top_k_recall)}</td>
                    <td>{pct(m.review_random)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="tiny muted table-note">Every row uses the same model settings so the fire types compare fairly. The lightning model on the rest of this page was tuned on its own validation years and catches {pct(m.review_recall)}.</p>
        </div>
        <div className="split">
          <div className="card">
            <h3 className="h-sm">Why lightning is different</h3>
            <p className="muted">People start fires where people are: roads, towns, power lines. Fire history already knows those places, so weather adds little. Lightning fires come from dry thunderstorms that roll through on particular days, which fire history can't see coming but weather can.</p>
          </div>
          <div className="card">
            <h3 className="h-sm">Why the official index misses them</h3>
            <p className="muted">The Energy Release Component (ERC) measures how dry and burnable fuels are. That says what season it is, but not where a storm will strike, so for picking places it does no better than chance.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export function LightningResults({ ov }) {
  const m = ov.metrics;
  const tiles = [
    { tone: "tone-violet", label: "Average precision", value: m.average_precision.toFixed(3), vs: `Random guessing: ${m.prevalence.toFixed(4)} · fire history alone: ${m.baseline_average_precision.toFixed(3)}` },
    { tone: "tone-canary", label: "ROC AUC", value: m.roc_auc.toFixed(3), vs: "Random guessing: 0.500" },
    { tone: "tone-mint", label: `Fire days caught checking ${m.review_k} of ${ov.cells.length} cells`, value: pct(m.review_recall), vs: `Random picks: ${pct(m.review_random, 1)} · fire history alone: ${pct(m.baseline_review_recall)}` },
    { tone: "tone-petal", label: "Lightning fires in the test period", value: nf.format(ov.data.replay_fires), vs: `${nf.format(m.positives)} cell-days with a lightning fire` },
  ];
  const charts = useMemo(() => ({
    years: {
      type: "bar",
      data: { labels: m.years.map((y) => y.year), datasets: [{ data: m.years.map((y) => y.top3_recall), backgroundColor: ["#e2ddfd", "#ffef99", "#b7efb2"], borderRadius: 6 }] },
      options: { scales: { y: { min: 0, max: 0.8, grid, ticks: { callback: (v) => pct(v) } }, x: { grid: { display: false }, ticks: { color: "#111" } } } },
    },
    importance: {
      type: "bar",
      data: { labels: ov.importance.slice(0, 8).map((d) => d.feature), datasets: [{ data: ov.importance.slice(0, 8).map((d) => d.importance), backgroundColor: "#e8400d", borderRadius: 4 }] },
      options: { indexAxis: "y", scales: { x: { grid, title: { display: true, text: "Drop in average precision" } }, y: { grid: { display: false }, ticks: { color: "#111" } } } },
    },
  }), [m, ov.importance]);

  return (
    <section id="results" className="section">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Results · 2022–2024 test period</p>
          <h2>The lightning model, on years it never saw</h2>
          <p className="muted">{nf.format(m.rows)} cell-days, of which only {pct(m.prevalence, 2)} had a lightning fire. Lightning fires come in bursts: a few storm days account for many of them.</p>
        </div>
        <div className="metric-grid">
          {tiles.map((x) => (
            <div key={x.label} className={`metric ${x.tone}`}><span className="label">{x.label}</span><span className="value">{x.value}</span><span className="vs">{x.vs}</span></div>
          ))}
        </div>
        <div className="split">
          <div className="card">
            <h3 className="h-sm">Fire days caught, by year</h3>
            <p className="muted small">Share of lightning-fire cell-days inside the top {m.review_k} (random = {pct(m.review_random, 1)}).</p>
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

export function LightningLimits({ ov }) {
  return (
    <section className="section band">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Honest limits</p>
          <h2>What this is, and isn't</h2>
        </div>
        <div className="limits">
          <div className="card"><h3 className="h-sm">Few fires to learn from</h3><p className="muted">Only {nf.format(ov.data.replay_fires)} lightning fires in the test years, many on a handful of storm days. The direction is solid; exact numbers are less certain.</p></div>
          <div className="card"><h3 className="h-sm">No storm forecasts yet</h3><p className="muted">The model sees heat, humidity, wind and rain, but not thunderstorm forecasts or atmospheric instability, the things that actually predict lightning. Adding them is the next step.</p></div>
          <div className="card"><h3 className="h-sm">Cause records have gaps</h3><p className="muted">About 4 in 10 fires have no recorded cause. Some of those were surely lightning, so the label misses some real lightning fires.</p></div>
          <div className="card"><h3 className="h-sm">A ranking, from hindsight</h3><p className="muted">Scores rank cells; they aren't chances of fire. gridMET weather is built after the fact, so live forecasts would likely do worse.</p></div>
        </div>
      </div>
    </section>
  );
}
