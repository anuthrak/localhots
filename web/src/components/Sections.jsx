import { useMemo } from "react";
import ChartBox, { grid } from "./ChartBox.jsx";
import { nf, pct } from "../lib.js";

const HOME_LINKS = [
  ["#replay", "Replay map"], ["#how", "How it works"], ["#results", "Results"], ["#linuxone", "LinuxONE"],
  ["#forecast", "Live forecast"],
];
const HOME_CTA = [`${import.meta.env.BASE_URL}norcal/`, "Detailed map"];

// links: [href, label] pairs; cta: the button at the far right, to the other page; chip: status text once
// the data has loaded (defaults to the server's).
export function Nav({ ov, error, links = HOME_LINKS, cta = HOME_CTA, chip }) {
  const r = ov?.runtime;
  return (
    <header className="nav">
      <div className="wrap nav-inner">
        <a href="#top" className="logo"><span className="flame" aria-hidden="true"></span>localhots</a>
        <nav aria-label="Sections">
          {links.map(([href, label]) => <a key={href} href={href}>{label}</a>)}
        </nav>
        <span className={`chip chip-live${r ? " ok" : ""}`}>
          <span className="dot"></span>
          {error ? "Data unavailable" : !r ? "Connecting…" : chip ?? (r.snapshot ? `Built on IBM LinuxONE · ${r.machine}` : `Live server · ${r.machine}`)}
        </span>
        <a href={cta[0]} className="btn btn-primary nav-cta">{cta[1]}</a>
      </div>
    </header>
  );
}

export function Hero({ ov }) {
  return (
    <section className="hero">
      <div className="wrap">
        <p className="eyebrow">IBM Z Datathon 2026 · Northern California</p>
        <h1 className="display">Where will the next fire start?</h1>
        <p className="lede">Fire crews can't watch everywhere. Each day our model ranks 24 areas of Northern California by how likely a new wildfire is <em>tomorrow</em>, using only weather and history up to today. Replay 2022–2024 and see how often it pointed the right way.</p>
        <div className="hero-actions">
          <a href="#replay" className="btn btn-primary">Replay a fire season</a>
          <a href="#results" className="btn btn-light">See the results</a>
        </div>
        <dl className="hero-stats">
          <div><dt>Fires in the record</dt><dd>{ov ? nf.format(ov.data.fire_records_total) : "18,303"}</dd></div>
          <div><dt>Areas ranked each day</dt><dd>24</dd></div>
          <div><dt>Fire days caught by checking 3 areas</dt><dd>{ov ? pct(ov.metrics.review_recall) : "34%"}</dd></div>
          <div><dt>Time to score a day on LinuxONE</dt><dd>{ov ? `${ov.runtime.score_day_ms.toFixed(1)} ms` : "~6 ms"}</dd></div>
        </dl>
      </div>
    </section>
  );
}

export function HowItWorks({ ov }) {
  const importance = useMemo(() => ({
    type: "bar",
    data: { labels: ov.importance.slice(0, 8).map((d) => d.feature), datasets: [{ data: ov.importance.slice(0, 8).map((d) => d.importance), backgroundColor: "#e8400d", borderRadius: 4 }] },
    options: { indexAxis: "y", scales: { x: { grid, title: { display: true, text: "Drop in average precision" } }, y: { grid: { display: false }, ticks: { color: "#111" } } } },
  }), [ov]);
  return (
    <section id="how" className="section band">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">How it works</p>
          <h2>From 13 years of fires and weather to a daily watch list</h2>
        </div>
        <div className="steps">
          <article className="tile tone-petal">
            <span className="step-n">01</span>
            <h3>Collect</h3>
            <p><b>{nf.format(ov.data.fire_records_total)}</b> fires from the US Forest Service fire-occurrence database, plus daily ERA5 weather for each area from 2012 to 2024.</p>
          </article>
          <article className="tile tone-canary">
            <span className="step-n">02</span>
            <h3>Describe each day</h3>
            <p>For every area and day: recent heat, humidity, wind, rain over 1–30 days, dry streaks, and how unusual each is <em>for that place and month</em>.</p>
          </article>
          <article className="tile tone-mint">
            <span className="step-n">03</span>
            <h3>Learn</h3>
            <p>A gradient-boosted tree model ({ov.model.trees} trees, {ov.model.features} inputs), trained on 2013–2021, blended 75/25 with each area's past fire rate.</p>
          </article>
          <article className="tile tone-violet">
            <span className="step-n">04</span>
            <h3>Rank</h3>
            <p>Every evening, score all 24 areas for tomorrow and send crews to the top 3. Tested on 2022–2024, which the model never saw during training.</p>
          </article>
        </div>
        <div className="split">
          <div className="card">
            <h3 className="h-sm">No peeking at the future</h3>
            <p className="muted">Every input for a target day uses data up to the day before. Training history stops at 31 Dec 2021 and the model is frozen after that, so the model never trained on any day shown on the map.</p>
          </div>
          <div className="card">
            <h3 className="h-sm">What the model leans on</h3>
            <p className="muted small">How much the score quality drops when each input is scrambled (permutation importance on the 2022–2024 test years).</p>
            <ChartBox config={importance} short />
          </div>
        </div>
      </div>
    </section>
  );
}

export function Results({ ov }) {
  const m = ov.metrics, c = m.confusion;
  const tiles = [
    { tone: "tone-petal", label: "Average precision", value: m.average_precision.toFixed(3), vs: `Random guessing: ${m.prevalence.toFixed(3)} · past fire rate alone: ${m.baseline_average_precision.toFixed(3)}` },
    { tone: "tone-canary", label: "ROC AUC", value: m.roc_auc.toFixed(3), vs: "Random guessing: 0.500" },
    { tone: "tone-mint", label: `Fire days caught checking ${m.review_k} of 24 areas`, value: pct(m.review_recall), vs: `Random picks: ${pct(m.review_random, 1)} · ${(m.review_recall / m.review_random).toFixed(1)}× better` },
    { tone: "tone-violet", label: "Fires in the test period", value: nf.format(ov.data.replay_fires), vs: `${nf.format(m.positives)} area-days with fire, none seen in training` },
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
        scales: { x: { type: "linear", min: 0, max: 1, grid, title: { display: true, text: "Recall (share of fire days caught)" } }, y: { min: 0, max: 1, grid, title: { display: true, text: "Precision" } } },
      },
    },
    years: {
      type: "bar",
      data: { labels: m.years.map((y) => y.year), datasets: [{ data: m.years.map((y) => y.ap), backgroundColor: ["#ffd7f0", "#ffef99", "#b7efb2"], borderRadius: 6 }] },
      options: { scales: { y: { min: 0, max: 0.6, grid }, x: { grid: { display: false }, ticks: { color: "#111" } } } },
    },
    review: {
      type: "bar",
      data: { labels: ["Random 3 areas", "Model's top 3"], datasets: [{ data: [m.review_random, m.review_recall], backgroundColor: ["#ecebea", "#e8400d"], borderRadius: 6 }] },
      options: { indexAxis: "y", scales: { x: { min: 0, max: 0.5, grid, ticks: { callback: (v) => pct(v) } }, y: { grid: { display: false }, ticks: { color: "#111" } } } },
    },
  }), [m]);

  return (
    <section id="results" className="section">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Results · 2022–2024 test period</p>
          <h2>Ranking the right places first</h2>
          <p className="muted">{nf.format(m.rows)} area-days (24 areas × 3 years), {nf.format(m.positives)} of them with at least one fire. Recomputed from the model {ov.runtime.snapshot ? "when this site was built" : "each time the server starts"}.</p>
        </div>

        <div className="metric-grid">
          {tiles.map((t) => (
            <div key={t.label} className={`metric ${t.tone}`}><span className="label">{t.label}</span><span className="value">{t.value}</span><span className="vs">{t.vs}</span></div>
          ))}
        </div>

        <div className="split">
          <div className="card">
            <h3 className="h-sm">The accuracy trap</h3>
            <p className="muted">Only <b>{pct(m.prevalence)}</b> of area-days have a fire. A "model" that always says <em>no fire</em> is <b>{pct(m.always_no_accuracy)}</b> accurate and useless. Ours is <b>{pct(m.accuracy)}</b> accurate at its alert setting, because it deliberately raises alerts to catch more fires. So we judge it on how well it <em>ranks</em> risky places, not on accuracy.</p>
            <div className="confusion">
              <div className="hd"></div><div className="hd">Fire happened</div><div className="hd">No fire</div>
              <div className="hd">Alert raised</div><div className="good"><b>{nf.format(c.tp)}</b>caught</div><div className="bad"><b>{nf.format(c.fp)}</b>false alarms</div>
              <div className="hd">No alert</div><div className="bad"><b>{nf.format(c.fn)}</b>missed</div><div className="good"><b>{nf.format(c.tn)}</b>correctly quiet</div>
            </div>
          </div>
          <div className="card">
            <h3 className="h-sm">Precision vs. recall</h3>
            <p className="muted small">Above the dashed line means better than random guessing at every setting.</p>
            <ChartBox config={charts.pr} />
          </div>
        </div>

        <div className="split">
          <div className="card">
            <h3 className="h-sm">Consistent every year</h3>
            <p className="muted small">Average precision by test year (random = about 0.12).</p>
            <ChartBox config={charts.years} short />
          </div>
          <div className="card">
            <h3 className="h-sm">Checking only 3 of 24 areas a day</h3>
            <p className="muted small">Share of all fire area-days covered.</p>
            <ChartBox config={charts.review} short />
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
          <p className="on-dark-muted">At the datathon, data, training, model selection, inference and the website all ran on a LinuxONE Community Cloud instance. {r.snapshot ? "The numbers below were measured there." : "The numbers below are reported live by the server you're running."}</p>
        </div>
        <div className="run-grid">
          {runs.map(([k, v, s]) => <div key={k}><span>{k}</span><b>{v}</b><small>{s}</small></div>)}
        </div>
        <div className="flow">
          <div><b>USFS fires + ERA5 weather</b><span>public data</span></div>
          <div><b>Feature builder</b><span>past-only, per area/day</span></div>
          <div><b>HistGradientBoosting</b><span>trained on s390x</span></div>
          <div><b>Docker container</b><span>read-only, 3 CPU / 2 GB</span></div>
          <div><b>This website</b><span>served from LinuxONE at the datathon</span></div>
        </div>
      </div>
    </section>
  );
}

export function Forecast() {
  return (
    <section id="forecast" className="section">
      <div className="wrap">
        <div className="forecast card-dark">
          <div>
            <span className="chip chip-soon">Coming soon</span>
            <h2>Live forecast</h2>
            <p className="on-dark-muted">Today the model replays history. Next, we'll pull tomorrow's weather forecast every evening, build the same 27 inputs, and publish a real outlook for the next day, with the same map and watch list.</p>
            <ol className="forecast-steps">
              <li>Fetch the daily forecast for all 24 areas (Open-Meteo)</li>
              <li>Build features exactly like training</li>
              <li>Score every evening and publish tomorrow's top 3</li>
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

export function Limits() {
  return (
    <section className="section band">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Honest limits</p>
          <h2>What this is, and isn't</h2>
        </div>
        <div className="limits">
          <div className="card"><h3 className="h-sm">A ranking, not a probability</h3><p className="muted">Scores say which areas are riskier than others. They are not calibrated chances of fire.</p></div>
          <div className="card"><h3 className="h-sm">Large areas</h3><p className="muted">Each area is about 45 × 56 km with one weather point. It can't see terrain, fuels, power lines, lightning or people.</p></div>
          <div className="card"><h3 className="h-sm">Hindsight weather</h3><p className="muted">ERA5 is reanalysis made after the fact. Real forecasts are noisier, which is why the live version is the next step.</p></div>
          <div className="card"><h3 className="h-sm">One region</h3><p className="muted">Trained and tested on one part of Northern California. The test years were also looked at during development.</p></div>
        </div>
      </div>
    </section>
  );
}

const SOURCES = "Fire data: US Forest Service, Fire Program Analysis Fire-Occurrence Database (FPA FOD), 7th edition. Weather: Open-Meteo / Copernicus ERA5, CC BY 4.0. Map tiles © OpenStreetMap contributors. Research model, not an official warning service.";

export function Footer({ sources = SOURCES }) {
  return (
    <footer className="footer">
      <div className="wrap">
        <p><b>localhots</b> · built at IBM Z Datathon 2026</p>
        <p className="tiny">{sources}</p>
      </div>
    </footer>
  );
}
