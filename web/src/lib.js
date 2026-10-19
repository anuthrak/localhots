// Approximate place names for the 24 half-degree areas (centroid-nearest town or landmark).
export const NAMES = {
  r0c0: "Healdsburg", r0c1: "Lake Berryessa", r0c2: "Woodland", r0c3: "Roseville",
  r1c0: "Clear Lake", r1c1: "Williams", r1c2: "Yuba City", r1c3: "Grass Valley",
  r2c0: "Mendocino NF", r2c1: "Orland", r2c2: "Chico", r2c3: "Feather River",
  r3c0: "Yolla Bolly", r3c1: "Red Bluff", r3c2: "Lassen foothills", r3c3: "Lake Almanor",
  r4c0: "Weaverville", r4c1: "Redding", r4c2: "Burney", r4c3: "Hat Creek",
  r5c0: "Trinity Alps", r5c1: "Mount Shasta", r5c2: "McCloud River", r5c3: "Fall River Valley",
};
export const RISK = [0.05, 0.12, 0.2463, 0.4]; // ramp breaks; 0.2463 is the model's alert threshold
// Score tag colours, picked from the map's heat ramp.
const RISK_COLORS = ["#fff7cc", "#ffef99", "#ffb35c", "#e8400d", "#af1e00"];

let cellNames = NAMES;
// Pages whose cells carry their own names (overview.cells[].name) register them here.
export const setCellNames = (cells) => { cellNames = Object.fromEntries(cells.map((c) => [c.id, c.name])); };

export const nf = new Intl.NumberFormat("en-US");
export const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
export const name = (id) => `${cellNames[id] ?? id}`;
export const fmtDay = (s, opts) => new Date(`${s}T12:00:00`).toLocaleDateString("en-US", opts);
export const fires = (n) => `${n} fire${n > 1 ? "s" : ""}`;

export function riskColor(s) {
  const i = RISK.findIndex((b) => s < b);
  return RISK_COLORS[i === -1 ? 4 : i];
}

// Compact day files (one array per value, in overview.cells order) expanded into the per-cell
// objects the replay components use. Cells come back sorted by rank, like the main site's API.
export function expandDay(ov, d) {
  const k = ov.metrics.review_k;
  const rank = new Array(d.score.length);
  d.score.map((s, i) => i).sort((a, b) => d.score[b] - d.score[a]).forEach((i, r) => { rank[i] = r + 1; });
  const cells = ov.cells.map((c, i) => ({
    id: c.id, score: d.score[i], rank: rank[i], alert: rank[i] <= k, fires: d.fires_by_cell[i] ?? 0,
    past_rate: d.weather.past_rate[i], reasons: d.reasons[i] ?? [],
    weather: { tmax: d.weather.tmax[i], rh: d.weather.rh[i], wind: d.weather.wind[i], rain30: d.weather.rain30[i], dry_days: d.weather.dry_days[i] },
    anomaly: { heat: d.anomaly.heat[i], humidity: d.anomaly.humidity[i], wind: d.anomaly.wind[i], rain30: d.anomaly.rain30[i] },
  })).sort((a, b) => a.rank - b.rank);
  const top = cells.slice(0, k);
  return {
    date: d.date, issued: d.issued, cells,
    fires: d.fires.map(([lat, lon, i]) => [lat, lon, ov.cells[i].id]),
    summary: { fire_cells: cells.filter((c) => c.fires).length, fires: cells.reduce((n, c) => n + c.fires, 0), caught: top.filter((c) => c.fires).length },
  };
}

export async function getJSON(path) {
  const res = await fetch(`${import.meta.env.BASE_URL}${path}`);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}
