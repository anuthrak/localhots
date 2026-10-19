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

export const nf = new Intl.NumberFormat("en-US");
export const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
export const name = (id) => `${NAMES[id] ?? id}`;
export const fmtDay = (s, opts) => new Date(`${s}T12:00:00`).toLocaleDateString("en-US", opts);
export const fires = (n) => `${n} fire${n > 1 ? "s" : ""}`;

export function riskColor(s) {
  const i = RISK.findIndex((b) => s < b);
  return RISK_COLORS[i === -1 ? 4 : i];
}

export async function getJSON(path) {
  const res = await fetch(`${import.meta.env.BASE_URL}${path}`);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}
