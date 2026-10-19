// Canvas heat layer for the replay map. Same drawing code as the original site.

// Heat colour ramp: [score, r, g, b, alpha]. Low risk stays nearly clear so the map shows through.
const HEAT = [
  [0.0, 255, 239, 153, 0.0],
  [0.06, 255, 239, 153, 0.28],
  [0.14, 255, 179, 92, 0.45],
  [0.25, 232, 64, 13, 0.55],
  [0.45, 175, 30, 0, 0.66],
  [0.75, 120, 20, 0, 0.74],
];
export const VIEW_CELLS = 8; // the view spans 8 x 8 cells with the modelled 4 x 6 area in the middle
const CANVAS_PX = 256;
export const WASH = 0.8; // white wash strength over everything the model doesn't cover

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

function heatColor(v) {
  if (v <= HEAT[0][0]) return HEAT[0].slice(1);
  for (let i = 1; i < HEAT.length; i++) {
    if (v <= HEAT[i][0]) {
      const [a, b] = [HEAT[i - 1], HEAT[i]];
      const t = (v - a[0]) / (b[0] - a[0]);
      return [1, 2, 3, 4].map((k) => a[k] + (b[k] - a[k]) * t);
    }
  }
  return HEAT[HEAT.length - 1].slice(1);
}
const HEAT_LUT = Array.from({ length: 256 }, (_, i) => heatColor(i / 255)); // ramp sampled once
const NO_HEAT = [0, 0, 0, 0];

// Grid geometry from the area bounds: the modelled block centred in a VIEW_CELLS-wide square.
export function gridFor(cells) {
  const south = Math.min(...cells.map((c) => c.bounds[0][0]));
  const west = Math.min(...cells.map((c) => c.bounds[0][1]));
  const north = Math.max(...cells.map((c) => c.bounds[1][0]));
  const east = Math.max(...cells.map((c) => c.bounds[1][1]));
  const step = cells[0].bounds[1][0] - cells[0].bounds[0][0];
  const rows = Math.round((north - south) / step), cols = Math.round((east - west) / step);
  const s0 = south - Math.floor((VIEW_CELLS - rows) / 2) * step;
  const w0 = west - Math.floor((VIEW_CELLS - cols) / 2) * step;
  return { south, west, north, east, step, rows, cols, s0, w0, span: VIEW_CELLS * step };
}

// Per-pixel geometry for the heat canvas, worked out once: which 2 x 2 block of area centres each
// pixel blends between, and how much heat fade and white wash it gets from its distance outside
// the modelled area. Drawing a frame then only has to mix scores.
function prepareHeat(g) {
  const N = CANVAS_PX;
  const px = { cell: new Uint8Array(N * N), tr: new Float32Array(N * N), tc: new Float32Array(N * N),
    fade: new Float32Array(N * N), wash: new Float32Array(N * N) };
  for (let y = 0; y < N; y++) {
    const lat = g.s0 + g.span - ((y + 0.5) / N) * g.span;
    const fr = Math.min(g.rows - 1, Math.max(0, (lat - g.south) / g.step - 0.5));
    const r0 = Math.min(g.rows - 2, Math.floor(fr));
    const dy = Math.max(g.south - lat, 0, lat - g.north) / g.step;
    for (let x = 0; x < N; x++) {
      const lon = g.w0 + ((x + 0.5) / N) * g.span;
      const fc = Math.min(g.cols - 1, Math.max(0, (lon - g.west) / g.step - 0.5));
      const c0 = Math.min(g.cols - 2, Math.floor(fc));
      const d = Math.hypot(Math.max(g.west - lon, 0, lon - g.east) / g.step, dy);
      const p = y * N + x;
      px.cell[p] = r0 * g.cols + c0;
      px.tr[p] = fr - r0;
      px.tc[p] = fc - c0;
      px.fade[p] = 1 - smooth(0, 0.7, d);
      px.wash[p] = smooth(0.05, 0.9, d) * WASH; // full strength before the canvas edge (1 cell out)
    }
  }
  return px;
}

// Returns the canvas and a draw(scores) that paints scores as a smooth surface: bilinear blend
// between area centres, softly fading out past the modelled area, with a white wash over everything
// the model doesn't cover. Missing scores are left out of the blend (never treated as zero risk).
export function createHeat(g) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = CANVAS_PX;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(CANVAS_PX, CANVAS_PX);
  const px = prepareHeat(g);

  function draw(scores) {
    const data = img.data;
    const vals = new Float64Array(g.rows * g.cols).fill(NaN);
    for (let r = 0; r < g.rows; r++) for (let c = 0; c < g.cols; c++) {
      const s = scores[`r${r}c${c}`];
      if (Number.isFinite(s)) vals[r * g.cols + c] = s;
    }
    for (let p = 0, i = 0; p < px.cell.length; p++, i += 4) {
      const k = px.cell[p], tr = px.tr[p], tc = px.tc[p];
      let total = 0, weight = 0;
      const mix = (s, w) => { if (s === s) { total += s * w; weight += w; } }; // s === s skips NaN
      mix(vals[k], (1 - tr) * (1 - tc));
      mix(vals[k + 1], (1 - tr) * tc);
      mix(vals[k + g.cols], tr * (1 - tc));
      mix(vals[k + g.cols + 1], tr * tc);
      const [hr, hg, hb, ha0] = weight ? HEAT_LUT[Math.min(255, Math.max(0, Math.round((total / weight) * 255)))] : NO_HEAT;
      const ha = ha0 * px.fade[p], m = px.wash[p];
      const a = m + ha * (1 - m);
      if (a > 0) {
        data[i] = (255 * m + hr * ha * (1 - m)) / a;
        data[i + 1] = (255 * m + hg * ha * (1 - m)) / a;
        data[i + 2] = (255 * m + hb * ha * (1 - m)) / a;
      }
      data[i + 3] = a * 255;
    }
    ctx.putImageData(img, 0, 0);
  }
  return { canvas, draw };
}
