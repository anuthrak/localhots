import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { createHeat, gridFor, WASH } from "../heat.js";
import { fires, name } from "../lib.js";

const HOVER = { stroke: true, color: "#111111", weight: 1, opacity: 0.5, dashArray: "4 4" };
const SELECTED = { stroke: true, color: "#111111", weight: 2, opacity: 0.9, dashArray: "5 4" };

// Leaflet owns this div; React only tells it which day and area to show.
export default function ReplayMap({ ov, day, selected, onSelect, duration }) {
  const el = useRef(null);
  const m = useRef(null);
  const latest = useRef({});
  latest.current = { onSelect, selected, duration };

  useEffect(() => {
    const map = L.map(el.current, { zoomSnap: 0.25, scrollWheelZoom: false, attributionControl: true, maxZoom: 10, maxBoundsViscosity: 1 });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 10, className: "basemap",
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);

    const g = gridFor(ov.cells);
    const outer = L.latLngBounds([g.s0, g.w0], [g.s0 + g.span, g.w0 + g.span]);
    const heat = createHeat(g);
    // svgOverlay places any element over the bounds, so the canvas is drawn on directly each frame
    // (no PNG encode/decode as with an image overlay).
    L.svgOverlay(heat.canvas, outer, { interactive: false, className: "heat" }).addTo(map);
    // Carry the canvas's full-strength wash on past its edge so the overlay doesn't show as a box.
    const o = [outer.getSouthWest(), outer.getNorthWest(), outer.getNorthEast(), outer.getSouthEast()];
    L.polygon([[[-85, -180], [85, -180], [85, 180], [-85, 180]], o], {
      stroke: false, fillColor: "#ffffff", fillOpacity: WASH, interactive: false,
    }).addTo(map);

    // Invisible hit areas so each modelled area can still be hovered and clicked.
    const rects = {};
    const outline = (id) => rects[id].setStyle(latest.current.selected === id ? SELECTED : { stroke: false });
    for (const cell of ov.cells) {
      rects[cell.id] = L.rectangle(cell.bounds, { stroke: false, fillOpacity: 0, fillColor: "#111111" })
        .on("click", () => latest.current.onSelect(cell.id))
        .on("mouseover", () => rects[cell.id].setStyle(HOVER))
        .on("mouseout", () => outline(cell.id))
        .bindTooltip("", { sticky: true })
        .addTo(map);
    }

    // Lock the view: can't zoom out past the region or pan away from Northern California.
    const lock = () => {
      map.invalidateSize();
      const z = map.getBoundsZoom(outer, false);
      map.setMinZoom(z);
      map.setMaxBounds(outer.pad(0.02));
      if (map.getZoom() <= z || !map.getBounds().intersects(outer)) map.fitBounds(outer);
    };
    map.fitBounds(outer);
    lock();
    const observer = new ResizeObserver(lock);
    observer.observe(el.current);

    m.current = { map, rects, outline, heat, badges: L.layerGroup().addTo(map), fires: L.layerGroup().addTo(map), shown: null, anim: 0 };
    return () => {
      cancelAnimationFrame(m.current.anim);
      observer.disconnect();
      map.remove();
      m.current = null;
    };
  }, [ov]);

  useEffect(() => {
    const s = m.current;
    if (!s || !day) return;
    s.badges.clearLayers();
    s.fires.clearLayers();
    animateHeat(s, Object.fromEntries(day.cells.map((c) => [c.id, c.score])), latest.current.duration);
    for (const c of day.cells) {
      s.outline(c.id);
      s.rects[c.id].setTooltipContent(`<b>${name(c.id)}</b><br>Rank ${c.rank} of 24 · score ${c.score.toFixed(2)}${c.fires ? `<br>${fires(c.fires)} discovered` : ""}`);
      if (c.rank <= ov.metrics.review_k) {
        L.marker(s.rects[c.id].getBounds().getCenter(), {
          interactive: false,
          icon: L.divIcon({ className: "", html: `<span class="rank-badge">${c.rank}</span>`, iconSize: [26, 26], iconAnchor: [13, 13] }),
        }).addTo(s.badges);
      }
    }
    for (const [lat, lon] of day.fires) {
      L.circleMarker([lat, lon], { radius: 4.5, color: "#111111", weight: 2, fillColor: "#ffffff", fillOpacity: 1, interactive: false }).addTo(s.fires);
    }
  }, [ov, day, selected]);

  return <div id="map" ref={el} className="card map" role="img" aria-label="Map of the 24 forecast areas" />;
}

// Ease from the scores on screen to the new day's scores.
function animateHeat(s, target, dur) {
  cancelAnimationFrame(s.anim);
  target = Object.fromEntries(Object.entries(target).filter(([, v]) => Number.isFinite(v)));
  const from = Object.fromEntries(Object.keys(target).map((k) => [k, Number.isFinite(s.shown?.[k]) ? s.shown[k] : 0]));
  const t0 = performance.now();
  const frame = (now) => {
    const t = Math.min(1, (now - t0) / dur), e = t * (2 - t);
    s.shown = Object.fromEntries(Object.keys(target).map((k) => [k, from[k] + (target[k] - from[k]) * e]));
    s.heat.draw(s.shown);
    if (t < 1) s.anim = requestAnimationFrame(frame);
  };
  s.anim = requestAnimationFrame(frame);
}
