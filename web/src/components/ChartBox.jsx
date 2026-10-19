import { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.color = "#6d6c6b";
Chart.defaults.plugins.legend.display = false;
Chart.defaults.maintainAspectRatio = false;

export const grid = { color: "rgba(17,17,17,0.06)" };

export default function ChartBox({ config, short = false }) {
  const canvas = useRef(null);
  useEffect(() => {
    const chart = new Chart(canvas.current, config);
    return () => chart.destroy();
  }, [config]);
  return <div className={`chart-box${short ? " short" : ""}`}><canvas ref={canvas} /></div>;
}
