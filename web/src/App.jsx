import { useEffect, useState } from "react";
import Replay from "./components/Replay.jsx";
import { Details, Findings, Footer, Forecast, Hero, HowItWorks, LinuxOne, Limits, Nav } from "./components/Sections.jsx";
import { expandDay, getJSON, setCellNames } from "./lib.js";

// Both map modes share the same 299-area grid; "base" is the datathon's 24-area model (LinuxONE numbers).
const MODES = [
  { key: "lightning", label: "Lightning fires", api: "api-lightning", note: "Showing the lightning model, where weather adds the most. Dots are fires started by lightning." },
  { key: "all", label: "All fires", api: "api-norcal", note: "Showing the all-fire model. It mostly follows each area's fire history, because weather adds little for fires people start. Dots are every fire discovered." },
];

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const [mode, setMode] = useState("lightning");

  useEffect(() => {
    Promise.all(["api-lightning", "api-norcal", "api"].map((a) => getJSON(`${a}/overview.json`))).then(([lightning, all, base]) => {
      setCellNames(all.cells);
      setData({ lightning, all, base });
    }, () => setError(true));
  }, []);

  const m = MODES.find((x) => x.key === mode);
  const ov = data && (mode === "lightning" ? data.lightning : data.all);

  return (
    <>
      <Nav base={data?.base} error={error} />
      <main id="top">
        <Hero all={data?.all} lightning={data?.lightning} />
        {error && <section className="section"><div className="wrap"><h3>Couldn't load the model data.</h3></div></section>}
        {data && (
          <>
            {/* Keyed by mode so the replay starts fresh; it keeps the date through the URL hash. */}
            <Replay key={mode} ov={ov} api={m.api} expand={(d) => expandDay(ov, d)} note={m.note}
              modes={MODES} mode={mode} onMode={setMode} />
            <Findings lightning={data.lightning} />
            <HowItWorks all={data.all} />
            <LinuxOne ov={data.base} />
          </>
        )}
        <Forecast all={data?.all} />
        {data && (
          <>
            <Limits lightning={data.lightning} />
            <Details all={data.all} lightning={data.lightning} base={data.base} />
          </>
        )}
      </main>
      <Footer />
    </>
  );
}
