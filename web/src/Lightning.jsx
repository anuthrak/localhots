import { useEffect, useState } from "react";
import Replay from "./components/Replay.jsx";
import { Footer, Nav } from "./components/Sections.jsx";
import { LightningExperiment, LightningHero, LightningLimits, LightningResults } from "./components/LightningSections.jsx";
import { NORCAL_SOURCES } from "./Norcal.jsx";
import { expandDay, getJSON, setCellNames } from "./lib.js";

const API = "api-lightning";
const LINKS = [["#replay", "Replay map"], ["#experiment", "Why lightning"], ["#results", "Results"]];
const CTA = [`${import.meta.env.BASE_URL}norcal/`, "All-fire map"];

export default function Lightning() {
  const [ov, setOv] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getJSON(`${API}/overview.json`).then((o) => { setCellNames(o.cells); setOv(o); }, () => setError(true));
  }, []);

  return (
    <>
      <Nav ov={ov} error={error} links={LINKS} cta={CTA} chip={ov && `Lightning only · ${ov.cells.length} cells`} />
      <main id="top">
        <LightningHero ov={ov} />
        {error && <section className="section"><div className="wrap"><h3>Couldn't load the model data.</h3></div></section>}
        {ov && (
          <>
            <Replay ov={ov} api={API} expand={(d) => expandDay(ov, d)} />
            <LightningExperiment ov={ov} />
            <LightningResults ov={ov} />
            <LightningLimits ov={ov} />
          </>
        )}
      </main>
      <Footer sources={NORCAL_SOURCES} />
    </>
  );
}
