import { useEffect, useState } from "react";
import Replay from "./components/Replay.jsx";
import { Footer, Nav } from "./components/Sections.jsx";
import { NorcalChanges, NorcalHero, NorcalLimits, NorcalResults } from "./components/NorcalSections.jsx";
import { expandDay, getJSON, setCellNames } from "./lib.js";

const API = "api-norcal";
const LINKS = [["#replay", "Replay map"], ["#how", "What changed"], ["#results", "Results"]];
const CTA = [import.meta.env.BASE_URL, "Original map"];

export default function Norcal() {
  const [ov, setOv] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getJSON(`${API}/overview.json`).then((o) => { setCellNames(o.cells); setOv(o); }, () => setError(true));
  }, []);

  return (
    <>
      <Nav ov={ov} error={error} links={LINKS} cta={CTA} chip={ov && `${ov.cells.length} cells · ${ov.region.step}°`} />
      <main id="top">
        <NorcalHero ov={ov} />
        {error && <section className="section"><div className="wrap"><h3>Couldn't load the model data.</h3></div></section>}
        {ov && (
          <>
            <Replay ov={ov} api={API} expand={(d) => expandDay(ov, d)} />
            <NorcalChanges ov={ov} />
            <NorcalResults ov={ov} />
            <NorcalLimits ov={ov} />
          </>
        )}
      </main>
      <Footer sources="Fire data: US Forest Service, Fire Program Analysis Fire-Occurrence Database (FPA FOD), 7th edition. Weather: gridMET (University of California Merced / Climatology Lab). Place names: US Census Bureau Gazetteer. Map tiles © OpenStreetMap contributors. Research model, not an official warning service." />
    </>
  );
}
