import { useEffect, useState } from "react";
import Replay from "./components/Replay.jsx";
import { Footer, Forecast, Hero, HowItWorks, LinuxOne, Limits, Nav, Results } from "./components/Sections.jsx";
import { getJSON } from "./lib.js";

export default function App() {
  const [ov, setOv] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getJSON("api/overview.json").then(setOv, () => setError(true));
  }, []);

  return (
    <>
      <Nav ov={ov} error={error} />
      <main id="top">
        <Hero ov={ov} />
        {error && <section className="section"><div className="wrap"><h3>Couldn't load the model data.</h3></div></section>}
        {ov && (
          <>
            <Replay ov={ov} />
            <HowItWorks ov={ov} />
            <Results ov={ov} />
            <LinuxOne ov={ov} />
          </>
        )}
        <Forecast />
        <Limits />
      </main>
      <Footer />
    </>
  );
}
