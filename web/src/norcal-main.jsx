import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Norcal from "./Norcal.jsx";
import "./style.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Norcal />
  </StrictMode>,
);
