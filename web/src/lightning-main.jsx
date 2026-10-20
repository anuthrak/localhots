import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Lightning from "./Lightning.jsx";
import "./style.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Lightning />
  </StrictMode>,
);
