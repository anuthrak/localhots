# localhots

Where will the next wildfire start? **localhots** ranks 24 areas of Northern California each day by how likely a new wildfire is *tomorrow*, and lets you replay the 2022–2024 fire seasons to see how often it pointed the right way.

Built by a team of two at the **IBM Z Datathon 2026** (Oct 17–18, 2026), originally served from an IBM LinuxONE (s390x) mainframe. This repo is the continuation, being polished as a portfolio project.

> The name: during the datathon we kept accidentally sending each other `localhost` links. It's also *local hots*: local hot spots.

## What's in it

| Part | Tech |
|---|---|
| Model | Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`), 27 weather and history features, trained on 2013–2021, tested on 2022–2024 |
| Server | Python standard-library HTTP server with a small JSON API; scores each day live |
| Website | HTML/CSS/JavaScript, Leaflet map with a canvas heat layer, Chart.js |
| Original hosting | Docker on IBM LinuxONE (s390x) |

Results on the 2022–2024 test years: average precision 0.438 (random: 0.119), ROC AUC 0.848, and checking just the top 3 of 24 areas catches 34% of fire days (random: 12.5%).

## Layout

| Path | What it is |
|---|---|
| `app/server.py` | API + static file server (`/api/overview`, `/api/day?date=YYYY-MM-DD`, `/api/health`) |
| `app/static/` | The website |
| `ml/train.py` | Rebuilds the model from the dataset; reproduces the original scores exactly (`--compare`) |
| `deploy/linuxone/run.sh` | How it ran on LinuxONE during the datathon |
| `design.md` | Page plan, API and visual style |

The trained model and datasets are not in this repo yet.

## Credits

- Team: Anuthrak (website, map and replay experience); teammate (model training, data pipeline, LinuxONE setup).
- Fire records: US Forest Service, Fire Program Analysis Fire-Occurrence Database (FPA FOD), 7th edition.
- Weather: Open-Meteo historical API (Copernicus ERA5 reanalysis), CC BY 4.0.
- Fuel moisture: gridMET. Elevation: USGS 3DEP.
- Libraries: [scikit-learn](https://scikit-learn.org), [NumPy](https://numpy.org), [pandas](https://pandas.pydata.org), [Leaflet](https://leafletjs.com), [Chart.js](https://www.chartjs.org). Map tiles © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors.
- Built with help from AI coding assistants (Claude Code).
- Infrastructure: [IBM LinuxONE Community Cloud](https://community.ibm.com/zsystems/l1cc/).
