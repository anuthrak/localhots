# localhots

Where will the next wildfire start? **localhots** ranks 299 areas of Northern California each day by how likely a new wildfire is *tomorrow*, and lets you replay the 2022–2024 fire seasons to see how often it pointed the right way.

The main finding: each area's fire history explains most fires, because people start them where people are, so weather adds little. Lightning fires are the exception. A lightning-only model catches 40% of lightning fire days by checking one area in eight, against 32% for fire history alone and 12% for random picks.

Built by a team of four at the **IBM Z Datathon 2026** (Oct 17–18, 2026), originally served from an IBM LinuxONE (s390x) mainframe. This repo is the continuation, being polished as a portfolio project.

> The name: during the datathon we kept accidentally sending each other `localhost` links. It's also *local hots*: local hot spots.

## What's in it

| Part | Tech |
|---|---|
| Model | Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`), 27 weather and history features, trained on 2013–2021, tested on 2022–2024 |
| Server | Python standard-library HTTP server with a small JSON API; scores each day live |
| Website | React + Vite, Leaflet map with a canvas heat layer, Chart.js; hosted on Vercel as static files |
| Original hosting | Docker on IBM LinuxONE (s390x) |

The original datathon model ranked 24 areas: on the 2022–2024 test years, checking its top 3 caught 34% of fire days (random: 12.5%). The site now shows the 299-area models, with the datathon numbers under its technical details.

## Layout

| Path | What it is |
|---|---|
| `web/` | The website (React + Vite). `npm run dev` after exporting the data |
| `app/server.py` | Python API that scores the model (`/api/overview.json`, `/api/day/YYYY-MM-DD.json`, `/api/health`); also serves the archived site |
| `deploy/vercel/export.py` | Writes every API answer to `web/public/api/` as static JSON for the React site (the datathon model) |
| `ml/forecast.py` | The live lightning forecast: `pack` bundles the model for nightly runs, `run` scores tomorrow from gridMET plus the Open-Meteo forecast |
| `ml/track.py` | The track record: scores each forecast's crew list against lightning fires reported to NIFC, against fire history alone and random picks; `backfill` reconstructs past days' forecasts |
| `.github/workflows/forecast.yml` | Runs the forecast and the track record every evening and publishes them to Vercel Blob, where the site reads them |
| `deploy/vercel/export_norcal.py` | Writes the 299-area all-fire and lightning models' data to `web/public/api-norcal/` and `web/public/api-lightning/` |
| `archive/static-site/` | The original plain HTML/CSS/JS site from the datathon |
| `ml/train.py` | Rebuilds the model from the dataset; reproduces the original scores exactly (`--compare`) |
| `deploy/linuxone/run.sh` | How it ran on LinuxONE during the datathon |
| `design.md` | Page plan, API and visual style |

The trained model and datasets are not in this repo. The nightly forecast gets its model from the `forecast-kit` release asset; after retraining, re-pack and replace it:

```
python ml/forecast.py pack --data private/norcal --out private/forecast_kit.joblib
gh release upload forecast-kit private/forecast_kit.joblib --clobber
```

## Credits

- Team: Aleksandr Evtushenko, Bing He, Anuthrak Ratana, Kevin Ye.
- Fire records: US Forest Service, Fire Program Analysis Fire-Occurrence Database (FPA FOD), 7th edition.
- Weather: Open-Meteo historical API (Copernicus ERA5 reanalysis), CC BY 4.0.
- Fuel moisture: gridMET. Elevation: USGS 3DEP.
- Libraries: [scikit-learn](https://scikit-learn.org), [NumPy](https://numpy.org), [pandas](https://pandas.pydata.org), [Leaflet](https://leafletjs.com), [Chart.js](https://www.chartjs.org). Map tiles © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors.
- Built with help from AI coding assistants (Claude Code).
- Infrastructure: [IBM LinuxONE Community Cloud](https://community.ibm.com/zsystems/l1cc/).
