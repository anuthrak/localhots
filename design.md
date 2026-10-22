# localhots — design

## Goal
One page (`web/`) with one story: fire history explains most fires; weather adds real value for lightning fires. It has to (1) show the ML project, (2) show useful data from the model and (3) stay simple to read. Anything that doesn't serve those goes in the collapsed technical details or the README.

## Page sections (web/src/App.jsx)
1. Hero: the question, the finding in two sentences, two buttons, 4 headline numbers (fires in the record, areas ranked, lightning fire days caught vs history alone and random).
2. Replay map (`#replay`): toggle between the lightning model (default) and the all-fire model, both on the same 299-area grid; it keeps the date through the URL hash. Date picker, prev/next, play season, busy-day jumps, heat glow, numbered badges on the top picks, actual fires as dots, season strip, side panel with the watch list and per-area weather.
3. What we found (`#findings`): one bar chart of fire days caught per fire type, fire history only vs the model, plus three takeaways (history wins for most fires, weather is the signal for lightning, ERC tells you when, not where).
4. How it works (`#how`): collect → describe → learn → rank; no peeking; always checked against a simple guess.
5. LinuxONE (`#linuxone`, dark section): the datathon's 24-area model and its runtime numbers on LinuxONE, pipeline diagram.
6. Live forecast (`#forecast`): tomorrow's lightning outlook on the same map and watch list, read from Vercel Blob (`FORECAST_URL` in `web/src/lib.js`), with how it's made and a backtest. Falls back to the "coming soon" plan if the file can't be loaded; shows the last forecast's date if tonight's hasn't run. Nightly: `.github/workflows/forecast.yml` → `ml/forecast.py run` (gridMET through its last day, about two days behind, then Open-Meteo up to today, rain averaged over five points per cell) → `forecast/lightning/latest.json` and `<date>.json`. Track record (`ml/track.py` → `forecast/lightning/track.json`): each night re-scores every forecast day against NIFC WFIGS fires with cause Natural (causes are often filled in late), showing the model's crew list vs fire history's vs random, live days apart from the reconstructed 2026 season. WFIGS labels only about 1 in 4 lightning fires, which the page says. Next: add thunderstorm signals.
7. Honest limits.
8. Technical details (`#details`, collapsed): lightning model metrics, PR curve, accuracy trap, per-year, importance; the three models compared against random; what changed after the datathon.

`/norcal/` and `/lightning/` were separate pages; they now redirect to `/` (`web/vercel.json`).

## API (app/server.py)
- `GET /api/overview`: model info, cells, dates, metrics, PR curve, years, importance, timeline, runtime.
- `GET /api/day?date=`: per-cell score, rank, alert, weather, anomalies, reasons; fires that day; summary. Re-scored live per request.
- `GET /api/health`.

## Style
Light, airy and editorial: lots of white space, near-black text, one warm orange accent, soft pastel tiles. All values live as CSS variables at the top of `app/static/style.css`.

**Colours**

| Token | Value | Used for |
|---|---|---|
| `--ink` | `#111111` | Text, primary buttons, rank badges |
| `--canvas` | `#ffffff` | Page and card background |
| `--cream` | `#f6f5f3` | Soft section bands, chips, map background |
| `--ash` | `#6d6c6b` | Secondary text, labels |
| `--stone` / `--pearl` | `#b1b1af` / `#ecebea` | Borders, muted controls |
| `--hairline` | `rgba(17,17,17,0.08)` | Card borders, dividers |
| `--phoenix` | `#e8400d` | The one accent: logo, highlights, "caught" bars |
| `--mint` / `--petal` | `#b7efb2` / `#ffd7f0` | Good / bad (verdicts, confusion matrix) |
| `--canary` / `--violet` | `#ffef99` / `#e2ddfd` | Pastel tiles, "coming soon" chip, hover |
| `--charcoal` | `#272625` | Dark LinuxONE section, footer |
| `--midnight` / `--indigo-deep` | `#10054d` / `#2e2460` | Dark cards inside the LinuxONE section |

Map heat ramp (in `HEAT` at the top of the map code in `app.js`, not CSS): clear → pale yellow → orange → phoenix → deep red, with low risk nearly transparent so the basemap shows through. The basemap is greyscaled so only the heat carries colour.

**Type**: Inter (Google Fonts). Headings are weight 400 with tight negative letter-spacing; 500–600 for labels and emphasis; 900 only for the hero display line. Eyebrows are 12px uppercase, ash.

**Shape**: 8px radius for buttons and inputs (`--r-btn`), 12px for cards and tiles (`--r-card`), full pills for chips and score tags. Flat cards with hairline borders, almost no shadows. Content max width 1200px with 24px side padding.

**Rules**
- No emojis anywhere on the site.
- One accent colour. Pastels are flat backgrounds, never text.
- Numbers come from the API; the HTML only holds fallbacks.

## Rules
- Never change the model or datasets in `~/wildfire-outlook` (mounted read-only).
- Scores are relative rankings, not probabilities. Don't claim accuracy as the headline.
- Change code in git, then deploy. Never edit files on the server.
