# localhots — design

## Goal
One page that shows judges (1) the replay map, (2) the results, (3) that everything runs on LinuxONE, and (4) the planned live forecast.

## Page sections (app/static/index.html)
1. Hero: the question, two buttons, 4 headline numbers on a warm orange-to-violet wash.
2. Replay map (`#replay`): date picker, prev/next, play season, busy-day jumps. A smooth heat glow blended from the 24 area scores, with a white wash over everything the model doesn't cover; numbered badges on the top 3 areas; actual fires as dots. Season strip below (fires per day, orange = top 3 caught one). Side panel: watch list with reasons and per-area weather.
3. How it works (`#how`): collect → describe → learn → rank; permutation importance chart.
4. Results (`#results`): AP, ROC AUC, top-3 recall, accuracy trap + confusion matrix, PR curve, per-year AP, top-3 vs random.
5. LinuxONE (`#linuxone`, dark section): live runtime numbers from the server, pipeline diagram.
6. Live forecast (`#forecast`): RESERVED ("coming soon"). Plan: nightly Open-Meteo forecast → same 27 features → score → publish tomorrow's map → score it next day. Needs a forecast-feature builder that matches training, plus `/api/forecast`.
7. Honest limits, footer with data attribution.

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
