"""Model regions: bounding box (west, south, east, north) in degrees and grid cell size."""

REGIONS = {
    # The datathon model: 24 cells of 0.5 degrees. Weather from Open-Meteo (ERA5), one point per cell.
    "norcal24": {"name": "Northern California (datathon)", "bounds": (-123.0, 38.5, -121.0, 41.5), "step": 0.5},
    # Bay Area to the Oregon border, coast to the Sierra. Weather from gridMET, averaged over each cell.
    "norcal": {"name": "Northern California", "bounds": (-124.5, 37.0, -120.0, 42.0), "step": 0.25},
}

START, END = "2012-01-01", "2024-12-31"  # target days
WEATHER_START = "2011-12-01"             # 30-day windows need a month of warm-up
