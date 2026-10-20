"""Download fire records and gridMET weather for a region.

    python ml/fetch_data.py --region norcal --out <data dir>

Writes <out>/fires.csv (FPA FOD points in the region's box), <out>/gridmet/<var>_<year>.nc,
<out>/places.tsv (US Census places, for naming cells) and <out>/sources.json (URLs, counts, checksums).
Files already downloaded are skipped, so it can be re-run after a dropped connection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from regions import END, REGIONS, WEATHER_START

FIRE_URL = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_FireOccurrenceCurrentEdition_01/MapServer/0/query"
FIRE_FIELDS = ["objectid", "fod_id", "fire_year", "discovery_date", "latitude", "longitude", "state"]
GRIDMET_URL = "https://www.reacchpna.org/thredds/ncss/grid/MET/{var}/{var}_{year}.nc"
GRIDMET_VARS = {"tmmx": "air_temperature", "rmin": "relative_humidity", "vs": "wind_speed", "pr": "precipitation_amount",
                # US fire danger indices (NFDRS fuel model G), used as baselines in ml/compare_indices.py
                "erc": "energy_release_component-g", "bi": "burning_index_g"}
PLACES_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_place_national.zip"
PAGE = 500


def log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


def request(url: str, params: dict | None = None, post: bool = False, tries: int = 5) -> bytes:
    query = urllib.parse.urlencode(params or {})
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=query.encode()) if post else urllib.request.Request(f"{url}?{query}" if query else url)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except OSError as e:
            if attempt == tries - 1:
                raise
            log(f"retrying after {e}")
            time.sleep(5 * (attempt + 1))


def fetch_fires(bounds, out: Path) -> dict:
    path = out / "fires.csv"
    base = {"where": f"fire_year >= 2012 AND fire_year <= {END[:4]}", "geometry": ",".join(map(str, bounds)),
            "geometryType": "esriGeometryEnvelope", "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "f": "json"}
    if path.exists():
        log(f"{path.name} exists, skipping")
    else:
        ids = sorted(json.loads(request(FIRE_URL, {**base, "returnIdsOnly": "true"}, post=True))["objectIds"])
        log(f"{len(ids):,} fire records to download")
        rows = []
        for i in range(0, len(ids), PAGE):
            page = json.loads(request(FIRE_URL, {"objectIds": ",".join(map(str, ids[i:i + PAGE])), "outFields": ",".join(FIRE_FIELDS),
                                                 "returnGeometry": "false", "f": "json"}, post=True))
            rows += [f["attributes"] for f in page["features"]]
            if i // PAGE % 20 == 0:
                log(f"fires {len(rows):,}/{len(ids):,}")
        if len(rows) != len(ids):
            raise RuntimeError(f"expected {len(ids)} fire records, got {len(rows)}")
        tmp = path.with_suffix(".part")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, FIRE_FIELDS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["objectid"]))
        tmp.replace(path)
    return {"url": FIRE_URL, "query": base, "records": sum(1 for _ in path.open()) - 1, "sha256": sha256(path)}


def fetch_gridmet(bounds, out: Path) -> list[dict]:
    west, south, east, north = bounds
    folder = out / "gridmet"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for year in range(int(WEATHER_START[:4]), int(END[:4]) + 1):
        start = WEATHER_START if year == int(WEATHER_START[:4]) else f"{year}-01-01"
        for var, name in GRIDMET_VARS.items():
            path = folder / f"{var}_{year}.nc"
            url = GRIDMET_URL.format(var=var, year=year)
            params = {"var": name, "north": north, "south": south, "west": west, "east": east,
                      "time_start": f"{start}T00:00:00Z", "time_end": f"{year}-12-31T00:00:00Z", "accept": "netcdf"}
            if not path.exists():
                data = request(url, params)
                if not data.startswith(b"CDF"):
                    raise RuntimeError(f"{path.name}: not a netCDF file: {data[:200]!r}")
                path.with_suffix(".part").write_bytes(data)
                path.with_suffix(".part").replace(path)
                log(f"{path.name} {len(data) / 1e6:.1f} MB")
            files.append({"path": f"gridmet/{path.name}", "url": f"{url}?{urllib.parse.urlencode(params)}", "sha256": sha256(path)})
    return files


def fetch_places(bounds, out: Path) -> dict:
    """Census places (cities, towns, CDPs) inside the box; public domain."""
    path = out / "places.tsv"
    if not path.exists():
        with zipfile.ZipFile(io.BytesIO(request(PLACES_URL))) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        lines = text.splitlines()
        header = [h.strip() for h in lines[0].split("\t")]
        lat, lon = header.index("INTPTLAT"), header.index("INTPTLONG")
        west, south, east, north = bounds
        keep = [line for line in lines[1:] if south <= float(line.split("\t")[lat]) < north and west <= float(line.split("\t")[lon]) < east]
        path.write_text("\n".join(["\t".join(header)] + [line.rstrip() for line in keep]) + "\n", encoding="utf-8")
        log(f"{len(keep):,} places")
    return {"url": PLACES_URL, "sha256": sha256(path)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--region", choices=REGIONS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    region = REGIONS[args.region]
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {"region": args.region, **region, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
               "fires": fetch_fires(region["bounds"], args.out), "gridmet": fetch_gridmet(region["bounds"], args.out),
               "places": fetch_places(region["bounds"], args.out)}
    (args.out / "sources.json").write_text(json.dumps(sources, indent=2))
    log(f"done: {sources['fires']['records']:,} fires, {len(sources['gridmet'])} gridMET files")


if __name__ == "__main__":
    main()
