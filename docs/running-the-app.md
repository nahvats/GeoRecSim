# Running GeoRecSim

How to install the engine, run the web app locally, and what the backend does
under the hood. For a hosted version, use <https://airecsim.cusp.nyu.edu/>.

## Install

You need Python 3.10 or newer, and Node 18 or newer for the frontend.

```bash
# engine + backend, editable
pip install -e .[server]

# or everything used in this repo (engine + backend + paper notebook)
pip install -r requirements.txt
```

The engine depends on `osmnx`, which brings in `geopandas`, `pandas` and
`shapely`. The first run for an area downloads that area's OpenStreetMap
network and caches it under `.cache/` (set `WELFARE_RS_CACHE` to move it).
After that, runs are fully local.

A metro's first build also geocodes its boundaries (Nominatim) and downloads
its graph (Overpass). That takes minutes to tens of minutes per metro, so it is
worth prefetching them once:

```bash
python scripts/warm_metros.py              # all eight metros
python scripts/warm_metros.py miami seattle
```

> **NumPy warning.** If you see `A module compiled using NumPy 1.x cannot be
> run in NumPy 2.x … _ARRAY_API not found`, that's `pyarrow` built against
> NumPy 1.x running under NumPy 2.x. It is a warning, not a failure. Fix it with
> `pip install -U pyarrow` or `pip install "numpy<2"`.

## Development: two processes

Backend:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Frontend (the Vite dev server proxies `/api` to port 8000):

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Open <http://localhost:5173>, pick a city, set up your recommenders and press
**Run**.

## Production-style: one server

Build the frontend once and FastAPI serves it as static files at `/`:

```bash
cd frontend && npm install && npm run build      # -> frontend/dist
cd ../backend && uvicorn main:app --port 8000     # http://localhost:8000
```

## How the map stays fast

The browser never holds a whole run. It streams only the agent positions for
the current time window and map viewport, and fetches a trip's full street
geometry on demand when you click a moving dot. That is what lets the map keep
up with large populations and long runs.

## Several studies at once

Studies don't queue behind each other. They share one process pool sized to
the machine, and `ResourceGovernor` splits its slots evenly between whoever is
running: one study gets all 60 slots, a second one arriving halves both to 30,
and the first widens back out when the second finishes. Nothing restarts; the
quota is re-read every time a task starts.

One shared pool, rather than a pool per study, is what keeps memory bounded.
Separate pools would each hold their own copy of every metro graph. A RAM floor
sits on the same path, so memory pressure delays new tasks instead of
triggering an out-of-memory kill that would take every running study with it.

| Variable | Default | Meaning |
|---|---|---|
| `WELFARE_RS_MAX_WORKERS` | CPU count | total process slots |
| `WELFARE_RS_MAX_STUDIES` | 10 | studies allowed to run at once |
| `WELFARE_RS_MIN_FREE_RAM_GB` | 24 | free RAM required before starting a task |
| `WELFARE_RS_CACHE` | `.cache/` | where networks and POI files are cached |
| `WELFARE_RS_DATASOURCE` | `local` | `local` or `postgres` (see [data.md](data.md)) |
| `WELFARE_RS_PG_DSN` | | Postgres connection string when using `postgres` |

## HTTP API

Everything lives under `/api`. Interactive docs are at `/docs` (Swagger).
`bbox` is `south,west,north,east`.

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/cities` | the 8 metros, recommender presets, POI availability, default config |
| POST | `/api/runs` | start a run (or return a cached one) and get its `run_id` |
| GET  | `/api/runs/{id}` | run summary, view bounds and metrics |
| GET  | `/api/runs/{id}/events` | server-sent events with per-day progress |
| GET  | `/api/runs/{id}/timeline?t0&t1&bbox` | trip and stay keyframes for a time window, clipped to the viewport |
| GET  | `/api/runs/{id}/positions?t&bbox` | agent positions at one instant, interpolated along real routes |
| GET  | `/api/runs/{id}/trips/{trip_id}/geometry` | full street polyline for one trip |
| GET  | `/api/runs/{id}/pois?bbox` | venues inside the viewport |
| GET  | `/api/resources` | studies running, slots each, slots busy, RAM headroom |
