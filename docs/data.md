# Data: where everything comes from

**No data lives in git.** The census inputs are downloaded by scripts, the
venue catalog is a licensed dataset you supply yourself, and the survey
population is IRB-protected and never leaves institutional storage. The repo
carries the scripts that rebuild each piece. To see the engine run without any
of it, use the demo described in the main [README](../README.md#demo).

## Eight metros, two layers each

GeoRecSim covers 8 US metros: New York, Seattle, San Francisco Bay Area,
Chicago, Houston, Washington DC, Miami and Los Angeles. Each is one road
network with two layers:

- **Core**: the principal city (Manhattan for NYC; SF + Oakland for the Bay
  Area), at full street detail. All venues and workplaces are here, and
  leisure trips are routed street by street.
- **Shell**: the surrounding counties that together send 90% of commuters into
  the core, reduced to major roads (motorway, trunk, primary, secondary).
  Dropping residential streets outside the core is the standard abstraction in
  regional travel models and keeps metro graphs tractable.

Cars route on the fastest network path using per-edge OSM speeds, so freeway
commutes behave like freeway commutes. Congestion and weather scale that time.
Walk, bike-share and taxi are described in [multimodal.md](multimodal.md).

## Where agents live and work

Homes and workplaces come from real LODES8 commutes (2023, JT01 primary jobs).
Each agent is one draw from `metro_od_pairs`, a home census block → work census
block pair weighted by its job count, with both blocks snapped onto the graph.
Because the pair is drawn whole, home and work stay correlated and commute
lengths follow the published distribution. The counties that make up the shell
come from the same data grouped by home county, so the graph always covers the
homes it needs. See `db/07_commute_tables.sql`.

Without commute data the engine falls back to a county-weighted home and a
uniform workplace inside the core.

## Who the agents are

Populations are built offline and stored, one row per agent, from three
sources, each used for what only it can provide:

- **LODES** places the agent: a real home → work commute, which also fixes that
  job's age, earnings and industry band.
- **ACS PUMS**, reweighted to the home tract, says who lives there: one real
  person's age, earnings, household income, vehicles, disability and household
  size, jointly.
- **The survey** (473 respondents, IRB-FY2026-11354) supplies personality and
  platform attitudes, matched to that person on sex, employment, income band
  and age band.

People who don't work are drawn from core blocks only. Their leisure trip
starts from home rather than a workplace, and metro-wide homes would put many
of them 30–60 km from every venue. The population is therefore people whose day
is anchored in the core.

Build it with:

```bash
python db/fetch_census_data.py
python db/load_pums.py
python db/load_acs.py
python db/build_tract_weights.py
python db/build_population.py
python scripts/verify_acs_population.py
```

Full design and limitations: [acs-population.md](acs-population.md). What the
survey does and doesn't ground: [survey-coverage.md](survey-coverage.md).

## Venues

Real venues come from per-metro CSVs (`<cache>/pois/<metro>_leisure_pois.csv`),
each produced by one streaming pass of `data/filter_metro_pois.py` over the raw
Placekey/NAICS dump (`data/poi_children_merged_by_wkt.csv`, about 375 MB,
gitignored), clipped to that metro's core polygon.

The raw dump on hand covers the NYC region only, so locally only NYC has real
venues; the other metros fall back to synthetic venues placed on core-city
intersections (the "Real POIs" toggle disables itself per city). Nationwide
venues come from the server database (`metro_pois` table).

## Local files or Postgres

`welfare_rs/datasource.py` is the one place that knows where data comes from:

- `LocalDataSource` (default) reads the local files above.
- `PostgresDataSource` serves identical shapes from the server database. Set
  `WELFARE_RS_DATASOURCE=postgres` and `WELFARE_RS_PG_DSN=postgresql://…`.
  Table schemas are documented in that module and in `db/`.
