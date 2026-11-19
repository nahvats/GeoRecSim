"""City-level covariates for the cross-metro analysis in sim_results_analysis.ipynb.

Every quantity describes the *principal city* (the simulation's core: POIs and
work locations live there), matching params.METRO_PARAMS["core_places"]:
NYC = Manhattan, Bay Area = San Francisco + Oakland, others = city proper.

Sources (all public; nothing about the user is sent anywhere):
  * Population  -- Census Population Estimates Program (PEP) vintage 2023 CSVs:
                   2020 census base and July-2023 estimate. No API key needed.
  * Land area   -- Census 2023 Gazetteer files (ALAND_SQMI).       No key.
  * ACS 5-year  -- median household income, zero-vehicle household share,
                   transit and walk commute shares. The Census API now needs a
                   key: set CENSUS_API_KEY to fetch these, else they are NaN.
  * Network     -- OSM drive network of the core, via osmnx: intersection
                   density, street density, circuity, node degree, 4-way and
                   dead-end shares, street-orientation entropy.

Results are cached as CSV under <repo>/.cache/city_covariates/ (gitignored),
so only the first run downloads anything.
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, ".cache", "city_covariates")

# Census geography of each principal city. kind="county" uses the county file
# (Manhattan is New York County; SF city == SF county); kind="place" the place file.
CORE_GEOS = {
    "nyc":     [("county", "36", "061")],
    "seattle": [("place", "53", "63000")],
    "sf_bay":  [("county", "06", "075"), ("place", "06", "53000")],  # SF + Oakland
    "chicago": [("place", "17", "14000")],
    "houston": [("place", "48", "35000")],
    "dc":      [("place", "11", "50000")],
    "miami":   [("place", "12", "45000")],
    "la":      [("place", "06", "44000")],
}

# Same strings as params.METRO_PARAMS[...]["core_places"].
CORE_PLACES = {
    "nyc": ["Manhattan, New York, USA"],
    "seattle": ["Seattle, Washington, USA"],
    "sf_bay": ["San Francisco, California, USA", "Oakland, California, USA"],
    "chicago": ["Chicago, Illinois, USA"],
    "houston": ["Houston, Texas, USA"],
    "dc": ["Washington, District of Columbia, USA"],
    "miami": ["Miami, Florida, USA"],
    "la": ["Los Angeles, California, USA"],
}

PEP_COUNTY = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv"
PEP_PLACE = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/cities/totals/sub-est2023.csv"
GAZ_PLACE = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_place_national.zip"
GAZ_COUNTY = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip"
SQMI_TO_KM2 = 2.589988


def _cached_download(url: str) -> bytes:
    os.makedirs(os.path.join(CACHE, "raw"), exist_ok=True)
    fp = os.path.join(CACHE, "raw", os.path.basename(url))
    if not os.path.exists(fp):
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(fp, "wb") as f:
            f.write(r.content)
    with open(fp, "rb") as f:
        return f.read()


def _read_gazetteer(url: str) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(_cached_download(url)))
    df = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype={"GEOID": str})
    df.columns = [c.strip() for c in df.columns]
    return df


def census_basics() -> pd.DataFrame:
    """Population (2020 base, 2023 est.) and land area for each principal city."""
    co = pd.read_csv(io.BytesIO(_cached_download(PEP_COUNTY)), encoding="latin-1",
                     dtype={"STATE": str, "COUNTY": str})
    pl = pd.read_csv(io.BytesIO(_cached_download(PEP_PLACE)), encoding="latin-1",
                     dtype={"STATE": str, "PLACE": str, "SUMLEV": str})
    pl = pl[pl.SUMLEV == "162"]  # incorporated place totals
    gp = _read_gazetteer(GAZ_PLACE).set_index("GEOID")
    gc = _read_gazetteer(GAZ_COUNTY).set_index("GEOID")

    rows = []
    for city, geos in CORE_GEOS.items():
        pop20 = pop23 = area = 0.0
        for kind, st, code in geos:
            if kind == "county":
                r = co[(co.STATE == st) & (co.COUNTY == code)].iloc[0]
                g = gc.loc[st + code]
            else:
                r = pl[(pl.STATE == st) & (pl.PLACE == code)].iloc[0]
                g = gp.loc[st + code]
            pop20 += r["ESTIMATESBASE2020"]
            pop23 += r["POPESTIMATE2023"]
            area += float(g["ALAND_SQMI"])
        # pop_2020 is PEP's April-2020 estimates base (≈ census count, not identical)
        rows.append({"city": city, "pop_2020": pop20, "pop_2023": pop23,
                     "land_km2": area * SQMI_TO_KM2})
    df = pd.DataFrame(rows).set_index("city")
    df["density_per_km2"] = df.pop_2023 / df.land_km2
    df["pop_growth_20_23"] = df.pop_2023 / df.pop_2020 - 1
    return df


ACS_VARS = {
    "B19013_001E": "med_hh_income",
    "B08301_001E": "_workers", "B08301_010E": "_transit", "B08301_019E": "_walk",
    "B25044_001E": "_hh", "B25044_003E": "_own0", "B25044_010E": "_rent0",
}


def acs_optional(year: int = 2022) -> pd.DataFrame:
    """ACS 5-year variables; empty frame (all NaN later) when no CENSUS_API_KEY."""
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        return pd.DataFrame(index=list(CORE_GEOS))
    get = ",".join(ACS_VARS)
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    rows = []
    for city, geos in CORE_GEOS.items():
        tot = {v: 0.0 for v in ACS_VARS}
        inc_w = 0.0
        for kind, st, code in geos:
            geo = {"for": f"county:{code}"} if kind == "county" else {"for": f"place:{code}"}
            r = requests.get(base, params={"get": get, **geo, "in": f"state:{st}", "key": key},
                             timeout=60)
            r.raise_for_status()
            hdr, vals = r.json()
            d = {h: float(v) for h, v in zip(hdr, vals) if h in ACS_VARS}
            for v in ACS_VARS:
                if v != "B19013_001E":
                    tot[v] += d[v]
            # household-weighted median income when two geographies are pooled
            tot["B19013_001E"] += d["B19013_001E"] * d["B25044_001E"]
            inc_w += d["B25044_001E"]
        rows.append({
            "city": city,
            "med_hh_income": tot["B19013_001E"] / inc_w,
            "transit_commute_share": tot["B08301_010E"] / tot["B08301_001E"],
            "walk_commute_share": tot["B08301_019E"] / tot["B08301_001E"],
            "zero_vehicle_hh_share": (tot["B25044_003E"] + tot["B25044_010E"]) / tot["B25044_001E"],
        })
    return pd.DataFrame(rows).set_index("city")


def network_metrics_one(city: str) -> dict:
    import osmnx as ox

    fp = os.path.join(CACHE, f"network_{city}.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp).iloc[0].to_dict()
    ox.settings.use_cache = True
    ox.settings.cache_folder = os.path.join(CACHE, "osmnx_http")
    boundary = ox.geocode_to_gdf(CORE_PLACES[city])
    poly = boundary.union_all()
    area_m2 = float(ox.projection.project_gdf(boundary).union_all().area)
    # The public Overpass server sheds load with 502/429; back off and retry.
    # osmnx's HTTP cache keeps sub-queries that already succeeded.
    import time
    for attempt in range(8):
        try:
            G = ox.graph_from_polygon(poly, network_type="drive")
            break
        except Exception as e:  # osmnx raises ResponseStatusCodeError / ConnectionError
            if attempt == 7:
                raise
            wait = 30 * 2 ** attempt
            print(f"[{city}] Overpass error ({type(e).__name__}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    s = ox.stats.basic_stats(G, area=area_m2)
    Gu = ox.bearing.add_edge_bearings(ox.convert.to_undirected(G))
    spn = s.get("streets_per_node_proportions", {})
    out = {
        "city": city,
        "osm_area_km2": area_m2 / 1e6,
        "net_nodes": s["n"],
        "net_edges": s["m"],
        "intersection_density_km2": s["intersection_density_km"],
        "street_density_km_per_km2": s["street_density_km"] / 1000.0,
        "street_length_avg_m": s["street_length_avg"],
        "circuity_avg": s["circuity_avg"],
        "streets_per_node_avg": s["streets_per_node_avg"],
        "four_way_share": spn.get(4, 0.0),
        "dead_end_share": spn.get(1, 0.0),
        "orientation_entropy": ox.bearing.orientation_entropy(Gu),
    }
    os.makedirs(CACHE, exist_ok=True)
    pd.DataFrame([out]).to_csv(fp, index=False)
    return out


def network_metrics(cities=None) -> pd.DataFrame:
    cities = cities or list(CORE_PLACES)
    return pd.DataFrame([network_metrics_one(c) for c in cities]).set_index("city")


def all_covariates(with_network: bool = True) -> pd.DataFrame:
    df = census_basics().join(acs_optional())
    if with_network:
        net = network_metrics(list(df.index))
        df = df.join(net)
        # OSM/Nominatim admin polygons include water (e.g. Manhattan 87 km2 vs
        # 59 km2 of land; Seattle 373 vs 218), by a different factor per city.
        # Re-normalize per-area densities to Census LAND area so they are comparable.
        scale = df["osm_area_km2"] / df["land_km2"]
        df["intersection_density_km2"] = df["intersection_density_km2"] * scale
        df["street_density_km_per_km2"] = df["street_density_km_per_km2"] * scale
    return df


if __name__ == "__main__":
    import sys
    cities = sys.argv[1:] or None
    print(census_basics().round(2))
    for c in cities or list(CORE_PLACES):
        print(network_metrics_one(c), flush=True)
