"""City-level covariates for the cross-metro analysis in sim_results_analysis.ipynb.

Every quantity describes the *principal city* (the simulation's core: POIs and
work locations live there), matching params.METRO_PARAMS["core_places"]:
NYC = Manhattan, Bay Area = San Francisco + Oakland, others = city proper.

Sources (all public; nothing about the user is sent anywhere):
  * Population  -- Census Population Estimates Program (PEP) vintage 2023 CSVs:
                   2020 census base and July-2023 estimate. No API key needed.
  * Land area   -- Census 2023 Gazetteer files (ALAND_SQMI).       No key.
  * ACS 5-year  -- 2019-2023: interpolated median household income, zero-vehicle
                   household share, transit and walk commute shares. Read from
                   the welfare Postgres (WELFARE_RS_PG_DSN); see acs_postgres().
  * Network     -- OSM drive network of the core (the simulation's own cached
                   metro graph clipped to the core; Overpass only as fallback): intersection
                   density, street density, circuity, node degree, 4-way and
                   dead-end shares, street-orientation entropy.

Results are cached as CSV under <repo>/.cache/city_covariates/ (gitignored),
so only the first run downloads anything.
"""
from __future__ import annotations

import io
import os
import sys
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


# ---------------------------------------------------------------------------
# ACS 2019-2023 5-year, from the welfare Postgres (census.acs_estimates).
#
# That table holds counties (sumlevel 50), tracts (140) and block groups (150)
# only -- no places -- and does not include B19013 (median household income).
# So:
#   * county geographies (Manhattan, San Francisco) are read exactly at sumlevel 50;
#   * places are built from 2020 block groups, apportioned by the share of each
#     block group's 2020 housing units (household tables) or 2020 population
#     (B08301, workers) that lies inside the place. Block -> place comes from the
#     Census 2020 Block Assignment Files (public download, no key); block counts
#     from census.blocks (2020 decennial). 2020 blocks nest exactly in places,
#     so the only approximation is splitting block groups that straddle a city
#     boundary, which assumes ACS characteristics are uniform within that BG;
#   * median household income is linearly interpolated from the 16-bin B19001
#     distribution (pooled for SF + Oakland). It approximates, but is not, the
#     published B19013 median. For rank-based (Spearman) use this is immaterial.
# Commute shares are over all workers 16+ (B08301_001, includes work-from-home).
# ---------------------------------------------------------------------------
ACS_TABLE_VINTAGE = "ACS 2019-2023 5-year"
BAF_URL = "https://www2.census.gov/geo/docs/maps-data/data/baf2020/BlockAssign_ST{st}_{abbr}.zip"
STATE_ABBR = {"06": "CA", "11": "DC", "12": "FL", "17": "IL", "48": "TX", "53": "WA", "36": "NY"}
B19001_EDGES = [0, 10e3, 15e3, 20e3, 25e3, 30e3, 35e3, 40e3, 45e3, 50e3, 60e3, 75e3,
                100e3, 125e3, 150e3, 200e3, np.inf]   # varnum 2..17


def _place_blocks(st: str, place: str) -> set[str]:
    raw = _cached_download(BAF_URL.format(st=st, abbr=STATE_ABBR[st]))
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = next(n for n in z.namelist() if n.endswith("_INCPLACE_CDP.txt"))
    df = pd.read_csv(z.open(name), sep="|", dtype=str)
    return set(df.loc[df.PLACEFP == place, "BLOCKID"])


def _interp_median(counts: np.ndarray) -> float:
    """Median of a binned distribution by linear interpolation within the bin."""
    cum = np.cumsum(counts)
    half = cum[-1] / 2
    i = int(np.searchsorted(cum, half))
    lo, hi = B19001_EDGES[i], B19001_EDGES[i + 1]
    if not np.isfinite(hi):
        return np.nan  # median in the open-ended top bin: undefined by interpolation
    below = cum[i - 1] if i else 0.0
    return lo + (half - below) / counts[i] * (hi - lo)


def _acs_counts(conn, kind: str, st: str, code: str) -> tuple[dict, dict]:
    """{(table, varnum): estimate} for one geography, plus diagnostics."""
    tabs = ("b19001", "b08301", "b25044")
    if kind == "county":
        rows = conn.execute(
            "SELECT table_id, varnum, estimate FROM census.acs_estimates "
            "WHERE sumlevel = 50 AND geoid = %s AND table_id = ANY(%s)", (st + code, list(tabs))).fetchall()
        return {(t, v): e for t, v, e in rows}, {"method": "county (exact)"}
    blocks = _place_blocks(st, code)
    b = pd.DataFrame(conn.execute(
        "SELECT geoid20, pop20, housing20 FROM census.blocks WHERE statefp = %s", (st,)).fetchall(),
        columns=["geoid20", "pop20", "housing20"])
    b["bg"] = b.geoid20.str[:12]
    b["inside"] = b.geoid20.isin(blocks)
    g = b.groupby("bg").agg(pop_all=("pop20", "sum"), hu=("housing20", "sum"))
    gi = b[b.inside].groupby("bg").agg(pop_in=("pop20", "sum"), hu_in=("housing20", "sum"))
    g = gi.join(g)
    g["w_pop"] = (g.pop_in / g.pop_all.where(g.pop_all > 0)).fillna(0)
    g["w_hu"] = (g.hu_in / g.hu.where(g.hu > 0)).fillna(0)
    acs = pd.DataFrame(conn.execute(
        "SELECT geoid, table_id, varnum, estimate FROM census.acs_estimates "
        "WHERE sumlevel = 150 AND geoid = ANY(%s) AND table_id = ANY(%s)",
        (list(g.index), list(tabs))).fetchall(), columns=["bg", "table_id", "varnum", "est"])
    acs = acs.join(g[["w_pop", "w_hu"]], on="bg")
    acs["w"] = np.where(acs.table_id == "b08301", acs.w_pop, acs.w_hu)
    out = (acs.est * acs.w).groupby([acs.table_id, acs.varnum]).sum().to_dict()
    have = set(acs.bg)
    diag = {"method": "block-group apportionment",
            "place_pop20_blocks": int(b.loc[b.inside, "pop20"].sum()),
            "bgs_touching_place": len(g),
            "bgs_split_by_boundary": int(((g.w_hu > 0) & (g.w_hu < 1)).sum()),
            "pop20_in_split_bgs_share": float(g.loc[(g.w_pop > 0) & (g.w_pop < 1), "pop_in"].sum() / g.pop_in.sum()),
            "pop20_missing_acs_share": float(g.loc[~g.index.isin(have), "pop_in"].sum() / g.pop_in.sum())}
    return out, diag


def acs_postgres(dsn: str | None = None) -> pd.DataFrame:
    """ACS covariates per principal city from census.acs_estimates (see block comment)."""
    fp = os.path.join(CACHE, "acs_postgres.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp).set_index("city")
    import psycopg

    dsn = dsn or os.environ.get("WELFARE_RS_PG_DSN")
    if not dsn:
        raise RuntimeError("WELFARE_RS_PG_DSN is not set (source .env.server)")
    rows = []
    with psycopg.connect(dsn) as conn:
        for city, geos in CORE_GEOS.items():
            tot: dict = {}
            diags = []
            for kind, st, code in geos:
                d, diag = _acs_counts(conn, kind, st, code)
                for k, v in d.items():
                    tot[k] = tot.get(k, 0.0) + v
                diags.append(diag)
            inc = np.array([tot[("b19001", v)] for v in range(2, 18)])
            rows.append({
                "city": city,
                "med_hh_income_interp": _interp_median(inc),
                "transit_commute_share": tot[("b08301", 10)] / tot[("b08301", 1)],
                "walk_commute_share": tot[("b08301", 19)] / tot[("b08301", 1)],
                "zero_vehicle_hh_share": (tot[("b25044", 3)] + tot[("b25044", 10)]) / tot[("b25044", 1)],
                "acs_households": tot[("b25044", 1)],
                "acs_workers": tot[("b08301", 1)],
                "acs_method": "; ".join(x["method"] for x in diags),
                "acs_pop20_in_split_bgs_share": max((x.get("pop20_in_split_bgs_share", 0.0) for x in diags)),
                "acs_pop20_missing_share": max((x.get("pop20_missing_acs_share", 0.0) for x in diags)),
                "place_pop20_blocks": sum(x.get("place_pop20_blocks", 0) for x in diags) or np.nan,
            })
    df = pd.DataFrame(rows).set_index("city")
    os.makedirs(CACHE, exist_ok=True)
    df.to_csv(fp)
    return df


ACS_COLS = ["med_hh_income_interp", "transit_commute_share", "walk_commute_share", "zero_vehicle_hh_share"]


def acs_optional() -> pd.DataFrame:
    """ACS columns from Postgres; empty frame (NaN later) if the DB is unreachable."""
    try:
        return acs_postgres()[ACS_COLS]
    except Exception as e:  # off the server: no DSN / no psycopg
        print(f"ACS skipped ({type(e).__name__}: {e})")
        return pd.DataFrame(index=list(CORE_GEOS))


SIM_CACHE = os.path.join(REPO, ".cache")   # where welfare_rs keeps boundaries/ and networks/


def _core_graph(city: str):
    """Drive network of the principal city, and the core polygon it covers.

    Preferred source: the simulation's own metro graph
    (``.cache/networks/<city>_metro.graphml``), clipped to the core polygon that
    welfare_rs.metro built it from (union of the geocoded ``core_places``, cached
    in ``.cache/boundaries``). Inside the core that graph is the full osmnx
    ``network_type="drive"`` download, so the clip is the core drive network
    the agents actually route on -- no Overpass call. Two differences from a
    fresh ``graph_from_polygon``: the metro graph was cut to its largest
    *strongly* connected component (drops a few one-way stubs), and all
    components inside the core are kept (SF and Oakland are separate pieces).
    Node ``street_count`` was computed by osmnx on the buffered download before
    truncation, so boundary nodes keep their true street counts.

    Fallback (no cached graph, e.g. off the server): Overpass via osmnx.
    """
    import osmnx as ox
    from shapely.ops import unary_union

    sys.path.insert(0, REPO)
    from welfare_rs import metro as wm

    poly = unary_union([wm.geocode_polygon(q, SIM_CACHE) for q in CORE_PLACES[city]])
    gml = os.path.join(SIM_CACHE, "networks", f"{city}_metro.graphml")
    if os.path.exists(gml):
        G = ox.load_graphml(gml)
        G = ox.truncate.truncate_graph_polygon(G, poly, truncate_by_edge=False)
        return G, poly, "sim metro graph clipped to core"
    ox.settings.use_cache = True
    ox.settings.cache_folder = os.path.join(CACHE, "osmnx_http")
    # On the welfare server overpass-api.de IP-blocks us; netfix pins a healthy mirror.
    from welfare_rs import netfix
    netfix.prefer_ipv4()
    netfix.configure_overpass()
    import time
    for attempt in range(8):
        try:
            return ox.graph_from_polygon(poly, network_type="drive", retain_all=True), poly, "overpass"
        except Exception as e:  # osmnx raises ResponseStatusCodeError / ConnectionError
            if attempt == 7:
                raise
            wait = 30 * 2 ** attempt
            print(f"[{city}] Overpass error ({type(e).__name__}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def network_metrics_one(city: str) -> dict:
    import geopandas as gpd
    import osmnx as ox

    fp = os.path.join(CACHE, f"network_{city}.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp).iloc[0].to_dict()
    G, poly, source = _core_graph(city)
    area_m2 = float(ox.projection.project_gdf(gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")).area.sum())
    s = ox.stats.basic_stats(G, area=area_m2)
    Gu = ox.bearing.add_edge_bearings(ox.convert.to_undirected(G))
    spn = s.get("streets_per_node_proportions", {})
    out = {
        "city": city,
        "network_source": source,
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
        print("network source:", net.pop("network_source").to_dict())
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
