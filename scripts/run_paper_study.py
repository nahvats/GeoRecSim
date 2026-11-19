#!/usr/bin/env python3
"""Produce every number in the paper's results tables, headless.

Runs the study described in the Experiments section --- 4 metros x 5 scenarios
x 5 seeds, 2,000 agents, 50 days --- and writes one JSON per metro plus the
LaTeX-ready rows for Tables 5, 6 and 7.

This deliberately does NOT go through backend/service.py. That path builds the
deck.gl geometry for every agent-day (viz.build_timelines) and holds it in the
parent process; at 2,000 agents x 50 days x 5 seeds x 5 scenarios that payload
is tens of GB and has to be pickled back from every worker. Nothing in the
tables needs it. Everything here uses the same public engine functions the
service uses, so the numbers are identical to what the dashboard would show.

    python scripts/run_paper_study.py --out results/
    python scripts/run_paper_study.py --metros la --agents 200 --days 5 --seeds 2

Aggregation across seeds mirrors backend/service.py exactly:
  * Table 1 quantities: mean over seeds, sigma_U = std of the per-seed means.
  * Table 2 quantities: paired per seed against that seed's No-RS run, then
    averaged.
  * Spatial panel: sample-size-weighted mean, since a seed where nine agents
    went out must not count as much as one where sixty did.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from welfare_rs import metrics as sim_metrics
from welfare_rs import params
from welfare_rs.datasource import get_datasource
from welfare_rs.experiment_harness import (
    eval_day_leisure_utilities,
    table1_metrics,
    table2_from_utilities,
)
from welfare_rs.metro import build_network
from welfare_rs.recommender_systems import RecommenderConfig
from welfare_rs.simulation import Simulation

CONTROL = "No RS"

#: Table 4 of the paper. Weights are renormalized engine-side; delta and the
#: learning rate are the defaults and are stated in the table caption.
SCENARIOS: Dict[str, Optional[dict]] = {
    CONTROL: None,
    "Proximity-led": dict(
        w_rating=0.25, w_reviews=0.20, w_relevance=0.10,
        w_proximity=0.35, w_popularity=0.00, w_personalization=0.10,
        distance_scale_km=5.0, popularity_gamma=1.0,
    ),
    "Footfall-led": dict(
        w_rating=0.20, w_reviews=0.25, w_relevance=0.10,
        w_proximity=0.00, w_popularity=0.35, w_personalization=0.10,
        distance_scale_km=5.0, popularity_gamma=1.0,
    ),
    "Personalized": dict(
        w_rating=0.15, w_reviews=0.10, w_relevance=0.10,
        w_proximity=0.20, w_popularity=0.10, w_personalization=0.35,
        distance_scale_km=5.0, popularity_gamma=1.0,
    ),
    "Viral": dict(
        w_rating=0.10, w_reviews=0.10, w_relevance=0.05,
        w_proximity=0.10, w_popularity=0.60, w_personalization=0.05,
        distance_scale_km=5.0, popularity_gamma=2.0,
    ),
}

METROS = ["nyc", "la", "houston", "seattle"]

_NET_CACHE: Dict[str, object] = {}


# ── Worker ───────────────────────────────────────────────────────────────────

def _run_one(city: str, seed: int, scenario: str, n_agents: int, n_days: int) -> dict:
    """One simulation: one scenario at one seed. Returns only summary numbers."""
    params.SIMPLIFICATION_TOGGLES["CAR_ONLY_MODE"] = True

    ds = get_datasource()
    poi_rows = ds.poi_rows(city)
    if city not in _NET_CACHE:
        _NET_CACHE[city] = build_network(city)
    network = _NET_CACHE[city]

    weights = SCENARIOS[scenario]
    rec_config = (
        RecommenderConfig(label=scenario, **weights) if weights is not None else None
    )

    sim = Simulation(
        num_agents=n_agents,
        seed=seed,
        # The control still BUILDS a recommender it never consults, so the
        # catalog, the tastes and every RNG stream stay identical to the
        # treated scenarios. This flag is what withholds it from the agents.
        use_recommenders=(scenario != CONTROL),
        persona_csv_path=ds.persona_csv_path() or params.NYC_PERSONA_CSV_PATH,
        road_network=network,
        poi_rows=poi_rows,
        disabled_modes=("transit",),
        recommender_config=rec_config,
        recommender_factory=None,
    )

    run_metrics = sim_metrics.RunMetrics(sim)
    sim.run_days(n_days, on_day_complete=lambda _d, s: run_metrics.observe_day(s))

    summary = sim.summarize()
    return {
        "city": city,
        "seed": int(seed),
        "scenario": scenario,
        # Eval-day welfare (paper Table 6 levels).
        "t1": table1_metrics(sim),
        # agent_id -> eval-day mean leisure net utility; the parent pairs these
        # against the control to get Harmed / Improved / ORC.
        "utils": eval_day_leisure_utilities(sim.agents),
        # Run-cumulative concentration, pooled travel, per-day series, segments.
        "metrics": run_metrics.finalize(sim),
        "acceptance_rate": float(summary.get("recommendation_acceptance_rate", 0.0)),
        "avg_travel_time_min": float(summary.get("avg_travel_time_min", 0.0)),
        "n_catalog": len(sim.place_catalog),
    }


# ── Aggregation (mirrors backend/service.py) ─────────────────────────────────

def _pooled_mean(rows: List[dict], key: str, weight: str = "n") -> float:
    total = sum(float(r.get(weight, 0) or 0) for r in rows)
    if total <= 0:
        return 0.0
    return sum(float(r.get(key, 0.0)) * float(r.get(weight, 0) or 0) for r in rows) / total


def aggregate(city: str, seeds: List[int], results: Dict[Tuple[int, str], dict]) -> dict:
    scenarios = list(SCENARIOS)
    ok = [s for s in seeds if all((s, c) in results for c in scenarios)]
    if not ok:
        raise RuntimeError(f"{city}: no seed produced every scenario")

    out = {"city": city, "seeds": ok, "rows": {}}
    for c in scenarios:
        t1 = [results[(s, c)]["t1"] for s in ok]
        foot = [results[(s, c)]["metrics"]["footfall"] for s in ok]
        spat = [results[(s, c)]["metrics"]["spatial"]["overall"] for s in ok]
        taste = [results[(s, c)]["metrics"]["taste"] for s in ok]
        means = [r["mean_utility"] for r in t1]

        row = {
            # ── Table 5 left: concentration (run-cumulative) ──
            "effective_places": float(np.mean([f["effective_places"] for f in foot])),
            "top10_share":      float(np.mean([f["top10_share"] for f in foot])),
            "gini_visited":     float(np.mean([f["gini"] for f in foot])),
            "coverage":         float(np.mean([f["coverage"] for f in foot])),
            "repeat_rate":      float(np.mean([f["repeat_rate"] for f in foot])),
            "visits_per_place": float(np.mean([f["visits_per_place"] for f in foot])),
            "total_visits":     float(np.mean([f["total_visits"] for f in foot])),
            "n_places":         float(np.mean([f["n_places"] for f in foot])),
            # ── Table 5 right: travel (pooled over run) ──
            "mean_distance_km":  _pooled_mean(spat, "mean_distance_km"),
            "mean_travel_min":   _pooled_mean(spat, "mean_travel_min"),
            "mean_emissions_g":  _pooled_mean(spat, "mean_emissions_g"),
            "median_detour":     _pooled_mean(spat, "median_detour"),
            "median_excess_km":  _pooled_mean(spat, "median_excess_km"),
            "n_leisure_trips_pooled": int(sum(r["n"] for r in spat)),
            # ── Table 6: welfare (eval day) ──
            "acceptance_rate": float(np.mean([results[(s, c)]["acceptance_rate"] for s in ok])),
            "mean_utility":    float(np.mean(means)),
            "sigma_u":         float(np.std(means)) if len(means) > 1 else 0.0,
            "neg_rate":        float(np.mean([r["neg_rate"] for r in t1])),
            "gini_utility":    float(np.mean([r["gini"] for r in t1])),
            "abstention_rate": float(np.mean([r["abstention_rate"] for r in t1])),
            "n_eval_trips":    int(sum(r["n_leisure_trips"] for r in t1)),
            # ── Taste discovery ──
            "taste_recommended": float(np.mean([t["recommended_rate"] for t in taste])),
            "taste_organic":     float(np.mean([t["organic_rate"] for t in taste])),
        }

        if c != CONTROL:
            paired = [
                table2_from_utilities(results[(s, c)]["utils"], results[(s, CONTROL)]["utils"])
                for s in ok
            ]
            row.update({
                "harmed_pct":   float(np.mean([p["harmed_pct"] for p in paired])),
                "improved_pct": float(np.mean([p["improved_pct"] for p in paired])),
                "mean_orc":     float(np.mean([p["mean_orc"] for p in paired])),
                "n_matched":    int(sum(p["n_matched"] for p in paired)),
            })

        out["rows"][c] = row

    # Per-day series, averaged over seeds: the input to the by-day figures.
    out["per_day"] = {}
    for c in scenarios:
        series = [results[(s, c)]["metrics"]["per_day"] for s in ok]
        length = min(len(x) for x in series)
        keys = sorted({k for x in series for row in x for k in row})
        out["per_day"][c] = [
            {"day": d + 1, **{k: float(np.mean([x[d][k] for x in series if k in x[d]]))
                              for k in keys}}
            for d in range(length)
        ]

    # By-income and by-trust welfare panels, for the two segment figures.
    out["segments"] = {}
    for c in scenarios:
        per_seed = [results[(s, c)]["metrics"]["segments"] for s in ok]
        merged = {}
        for seg in ("income", "trust"):
            groups = {g for s_ in per_seed for g in s_.get(seg, {})}
            merged[seg] = {
                g: {
                    "n": int(sum(s_[seg][g]["n"] for s_ in per_seed if g in s_.get(seg, {}))),
                    "mean_utility": _pooled_mean(
                        [s_[seg][g] for s_ in per_seed if g in s_.get(seg, {})], "mean_utility"),
                    "neg_rate": _pooled_mean(
                        [s_[seg][g] for s_ in per_seed if g in s_.get(seg, {})], "neg_rate"),
                }
                for g in sorted(groups)
            }
        out["segments"][c] = merged

    return out


# ── LaTeX emission ───────────────────────────────────────────────────────────

def latex_table5(agg: dict) -> str:
    lines = []
    for c, r in agg["rows"].items():
        lines.append(
            f"{c:<14}& {r['effective_places']:.0f} & {100*r['top10_share']:.1f}\\% "
            f"& {r['gini_visited']:.3f} & {100*r['coverage']:.1f}\\% "
            f"& {100*r['repeat_rate']:.1f}\\% & {r['mean_distance_km']:.2f} "
            f"& {r['mean_travel_min']:.1f} & {r['mean_emissions_g']:.0f} "
            f"& {r['median_detour']:.2f} & {r['median_excess_km']:.2f} \\\\"
        )
    head = (f"% {agg['city']}: {agg['rows'][CONTROL]['visits_per_place']:.2f} "
            f"visits per catalog venue, {agg['rows'][CONTROL]['n_places']:.0f} venues\n")
    return head + "\n".join(lines)


def latex_table6(agg: dict) -> str:
    lines = []
    for c, r in agg["rows"].items():
        acc = "---" if c == CONTROL else f"{100*r['acceptance_rate']:.1f}\\%"
        harm = "---" if c == CONTROL else f"{100*r['harmed_pct']:.1f}\\%"
        orc = "---" if c == CONTROL else f"{r['mean_orc']:.3f}"
        lines.append(
            f"{c:<14}& {acc} & {r['mean_utility']:.3f} ({r['sigma_u']:.3f}) "
            f"& {100*r['neg_rate']:.1f}\\% "
            f"& {harm} & {orc} \\\\"
        )
    return "\n".join(lines)


def latex_table7(by_city: Dict[str, dict]) -> str:
    label = {"nyc": "New York", "la": "Los Angeles",
             "houston": "Houston", "seattle": "Seattle"}
    treated = [c for c in SCENARIOS if c != CONTROL]
    out = ["\\multicolumn{6}{@{}l}{\\emph{Effective venues $D$, ratio to control}} \\\\"]
    for city, agg in by_city.items():
        base = agg["rows"][CONTROL]["effective_places"]
        cells = " & ".join(f"{agg['rows'][c]['effective_places'] / base:.2f}" for c in treated)
        out.append(f"{label.get(city, city):<12}& {base:.0f} & {cells} \\\\")
    out.append("\\midrule")
    out.append("\\multicolumn{6}{@{}l}{\\emph{Median excess $\\Delta$, difference from control (km)}} \\\\")
    for city, agg in by_city.items():
        base = agg["rows"][CONTROL]["median_excess_km"]
        cells = " & ".join(
            f"{agg['rows'][c]['median_excess_km'] - base:+.2f}" for c in treated)
        out.append(f"{label.get(city, city):<12}& {base:.2f} & {cells} \\\\")
    return "\n".join(out)


# ── Driver ───────────────────────────────────────────────────────────────────

def run_city(city: str, agents: int, days: int, base_seed: int, n_seeds: int,
             workers: int) -> dict:
    seeds = [base_seed + i for i in range(n_seeds)]
    tasks = [(s, c) for s in seeds for c in SCENARIOS]

    # Build the network once in the parent so no worker races to download it
    # from Overpass or to write the warm pickle.
    print(f"[{city}] building/loading network ...", flush=True)
    t0 = time.time()
    build_network(city)
    print(f"[{city}] network ready in {time.time() - t0:.0f}s; "
          f"{len(tasks)} simulations on {workers} workers", flush=True)

    results: Dict[Tuple[int, str], dict] = {}
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {
            pool.submit(_run_one, city, s, c, agents, days): (s, c)
            for s, c in tasks
        }
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results[(res["seed"], res["scenario"])] = res
            print(f"[{city}] {i}/{len(tasks)} done "
                  f"(seed {res['seed']}, {res['scenario']}) "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)

    return aggregate(city, seeds, results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metros", nargs="+", default=METROS)
    ap.add_argument("--agents", type=int, default=2000)
    ap.add_argument("--days", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("WELFARE_RS_MAX_WORKERS", "0") or 0)
                    or (os.cpu_count() or 1))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    by_city: Dict[str, dict] = {}

    for city in args.metros:
        agg = run_city(city, args.agents, args.days, args.base_seed,
                       args.seeds, args.workers)
        by_city[city] = agg
        with open(os.path.join(args.out, f"{city}.json"), "w") as fh:
            json.dump(agg, fh, indent=2)
        print(f"[{city}] wrote {args.out}/{city}.json", flush=True)

    primary = "la" if "la" in by_city else args.metros[0]
    with open(os.path.join(args.out, "tables.tex"), "w") as fh:
        fh.write("% ── Table 5 rows (concentration + travel), "
                 f"{primary} ──\n{latex_table5(by_city[primary])}\n\n")
        fh.write("% ── Table 6 rows (welfare) ──\n"
                 f"{latex_table6(by_city[primary])}\n\n")
        if len(by_city) > 1:
            fh.write(f"% ── Table 7 rows (cross-metro) ──\n{latex_table7(by_city)}\n")
    print(f"wrote {args.out}/tables.tex", flush=True)


if __name__ == "__main__":
    main()
