"""Run management for the FastAPI backend (multi-condition, multi-seed studies).

A *study* is a set of recommender **conditions** swept across N random seeds. The
No-RS control is always one of them — it is the baseline every Table-2 row is
paired against, and its movement is worth watching on the map in its own right.

The unit of parallelism is one simulation: every ``(seed, condition)`` pair is a
separate process. A 3-seed study over 4 recommenders is 15 simulations running
at once, not 3 sequential pairs. Why processes, and what
they do and don't share:

* Threads wouldn't help — the agent loop is pure-Python (GIL-bound). Separate
  processes each get their own module globals, so there is no cross-task
  contention.
* Each metro with a study running has one host process, which loads the metro
  network once (0.8-3.2 GB) and forks a child per simulation; the children
  share the network instead of each loading a copy (see ``SimulationHost``).
  A child exits when its simulation ends, so every byte goes back to the OS,
  and it saves its own result to disk (see the checkpoint section). The web
  process keeps no map data in memory.
* The routing matrices (welfare_rs.routing_matrix, 3-35 GB per metro) ARE shared:
  they are opened ``mmap_mode="r"``, so every simulation reads one page-cache copy.
* Memory, not cores, is what limits how many simulations run at once: each
  reserves its estimated peak before it starts (``ResourceGovernor``).

Table 2 pairs a recommender against No RS agent-by-agent, which would otherwise
force both runs into one process. Instead each task returns its per-agent eval-day
utilities and the pairing happens here, in the parent (see ``_assemble``).

Metrics are computed the paper's way (welfare_rs.experiment_harness): Table 1 over
the **last (evaluation) day's** leisure trips per run, then aggregated across seeds
(σ_U = std of per-seed Ū); Table 2 paired against the No-RS run per seed.
"""

from __future__ import annotations

import calendar
import functools
import gzip
import ctypes
import gc
import hashlib
import itertools
import json
import math
import pickle
import os
import warnings
import sys

# Make ``viz`` and ``welfare_rs`` importable even in spawned worker processes,
# which re-import this module fresh (macOS uses the 'spawn' start method).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp
import queue
import signal
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from multiprocessing import connection as mp_connection
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Tuple

import numpy as np

from welfare_rs import metrics as sim_metrics
from welfare_rs import params, routing_matrix
from welfare_rs.datasource import get_datasource
from welfare_rs.experiment_harness import (
    eval_day_leisure_utilities,
    leisure_net_utility,
    table1_metrics,
    table2_from_utilities,
)
from welfare_rs.geo import haversine_km
from welfare_rs.metro import build_network
from welfare_rs.recommender_systems import RECOMMENDER_PRESETS, RecommenderConfig
from welfare_rs.simulation import Simulation, SimulationCancelled

import viz

# ── Recommenders / cities ────────────────────────────────────────────────────

#: The No-RS control. Always run, never a choice: it is the baseline every
#: paired comparison is defined against.
CONTROL = "No RS"

#: The Oracle control (welfare_rs.oracle_rs). Also always run: the
#: perfect-knowledge upper bound every built recommender is read against, as in
#: Uğurel & Yabe (2026), whose tables report it beside every condition.
ORACLE = "Oracle"

#: Labels a user-built recommender may not take, since condition labels are
#: the identity of an arm everywhere downstream.
RESERVED_LABELS = (CONTROL, ORACLE)

#: Welfare gates a built recommender may wear on top of its ranking.
WELFARE_GATES = ("pup", "rm", "pup_rm")

#: Ceiling on user-built recommenders in one study (the controls are extra).
#: Cost and retained map geometry both scale linearly in arms.
MAX_RECOMMENDERS = 5

#: Ceiling on points in the spatial welfare map. Well past what a hexbin can
#: resolve, and it keeps the run payload bounded regardless of population.
WELFARE_MAP_MAX_POINTS = 20000

#: Where every finished study is saved (see ``_save_study_record``). Studies
#: are otherwise held only in memory, so a service restart or LRU eviction
#: would lose the per-agent detail that later figures are made from.
STUDY_RECORD_DIR = os.path.join(params.GEO_PARAMS["cache_dir"], "studies")

#: Deprecated fixed vocabulary, kept so saved API calls still resolve.
TREATMENTS = ["No RS", "Standard RS", "PUP", "RM", "PUP+RM"]

# (metro key, friendly label) — the two-layer metros surfaced to the frontend.
#
# ALL metros are offered, warmed or not. Picking an unwarmed one makes the run
# build its network on demand, which goes to Overpass from inside the web
# worker — slow at best, and while this host is blocked by overpass-api.de it
# hangs on a 180 s connect timeout per sub-query before failing. That is a
# deliberate development-time choice: the operator knows which metros are warm.
# ``warmed`` in the /api/cities payload says which are ready without having to
# guess. Warming itself stays an offline step (scripts/warm_metros.py).
CITIES = [(key, spec["label"]) for key, spec in params.METRO_PARAMS["metros"].items()]
_CITY_LABELS = dict(CITIES)
DEFAULT_CITY = params.METRO_PARAMS["default_metro"]


def warmed_cities() -> dict:
    """``{metro key: warm pickle exists}`` — advisory, read fresh each call so
    a metro warmed while the service is up shows up without a restart."""
    cache_dir = params.GEO_PARAMS["cache_dir"]
    return {
        key: os.path.exists(os.path.join(cache_dir, "warmed", f"{key}_metro.pkl"))
        for key, _ in CITIES
    }


def precomputed_cities() -> dict:
    """``{metro key: routing matrices present}``.

    A warmed metro without matrices still runs — it just falls back to computing
    Dijkstra trees per run, which is what makes runs expensive and serialised.
    Read fresh each call so a metro precomputed while the service is up is
    reflected without a restart.
    """
    cache_dir = params.GEO_PARAMS["cache_dir"]
    out = {}
    for key, _ in CITIES:
        meta = routing_matrix.read_meta(cache_dir, key)
        out[key] = bool(meta and meta.get("version") == routing_matrix.MATRIX_VERSION)
    return out

MAX_SEEDS = 12

# Ceiling on simulations running at once, one per core by default. Memory
# usually binds first (see ResourceGovernor). Set WELFARE_RS_MAX_WORKERS to
# leave the box headroom for other work.
MAX_WORKERS = int(os.environ.get("WELFARE_RS_MAX_WORKERS", "0") or 0) or (os.cpu_count() or 1)


def _worker_count(n_tasks: int) -> int:
    """Slots a study of this size could use if it had the box to itself.

    Reporting only since studies became concurrent — ResourceGovernor decides
    what a study actually gets, and that depends on how many others are running.
    """
    return max(1, min(n_tasks, os.cpu_count() or 1, MAX_WORKERS))


# Studies allowed to run at once. They no longer queue behind each other: the
# box's CPU budget is split evenly between however many are running, so two
# studies get half the cores each and widen back out as one finishes.
MAX_STUDIES = int(os.environ.get("WELFARE_RS_MAX_STUDIES", "0") or 0) or 10

# Memory the simulations may reserve between them, in GB. Leaves the web
# process (which holds every running study's per-run results, without their map
# data) and the box's other work room under the service's MemoryMax.
MEMORY_BUDGET_GB = float(os.environ.get("WELFARE_RS_MEMORY_BUDGET_GB", "0") or 0) or 160.0

# Last guard: no simulation starts while the box has less than this available,
# whatever the reservations say. Catches memory used outside the service.
MIN_FREE_RAM_GB = float(os.environ.get("WELFARE_RS_MIN_FREE_RAM_GB", "24"))

# Ceiling on Oracle simulations at once, across all studies. Oracle is the
# slowest arm, so its tasks start first; without a ceiling they would take the
# whole memory budget and leave the other arms queueing for hours.
MAX_ORACLE_TASKS = int(os.environ.get("WELFARE_RS_MAX_ORACLE_TASKS", "0") or 0) or MAX_WORKERS

# A simulation whose process dies (killed for memory, or crashed) is retried
# this many times in all before its study fails.
MAX_TASK_ATTEMPTS = 3


# Memory estimates, in GB. A metro's host holds its network once (_NETWORK_GB);
# each simulation, forked from the host, holds its agents and what it builds up
# over the run, plus the part of the shared network it ends up copying (Python
# writes to an object's memory just by referencing it). Measured 2026-10-12 on
# simulations forked from a host, 2,500 agents, Personalized and Oracle over
# 3-8 days: per-1,000-agent memory on day 1 and growth per day, taking the
# larger of the two arms. Growth is projected linearly, which overstates it (the
# copying of the shared network tails off); where a 15-day probe without
# sharing gave a lower 60-day peak, growth is capped to match it.
_NETWORK_GB = {"la": 3.0, "nyc": 0.8, "houston": 3.2, "chicago": 2.9,
               "sf_bay": 1.0, "dc": 0.9, "seattle": 1.2, "miami": 0.6}
_SIM_BASE_GB = {"la": 1.27, "nyc": 0.48, "houston": 1.12, "chicago": 1.19,
                "sf_bay": 0.74, "dc": 0.50, "seattle": 0.79, "miami": 0.42}
_SIM_GROWTH_GB_PER_1000_AGENT_DAYS = {"la": 0.012, "nyc": 0.025, "houston": 0.046, "chicago": 0.028,
                                      "sf_bay": 0.021, "dc": 0.043, "seattle": 0.037, "miami": 0.025}
_MEMORY_MARGIN = 1.15                   # headroom on every estimate


def _host_memory_gb(city: str) -> float:
    """Memory a metro's host holds: its network, with margin."""
    return _MEMORY_MARGIN * (_NETWORK_GB.get(city, max(_NETWORK_GB.values())) + 0.2)


def _task_memory_gb(cfg: dict, condition: str) -> float:
    """Expected peak memory of one simulation of this study, in GB, with margin
    (its host's network not included)."""
    city = cfg["city"]
    agents_k = int(cfg["num_agents"]) / 1000.0
    base = _SIM_BASE_GB.get(city, max(_SIM_BASE_GB.values()))
    growth = _SIM_GROWTH_GB_PER_1000_AGENT_DAYS.get(city, max(_SIM_GROWTH_GB_PER_1000_AGENT_DAYS.values()))
    return _MEMORY_MARGIN * (0.5 + agents_k * (base + growth * int(cfg["num_days"])))


def _cgroup_headroom() -> Optional[int]:
    """Bytes left under this service's cgroup memory cap, or None if uncapped.

    systemd runs the service under MemoryMax, which is the limit that actually
    kills it — usually lower than the box's free memory, so it has to be checked
    separately from /proc/meminfo.
    """
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as f:
            rel = f.read().strip().split(":")[-1]
        base = os.path.join("/sys/fs/cgroup", rel.lstrip("/"))
        with open(os.path.join(base, "memory.max"), encoding="utf-8") as f:
            cap = f.read().strip()
        if cap == "max":
            return None
        with open(os.path.join(base, "memory.current"), encoding="utf-8") as f:
            used = int(f.read().strip())
        return max(0, int(cap) - used)
    except (OSError, ValueError, IndexError):
        return None


def _available_ram() -> Optional[int]:
    """Usable headroom in bytes: the tighter of box-free and cgroup-free."""
    headrooms = []
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    headrooms.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError):
        pass
    cg = _cgroup_headroom()
    if cg is not None:
        headrooms.append(cg)
    return min(headrooms) if headrooms else None


@dataclass(frozen=True)
class TaskCost:
    """What one simulation holds while it runs."""
    memory_gb: float
    oracle: bool


class ResourceGovernor:
    """Decides which simulation may START, across every running study.

    A simulation starts only if all of these hold, and keeps its share until it
    ends:

    * CPU: each registered study gets ``MAX_WORKERS // n_studies`` slots (at
      least one), recomputed on every start and finish, so a study widens as
      others finish and narrows as they start.
    * Memory: the simulation reserves its expected PEAK memory
      (``_task_memory_gb``) and all reservations, plus those of the metro hosts'
      networks, must fit in ``MEMORY_BUDGET_GB``. Reserving the peak is the
      point: a simulation starts small and reaches several GB minutes later, so
      a check of free memory at start let 56 start together and then outgrow
      the box together (the service was killed for memory on 2026-10-11).
    * Fairness: each registered study may reserve at most an equal share of the
      memory left after the hosts, so every study runs at once instead of the
      first one to register taking it all. A study with nothing running may
      always start one simulation that fits, so a small share never stalls it.
    * Oracle: at most ``MAX_ORACLE_TASKS`` Oracle simulations at once.
    * The box must still have ``MIN_FREE_RAM_GB`` available.

    Studies register when they are submitted, not when they start running, so
    a batch submitted together shares from the first simulation on.
    """

    def __init__(self, total_slots: int, memory_budget_gb: float, max_oracle: int,
                 min_free_ram_bytes: int):
        self._total = max(1, total_slots)
        self._memory_budget = memory_budget_gb
        self._max_oracle = max(1, max_oracle)
        self._min_free_ram = max(0, min_free_ram_bytes)
        self._held: Dict[str, int] = {}          # run id -> simulations running
        self._study_gb: Dict[str, float] = {}    # run id -> memory reserved
        self._host_gb: Dict[str, float] = {}     # metro -> its host's network
        self._oracles = 0
        self._cond = threading.Condition()
        self._ram_waits = 0

    def register(self, run_id: str) -> None:
        with self._cond:
            self._held.setdefault(run_id, 0)
            self._study_gb.setdefault(run_id, 0.0)
            self._cond.notify_all()      # everyone's share just shrank

    def unregister(self, run_id: str) -> None:
        with self._cond:
            self._held.pop(run_id, None)
            self._study_gb.pop(run_id, None)
            self._cond.notify_all()      # everyone's share just grew

    def reserve_host(self, city: str, memory_gb: float) -> None:
        with self._cond:
            self._host_gb[city] = memory_gb

    def release_host(self, city: str) -> None:
        with self._cond:
            self._host_gb.pop(city, None)
            self._cond.notify_all()

    def _quota_locked(self) -> int:
        return max(1, self._total // max(1, len(self._held)))

    def _reserved_locked(self) -> float:
        return sum(self._study_gb.values()) + sum(self._host_gb.values())

    def _ram_ok(self) -> bool:
        if not self._min_free_ram:
            return True
        free = _available_ram()
        return free is None or free >= self._min_free_ram

    def try_acquire(self, run_id: str, cost: TaskCost) -> bool:
        """Take a slot and reserve ``cost`` if everything fits right now; never block."""
        with self._cond:
            held = self._held.get(run_id, 0)
            if held >= self._quota_locked() or sum(self._held.values()) >= self._total:
                return False
            if self._reserved_locked() + cost.memory_gb > self._memory_budget:
                return False
            share = (self._memory_budget - sum(self._host_gb.values())) / max(1, len(self._held))
            if held and self._study_gb.get(run_id, 0.0) + cost.memory_gb > share:
                return False
            if cost.oracle and self._oracles >= self._max_oracle:
                return False
            if not self._ram_ok():
                self._ram_waits += 1
                return False
            self._held[run_id] = held + 1
            self._study_gb[run_id] = self._study_gb.get(run_id, 0.0) + cost.memory_gb
            self._oracles += cost.oracle
            return True

    def release(self, run_id: str, cost: TaskCost) -> None:
        with self._cond:
            if self._held.get(run_id):
                self._held[run_id] -= 1
            if run_id in self._study_gb:
                self._study_gb[run_id] = max(0.0, self._study_gb[run_id] - cost.memory_gb)
            self._oracles = max(0, self._oracles - cost.oracle)
            self._cond.notify_all()

    def wait(self, timeout: float) -> None:
        """Sleep until something is released, or ``timeout`` seconds pass."""
        with self._cond:
            self._cond.wait(timeout=timeout)

    def snapshot(self) -> dict:
        with self._cond:
            free = _available_ram()
            return {
                "total_slots": self._total,
                "studies_running": len(self._held),
                "slots_per_study": self._quota_locked(),
                "slots_in_use": sum(self._held.values()),
                "per_study": dict(self._held),
                "memory_budget_gb": self._memory_budget,
                "memory_reserved_gb": round(self._reserved_locked(), 1),
                "memory_per_study_gb": {k: round(v, 1) for k, v in self._study_gb.items()},
                "memory_hosts_gb": {k: round(v, 1) for k, v in self._host_gb.items()},
                "oracle_tasks": self._oracles,
                "ram_free_gb": round(free / 1e9, 1) if free is not None else None,
                "ram_floor_gb": round(self._min_free_ram / 1e9, 1),
                "ram_throttle_events": self._ram_waits,
            }


GOVERNOR = ResourceGovernor(MAX_WORKERS, MEMORY_BUDGET_GB, MAX_ORACLE_TASKS,
                            int(MIN_FREE_RAM_GB * 1e9))


# ── Simulation checkpoints ───────────────────────────────────────────────────
#
# Every simulation saves its own result as soon as it finishes, in two files
# under TASK_RESULT_DIR/<run id>/: the map data (by far the largest part, ~0.2 GB
# pickled for 2,500 agents x 60 days) and everything else. The web process only
# ever loads the second: the map data is read from disk when a map is opened.
# A study that dies part-way loses only the simulations still running, since
# resubmitting the same config (same run id) reloads the finished ones.

TASK_RESULT_DIR = os.path.join(params.GEO_PARAMS["cache_dir"], "study_tasks")


def _task_checkpoint(run_id: str, seed: int, condition: str) -> str:
    """Path prefix of one simulation's checkpoint files."""
    name = hashlib.sha1(condition.encode()).hexdigest()[:12]
    return os.path.join(TASK_RESULT_DIR, run_id, f"seed{seed}_{name}")


def _write_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # never leave a half-written file under the final name


def _save_task_result(checkpoint: str, res: dict) -> dict:
    """Save a finished simulation; return it without its map data.

    The map data is written first, so the results file existing means both do.
    """
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    _write_atomic(checkpoint + ".viz.pkl", res["viz"])
    light = {k: v for k, v in res.items() if k != "viz"}
    light["time_span"] = res["viz"]["time_span"]
    light["viz_file"] = checkpoint + ".viz.pkl"
    _write_atomic(checkpoint + ".pkl", light)
    return light


def _load_task_result(checkpoint: str, seed: int, condition: str) -> Optional[dict]:
    """A finished simulation's result (without map data), or None."""
    if not os.path.exists(checkpoint + ".pkl"):
        return None
    with open(checkpoint + ".pkl", "rb") as fh:
        res = pickle.load(fh)
    return res if (res["seed"], res["condition"]) == (seed, condition) else None


@functools.lru_cache(maxsize=4)
def _load_seed_viz(path: str, seed: int) -> "SeedViz":
    """One simulation's map data, read from its checkpoint (the few most
    recently viewed are kept)."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return SeedViz(seed=seed, merged={int(k): v for k, v in data["merged"].items()},
                   trip_index=data["trip_index"], time_span=data["time_span"])


# ── Simulation hosts ─────────────────────────────────────────────────────────
#
# One host process per metro with a study running. It loads the metro network
# once and starts each simulation as a forked child, which shares the network's
# memory instead of loading its own copy (0.8-3.2 GB). A child exits when its
# simulation ends, so every byte it used goes back to the OS. The host is
# spawned (fork-after-threads under uvicorn is unsafe) and single-threaded, so
# forking from it is safe; it is shut down when no study needs its metro.


class SimulationProcessDied(RuntimeError):
    """A simulation's process, or its host, ended without reporting back:
    killed (most likely for memory) or crashed. The task can be retried."""


_HOST_PID = None  # set in a host process; inherited by the simulations it forks


def _host_main(conn, city: str) -> None:
    """Body of a SimulationHost process: serve simulation requests for ``city``.

    A request is ``(task_id, args)`` for _simulate_and_save; each finished child
    is answered with ``(task_id, (outcome, payload))``. ``None``, or the web
    process going away, ends the host.
    """
    global _HOST_PID
    _HOST_PID = os.getpid()
    network = build_network(city)
    gc.collect()
    gc.freeze()  # keep the collector off the network's objects, or every child would copy them
    fork = mp.get_context("fork")
    running: Dict[object, Tuple[int, object]] = {}  # child's pipe -> (task id, child)
    try:
        while True:
            for ready in mp_connection.wait([conn, *running]):
                if ready is conn:
                    request = conn.recv()
                    if request is None:
                        return
                    task_id, args = request
                    receive, send = fork.Pipe(duplex=False)
                    child = fork.Process(target=_simulate_and_save, args=(send, *args, network))
                    child.start()
                    send.close()  # the child holds the only writer, so its death ends the pipe
                    running[receive] = (task_id, child)
                else:
                    task_id, child = running.pop(ready)
                    try:
                        outcome = ready.recv()
                    except EOFError:
                        child.join()
                        outcome = ("died", f"exit code {child.exitcode}")
                    else:
                        child.join()
                    ready.close()
                    conn.send((task_id, outcome))
    except (EOFError, OSError):
        return  # the web process is gone
    finally:
        for _task_id, child in running.values():
            child.kill()  # never leave orphans behind


class SimulationHost:
    """The web process's handle on one metro's host process."""

    def __init__(self, city: str):
        self.city = city
        ctx = mp.get_context("spawn")
        self._conn, host_end = ctx.Pipe()
        # Not a daemon: a daemonic process may not start children.
        self._proc = ctx.Process(target=_host_main, args=(host_end, city), name=f"simhost-{city}")
        self._proc.start()
        host_end.close()
        self._lock = threading.Lock()
        self._pending: Dict[int, Tuple[Future, str]] = {}
        self._ids = itertools.count()
        self.alive = True
        threading.Thread(target=self._read, name=f"simhost-{city}", daemon=True).start()

    def submit(self, cfg: dict, seed: int, condition: str, cancel_event, progress_q,
               checkpoint: str) -> Future:
        """Start one simulation; the Future resolves to its result without map data."""
        fut: Future = Future()
        fut.set_running_or_notify_cancel()
        what = f"{cfg['city']} seed {seed} {condition}"
        with self._lock:
            if not self.alive:
                fut.set_exception(SimulationProcessDied(f"{what}: simulation host for {self.city} is gone"))
                return fut
            task_id = next(self._ids)
            self._pending[task_id] = (fut, what)
            try:
                self._conn.send((task_id, (cfg, seed, condition, cancel_event, progress_q, checkpoint)))
            except OSError:
                self._pending.pop(task_id)
                fut.set_exception(SimulationProcessDied(f"{what}: simulation host for {self.city} is gone"))
        return fut

    def _read(self) -> None:
        while True:
            try:
                task_id, (outcome, payload) = self._conn.recv()
            except (EOFError, OSError):
                break
            with self._lock:
                fut, what = self._pending.pop(task_id)
            if outcome == "ok":
                fut.set_result(payload)
            elif outcome == "cancelled":
                fut.set_exception(SimulationCancelled())
            elif outcome == "died":
                fut.set_exception(SimulationProcessDied(f"{what}: simulation process ended with {payload} without a result"))
            else:
                fut.set_exception(RuntimeError(f"{what}: {payload}"))
        # The host has gone: whatever it was running will never report back.
        with self._lock:
            self.alive = False
            lost, self._pending = list(self._pending.values()), {}
        for fut, what in lost:
            fut.set_exception(SimulationProcessDied(f"{what}: simulation host for {self.city} ended"))

    def close(self) -> None:
        with self._lock:
            if self.alive:
                try:
                    self._conn.send(None)
                except OSError:
                    pass
        self._proc.join(timeout=30)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join()


class _HostRegistry:
    """One live host per metro while any study of that metro runs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._hosts: Dict[str, SimulationHost] = {}
        self._users: Dict[str, int] = {}

    def acquire(self, city: str) -> None:
        """A study of ``city`` starts using its host (started, and its network's
        memory reserved, on first use)."""
        with self._lock:
            if city not in self._hosts:
                GOVERNOR.reserve_host(city, _host_memory_gb(city))
                self._hosts[city] = SimulationHost(city)
            self._users[city] = self._users.get(city, 0) + 1

    def host(self, city: str) -> SimulationHost:
        """The live host for ``city``; a host that died is replaced."""
        with self._lock:
            host = self._hosts[city]
            if not host.alive:
                host.close()
                host = self._hosts[city] = SimulationHost(city)
            return host

    def release(self, city: str) -> None:
        """A study of ``city`` is done with its host; the last one shuts it down."""
        with self._lock:
            self._users[city] -= 1
            if self._users[city]:
                return
            del self._users[city]
            host = self._hosts.pop(city)
        host.close()
        GOVERNOR.release_host(city)


HOSTS = _HostRegistry()


# ── Config / run records ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RecommenderSpec:
    """One study arm: a full configuration of the single unified scorer.

    Frozen and built only of scalars, so it is hashable (the run id folds it in)
    and picklable (it crosses into the pool workers as-is).
    """

    label: str = "Custom RS"
    w_rating: float = 0.20
    w_relevance: float = 0.10
    w_proximity: float = 0.25
    w_popularity: float = 0.20
    personalization_lambda: float = 0.10
    distance_scale_km: float = 5.0
    popularity_gamma: float = 1.0
    prior_strength: float = 2.0
    exploration: float = 0.10
    welfare_gate: str = ""          # "" | "pup" | "rm" | "pup_rm"
    pup_alpha: float = params.WELFARE_GATE_DEFAULTS["pup_alpha"]
    rm_epsilon: float = params.WELFARE_GATE_DEFAULTS["rm_epsilon"]
    random_ranking: bool = False
    #: The perfect-knowledge upper bound (welfare_rs.oracle_rs): ranks by each
    #: agent's true net utility, so every weight, random_ranking and the
    #: welfare gate are ignored.
    oracle: bool = False

    def to_engine_config(self) -> RecommenderConfig:
        return RecommenderConfig(
            label=self.label,
            w_rating=self.w_rating,
            w_relevance=self.w_relevance, w_proximity=self.w_proximity,
            w_popularity=self.w_popularity,
            personalization_lambda=self.personalization_lambda,
            distance_scale_km=self.distance_scale_km,
            popularity_gamma=self.popularity_gamma,
            prior_strength=self.prior_strength,
            exploration=self.exploration,
            random_ranking=self.random_ranking,
        )

    def identity(self) -> tuple:
        """Everything that changes behaviour, for the run-id hash.

        The label is included: two arms differing only by name are two rows in
        the results, so they must not collapse to one cached study.
        """
        return (
            self.label,
            round(self.w_rating, 4),
            round(self.w_relevance, 4), round(self.w_proximity, 4),
            round(self.w_popularity, 4), round(self.personalization_lambda, 4),
            round(self.distance_scale_km, 4), round(self.popularity_gamma, 4),
            round(self.prior_strength, 4), round(self.exploration, 4),
            self.welfare_gate,
            round(self.pup_alpha, 4), round(self.rm_epsilon, 4),
            bool(self.random_ranking),
        # Appended only when set, so every non-oracle spec keeps the identity,
        # and therefore the run id, it had before the oracle existed.
        ) + ((("oracle", True),) if self.oracle else ())


def spec_from_legacy_name(
    name: str, pup_alpha: float, rm_epsilon: float
) -> Optional[RecommenderSpec]:
    """Map a pre-builder treatment name onto the equivalent spec.

    "Standard RS" was the four-platform stack with default weights; the
    balanced preset is its closest single-scorer equivalent. The gated
    treatments are that same ranking with the welfare filter switched on, which
    is exactly what they always were.
    """
    if name == CONTROL:
        return None
    gate = {"PUP": "pup", "RM": "rm", "PUP+RM": "pup_rm"}.get(name, "")
    base = dict(RECOMMENDER_PRESETS["Balanced hybrid"])
    return RecommenderSpec(
        label=name, **base,
        welfare_gate=gate, pup_alpha=pup_alpha, rm_epsilon=rm_epsilon,
    )


@dataclass(frozen=True)
class RunConfig:
    city: str = DEFAULT_CITY
    num_agents: int = 80
    num_days: int = 30
    seed: int = 42          # base seed; the study sweeps seed .. seed+num_seeds-1
    num_seeds: int = 3
    # Recommenders to run *besides* the No-RS control, which is always included.
    # A tuple, not a list, so the frozen dataclass stays hashable.
    recommenders: Tuple[RecommenderSpec, ...] = ()
    use_real_pois: bool = True

    def seeds(self) -> List[int]:
        n = max(1, min(int(self.num_seeds), MAX_SEEDS))
        return [int(self.seed) + i for i in range(n)]

    def all_conditions(self) -> List[str]:
        """``[CONTROL, ORACLE, *built recommenders]``, the built ones in the
        order the user built them.

        The No-RS control is always present and always first: every Table-2 row
        is defined relative to it. The Oracle is always present and second, so
        each study carries its own upper bound. Order is the user's, not canonical — with
        free-form labels there is no canonical order to impose, and the run id
        folds in the sequence so a reordered study is a different (correctly
        re-run) study rather than a stale cache hit.
        """
        return [CONTROL, ORACLE] + [s.label for s in self.recommenders]

    def run_id(self) -> str:
        key = (
            self.city, int(self.num_agents), int(self.num_days), int(self.seed),
            int(self.num_seeds), tuple(s.identity() for s in self.recommenders),
            bool(self.use_real_pois),
            # A study cached before the Oracle control existed has no Oracle
            # arm, so it must not be handed back for the same settings.
            "oracle_control",
        )
        return hashlib.sha1(repr(key).encode()).hexdigest()[:16]


@dataclass
class SeedViz:
    """One condition's geometry at one seed — what the map plays back.

    POIs are deliberately *not* held here. The catalog varies with the seed (in
    synthetic mode) but never with which recommender ran, so it lives once per
    seed on the Run rather than being duplicated across every condition.
    """
    seed: int
    merged: Dict[int, dict]
    trip_index: Dict[str, dict]
    time_span: int


@dataclass
class Run:
    run_id: str
    config: RunConfig
    seeds: List[int]
    conditions: List[str]                        # No RS first, then recommenders
    # condition -> seed -> the map data's checkpoint file, and its time span.
    # The data itself stays on disk until a map asks for it (see viz()).
    viz_files: Dict[str, Dict[int, str]]
    time_spans: Dict[str, Dict[int, int]]
    pois_by_seed: Dict[int, List[dict]]          # condition-independent
    table1: List[dict]            # one row per condition (No RS first)
    table2: List[dict]            # one row per recommender, paired vs No RS
    headlines: Dict[str, dict]    # condition -> headline metrics
    aggregate: Dict[str, dict]    # condition -> acceptance / travel-time summary
    metrics: Dict[str, dict]      # condition -> dashboard panels
    welfare_map: List[dict]       # per-agent home + per-condition utility delta
    view: dict
    bounds: dict                  # full metro network extent
    core_bounds: dict             # principal-city extent (initial map frame)
    num_days: int
    num_intersections: int
    poi_count: int
    poi_source: str
    # Wall-clock cost of the study, filled in once it completes.
    created_at: Optional[float] = None
    duration_sec: Optional[float] = None

    def default_condition(self) -> str:
        """What the map opens on: the first built recommender, else the Oracle,
        else the No-RS control (a study saved before the Oracle existed)."""
        for c in self.conditions:
            if c not in RESERVED_LABELS:
                return c
        return ORACLE if ORACLE in self.conditions else CONTROL

    def meta(self) -> dict:
        """Aggregated metrics + per-condition/seed index (no geometry)."""
        return {
            "run_id": self.run_id,
            "config": asdict(self.config),
            "seeds": self.seeds,
            "default_seed": self.seeds[0] if self.seeds else self.config.seed,
            "conditions": self.conditions,
            "default_condition": self.default_condition(),
            # {condition: {seed: span}} — the scrubber's extent depends on both.
            "time_spans": {
                cond: {str(s): span for s, span in by_seed.items()}
                for cond, by_seed in self.time_spans.items()
            },
            "table1": self.table1,
            "table2": self.table2,
            "headlines": self.headlines,
            "aggregate": self.aggregate,
            "metrics": self.metrics,
            "welfare_map": self.welfare_map,
            "income_bands": sim_metrics.income_band_order(),
            "trust_groups": sim_metrics.trust_group_labels(),
            "view": self.view,
            "bounds": self.bounds,
            "core_bounds": self.core_bounds,
            "num_days": self.num_days,
            "num_intersections": self.num_intersections,
            "poi_count": self.poi_count,
            "poi_source": self.poi_source,
            "city_label": _CITY_LABELS.get(self.config.city, self.config.city),
            "created_at": self.created_at,
            "duration_sec": self.duration_sec,
        }

    def has_condition(self, condition: Optional[str]) -> bool:
        return condition in self.viz_files

    def viz(self, condition: Optional[str], seed: Optional[int]) -> Optional[SeedViz]:
        """Geometry for (condition, seed), read from its checkpoint; unknown
        values fall back to defaults."""
        by_seed = self.viz_files.get(condition) if condition is not None else None
        if by_seed is None:
            by_seed = self.viz_files.get(self.default_condition())
        if not by_seed:
            return None
        if seed is None or int(seed) not in by_seed:
            seed = self.seeds[0] if self.seeds else None
        if seed is None or int(seed) not in by_seed:
            return None
        return _load_seed_viz(by_seed[int(seed)], int(seed))

    def pois_for(self, seed: Optional[int]) -> List[dict]:
        """The POI catalog for a seed (identical across conditions)."""
        if seed is not None and int(seed) in self.pois_by_seed:
            return self.pois_by_seed[int(seed)]
        if self.seeds and self.seeds[0] in self.pois_by_seed:
            return self.pois_by_seed[self.seeds[0]]
        return next(iter(self.pois_by_seed.values()), [])


# ── Job (background study execution + progress) ──────────────────────────────

@dataclass
class Job:
    run_id: str
    status: str = "running"  # running | done | error | cancelled
    # Progress is counted in simulated DAYS across the whole (seed x condition)
    # grid — the finest unit the parent can observe now that a task is a whole
    # simulation. `per_condition` breaks that down per arm (days done, out of
    # `cond_total` = seeds x days each), because every condition is running at
    # the same time: a single "most recent tick" label would show one arm at
    # random and read as if the study were sequential.
    current: int = 0
    total: int = 0
    per_condition: Dict[str, int] = field(default_factory=dict)
    cond_total: int = 0
    error: Optional[str] = None
    # Retained so /api/runs can describe a study that is still computing (and
    # therefore has no Run record yet), and so a browser that never saw the
    # submission can still show what it is.
    config: Optional["RunConfig"] = None
    # True for a stored study being reopened from its checkpoints (Storage tab):
    # nothing is simulated, so its timestamps say nothing about run time.
    reopened: bool = False
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    # Cancellation: `cancel` is the in-process signal, checked by the study's own
    # loop; `mp_cancel` is a manager Event shared with the simulation processes,
    # created per study while it runs and checked between days by the engine's
    # should_stop, so a stop takes effect within one simulated day.
    cancel: threading.Event = field(default_factory=threading.Event)
    mp_cancel: Optional[object] = None
    _subs: List["queue.Queue"] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def request_cancel(self) -> None:
        self.cancel.set()
        if self.mp_cancel is not None:
            try:
                self.mp_cancel.set()
            except Exception:
                pass

    def publish(self, event: dict) -> None:
        with self._lock:
            for q in self._subs:
                q.put(event)

    def snapshot(self) -> dict:
        """Status + progress only (no geometry), for the run list."""
        with self._lock:
            return {
                "status": self.status,
                "reopened": self.reopened,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "progress": {
                    "current": self.current,
                    "total": self.total,
                    "cond_total": self.cond_total,
                    "per_condition": dict(self.per_condition),
                },
            }

    def finish(self, status: str) -> None:
        self.status = status
        self.finished_at = time.time()

    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue()
        with self._lock:
            self._subs.append(q)
            if self.status == "running":
                q.put({"type": "progress", "current": self.current,
                       "total": self.total, "cond_total": self.cond_total,
                       "per_condition": dict(self.per_condition),
                       "elapsed_sec": time.time() - self.created_at})
            elif self.status == "done":
                q.put({"type": "done", "run_id": self.run_id})
            elif self.status == "error":
                q.put({"type": "error", "message": self.error or "run failed"})
            elif self.status == "cancelled":
                q.put({"type": "cancelled", "run_id": self.run_id})
        return q


# ── Recommender wiring (welfare gate) ────────────────────────────────────────

def _make_recommender_factory(spec: Optional[RecommenderSpec]):
    """Wrap the built recommender in its welfare gate, if it has one, or swap
    it for the oracle.

    Returns None when the spec is ungated (or absent), which leaves the
    simulation's own configurable recommender in place unwrapped.
    """
    if spec is not None and spec.oracle:
        # Ungated, as in the paper: the gate scores from rec.score, which the
        # oracle holds at a neutral constant, so a gate would filter on noise.
        from welfare_rs.oracle_rs import oracle_factory

        return oracle_factory(spec.label)
    if spec is None or spec.welfare_gate not in WELFARE_GATES:
        return None
    from welfare_rs import TravelCostEstimator, WelfareAwareOrchestrator

    mode = spec.welfare_gate
    pup_alpha = spec.pup_alpha
    rm_epsilon = spec.rm_epsilon

    def factory(sim, base_stack):
        return WelfareAwareOrchestrator(
            base_stack,
            TravelCostEstimator(metro=sim.metro),
            mode=mode,
            pup_alpha=pup_alpha,
            rm_epsilon=rm_epsilon,
            coord_scale_km=1.0,
            coord_distance_km=haversine_km,
        )

    return factory


def pois_available() -> Dict[str, bool]:
    """Per-city real-POI availability (drives the frontend toggle)."""
    ds = get_datasource()
    out = {}
    for key, _label in CITIES:
        try:
            out[key] = ds.has_pois(key)
        except Exception:
            out[key] = False
    return out


# ── Worker: run ONE simulation (one seed, one condition) ─────────────────────


def _spec_for(cfg: dict, condition: str) -> Optional[RecommenderSpec]:
    """The spec driving ``condition``, or None for the No-RS control.

    Specs travel into the worker as plain dicts (``asdict`` on the config), so
    they are rebuilt here rather than unpickled as dataclasses — keeping the
    task payload to primitives.
    """
    if condition == CONTROL:
        return None
    if condition == ORACLE:
        return RecommenderSpec(label=ORACLE, oracle=True)
    for raw in cfg.get("recommenders", ()):
        if raw.get("label") == condition:
            return RecommenderSpec(**raw)
    return None


_NO_POPULATION_WARNED: set = set()


def _population_rows(ds, city: str, seed: int, num_agents: int):
    """The pre-built ACS/PUMS population for this run, or None to fall back.

    A seed picks a replicate: replicates are what a seed used to do to the
    population, so seed s reads replicate s (mod however many were built) and
    seeds keep varying who the agents are, not only how they behave.

    Falls back to the survey persona population — loudly — when nothing is
    built for this metro, or when the built population is smaller than the run
    asks for. Falling back is the right call rather than failing the run, but
    it changes what the agents ARE, so it must never happen quietly.
    """
    n_reps = ds.population_replicates(city)
    if not n_reps:
        # Said once per metro per process, not per run: a metro with no built
        # population is a normal state, but a run silently using a different
        # population than the operator believes is exactly what this codebase
        # refuses to let happen quietly elsewhere.
        if city not in _NO_POPULATION_WARNED:
            _NO_POPULATION_WARNED.add(city)
            warnings.warn(
                f"{city}: no ACS/PUMS population built; agents come from the "
                "survey persona population with run-time commute sampling. "
                "Build one with db/build_population.py.",
                RuntimeWarning, stacklevel=2,
            )
        return None
    # More rows than agents: carless people outside the NYC exception who
    # cannot walk or cycle to work are left out by the Simulation, and the
    # next rows take their places (1-2% of a metro's population).
    rows = ds.population(city, replicate=int(seed) % n_reps,
                         limit=num_agents + max(100, num_agents // 5))
    if rows is None:
        return None
    if len(rows) < num_agents:
        warnings.warn(
            f"{city}: built population holds {len(rows):,} agents but the run asks "
            f"for {num_agents:,}; falling back to the survey persona population. "
            "Rebuild with db/build_population.py --agents to use census geography.",
            RuntimeWarning, stacklevel=2,
        )
        return None
    return rows


def _build_sim(cfg: dict, seed: int, condition: str, network, poi_rows,
               spec: Optional[RecommenderSpec]):
    ds = get_datasource()
    population_rows = _population_rows(ds, cfg["city"], seed, int(cfg["num_agents"]))
    return Simulation(
        num_agents=int(cfg["num_agents"]),
        seed=int(seed),
        use_recommenders=(condition != CONTROL),
        # Population from the datasource: under WELFARE_RS_DATASOURCE=postgres
        # this is the survey.personas view, otherwise the persona CSV. The path
        # is still passed as the fallback for a worker whose datasource has
        # neither (personas() returns None).
        persona_rows=ds.personas(),
        persona_csv_path=ds.persona_csv_path() or params.SURVEY_PERSONA_CSV_PATH,
        # When present this decides the agents entirely — census geography and
        # ACS/PUMS demographics — and the personas above are consulted only for
        # the psychometrics of the respondent each agent matched.
        population_rows=population_rows,
        road_network=network,
        poi_rows=poi_rows,
        disabled_modes=("transit",),
        # The control still builds a recommender it never consults, so that the
        # catalog, tastes and RNG streams are identical to the treated arms;
        # `use_recommenders=False` is what actually withholds it from agents.
        recommender_config=(spec.to_engine_config() if spec is not None else None),
        recommender_factory=_make_recommender_factory(spec),
    )


def _agent_segments(sim) -> Dict[int, dict]:
    """``agent_id -> {income, trust, paradigm, car_access}``. The paradigm only
    acts on mode choice, so it matters for agents with more than one mode."""
    segments = sim_metrics.agent_segments(sim)
    return {a.id: {**segments[a.id], "car_access": a.car_access_type} for a in sim.agents}


def _die_with_host() -> None:
    """Have the kernel kill this process the moment its host dies (Linux).

    A dead host's simulations are retried on a new one; left running, the old
    ones would hold memory the governor has already released.
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        return
    if os.getppid() != _HOST_PID:  # the host died before the signal was armed
        os._exit(1)


def _simulate_and_save(conn, cfg, seed, condition, cancel_event, progress_q, checkpoint,
                       network) -> None:
    """Body of a simulation child (forked by its metro's host, see _host_main):
    simulate on the host's network, save, report one (outcome, payload)."""
    _die_with_host()
    try:
        # If memory does run out, the kernel should kill a simulation, which is
        # retried, rather than the web process, which holds every study.
        with open("/proc/self/oom_score_adj", "w") as fh:
            fh.write("800")
    except OSError:
        pass
    try:
        res = _simulate(cfg, seed, condition, cancel_event, progress_q, network)
        conn.send(("ok", _save_task_result(checkpoint, res)))
    except SimulationCancelled:
        conn.send(("cancelled", None))
    except BaseException:  # report any failure as text: not every exception pickles
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def _simulate(cfg: dict, seed: int, condition: str, cancel_event=None,
              progress_q=None, network=None) -> dict:
    """Run ONE simulation: one condition at one seed. The unit of parallelism.

    Returns a picklable dict of that run's metrics, its per-agent utilities and
    its viz payload. ``utils`` is the piece that makes the fan-out possible:
    Table 2 pairs a recommender against No RS agent-by-agent, and shipping the
    ``{agent_id: eval-day leisure utility}`` map back lets the parent do that
    pairing across process boundaries (one float per agent).

    Raises SimulationCancelled if ``cancel_event`` is set between days. ``network``
    is the metro's network when the caller already holds it (a simulation host).
    """
    poi_rows = get_datasource().poi_rows(cfg["city"]) if bool(cfg["use_real_pois"]) else None
    use_real_pois = bool(poi_rows)
    if network is None:
        network = build_network(cfg["city"])  # metro two-layer or legacy preset

    # Cooperative cancellation: checked between days by the engine. `cancel_event`
    # is the study's manager Event, set when the study is cancelled or fails.
    should_stop = (lambda: cancel_event.is_set()) if cancel_event is not None else None

    per_day: List[Dict[int, dict]] = []
    spec = _spec_for(cfg, condition)
    sim = _build_sim(cfg, seed, condition, network, poi_rows, spec)
    # Dashboard panels accumulate across every day (see metrics.RunMetrics);
    # only Table 1 / Table 2 stay eval-day quantities.
    run_metrics = sim_metrics.RunMetrics(sim)

    def _on_day(day_index, sim_) -> None:
        per_day.append(viz.build_timelines(sim_))
        run_metrics.observe_day(sim_)
        # Day-granular progress. With one simulation per task there is no other
        # way for the parent to see inside a task, and tasks are long.
        if progress_q is not None:
            try:
                progress_q.put({"seed": int(seed), "condition": condition,
                                "day": int(day_index) + 1})
            except Exception:
                pass  # progress is advisory — never fail a run over it

    sim.run_days(int(cfg["num_days"]), on_day_complete=_on_day, should_stop=should_stop)

    merged = viz.merge_day_timelines(per_day)
    summ = sim.summarize()
    out = {
        "seed": int(seed),
        "condition": condition,
        "t1": table1_metrics(sim),
        "utils": eval_day_leisure_utilities(sim.agents),
        "metrics": run_metrics.finalize(sim),
        "summary": {
            "recommendation_acceptance_rate": summ.get("recommendation_acceptance_rate", 0.0),
            "avg_travel_time_min": summ.get("avg_travel_time_min", 0.0),
        },
        # For the saved study record only: every eval-day leisure trip's net
        # utility, and each agent's segments, so Table 1 can be broken down by
        # decision paradigm (or income, trust, car access) after the fact.
        "eval_trips": [[a.id, leisure_net_utility(t)]
                       for a in sim.agents for t in a.trips if t.purpose == "leisure"],
        "agent_segments": _agent_segments(sim),
        "viz": {
            "merged": merged,
            "trip_index": viz.build_trip_index(merged),
            "time_span": viz.time_span(merged),
        },
    }

    # The POI catalog, the agent population and the network extent are identical
    # across conditions at a given seed, and the control always runs — so ship
    # them back from that task alone instead of pickling them once per arm.
    if condition == CONTROL:
        out["agents"] = sim_metrics.agent_frame(sim)
        net = sim.road_network
        south, west, north, east = net.bounds
        csouth, cwest, cnorth, ceast = net.core_bounds
        out["pois"] = [
            {"lon": round(p.location[1], 6), "lat": round(p.location[0], 6),
             "category": p.category, "label": f"{p.name} · {p.category}"}
            for p in sim.place_catalog
        ]
        out["net"] = {
            # Initial map frame: the principal city; agents commute in from
            # the wider metro bounds visible on zoom-out.
            "view": {"latitude": net.core_center[0], "longitude": net.core_center[1]},
            "bounds": {"south": south, "west": west, "north": north, "east": east},
            "core_bounds": {"south": csouth, "west": cwest, "north": cnorth, "east": ceast},
            "num_intersections": net.num_base_nodes,
            "poi_count": len(sim.place_catalog),
            "poi_source": "real dataset" if use_real_pois else "synthetic",
        }
    return out


# ── Study record (saved to disk) ─────────────────────────────────────────────

def _save_study_record(run_id: str, cfg: RunConfig, seeds: List[int],
                       results: Dict[Tuple[int, str], dict],
                       table1: List[dict], table2: List[dict]) -> str:
    """Write the study's per-run details to ``STUDY_RECORD_DIR`` and return the path.

    The website shows Table 1 and Table 2; this keeps what they were computed
    from, run by run, so the study can be re-analysed later without re-running
    it: each (seed, condition) run's Table-1 row, its Table-2 row against the
    same seed's No-RS run, every eval-day leisure trip utility, each agent's
    eval-day and whole-run mean utility, and each agent's segments (decision
    paradigm, income band, trust group, car access). Agent ids are JSON keys,
    so they come back as strings.
    """
    runs = []
    for seed in seeds:
        control_utils = results[(seed, CONTROL)]["utils"]
        for condition in cfg.all_conditions():
            res = results[(seed, condition)]
            runs.append({
                "seed": seed,
                "condition": condition,
                "table1": res["t1"],
                "table2": (None if condition == CONTROL
                           else table2_from_utilities(res["utils"], control_utils)),
                "summary": res["summary"],
                "eval_trips": res["eval_trips"],
                "eval_utility": res["utils"],
                "run_utility": res["metrics"]["run_utility"],
                "agent_segments": res["agent_segments"],
            })
    record = {
        "run_id": run_id,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": asdict(cfg),
        "conditions": cfg.all_conditions(),
        "seeds": seeds,
        "table1": table1,
        "table2": table2,
        "runs": runs,
    }
    os.makedirs(STUDY_RECORD_DIR, exist_ok=True)
    path = os.path.join(STUDY_RECORD_DIR, f"{cfg.city}_{run_id}.json.gz")
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(record, fh)
    os.replace(tmp, path)  # never leave a half-written record under the final name
    return path


# ── Stored studies (the Storage tab) ─────────────────────────────────────────
#
# Studies are held in memory only while they are among the few most recently
# used, and a service restart drops them all. On disk, though, every finished
# study keeps its record (STUDY_RECORD_DIR) and every simulation's checkpoint
# (TASK_RESULT_DIR/<run id>/). The Storage tab lists the records and reopens a
# study from its checkpoints, exactly as it was: nothing is re-simulated, so a
# study made with an older recommender setting stays what it was.

#: Recommender fields of records saved before the convex fix. The
#: personalization weight is now lambda (the same value); reviews and the
#: learning rate no longer exist and are dropped.
_LEGACY_SPEC_FIELDS = {"w_personalization": "personalization_lambda"}

_stored_cache: Dict[str, Tuple[float, Optional[dict]]] = {}  # path -> (mtime, summary)
_stored_lock = threading.Lock()


def _spec_from_record(rec: dict) -> RecommenderSpec:
    known = {f.name for f in fields(RecommenderSpec)}
    spec = {}
    for key, value in rec.items():
        key = _LEGACY_SPEC_FIELDS.get(key, key)
        if key in known:
            spec[key] = value
    return RecommenderSpec(**spec)


def _read_study_record(path: str) -> dict:
    """Summarise one study record for the Storage tab.

    Keys starting with ``_`` are for reopening and are not sent to the browser.
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        record = json.load(fh)
    raw = record["config"]
    run_id = record["run_id"]
    seeds = [int(s) for s in record["seeds"]]
    conditions = list(record["conditions"])
    run_fields = {f.name for f in fields(RunConfig)}
    config = RunConfig(
        **{k: v for k, v in raw.items() if k in run_fields and k != "recommenders"},
        recommenders=tuple(_spec_from_record(r) for r in raw.get("recommenders", [])),
    )

    reason = None
    if config.all_conditions() != conditions:
        reason = "saved before the Oracle control existed"
    else:
        missing = [
            (s, c) for s in seeds for c in conditions
            if not all(os.path.exists(_task_checkpoint(run_id, s, c) + ext)
                       for ext in (".pkl", ".viz.pkl"))
        ]
        if missing:
            reason = f"{len(missing)} of {len(seeds) * len(conditions)} simulation checkpoints are missing"
    saved = record.get("saved_at")
    return {
        "run_id": run_id,
        "city": config.city,
        "city_label": _CITY_LABELS.get(config.city, config.city),
        "saved_at": (calendar.timegm(time.strptime(saved, "%Y-%m-%dT%H:%M:%SZ"))
                     if saved else os.path.getmtime(path)),
        "num_agents": config.num_agents,
        "num_days": config.num_days,
        "num_seeds": len(seeds),
        "conditions": conditions,
        # Each built arm's settings as it ran, so studies made with different
        # recommender settings can be told apart.
        "recommenders": [asdict(spec) for spec in config.recommenders],
        "table1": [
            {k: row.get(k) for k in ("condition", "mean_utility", "sigma_u", "abstention_rate")}
            for row in record.get("table1", [])
        ],
        "reopenable": reason is None,
        "reason": reason,
        "_config": config,
        "_seeds": seeds,
    }


def stored_studies() -> List[dict]:
    """Every study record on disk, newest first. Records are parsed once and
    re-read only when the file changes."""
    if not os.path.isdir(STUDY_RECORD_DIR):
        return []
    paths = sorted(os.path.join(STUDY_RECORD_DIR, n) for n in os.listdir(STUDY_RECORD_DIR)
                   if n.endswith(".json.gz"))
    out = []
    with _stored_lock:
        for path in paths:
            mtime = os.path.getmtime(path)
            cached = _stored_cache.get(path)
            if cached is None or cached[0] != mtime:
                try:
                    summary = _read_study_record(path)
                except Exception as exc:  # one unreadable record must not hide the rest
                    print(f"stored study {path}: unreadable ({exc!r})", flush=True)
                    summary = None
                _stored_cache[path] = cached = (mtime, summary)
            if cached[1] is not None:
                out.append(cached[1])
    out.sort(key=lambda s: s["saved_at"], reverse=True)
    return out


# ── Aggregation across seeds ─────────────────────────────────────────────────

def _agg_table1_row(condition: str, per_seed_rows: List[dict]) -> dict:
    """Average a Table-1 condition across seeds; σ_U = std of per-seed Ū."""
    means = [r["mean_utility"] for r in per_seed_rows]
    return {
        "condition": condition,
        "mean_utility": float(np.mean(means)),
        "sigma_u": float(np.std(means)) if len(means) > 1 else 0.0,
        "neg_rate": float(np.mean([r["neg_rate"] for r in per_seed_rows])),
        "abstention_rate": float(np.mean([r["abstention_rate"] for r in per_seed_rows])),
        "gini": float(np.mean([r["gini"] for r in per_seed_rows])),
        "n_leisure_trips": int(np.sum([r["n_leisure_trips"] for r in per_seed_rows])),
    }


def _agg_table2(condition: str, per_seed_t2: List[dict]) -> dict:
    return {
        "condition": condition,
        "harmed_pct": float(np.mean([t["harmed_pct"] for t in per_seed_t2])),
        "improved_pct": float(np.mean([t["improved_pct"] for t in per_seed_t2])),
        "mean_orc": float(np.mean([t["mean_orc"] for t in per_seed_t2])),
        "n_matched": int(np.sum([t["n_matched"] for t in per_seed_t2])),
    }


def _pooled_mean(rows: List[dict], value_key: str, weight_key: str = "n") -> float:
    """Sample-size-weighted mean across seeds.

    Weighted, not a mean of means: a seed where only nine agents in a band took
    a leisure trip would otherwise count as much as one where sixty did, which
    for the thinner income bands is most of the noise in the panel.
    """
    total = sum(float(r.get(weight_key, 0) or 0) for r in rows)
    if total <= 0:
        return 0.0
    return sum(float(r.get(value_key, 0.0)) * float(r.get(weight_key, 0) or 0)
               for r in rows) / total


def _agg_segment_panel(per_seed: List[dict]) -> Dict[str, dict]:
    """Merge one segmentation's ``{group: {n, mean_utility, neg_rate}}`` maps."""
    groups = {g for seed_rows in per_seed for g in seed_rows}
    out = {}
    for group in sorted(groups):
        rows = [s[group] for s in per_seed if group in s]
        out[group] = {
            "n": int(sum(r["n"] for r in rows)),
            "mean_utility": _pooled_mean(rows, "mean_utility"),
            "neg_rate": _pooled_mean(rows, "neg_rate"),
        }
    return out


def _agg_spatial(per_seed: List[dict]) -> Dict[str, dict]:
    """Merge the spatial panel across seeds."""
    def merge(rows: List[dict]) -> dict:
        return {
            "n": int(sum(r["n"] for r in rows)),
            "mean_distance_km": _pooled_mean(rows, "mean_distance_km"),
            "mean_emissions_g": _pooled_mean(rows, "mean_emissions_g"),
            "mean_travel_min": _pooled_mean(rows, "mean_travel_min"),
            # A median cannot be pooled from per-seed medians; the weighted mean
            # of them is the honest approximation and is labelled as such.
            "median_detour": _pooled_mean(rows, "median_detour"),
            "median_excess_km": _pooled_mean(rows, "median_excess_km"),
        }

    bands = {b for s in per_seed for b in s["by_income"]}
    return {
        "overall": merge([s["overall"] for s in per_seed]),
        "by_income": {
            band: merge([s["by_income"][band] for s in per_seed if band in s["by_income"]])
            for band in sorted(bands)
        },
    }


def _agg_per_day(per_seed: List[List[dict]]) -> List[dict]:
    """Average each day's record across seeds, truncated to the shortest run."""
    if not per_seed:
        return []
    length = min(len(s) for s in per_seed)
    keys = sorted({k for s in per_seed for row in s for k in row})
    out = []
    for day in range(length):
        row = {"day": day + 1}
        for key in keys:
            values = [s[day][key] for s in per_seed if key in s[day]]
            row[key] = float(np.mean(values)) if values else 0.0
        out.append(row)
    return out


def _agg_lorenz(curves: List[List[list]]) -> List[list]:
    """Average Lorenz curves pointwise.

    ``lorenz_points`` normally samples a fixed 41-point x-grid, which makes a
    pointwise mean well defined — but its no-visits early return emits just two
    points. A seed in which nobody took a leisure trip would therefore be a
    short curve, and zipping it against the others would run off its end. Only
    curves on the majority grid are averaged; the rest are dropped rather than
    interpolated, since a two-point curve carries no shape to preserve.
    """
    if not curves:
        return []
    widths = [len(c) for c in curves]
    grid = max(set(widths), key=widths.count)
    usable = [c for c in curves if len(c) == grid]
    if not usable:
        return []
    return [
        [usable[0][i][0], float(np.mean([c[i][1] for c in usable]))]
        for i in range(grid)
    ]


def _agg_metrics(per_seed: List[dict]) -> Dict[str, object]:
    """Everything the dashboard needs for one condition, merged over seeds."""
    category: Dict[str, Dict[str, int]] = {}
    for seed_rows in per_seed:
        for band, by_category in seed_rows["category_by_income"].items():
            row = category.setdefault(band, {})
            for name, count in by_category.items():
                row[name] = row.get(name, 0) + count

    segmentations = {k for s in per_seed for k in s["segments"]}
    lorenz = [s["lorenz"] for s in per_seed]
    return {
        "per_day": _agg_per_day([s["per_day"] for s in per_seed]),
        "segments": {
            key: _agg_segment_panel([s["segments"][key] for s in per_seed if key in s["segments"]])
            for key in sorted(segmentations)
        },
        "category_by_income": category,
        "spatial": _agg_spatial([s["spatial"] for s in per_seed]),
        "lorenz": _agg_lorenz(lorenz),
        "footfall": {
            key: float(np.mean([s["footfall"][key] for s in per_seed]))
            for key in per_seed[0]["footfall"]
        } if per_seed else {},
        "taste": {
            key: float(np.mean([s["taste"][key] for s in per_seed]))
            for key in per_seed[0]["taste"]
        } if per_seed else {},
    }


# ── Run manager ──────────────────────────────────────────────────────────────

class RunManager:
    """Builds and caches studies, running up to MAX_STUDIES of them at once and
    parallelising each (seed x condition) grid across a shared process pool."""

    # Studies now retain map geometry for EVERY condition, not just one, so a
    # cached study is several times larger than it used to be. Keep fewer.
    def __init__(self, max_runs: int = 0):
        # At least one resident slot per concurrently running study, or studies
        # would evict each other's results as they finish.
        max_runs = max_runs or max(4, MAX_STUDIES + 1)
        # Studies run concurrently, up to MAX_STUDIES. Memory and CPU stay
        # bounded not by serialising them but by ResourceGovernor, which splits
        # a single shared process pool evenly between whoever is running.
        self._executor = ThreadPoolExecutor(max_workers=MAX_STUDIES,
                                            thread_name_prefix="study")
        # Reopening a stored study only reads checkpoints, so it must not queue
        # behind studies that are simulating.
        self._reopen_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reopen")
        self._runs: Dict[str, Run] = {}
        self._jobs: Dict[str, Job] = {}
        self._access: Dict[str, float] = {}   # run_id -> last read, for LRU eviction
        self._lock = threading.Lock()
        self._max_runs = max_runs

    def pois_available(self) -> bool:
        return pois_available()

    def get_run(self, run_id: str) -> Optional[Run]:
        run = self._runs.get(run_id)
        if run is not None:
            # Touch on every read so an open tab keeps its study alive.
            self._access[run_id] = time.time()
        return run

    def list_runs(self) -> List[dict]:
        """Every run this process knows about, newest first — no geometry.

        Union of jobs (which exist from submission, so a study still computing
        appears immediately) and completed runs. ``available`` says whether the
        result is still resident: a study can be 'done' yet evicted, in which
        case it is listed but cannot be opened.
        """
        with self._lock:
            jobs = dict(self._jobs)
            resident = set(self._runs)
            runs = dict(self._runs)

        out: List[dict] = []
        for run_id, job in jobs.items():
            snap = job.snapshot()
            cfg = job.config or (runs[run_id].config if run_id in runs else None)
            out.append({
                "run_id": run_id,
                "available": run_id in resident,
                "city": cfg.city if cfg else None,
                "city_label": _CITY_LABELS.get(cfg.city, cfg.city) if cfg else None,
                "conditions": list(cfg.all_conditions()) if cfg else [],
                "num_agents": cfg.num_agents if cfg else None,
                "num_days": cfg.num_days if cfg else None,
                "num_seeds": cfg.num_seeds if cfg else None,
                **snap,
            })
        # A run whose job was pruned would otherwise vanish from the list.
        for run_id, run in runs.items():
            if run_id in jobs:
                continue
            out.append({
                "run_id": run_id, "available": True, "status": "done",
                "city": run.config.city,
                "city_label": _CITY_LABELS.get(run.config.city, run.config.city),
                "conditions": list(run.conditions),
                "num_agents": run.config.num_agents, "num_days": run.config.num_days,
                "num_seeds": run.config.num_seeds,
                "created_at": None, "finished_at": None, "error": None,
                "progress": {"current": 0, "total": 0, "cond_total": 0, "per_condition": {}},
            })
        out.sort(key=lambda r: (r.get("created_at") or 0.0), reverse=True)
        return out

    def get_job(self, run_id: str) -> Optional[Job]:
        return self._jobs.get(run_id)

    def cancel(self, run_id: str) -> bool:
        """Signal a running study to stop. Returns True if a running job was
        signalled. The study aborts (no result stored) within ~one day."""
        job = self._jobs.get(run_id)
        if job is not None and job.status == "running":
            job.request_cancel()
            return True
        return False

    def list_stored(self) -> List[dict]:
        """Every study saved on disk, for the Storage tab; ``open`` says whether
        it is currently loaded."""
        with self._lock:
            resident = set(self._runs)
        return [
            {**{k: v for k, v in s.items() if not k.startswith("_")},
             "open": s["run_id"] in resident}
            for s in stored_studies()
        ]

    def open_stored(self, run_id: str) -> tuple[str, str]:
        """Load a stored study from its checkpoints. Returns ``(run_id, status)``
        like ``submit``. Raises KeyError for an unknown id and ValueError for a
        study that cannot be reopened."""
        stored = next((s for s in stored_studies() if s["run_id"] == run_id), None)
        if stored is None:
            raise KeyError(run_id)
        if not stored["reopenable"]:
            raise ValueError(stored["reason"])
        config, seeds = stored["_config"], stored["_seeds"]
        with self._lock:
            if run_id in self._runs:
                return run_id, "ready"
            job = self._jobs.get(run_id)
            if job is not None and job.status == "running":
                return run_id, "running"
            job = Job(run_id=run_id, total=len(seeds), config=config, reopened=True)
            self._jobs[run_id] = job
        self._reopen_executor.submit(self._execute, run_id, config, seeds)
        return run_id, "running"

    def submit(self, config: RunConfig) -> tuple[str, str]:
        """Return ``(run_id, status)``. status: ready | running."""
        run_id = config.run_id()
        with self._lock:
            if run_id in self._runs:
                return run_id, "ready"
            if run_id in self._jobs and self._jobs[run_id].status == "running":
                return run_id, "running"
            job = Job(run_id=run_id, total=len(config.seeds()), config=config)
            self._jobs[run_id] = job
            # Registered now, not when it starts running, so a batch submitted
            # together shares memory fairly from its first simulation on.
            GOVERNOR.register(run_id)
        self._executor.submit(self._execute, run_id, config)
        return run_id, "running"

    def _execute(self, run_id: str, config: RunConfig,
                 reopen_seeds: Optional[List[int]] = None) -> None:
        job = self._jobs[run_id]
        try:
            run = self._build_study(run_id, config, job, reopen_seeds)
            run.created_at = job.created_at
            # A reopened study was only read from disk; its run time is unknown.
            run.duration_sec = None if job.reopened else time.time() - job.created_at
            with self._lock:
                self._runs[run_id] = run
                self._access[run_id] = time.time()
                while len(self._runs) > self._max_runs:
                    # Least-recently-ACCESSED, not oldest-inserted. Now that runs
                    # are listable and linkable (?run=<id>), the oldest study is
                    # quite likely the one someone left open in a tab; dropping it
                    # from under them would 410 a view they are actively using.
                    victim = min(self._runs, key=lambda r: self._access.get(r, 0.0))
                    self._runs.pop(victim, None)
                    self._access.pop(victim, None)
            job.finish("done")
            job.publish({"type": "done", "run_id": run_id})
        except SimulationCancelled:
            job.finish("cancelled")
            job.publish({"type": "cancelled", "run_id": run_id})
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.finish("error")
            job.publish({"type": "error", "message": job.error})
        finally:
            GOVERNOR.unregister(run_id)

    def _build_study(self, run_id: str, cfg: RunConfig, job: Job,
                     reopen_seeds: Optional[List[int]] = None) -> Run:
        """Run a study, reusing any finished checkpoints. With ``reopen_seeds``
        it only reloads a stored study: those seeds, from checkpoints alone."""
        seeds = cfg.seeds() if reopen_seeds is None else reopen_seeds
        conditions = cfg.all_conditions()
        # One task = one simulation = one (seed, condition) pair, No RS included.
        # Oracle tasks first: they are the slowest, so they set the study's length.
        tasks = [(s, c) for s in seeds for c in conditions]
        tasks.sort(key=lambda task: task[1] != ORACLE)
        num_days = max(1, int(cfg.num_days))
        job.total = len(tasks) * num_days   # progress counted in simulated days
        # Each condition runs once per seed, so every arm has the same
        # denominator. Seeded at zero up front so the UI lists all the arms the
        # moment the study starts, rather than having them pop in one by one.
        cond_total = len(seeds) * num_days
        job.cond_total = cond_total
        job.per_condition = {c: 0 for c in conditions}

        # Simulations a previous attempt at this same study already finished.
        cfg_dict = asdict(cfg)
        checkpoint = {t: _task_checkpoint(run_id, *t) for t in tasks}
        results: Dict[Tuple[int, str], dict] = {}
        for t in tasks:
            res = _load_task_result(checkpoint[t], *t)
            if res is not None:
                results[t] = res
                job.current += num_days
                job.per_condition[t[1]] += num_days
        tasks = [t for t in tasks if t not in results]
        if tasks and reopen_seeds is not None:
            raise RuntimeError(f"{len(tasks)} simulation checkpoints are missing; "
                               "a stored study is reopened, never re-simulated")

        # Ticks arrive from every worker at once, so the counter needs a lock.
        # Publishing happens outside it to keep the SSE fan-out off the critical
        # section.
        tick_lock = threading.Lock()

        def _tick(event: dict) -> None:
            with tick_lock:
                job.current = min(job.current + 1, job.total)
                cond = event.get("condition")
                if cond in job.per_condition:
                    job.per_condition[cond] = min(job.per_condition[cond] + 1, cond_total)
                # Snapshot under the lock; the publish below must not race a
                # concurrent worker mutating the dict mid-serialisation.
                cur, by_cond = job.current, dict(job.per_condition)
            # Elapsed is computed server-side and advanced locally by the client,
            # so the timer is immune to clock skew between browser and box.
            job.publish({"type": "progress", "current": cur, "total": job.total,
                         "cond_total": cond_total, "per_condition": by_cond,
                         "elapsed_sec": time.time() - job.created_at})

        if tasks:
            self._run_tasks(run_id, cfg_dict, tasks, checkpoint, results, job, _tick)
        return self._assemble(run_id, cfg, seeds, results,
                              save_record=reopen_seeds is None)

    @staticmethod
    def _run_tasks(run_id: str, cfg_dict: dict, tasks: List[Tuple[int, str]],
                   checkpoint: Dict[Tuple[int, str], str],
                   results: Dict[Tuple[int, str], dict], job: Job, tick) -> None:
        """Run ``tasks`` on the shared pool as the governor allows, adding each
        finished simulation to ``results``.

        Tasks start in order (Oracle first) whenever the governor admits them,
        re-checked on every pass. So a study that starts alone spreads across the
        box, gives slots back when another study starts, and takes them again
        when that one ends. A task whose process dies is retried, up to
        MAX_TASK_ATTEMPTS in all. If the study is cancelled or fails, its running
        simulations are told to stop and are waited for, so that their memory
        is reserved until it is actually free.
        """
        # 'spawn': fork-after-threads under uvicorn is unsafe. Manager-backed
        # Event and Queue are shareable with the workers: setting the Event
        # stops each simulation at its next day boundary; the Queue carries
        # day-level progress back.
        ctx = mp.get_context("spawn")
        manager_proc = ctx.Manager()
        job.mp_cancel = manager_proc.Event()
        progress_q = manager_proc.Queue()
        if job.cancel.is_set():  # cancel arrived during setup
            job.mp_cancel.set()

        stop_drain = threading.Event()

        def _drain() -> None:
            while not stop_drain.is_set():
                try:
                    tick(progress_q.get(timeout=0.2))
                except queue.Empty:
                    continue
                except Exception:
                    break  # manager torn down — expected at end of study

        drain = threading.Thread(target=_drain, name="progress", daemon=True)
        drain.start()

        city = cfg_dict["city"]
        cost = {t: TaskCost(_task_memory_gb(cfg_dict, t[1]), t[1] == ORACLE) for t in tasks}
        pending = list(tasks)
        attempts = {t: 0 for t in tasks}
        in_flight: Dict[Future, Tuple[int, str]] = {}
        HOSTS.acquire(city)

        def _start_what_fits() -> None:
            for task in list(pending):
                if not GOVERNOR.try_acquire(run_id, cost[task]):
                    continue
                pending.remove(task)
                attempts[task] += 1
                in_flight[HOSTS.host(city).submit(cfg_dict, task[0], task[1], job.mp_cancel,
                                                  progress_q, checkpoint[task])] = task

        def _finish(fut) -> Tuple[int, str]:
            task = in_flight.pop(fut)
            GOVERNOR.release(run_id, cost[task])
            return task

        try:
            while pending or in_flight:
                if job.cancel.is_set():
                    raise SimulationCancelled()
                _start_what_fits()
                if not in_flight:
                    GOVERNOR.wait(timeout=2.0)  # nothing of ours running: wait for room
                    continue
                done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED, timeout=2.0)
                for fut in done:
                    task = _finish(fut)
                    try:
                        results[task] = fut.result()
                    except SimulationProcessDied as exc:
                        if attempts[task] >= MAX_TASK_ATTEMPTS:
                            raise
                        print(f"study {run_id}: {exc}; retrying "
                              f"(attempt {attempts[task] + 1} of {MAX_TASK_ATTEMPTS})", flush=True)
                        pending.insert(0, task)
        except BaseException:
            # Stop this study's simulations and wait for them to wind down
            # (within one simulated day), keeping their memory reserved until then.
            job.mp_cancel.set()
            wait(list(in_flight))
            raise
        finally:
            for fut in list(in_flight):
                _finish(fut)
            HOSTS.release(city)
            stop_drain.set()
            drain.join(timeout=1.0)
            job.mp_cancel = None
            manager_proc.shutdown()


    @staticmethod
    def _welfare_map(conditions: List[str], ok_seeds: List[int],
                     results: Dict[Tuple[int, str], dict]) -> List[dict]:
        """Per-agent home point carrying each arm's utility delta vs the control.

        One row per (seed, agent) that took a leisure trip under the control and
        under at least one recommender; ``delta`` holds ``condition -> ΔU``. The
        map bins these into hexagons, so the payload is deliberately flat and
        rounded rather than pre-aggregated — the frontend chooses the bin size.

        Built from each arm's ``run_utility`` (mean leisure net utility over the
        WHOLE run), not from the eval-day ``utils`` that Table 2 uses. Table 2 is
        a paper-defined eval-day quantity and stays that way; a map is not, and
        on a single day only a minority of agents go out, so intersecting "went
        out under the control" with "went out under this arm" left barely a
        couple of dozen points to bin.

        Agents with no leisure trip in an arm are simply absent from that arm's
        delta rather than recorded as zero: a zero would read as "unaffected"
        when what actually happened is "did not go out", and averaged into a
        hexagon it would pull the cell toward neutral.

        Capped at ``WELFARE_MAP_MAX_POINTS``: this is the one part of the run
        payload that scales with agents x seeds, and at the top of the allowed
        range (10,000 agents, 12 seeds) it would be a 120,000-row, ~12 MB JSON
        body on every /api/runs/{id}. Over the cap it is thinned by an even
        stride rather than by truncation, which would otherwise drop whole seeds
        and, with them, whole parts of the map.
        """
        rows: List[dict] = []
        for seed in ok_seeds:
            frame = results[(seed, CONTROL)].get("agents")
            if not frame:
                continue
            control = results[(seed, CONTROL)]["metrics"]["run_utility"]
            for i, agent_id in enumerate(frame["agent_id"]):
                base = control.get(agent_id)
                if base is None:
                    continue
                deltas = {}
                for condition in conditions:
                    if condition == CONTROL:
                        continue
                    value = results[(seed, condition)]["metrics"]["run_utility"].get(agent_id)
                    if value is not None:
                        deltas[condition] = round(float(value) - float(base), 5)
                if not deltas:
                    continue
                rows.append({
                    "lat": frame["home_lat"][i],
                    "lon": frame["home_lon"][i],
                    "income": frame["income"][i],
                    "delta": deltas,
                })
        if len(rows) > WELFARE_MAP_MAX_POINTS:
            stride = math.ceil(len(rows) / WELFARE_MAP_MAX_POINTS)
            rows = rows[::stride]
        return rows

    def _assemble(self, run_id: str, cfg: RunConfig, seeds: List[int],
                  results: Dict[Tuple[int, str], dict], save_record: bool = True) -> Run:
        conditions = cfg.all_conditions()
        # Only seeds that produced *every* condition are usable: a seed missing
        # its No-RS run has nothing to pair against, and one missing a
        # recommender would silently drop that arm from a single row of the
        # cross-seed average.
        ok_seeds = [s for s in seeds if all((s, c) in results for c in conditions)]
        if not ok_seeds:
            raise RuntimeError("study produced no complete seed")

        # Table 1: one row per condition, No RS first (all_conditions order).
        table1 = [
            _agg_table1_row(c, [results[(s, c)]["t1"] for s in ok_seeds])
            for c in conditions
        ]

        # Table 2: each recommender paired against the SAME No-RS run, seed by
        # seed, then averaged. The pairing lives here rather than in the worker
        # because the two runs are now in different processes.
        table2 = [
            _agg_table2(c, [
                table2_from_utilities(results[(s, c)]["utils"],
                                      results[(s, CONTROL)]["utils"])
                for s in ok_seeds
            ])
            for c in conditions if c != CONTROL
        ]

        # A reopened study already has its record; rewriting it would only
        # change its saved date.
        if save_record:
            try:
                path = _save_study_record(run_id, cfg, ok_seeds, results, table1, table2)
                print(f"study {run_id}: record saved to {path}", flush=True)
            except Exception as exc:  # the study itself is still good; say so loudly
                print(f"study {run_id}: FAILED to save study record: {exc!r}", flush=True)

        t1_by_cond = {r["condition"]: r for r in table1}
        t2_by_cond = {r["condition"]: r for r in table2}
        # One headline per condition; the frontend shows whichever the map is on.
        # The No-RS control has no Table-2 row (it is the baseline), so its
        # over-recommendation fields are null rather than zero.
        headlines = {}
        for c in conditions:
            t1r, t2r = t1_by_cond[c], t2_by_cond.get(c)
            headlines[c] = {
                "mean_utility": t1r["mean_utility"],
                "sigma_u": t1r["sigma_u"],
                "neg_rate": t1r["neg_rate"],
                "harmed_pct": t2r["harmed_pct"] if t2r else None,
                "improved_pct": t2r["improved_pct"] if t2r else None,
                "mean_orc": t2r["mean_orc"] if t2r else None,
            }

        aggregate = {
            c: {
                "acceptance_rate": float(np.mean([
                    results[(s, c)]["summary"]["recommendation_acceptance_rate"]
                    for s in ok_seeds])),
                "avg_travel_time_min": float(np.mean([
                    results[(s, c)]["summary"]["avg_travel_time_min"]
                    for s in ok_seeds])),
            }
            for c in conditions
        }

        metrics = {
            c: _agg_metrics([results[(s, c)]["metrics"] for s in ok_seeds])
            for c in conditions
        }
        welfare_map = self._welfare_map(conditions, ok_seeds, results)

        viz_files = {c: {s: results[(s, c)]["viz_file"] for s in ok_seeds} for c in conditions}
        time_spans = {c: {s: results[(s, c)]["time_span"] for s in ok_seeds} for c in conditions}
        # POIs and the network extent came back on the No-RS task only.
        pois_by_seed = {s: results[(s, CONTROL)].get("pois", []) for s in ok_seeds}
        net0 = results[(ok_seeds[0], CONTROL)]["net"]

        return Run(
            run_id=run_id,
            config=cfg,
            seeds=ok_seeds,
            conditions=conditions,
            viz_files=viz_files,
            time_spans=time_spans,
            pois_by_seed=pois_by_seed,
            table1=table1,
            table2=table2,
            headlines=headlines,
            aggregate=aggregate,
            metrics=metrics,
            welfare_map=welfare_map,
            view=net0["view"],
            bounds=net0["bounds"],
            core_bounds=net0["core_bounds"],
            num_days=int(cfg.num_days),
            num_intersections=net0["num_intersections"],
            poi_count=net0["poi_count"],
            poi_source=net0["poi_source"],
        )


# Module-level singleton used by the FastAPI app.
manager = RunManager()
