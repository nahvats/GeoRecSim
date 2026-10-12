"""Submit the main welfare-gating study to the live website, one study per metro.

Each study: Personalized (the study's Standard RS) ungated, with PUP and with RM at the default thresholds
(welfare_rs.params.WELFARE_GATE_DEFAULTS), plus the No RS and Oracle controls the
website always adds. 2,500 agents, 60 days, 5 seeds (42-46).

The studies show on the website (Table 1, Table 2 / ORC) as they finish, and each
finished study is also saved to .cache/studies/<city>_<run_id>.json.gz with the
per-run, per-agent detail behind those tables (see backend/service.py,
_save_study_record). The run ids are written to .cache/studies/main_study.json.

Usage, on the server (posts to uvicorn directly, so no web credentials are needed):
    .venv/bin/python scripts/submit_main_study.py
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from welfare_rs import params  # noqa: E402
from welfare_rs.recommender_systems import RECOMMENDER_PRESETS  # noqa: E402

API = "http://127.0.0.1:8000/api/runs"
CITIES = ["houston", "la", "chicago", "sf_bay", "nyc", "dc", "seattle", "miami"]
BASE = {"label": "Personalized", **RECOMMENDER_PRESETS["Personalized"]}
GATES = params.WELFARE_GATE_DEFAULTS
RECOMMENDERS = [
    BASE,
    {**BASE, "label": "Personalized + PUP", "welfare_gate": "pup", "pup_alpha": GATES["pup_alpha"]},
    {**BASE, "label": "Personalized + RM", "welfare_gate": "rm", "rm_epsilon": GATES["rm_epsilon"]},
]
STUDY = {"num_agents": 2500, "num_days": 60, "seed": 42, "num_seeds": 5,
         "use_real_pois": True, "recommenders": RECOMMENDERS}
MANIFEST = REPO / ".cache" / "studies" / "main_study.json"


def post(body):
    request = urllib.request.Request(API, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def main():
    studies = {}
    for city in CITIES:
        reply = post({**STUDY, "city": city})
        studies[city] = reply["run_id"]
        print(f"{city:8s} {reply['run_id']}  {reply['status']}", flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "study": STUDY, "run_ids": studies,
    }, indent=2))
    print(f"run ids written to {MANIFEST}")


if __name__ == "__main__":
    main()
