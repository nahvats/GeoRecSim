#!/usr/bin/env bash
# Runs run_experiments.ipynb unattended, at low CPU priority so the web app stays responsive.
#
# Safe to call any number of times, and from cron at boot: it does nothing when the
# experiments are done (results/DONE.txt) or already running, and otherwise resumes from
# the per-run cache. Progress: results/progress.log. Notebook errors: results/nbconvert.log.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

[ -e results/DONE.txt ] && exit 0
exec 9> .run.lock
flock -n 9 || exit 0  # already running

mkdir -p results
nice -n 10 ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 run_experiments.ipynb >> results/nbconvert.log 2>&1
