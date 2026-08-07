#!/bin/bash
# One-shot setup on the CHTC access point (submit node).
#
#   bash setup.sh
#
# Does everything that is light enough for the AP:
#   1. records the AP's OS major version and pins sweep.sub to match
#   2. builds the portable venv (pip wheels only -- no compilation)
#   3. verifies PyROL/pyGSTi import and a GSTProblem can be constructed
#   4. writes joblist.txt
#
# It deliberately does NOT run a full POUNDERS fit here -- that is real compute and
# belongs on an execute node. Use  condor_submit smoke.sub  for that (one job).
#
# Why the OS pin: the venv is built with --copies, so it carries this machine's
# python binary. That binary needs a compatible glibc on the execute node. Building
# on an EL9 AP and landing on an EL8 node would fail. Pinning the requirement to
# this machine's major version removes the class of failure entirely.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

# CHTC APs ship python3 but usually no bare `python`, and not always 3.11.
. ./pick_python.sh
pick_python || exit 1
export PYVER="$PY"
echo "  interpreter: $PY  ($("$PY" --version 2>&1))"

echo "############ 1/4  environment ############"
OSVER=""
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    OSVER=${VERSION_ID%%.*}
    echo "  access point: ${PRETTY_NAME:-unknown}  (major version ${OSVER:-?})"
fi
echo "  glibc: $(ldd --version 2>/dev/null | head -1 || echo unknown)"

if [ -n "$OSVER" ]; then
    "$PY" - "$OSVER" <<'PY'
import re, sys, pathlib
osver = int(sys.argv[1])
p = pathlib.Path("sweep.sub")
s = p.read_text()
new = f"requirements            = (OpSysMajorVer == {osver})"
s2, n = re.subn(r"^requirements\s*=.*$", new, s, count=1, flags=re.M)
if n:
    p.write_text(s2)
    print(f"  sweep.sub pinned to OpSysMajorVer == {osver} "
          f"(the venv carries this machine's python binary)")
PY
fi

echo
echo "############ 2/4  build the portable venv ############"
CHTC_BUILD_ON_EXECUTE=1 bash build_env.sh

echo
echo "############ 3/4  verify the shipped code actually works ############"
rm -rf .verify && mkdir .verify && cd .verify
tar xzf ../code.tar.gz
if [ -d code ]; then ROOT="$PWD/code"; else ROOT="$PWD/IBCDFO"; fi
PYTHONPATH="$ROOT:$ROOT/pounders/py:$ROOT/GST_POUNDERS:$ROOT/GST_POUNDERS/seed_sweep_experiments" \
../rolenv/bin/python - "../experiment_config.json" <<'PY'
import sys, importlib.util
assert importlib.util.find_spec("pyrol") or importlib.util.find_spec("ROL"), \
    "PyROL not importable -- the sweep cannot run"
import gst_seed_experiment as G
import gradient_pounders, general_h_funs, adaptive_shots, adaptive_shot_hook  # noqa: F401
import near_minimal_fpr_reduction                                            # noqa: F401
cfg = G.ExperimentConfig.from_json(sys.argv[1])
prob = G.GSTProblem(cfg, data_seed=1)
print(f"  PyROL importable            : yes")
print(f"  all 7 local modules resolve : yes")
print(f"  GSTProblem builds           : {len(prob.circuits)} circuits, {prob.n} params")
print(f"  truth model                 : {cfg.noise_model}, truth_seed={cfg.truth_seed}")
PY
cd "$HERE" && rm -rf .verify

echo
echo "############ 4/4  job list ############"
"$PY" make_joblist.py
mkdir -p logs

echo
echo "############ ready ############"
echo "  next, run ONE real job on an execute node to be sure:"
echo "      condor_submit smoke.sub"
echo "      condor_watch_q                      # wait for it to finish"
echo "      cat logs/smoke.out                  # expect a 4-row table"
echo
echo "  then the full sweep:"
echo "      condor_submit sweep.sub"
echo "      condor_q"
echo
echo "  and afterwards:"
echo "      $PY collect.py"
