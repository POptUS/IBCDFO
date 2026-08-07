#!/bin/bash
# Build the two tarballs the jobs need. Run ONCE on a CHTC submit node
# (or any Linux box with glibc >= 2.28 and python3.11).
#
#   bash build_env.sh
#
# Produces:  rolenv.tar.gz   portable Python + PyROL + pyGSTi
#            code.tar.gz     the IBCDFO tree + experiment_config.json
set -euo pipefail

PYVER=${PYVER:-python3.11}
ROLVER=${ROLVER:-0.5.3}          # pin: matches the Docker image; 0.5.6 is newer
HERE=$(cd "$(dirname "$0")" && pwd)
IBCDFO=$(cd "$HERE/../../.." && pwd)     # .../IBCDFO

# --- preflight: this folder is NOT self-contained -------------------------------
# build_env.sh tars the IBCDFO source tree from three levels up, so the folder must
# sit inside the repo. Copying CHTC/ on its own will fail here.
VENDORED=0
if [ -f "$HERE/code/GST_POUNDERS/seed_sweep_experiments/gst_seed_experiment.py" ]; then
    VENDORED=1
    echo "== using the vendored source in ./code (self-contained mode)"
    sed "s/^/     /" "$HERE/code/PROVENANCE.txt" 2>/dev/null || true
elif [ ! -f "$IBCDFO/GST_POUNDERS/seed_sweep_experiments/gst_seed_experiment.py" ]; then
    echo "ERROR: no source found." >&2
    echo "  Either run  bash vendor_code.sh  on the machine that has the repo," >&2
    echo "  then scp this folder across (self-contained, ~200 KB of source);" >&2
    echo "  or put this folder back inside the repo at" >&2
    echo "     IBCDFO/GST_POUNDERS/seed_sweep_experiments/CHTC/" >&2
    exit 1
else
    echo "== IBCDFO source: $IBCDFO"
fi

echo "== python: $($PYVER --version)"
echo "== glibc : $(ldd --version | head -1)"

rm -rf rolenv rolenv.tar.gz code.tar.gz
$PYVER -m venv rolenv
./rolenv/bin/pip install --upgrade pip wheel

# PyROL: prebuilt manylinux_2_28 wheel, no compilation, no container.
# The import name is ROL (and pyrol); the PyPI name is pyroltrilinos.
./rolenv/bin/pip install --only-binary=:all: "pyroltrilinos==${ROLVER}"
./rolenv/bin/pip install numpy scipy pandas matplotlib pygsti

echo "== verifying PyROL imports the way gradient_pounders.py expects"
./rolenv/bin/python - <<'PY'
import importlib.util
assert importlib.util.find_spec("pyrol") or importlib.util.find_spec("ROL"), "neither pyrol nor ROL importable"
try:
    from pyrol import Objective, Bounds, Solver          # noqa: F401
    print("   pyrol OK")
except Exception:
    import ROL                                            # noqa: F401
    from ROL.numpy_vector import NumpyVector              # noqa: F401
    print("   ROL OK")
import pygsti, numpy, scipy
print("   pygsti", pygsti.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__)
PY

tar czf rolenv.tar.gz rolenv

# ship the code: the IBCDFO tree minus results/checkpoints/git
if [ "$VENDORED" = "1" ]; then
    tar czf code.tar.gz --exclude='__pycache__' --exclude='*.pyc' -C "$HERE" code
else
    tar czf code.tar.gz \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='all_methods_comparison*' --exclude='matched_results' \
        --exclude='criterion_comparison' --exclude='pounders_ladder' \
        --exclude='standard_gst_checkpoints' --exclude='.Trash-0' \
        -C "$(dirname "$IBCDFO")" "$(basename "$IBCDFO")"
fi

# use the frozen copy committed here; only fall back to the live one if absent.
# (the parent config is edited between runs -- pinning it keeps the sweep reproducible)
if [ -f experiment_config.json ]; then
    echo "== using the pinned experiment_config.json in this folder"
else
    echo "== no pinned config; copying the live one from the repo"
    cp "$IBCDFO/GST_POUNDERS/seed_sweep_experiments/experiment_config.json" .
fi

ls -lh rolenv.tar.gz code.tar.gz experiment_config.json
echo "== done. Now: python make_joblist.py  &&  condor_submit sweep.sub"
