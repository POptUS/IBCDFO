#!/bin/bash
# Runs on the execute node. Args: <seed> <tag> [extra run_one.py args...]
# <tag> must be unique per job -- it names the returned tarball. Several jobs can share a
# seed (e.g. one per baseline fraction), so seed alone is NOT unique.
set -uo pipefail
SEED=$1; TAG=$2; shift 2
OUT="out_${TAG}.tar.gz"

echo "=== $(date -u +%FT%TZ) host=$(hostname) seed=$SEED tag=$TAG"

# ALWAYS produce the tarball, however the job ends.
#
# sweep.sub names $OUT in transfer_output_files, so if the script exits without creating it
# HTCondor cannot transfer it and puts the job on HOLD -- which is far harder to diagnose than
# an empty result, because the logs stay on the execute node. A trap guarantees the file
# exists, so failures come back as data (an empty or partial results/ plus the .err log)
# rather than as a held job.
finish() {
    local rc=$?
    mkdir -p results
    tar czf "$OUT" \
        --exclude='fpr_selection_history.csv' \
        --exclude='iteration_gate_errors.csv' \
        --exclude='iteration_spam_errors.csv' \
        --exclude='x_evaluations.npy' \
        results 2>/dev/null || tar czf "$OUT" -T /dev/null
    echo "=== done status=$rc  $(du -h "$OUT" 2>/dev/null | cut -f1)"
    exit $rc
}
trap finish EXIT

tar xzf rolenv.tar.gz
tar xzf code.tar.gz

# The entry point has to be inside code.tar.gz -- sweep.sub transfers only the tarballs.
if [ ! -f code/run_one.py ]; then
    echo "ERROR: code/run_one.py missing -- re-run 'bash vendor.sh' on the submit node" >&2
    ls -la code/ >&2
    exit 1
fi

# One thread. Cluster nodes are heterogeneous and a single marginal accept/reject decides
# which basin a seed lands in; multithreaded BLAS would make runs irreproducible across
# machines. HTCondor also allocates one core per job.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"
export MPLBACKEND=Agg

# build_env.sh bundles libpythonX.Y.so into rolenv/lib because the venv's interpreter is
# dynamically linked to it and many execute nodes do not have that python installed. Without
# this the job dies with "error while loading shared libraries: libpython3.9.so.1.0" -- and
# only on the nodes that lack it, so the failures look random.
export LD_LIBRARY_PATH="$PWD/rolenv/lib:${LD_LIBRARY_PATH:-}"

# A venv carries no stdlib -- it points at the base installation, which most execute nodes do
# not have. build_env.sh copies the stdlib into rolenv/lib/pythonX.Y; PYTHONHOME makes the
# interpreter look there instead of at the (absent) system python.
PYMM=$(basename "$(ls -d rolenv/lib/python3.* 2>/dev/null | head -1)" | sed 's/^python//')
export PYTHONHOME="$PWD/rolenv"

# Do NOT swallow stderr here: the whole point is to see why it will not start.
if ! ./rolenv/bin/python -c "import sys, encodings" ; then
    echo "ERROR: the bundled interpreter will not start on this node (python${PYMM})." >&2
    echo "  'No module named encodings' means rolenv.tar.gz has no stdlib -- rebuild it" >&2
    echo "  with 'bash build_env.sh' on the submit node, then resubmit." >&2
    ldd ./rolenv/bin/python3 2>&1 | grep -E "not found|libpython" >&2
    exit 1
fi

./rolenv/bin/python code/run_one.py \
    --seed "$SEED" --config experiment_config.json --outdir results "$@"
