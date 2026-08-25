#!/bin/bash
# Runs on the execute node. Args: <seed> <tag> [extra run_one.py args...]
# <tag> must be unique per job -- it names the returned tarball. Several jobs can share a
# seed (e.g. one per baseline fraction), so seed alone is NOT unique.
set -euo pipefail
SEED=$1; TAG=$2; shift 2

echo "=== $(date -u +%FT%TZ) host=$(hostname) seed=$SEED tag=$TAG"
tar xzf rolenv.tar.gz
tar xzf code.tar.gz

# Fail loudly and immediately if the entry point did not make it into code.tar.gz. Without
# this the job "succeeds" into an empty 45-byte tarball and the failure only surfaces at
# collect time, after every job has run.
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

set +e
./rolenv/bin/python code/run_one.py \
    --seed "$SEED" --config experiment_config.json --outdir results "$@"
STATUS=$?
set -e

# Return only what the analysis needs. The four excluded files are ~90% of the bytes and
# nothing in the notebooks reads them, which is what keeps results committable to git.
tar czf "out_${TAG}.tar.gz" \
    --exclude='fpr_selection_history.csv' \
    --exclude='iteration_gate_errors.csv' \
    --exclude='iteration_spam_errors.csv' \
    --exclude='x_evaluations.npy' \
    results
echo "=== done status=$STATUS  $(du -h "out_${TAG}.tar.gz" | cut -f1)"
exit $STATUS
