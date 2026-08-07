#!/bin/bash
# Executed on the CHTC worker. Args: <pilot> <seed>
set -euo pipefail
PILOT=$1
SEED=$2

tar xzf rolenv.tar.gz
tar xzf code.tar.gz

# code.tar.gz unpacks to ./code (vendored) or ./IBCDFO (whole repo)
if [ -d "$PWD/code" ]; then ROOT="$PWD/code"; else ROOT="$PWD/IBCDFO"; fi
export PYTHONPATH="$ROOT:$ROOT/pounders/py:$ROOT/GST_POUNDERS:$ROOT/GST_POUNDERS/seed_sweep_experiments"

# deterministic numerics: execute nodes are heterogeneous
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "host=$(hostname)  pilot=$PILOT  seed=$SEED  $(./rolenv/bin/python -V)"

./rolenv/bin/python run_one.py \
    --pilot "$PILOT" --seed "$SEED" \
    --config experiment_config.json \
    --outdir "results" \
    ${SWEEP_ARGS:-}

# one small tarball back per job
tar czf "out_p${PILOT}_s${SEED}.tar.gz" results
rm -rf results rolenv IBCDFO code
