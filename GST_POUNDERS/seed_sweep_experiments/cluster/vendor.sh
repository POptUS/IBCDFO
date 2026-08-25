#!/bin/bash
# Copy the exact source files the jobs need into ./code/, so this folder is self-contained.
# Run on the machine that has the repo:   bash vendor.sh
#
# The list is not guesswork: it is every module that resolved inside the IBCDFO tree during a
# real run (GSTProblem + FPR + adaptive hook + POUNDERS). Everything else the jobs touch is
# pygsti / pyrol / numpy / scipy / pandas / cvxpy from pip.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SSE=$(cd "$HERE/.." && pwd)
IBCDFO=$(cd "$SSE/../.." && pwd)

FILES=(
    "GST_POUNDERS/adaptive_shot_hook.py"
    "GST_POUNDERS/adaptive_shots.py"
    "GST_POUNDERS/near_minimal_fpr_reduction.py"
    "GST_POUNDERS/seed_sweep_experiments/gst_seed_experiment.py"
    "GST_POUNDERS/seed_sweep_experiments/lm_arms.py"
    "pounders/py/general_h_funs.py"
    "pounders/py/gradient_pounders.py"
    "pounders/py/prepare_outputs_before_return_gradient.py"
    # the entry point itself: job.sh runs `python code/run_one.py`, and sweep.sub transfers
    # only the tarballs, so run_one.py must be INSIDE code.tar.gz or the job dies with
    # "can't open file .../code/run_one.py"
    "GST_POUNDERS/seed_sweep_experiments/cluster/run_one.py"
)

[ -f "$IBCDFO/GST_POUNDERS/seed_sweep_experiments/gst_seed_experiment.py" ] || {
    echo "ERROR: cannot see the repo at $IBCDFO" >&2; exit 1; }

rm -rf "$HERE/code"
mkdir -p "$HERE/code"
for f in "${FILES[@]}"; do
    cp "$IBCDFO/$f" "$HERE/code/$(basename "$f")"      # flat: job.sh puts code/ on PYTHONPATH
    printf '   %-58s %5s KB\n' "$(basename "$f")" "$(( ($(wc -c < "$IBCDFO/$f") + 1023) / 1024 ))"
done
cp "$SSE/experiment_config.json" "$HERE/experiment_config.json"
echo "   experiment_config.json"

# provenance: which commit these came from, so a result can always be traced back
{
  echo "vendored_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit:  $(git -C "$IBCDFO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty:   $(git -C "$IBCDFO" status --porcelain 2>/dev/null | wc -l) modified files"
} > "$HERE/code/PROVENANCE.txt"
cat "$HERE/code/PROVENANCE.txt"

tar czf "$HERE/code.tar.gz" -C "$HERE" code
echo "-> code.tar.gz ($(( $(wc -c < "$HERE/code.tar.gz") / 1024 )) KB)"
