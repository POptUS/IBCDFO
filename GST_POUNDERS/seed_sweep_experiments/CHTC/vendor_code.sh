#!/bin/bash
# Copy the ONLY source files the jobs need into ./code/, so this folder becomes
# self-contained and can be scp'd to CHTC on its own (no repo, no 3 GB of results).
#
# Run this on the machine that has the repo:   bash vendor_code.sh
#
# The 7 files below are the complete set: determined by importing
# gst_seed_experiment and recording every module resolved inside the IBCDFO tree,
# then cross-checking every function-level import in those files. Everything else
# they touch is pygsti / pyrol / numpy / scipy (pip) or guarded optional debug
# helpers (ipdb, poptus).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
IBCDFO=$(cd "$HERE/../../.." && pwd)

FILES=(
    "GST_POUNDERS/adaptive_shot_hook.py"
    "GST_POUNDERS/adaptive_shots.py"
    "GST_POUNDERS/near_minimal_fpr_reduction.py"
    "GST_POUNDERS/seed_sweep_experiments/gst_seed_experiment.py"
    "pounders/py/general_h_funs.py"
    "pounders/py/gradient_pounders.py"
    "pounders/py/prepare_outputs_before_return_gradient.py"
)

if [ ! -f "$IBCDFO/GST_POUNDERS/seed_sweep_experiments/gst_seed_experiment.py" ]; then
    echo "ERROR: run this from inside the repo (cannot see $IBCDFO/...)" >&2
    exit 1
fi

rm -rf "$HERE/code"
for f in "${FILES[@]}"; do
    mkdir -p "$HERE/code/$(dirname "$f")"
    cp "$IBCDFO/$f" "$HERE/code/$f"
    printf '   %-62s %6s KB\n' "$f" "$(( ($(wc -c < "$IBCDFO/$f") + 1023) / 1024 ))"
done
cp "$IBCDFO/GST_POUNDERS/seed_sweep_experiments/experiment_config.json" "$HERE/experiment_config.json"

# record provenance -- which commit these came from
{
  echo "vendored_from: $IBCDFO"
  echo "git_commit:    $(git -C "$IBCDFO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_branch:    $(git -C "$IBCDFO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty:     $(test -n "$(git -C "$IBCDFO" status --porcelain -- ${FILES[*]} 2>/dev/null)" && echo yes || echo no)"
  echo "vendored_at:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$HERE/code/PROVENANCE.txt"

echo
echo "vendored $(find "$HERE/code" -name '*.py' | wc -l) files -> $HERE/code  ($(du -sh "$HERE/code" | cut -f1))"
cat "$HERE/code/PROVENANCE.txt" | sed 's/^/   /'
echo
echo "this folder is now self-contained: scp -r CHTC user@submit.chtc.wisc.edu:~/"
