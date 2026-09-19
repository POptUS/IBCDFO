# Driver for the full hand-coded-vs-jax-hash benchmark sweep: runs every
# (dfo row, hfun, hand/jax) combo from bench_run_one_combo.py in its own subprocess
# (with a timeout), so a single slow/hanging combo can't take down the whole sweep.
# Covers all 53 More-Wild rows x 8 hfuns x {hand, jax} = 848 combos; already-completed
# combos (per _already_done below) are skipped, so this is safe to re-run/resume.
#
# Usage: python run_benchmark_hand_coded_vs_jax.py
# Parallel usage (each rank runs combos[i] where i % commsize == rank; every rank
# writes to a distinct {name}__row{row_idx}__{version}.npz, so ranks never collide):
#   mpirun -np 4 python run_benchmark_hand_coded_vs_jax.py
import os
import subprocess
import sys
import time

import numpy as np

from bench_run_one_combo import HFUN_PAIRS, OUT_DIR, PROBS_TO_SOLVE

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    commsize = 1

os.makedirs(OUT_DIR, exist_ok=True)

# Some jax-hash hfuns (e.g. create_piecewise_quadratic_hfun on the larger dfo rows) are
# slow due to branch_extended_AD's non-JIT, per-call tracing overhead rather than any bug --
# generous enough that a "did it finish" timeout doesn't masquerade as a hang, but still
# bounded so one combo can't stall the whole sweep indefinitely.
TIMEOUT_SEC = 1800


def _already_done(row_idx, name, version):
    path = f"{OUT_DIR}/{name}__row{row_idx}__{version}.npz"
    if not os.path.exists(path):
        return False
    try:
        data = np.load(path, allow_pickle=True)
        return not np.all(np.isnan(np.atleast_1d(np.squeeze(data["h"]))))
    except Exception:
        return False  # unreadable/corrupt -- treat as not done


failures = []
skipped = 0
t_start = time.time()

# create_piecewise_quadratic_hfun is much slower than the rest (branch_extended_AD's non-JIT,
# per-call tracing overhead scales with the ~90 pieces on the larger dfo rows) and has
# hit failures on some rows -- run it dead last, across all rows, so it can't stall or
# block progress on the other (cheap, reliable) hfuns.
SLOW_HFUN = "create_piecewise_quadratic_hfun"
hfun_order = [name for name in sorted(HFUN_PAIRS) if name != SLOW_HFUN] + [SLOW_HFUN]

combos = [(row_idx, name, version) for name in hfun_order for row_idx in range(len(PROBS_TO_SOLVE)) for version in ("hand", "jax")]
n_mine = len(combos[rank::commsize])
print(f"Sweeping {len(combos)} combos ({len(PROBS_TO_SOLVE)} rows x {len(HFUN_PAIRS)} hfuns x 2 versions), {SLOW_HFUN} last")
if commsize > 1:
    print(f"Rank {rank} of {commsize}: running {n_mine} of {len(combos)} combos")

for i, (row_idx, name, version) in enumerate(combos):
    if i % commsize != rank:
        continue
    if _already_done(row_idx, name, version):
        skipped += 1
        continue
    cmd = [
        sys.executable,
        "bench_run_one_combo.py",
        "--row-idx",
        str(row_idx),
        "--hfun",
        name,
        "--version",
        version,
    ]
    try:
        result = subprocess.run(cmd, timeout=TIMEOUT_SEC)
        if result.returncode != 0:
            print(f"FAILED (exit {result.returncode}): {name} row {row_idx} {version}")
            failures.append((row_idx, name, version, f"exit {result.returncode}"))
            np.savez(f"{OUT_DIR}/{name}__row{row_idx}__{version}.npz", h=np.full(1, np.nan), flag=None)
    except subprocess.TimeoutExpired:
        print(f"TIMED OUT after {TIMEOUT_SEC}s: {name} row {row_idx} {version}")
        failures.append((row_idx, name, version, "timeout"))
        np.savez(f"{OUT_DIR}/{name}__row{row_idx}__{version}.npz", h=np.full(1, np.nan), flag=None)

elapsed = time.time() - t_start
print(f"\nRank {rank}: ran {n_mine - skipped} combos ({skipped} already done, skipped) in {elapsed:.1f}s. Failures: {len(failures)}")
for row_idx, name, version, reason in failures:
    print(f"  {name} row {row_idx} {version}: {reason}")
