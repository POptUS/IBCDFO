# CHTC pilot-shot sweep — no container

Sweeps `adaptive_baseline_shots` (the *pilot shots per circuit*) across a range of
values, many seeds per value, on HTCondor. **No Docker and no Apptainer**: PyROL
ships a prebuilt `manylinux_2_28` wheel, so a plain portable venv is enough and the
jobs run on any EL8+ execute node.

## Why pilot shots

The pilot shots decide how much of the budget the adaptive criterion ever gets to
steer. Every circuit the FPR union reveals is charged its full pilot cost up front:

| pilot | baseline cost | % of a 465,600-shot budget | steerable | E[zero-count circuits] |
|------:|--------------:|---------------------------:|----------:|-----------------------:|
| 100 | 59,900 | 13% | **87%** | 37.6 |
| 200 | 119,800 | 26% | 74% | 11.4 |
| 350 | 209,650 | 45% | 55% | 4.4 |
| 500 | 299,500 | 64% | **36%** | 2.6 |
| 800 | 479,200 | 103% | **0%** | 1.1 |

At the current 500, D-optimality steers only 36% of the shots — the rest is spent
uniformly before it has a say. At 800 adaptive becomes *identical* to uniform. Low
pilot buys steering power, but also more zero-count circuits, each of which lands at
weight `1/variance_floor = 1e12` in the WLS objective. Mapping that tradeoff is the
point of the sweep.

## Files

| file | what it does |
|---|---|
| `build_env.sh` | run **once** on a submit node → `rolenv.tar.gz`, `code.tar.gz` |
| `make_joblist.py` | writes `joblist.txt`, one line per (pilot, seed) |
| `sweep.sub` | HTCondor submit description |
| `job.sh` | runs on the worker: untar, set env, call `run_one.py`, tar results |
| `run_one.py` | one (pilot, seed) cell — **all methods**, matched budget |
| `collect.py` | unpack the returned tarballs → `sweep_summary.csv` |

## Run it

```bash
bash build_env.sh                     # once; ~2 min, produces the tarballs
python make_joblist.py                # edit PILOTS / SEEDS inside first
mkdir -p logs
condor_submit sweep.sub
# ... later ...
./rolenv/bin/python collect.py --indir . --out sweep_summary.csv
```

## Design decisions worth knowing

**One job = one (pilot, seed), running every method.** The budget anchor comes from
`fixed_fpr`'s accounted shot count, so the methods for a given seed must run
together. Splitting them across jobs would silently break the matched-budget
comparison.

**Pin the budget.** `run_one.py --budget N` fixes the total shot budget instead of
deriving it from `fixed_fpr`. For a pilot sweep this matters: without it, changing
the pilot also changes the anchor, and you cannot attribute the result. Set it via
`SWEEP_ARGS` in `sweep.sub`:

```
environment = "SWEEP_ARGS=--budget 465600"
```

**Threads are pinned to 1** (`job.sh` and `run_one.py` both set it). CHTC nodes are
heterogeneous, and in this problem a single marginal accept/reject decides which
basin a seed lands in — multithreaded BLAS would make runs unreproducible across
machines and you would not be able to tell a real effect from numerical jitter.

**PyROL version is pinned to 0.5.3** in `build_env.sh`, matching the Docker image.
PyPI is at 0.5.6; upgrading silently changes the trust-region subproblem solver.

## Requirements

- `python3.11` on the submit node (the wheel exists for cp38–cp312)
- glibc ≥ 2.28 on execute nodes — `sweep.sub` already asks for `OpSysMajorVer >= 8`
- ~6 GB scratch and ~4 GB RAM per job (raise if cells OOM)

## Before submitting 640 jobs

Test one interactively:

```bash
bash build_env.sh
mkdir -p testrun && cd testrun
tar xzf ../rolenv.tar.gz && tar xzf ../code.tar.gz
cp ../run_one.py ../experiment_config.json .
export PYTHONPATH=$PWD/IBCDFO:$PWD/IBCDFO/pounders/py:$PWD/IBCDFO/GST_POUNDERS/seed_sweep_experiments
./rolenv/bin/python run_one.py --pilot 350 --seed 20001 --outdir results --nfmax 40
```

`--nfmax 40` keeps the smoke test short. If that produces
`results/seed_020001/result.csv` with four rows, the full sweep will work.

## Gotchas

- **Line endings.** These files are authored on Windows. If bash complains about
  `\r`, run `dos2unix *.sh *.py` on the submit node.
- **venvs are path-sensitive.** `job.sh` invokes `./rolenv/bin/python` directly,
  which avoids most of it. If you hit path errors, rebuild with `conda-pack` or
  `python -m venv --copies`.
- **`--budget` changes the comparison.** With it, every cell has an identical
  budget. Without it, `fixed_fpr` sets the anchor per cell, which is what the
  notebook does today but confounds a pilot sweep.
- **The variance floor is still 1e-12.** At low pilot you expect ~38 zero-count
  circuits per run at weight 1e12, so the most interesting corner of the sweep is
  also the one most affected by that. Consider sweeping `variance_smoothing` or
  fixing the floor as a second axis before trusting the low-pilot cells.
