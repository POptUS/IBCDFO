# CHTC sweep — no container

One HTCondor job = one data seed = **every arm**. No Docker, no Apptainer: PyROL ships a
prebuilt `manylinux_2_28` wheel, so a plain portable venv runs on any EL8+ execute node.

Results come back small enough to commit to git, which is how you get them onto your laptop.

## Files

| file | what it does |
|---|---|
| `build_env.sh` | run **once** on the submit node → `rolenv.tar.gz` (~300 MB) |
| `vendor.sh` | copy the 8 source files + config out of the repo → `code.tar.gz` (~56 KB) |
| `make_jobs.py` | write `joblist.txt`, one line per job. **Edit `SEEDS` / `ARGS` first.** |
| `sweep.sub` | HTCondor submit description |
| `job.sh` | runs on the worker: untar, pin threads, call `run_one.py`, tar results |
| `run_one.py` | one seed, every arm, matched budget |
| `collect.py` | unpack the returned tarballs → `sweep_summary.csv`, and stage for git |

## First time on the submit node

```bash
git clone <your fork> && cd .../seed_sweep_experiments/chtc
bash build_env.sh          # ~5 min, one time per cluster
```

## Every run

```bash
bash vendor.sh                       # ALWAYS: picks up your latest code edits
python make_jobs.py                  # edit SEEDS / ARGS inside first
mkdir -p logs
condor_submit sweep.sub
condor_q                             # watch
```

### Adding one arm to an existing sweep

`--from-results` reuses another run's per-seed budget anchor and settings, so you only pay for
the arm that changed. **Generate the joblist on the machine that HAS the results** --
`all_methods_comparison*/` is gitignored and never reaches the cluster:

```bash
# LOCAL, where the reference results live
python make_jobs.py --from-results ../all_methods_comparison_1000     --arms adaptive_D_nofpr --fracs 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8     --extra "--nfmax 500"
git add -f joblist.txt          # joblist.txt is gitignored by default
```

then on the submit node `git pull` and `condor_submit sweep.sub`.

`--extra` is appended after the copied settings, and argparse keeps the last occurrence of a
flag, so `--extra "--nfmax 500"` overrides a copied `--nfmax 300`. Anything you override this
way no longer matches the reference POUNDERS arms. Overriding `nfmax` is safe against
`lm`/`lm_mle`/`lm_fpr` (they never read it) and confounding against `fixed_fpr` /
`fixed_no_fpr` / `adaptive_D`.

When the jobs land:

```bash
./rolenv/bin/python collect.py --stage ../results_chtc_run1
```

`vendor.sh` is the step people forget. It is what copies your edited
`gst_seed_experiment.py` into the tarball — skip it and the cluster silently runs the previous
version. It stamps `code/PROVENANCE.txt` with the commit and the dirty-file count so a result
can always be traced back.

## Getting results onto your laptop, via git

`job.sh` already drops the four files that are ~90% of the bytes and that nothing in the
analysis notebooks reads:

```
fpr_selection_history.csv   iteration_gate_errors.csv
iteration_spam_errors.csv   x_evaluations.npy
```

What is left is about **0.3 MB per arm-run**, so a 20-seed × 7-arm sweep is ~40 MB — fine for
git. On the submit node:

```bash
./rolenv/bin/python collect.py --stage ../results_chtc_run1
cd ..
git add results_chtc_run1 chtc/sweep_summary.csv
git commit -m "CHTC run 1: 20 seeds, poisson_logl, allocate_every 5"
git push
```

and on your laptop `git pull`, then point the notebooks at it:

```python
RESULTS_DIR = EXPERIMENT_DIR / "results_chtc_run1"
```

`analyze_all_methods.ipynb` and `estimator_comparison.ipynb` both discover arms from the
folder, so `adaptive_D_nofpr` and anything else you add appear with no edits.

`collect.py` warns if the staged tree exceeds 80 MB. If you hit that, commit only
`sweep_summary.csv` plus each arm's `summary.json` — enough for every headline figure.

## Arms

| arm | what it is |
|---|---|
| `fixed_fpr` | POUNDERS + FPR, uniform shots. **Budget anchor** for everything else. |
| `fixed_no_fpr` | POUNDERS, uniform over all 1918 circuits |
| `adaptive_D` | POUNDERS + FPR + D-optimal adaptive allocation |
| `adaptive_D_nofpr` | the same allocation with FPR **off** |
| `lm` | pyGSTi LM, standard recipe: chi² ladder + one MLE at the end |
| `lm_mle` | pyGSTi LM, pure MLE at every stage, full design |
| `lm_fpr` | pure MLE on pyGSTi's built-in FPR design |

`adaptive_D` vs `adaptive_D_nofpr` is the point of this sweep: paired runs show adaptive
allocation helping (0.565×) while FPR hurts (1.712×), which cancels to the tie against LM.
Unbundling them is what the `_nofpr` arm measures.

## Things that will bite you

**All arms must run in one job.** The budget anchor is `fixed_fpr`'s accounted shot count, and
every arm draws from the same data seed. Splitting arms across jobs would silently break the
matched-budget comparison.

**Threads are pinned to 1**, in `job.sh` and again in `run_one.py` before numpy is imported.
Cluster nodes are heterogeneous and a single marginal accept/reject decides which basin a seed
lands in; multithreaded BLAS would make runs irreproducible across machines.

**`cvxopt` is not optional.** Diamond distance is an SDP and cvxpy's default CLARABEL cannot
solve it. Without cvxopt every diamond value comes back `NaN` — silently. `build_env.sh`
asserts a real diamond solve before it finishes.

**Do not mix sweeps with different settings in one folder.** `nfmax` and
`adaptive_allocate_every` change the trajectory, so runs at different values are not
comparable. Each arm's `config.json` records what it used; give each sweep its own
`results_chtc_runN` directory.

**Pin the budget for a pilot sweep.** `--pilot` changes `adaptive_baseline_shots`, which also
moves `fixed_fpr`'s anchor — so without `--budget` you cannot attribute the result. Add
`--budget 421000` (or whatever) to `ARGS` in `make_jobs.py` whenever `--pilot` varies.

## Sweeping an axis

`ARGS` in `make_jobs.py` is passed straight to `run_one.py`, so no new code is needed:

```python
ARGS = [f"--objective poisson_logl --allocate-every 5 --nfmax 150 --budget 421000 --pilot {p}"
        for p in (100, 200, 350, 500)]
```

Every (seed, args) pair becomes a job. Commas are rejected — HTCondor splits the queue line on
them.
