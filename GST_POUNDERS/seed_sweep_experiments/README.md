# GST/POUNDERS Paired-Seed Experiments

This folder turns the working paths in `GST_model.ipynb` into a reproducible,
resumable experiment. It compares three methods on the **same data seeds**:

- `adaptive_fpr`: online FPR with D/A-optimal adaptive shot allocation.
- `fixed_fpr`: online FPR with a fixed uniform shot count.
- `fixed_no_fpr`: full GST circuits with a fixed uniform shot count.

The default configuration matches the current 1Q `smq1Q_XYZI`, CPTPLND,
weighted-least-squares setup. The original notebook is not modified.

## Files

- `gst_seed_experiment.py`: reusable model, oracle, optimizer, and diagnostics.
- `run_seed_sweep.py`: resumable command-line runner.
- `run_seed_sweep_from_jupyter.ipynb`: JupyterLab front end with live output.
- `experiment_config.json`: all experiment knobs in one place.
- `analysis_utils.py`: aggregation and color-blind-safe plotting helpers.
- `analyze_seed_sweep.ipynb`: median/IQR, shot-efficiency, gate, and SPAM analysis.
- `results/`: one self-contained bundle per `(data seed, method)`.

## Run in the PyROL Docker container

From `/workspace/IBCDFO/GST_POUNDERS/seed_sweep_experiments`:

```bash
python run_seed_sweep.py --seeds 100:110
```

`100:110` means seeds 100 through 109. A comma list also works:

```bash
python run_seed_sweep.py --seeds 100,103,107 --methods adaptive_fpr,fixed_fpr
```

Completed bundles are skipped automatically. Use `--force` to rerun them and
`--stop-on-error` to stop at the first failure. A failed run writes
`failed.json`; a successful run writes `completed.json`.

## Reproducibility choice

By default, `truth_seed` is fixed and only multinomial sampling changes with
`data_seed`. This isolates robustness to shot noise. Set
`vary_truth_with_data_seed` to `true` only for a second study in which both the
synthetic device and sampled data vary.

## Saved metrics

Each run saves:

- optimizer/FPR/adaptive-shot histories;
- every evaluated parameter vector and the final fitted vector;
- final per-circuit shot ledger;
- final per-gate entanglement and average infidelity to truth and ideal;
- final preparation and POVM-effect vector errors to truth and ideal;
- per-iteration gate/SPAM errors to truth;
- weighted objective, reduced chi-square, likelihood badness-of-fit, and
  `N_sigma`;
- both `accounted_revealed_shots` and `physical_precomputed_shots`.

The last distinction matters. The current simulation precomputes a dataset for
all circuits so the full objective can be diagnosed, while FPR decisions use
only their active/union rows. `accounted_revealed_shots` is the experimental
cost of the circuits the method says must be revealed. `physical_precomputed_shots`
records what the Python simulation actually generated. Do not interchange them
in a paper or plot.

Ground-truth models are used only in post-run diagnostics. They do not enter FPR,
the shot allocator, the trust-region ratio, or the optimization objective.
