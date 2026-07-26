# Result bundles

The runner creates:

```text
results/
  all_runs_summary.csv
  seed_000100/
    adaptive_fpr/
    fixed_fpr/
    fixed_no_fpr/
```

Each method directory contains its configuration, optimizer history, FPR and
shot events, parameter vectors, gate/SPAM diagnostics, and completion/failure
marker. It is safe to stop and resume the sweep; completed bundles are skipped.
