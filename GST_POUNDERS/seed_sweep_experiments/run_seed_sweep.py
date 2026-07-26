#!/usr/bin/env python3
"""Resumable paired-seed runner for GST/POUNDERS experiments."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from gst_seed_experiment import ExperimentConfig, METHODS, run_one_experiment


HERE = Path(__file__).resolve().parent


def parse_seeds(spec: str) -> list[int]:
    """Parse ``100,102,110`` or a Python-style ``100:110[:2]`` range."""
    spec = spec.strip()
    if ":" not in spec:
        values = [int(value.strip()) for value in spec.split(",") if value.strip()]
    else:
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError("Seed range must be START:STOP or START:STOP:STEP.")
        start, stop = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        values = list(range(start, stop, step))
    if not values:
        raise ValueError("At least one data seed is required.")
    return values


def parse_methods(spec: str) -> list[str]:
    values = [value.strip() for value in spec.split(",") if value.strip()]
    unknown = sorted(set(values) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choose from {METHODS}.")
    return values


def collect_summaries(results_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_root.glob("seed_*/**/summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_columns = [column for column in ("data_seed", "method") if column in frame]
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "experiment_config.json",
        help="JSON configuration file.",
    )
    parser.add_argument(
        "--seeds",
        default="100:110",
        help="Comma list or Python-style range, e.g. 100,103 or 100:110.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help=f"Comma list chosen from {METHODS}.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=HERE / "results",
        help="Root directory for resumable run bundles.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed bundles.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed method instead of recording and continuing.",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    seeds = parse_seeds(args.seeds)
    methods = parse_methods(args.methods)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Data seeds:", seeds)
    print("Methods:", methods)
    print("Results:", args.results_dir.resolve())
    print(
        "Truth model:",
        "varies with data seed" if config.vary_truth_with_data_seed else f"fixed seed {config.truth_seed}",
    )

    failures = 0
    for data_seed in seeds:
        for method in methods:
            run_dir = args.results_dir / f"seed_{data_seed:06d}" / method
            complete_marker = run_dir / "completed.json"
            if complete_marker.exists() and not args.force:
                print(f"SKIP seed={data_seed} method={method}: completed bundle exists")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            for stale in ("failed.json", "completed.json"):
                path = run_dir / stale
                if path.exists():
                    path.unlink()

            print("=" * 78)
            print(f"RUN seed={data_seed} method={method}")
            try:
                summary = run_one_experiment(
                    config=config,
                    data_seed=data_seed,
                    method=method,
                    output_dir=run_dir,
                )
                print(
                    f"DONE seed={data_seed} method={method}: "
                    f"infidelity={summary['mean_gate_entanglement_infidelity_to_truth']:.3e}, "
                    f"revealed_shots={summary['accounted_revealed_shots']:,}"
                )
            except Exception as exc:
                failures += 1
                failure = {
                    "data_seed": data_seed,
                    "method": method,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                (run_dir / "failed.json").write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                print(f"FAILED seed={data_seed} method={method}: {exc}")
                if args.stop_on_error:
                    raise
            finally:
                summary_frame = collect_summaries(args.results_dir)
                summary_frame.to_csv(args.results_dir / "all_runs_summary.csv", index=False)

    print("=" * 78)
    print(f"Sweep finished with {failures} failed run(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
