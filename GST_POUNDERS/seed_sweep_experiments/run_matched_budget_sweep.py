"""Run paired GST comparisons using fixed FPR as the budget anchor.

For every data seed, fixed FPR runs first at its configured shots per circuit.
Its actual revealed-shot cost becomes the matched budget for adaptive FPR and
fixed no-FPR. Adaptive FPR receives this cost as its maximum acquisition
budget. Fixed no-FPR uses the largest uniform shots-per-circuit value that
does not exceed the same budget.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd

from gst_seed_experiment import ExperimentConfig, run_one_experiment


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            seeds.append(int(token))
            continue
        pieces = [int(value) for value in token.split(":")]
        if len(pieces) == 2:
            start, stop = pieces
            step = 1
        elif len(pieces) == 3:
            start, stop, step = pieces
        else:
            raise ValueError(f"Invalid seed specification {token!r}.")
        seeds.extend(range(start, stop, step))
    return sorted(set(seeds))


def read_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def completed(run_dir: Path) -> bool:
    return (run_dir / "completed.json").exists() and (run_dir / "summary.json").exists()


def saved_config_matches(run_dir: Path, config: ExperimentConfig) -> bool:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return False
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(asdict(config)))
    return saved == expected


def run_or_load(
    *, config: ExperimentConfig, seed: int, method: str, run_dir: Path, force: bool
) -> dict:
    if completed(run_dir) and not force and saved_config_matches(run_dir, config):
        print(f"SKIP seed={seed} method={method}: completed result exists")
        return read_summary(run_dir)
    if completed(run_dir) and not force:
        print(f"RERUN seed={seed} method={method}: saved configuration changed")
    print(f"RUN seed={seed} method={method}")
    return run_one_experiment(
        config=config,
        data_seed=seed,
        method=method,
        output_dir=run_dir,
    )


def budget_row(label: str, target: int, summary: dict, shots_per_circuit: int | None):
    actual = int(summary["accounted_revealed_shots"])
    return {
        "method": label,
        "target_revealed_shots": int(target),
        "actual_revealed_shots": actual,
        "absolute_budget_difference": int(actual - target),
        "relative_budget_difference": float((actual - target) / target),
        "uniform_shots_per_circuit": shots_per_circuit,
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiment_config.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("matched_results"))
    parser.add_argument("--seeds", default="100:110")
    parser.add_argument(
        "--methods",
        default="fixed_fpr,adaptive_fpr,fixed_no_fpr",
        help="Comma list from fixed_fpr,adaptive_fpr,fixed_no_fpr. NOTE: fixed_fpr is the "
        "budget anchor; it is auto-run/loaded whenever adaptive_fpr or fixed_no_fpr is chosen.",
    )
    parser.add_argument("--force", action="store_true")
    # Retained as ignored compatibility options for older commands.
    parser.add_argument("--max-calibration-runs", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--budget-tolerance", type=float, default=0.0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    fixed_fpr_shots = int(config.fixed_fpr_shots)
    if fixed_fpr_shots <= 0:
        raise ValueError("fixed_fpr_shots must be positive for a matched-budget sweep.")

    valid_methods = ["fixed_fpr", "adaptive_fpr", "fixed_no_fpr"]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in valid_methods]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choose from {valid_methods}.")
    if not methods:
        raise ValueError("At least one method is required.")
    need_budget = ("adaptive_fpr" in methods) or ("fixed_no_fpr" in methods)
    print("Methods:", methods)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for seed in parse_seeds(args.seeds):
        seed_dir = args.results_dir / f"seed_{seed:06d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        target = None
        total_circuits = None
        fixed_fpr_union = None
        no_fpr_shots = None

        # fixed_fpr is the budget anchor: run/load it whenever it is selected OR when
        # adaptive_fpr / fixed_no_fpr need its revealed-shot budget.
        if ("fixed_fpr" in methods) or need_budget:
            fixed_fpr = run_or_load(
                config=config, seed=seed, method="fixed_fpr",
                run_dir=seed_dir / "fixed_fpr", force=args.force,
            )
            target = int(fixed_fpr["accounted_revealed_shots"])
            if target <= 0:
                raise ValueError(
                    f"Fixed FPR produced a nonpositive revealed-shot budget for seed {seed}."
                )
            total_circuits = int(fixed_fpr["total_circuits"])
            fixed_fpr_union = max(1, int(fixed_fpr["revealed_circuits"]))
            if "fixed_fpr" in methods:
                all_rows.append(budget_row("fixed_fpr", target, fixed_fpr, fixed_fpr_shots))
            else:
                print(
                    f"NOTE seed={seed}: fixed_fpr not selected but run/loaded as the "
                    f"budget anchor (target={target:,})."
                )

        if "adaptive_fpr" in methods:
            adaptive_config = replace(config, adaptive_total_shot_budget=target)
            adaptive = run_or_load(
                config=adaptive_config, seed=seed, method="adaptive_fpr",
                run_dir=seed_dir / "adaptive_fpr", force=args.force,
            )
            all_rows.append(budget_row("adaptive_fpr", target, adaptive, None))

        if "fixed_no_fpr" in methods:
            no_fpr_shots = max(1, target // total_circuits)
            no_fpr_config = replace(config, fixed_no_fpr_shots=no_fpr_shots)
            no_fpr = run_or_load(
                config=no_fpr_config, seed=seed, method="fixed_no_fpr",
                run_dir=seed_dir / "fixed_no_fpr", force=args.force,
            )
            all_rows.append(budget_row("fixed_no_fpr", target, no_fpr, no_fpr_shots))

        if target is not None:
            pd.DataFrame(
                [
                    {
                        "data_seed": seed,
                        "target_revealed_shots": target,
                        "budget_anchor_method": "fixed_fpr",
                        "fixed_fpr_shots_per_circuit": fixed_fpr_shots,
                        "fixed_fpr_revealed_circuits": fixed_fpr_union,
                        "fixed_fpr_actual_revealed_shots": target,
                        "adaptive_total_shot_budget": target,
                        "fixed_no_fpr_shots_per_circuit": no_fpr_shots,
                    }
                ]
            ).to_csv(seed_dir / "budget_anchor.csv", index=False)

        seed_rows = [row for row in all_rows if int(row["data_seed"]) == seed]
        if seed_rows:
            pd.DataFrame(seed_rows).to_csv(seed_dir / "matched_budget_summary.csv", index=False)
        print(f"DONE seed={seed}: methods={methods}, target={target}")

    frame = pd.DataFrame(all_rows)
    frame.to_csv(args.results_dir / "matched_budget_summary.csv", index=False)
    print(f"Saved {args.results_dir / 'matched_budget_summary.csv'}")


if __name__ == "__main__":
    main()
