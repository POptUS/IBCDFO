"""Loading, aggregation, and color-blind-safe plots for seed-sweep results."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "adaptive_fpr": "#0072B2",
    "fixed_fpr": "#E69F00",
    "fixed_no_fpr": "#009E73",
}
MARKERS = {"adaptive_fpr": "o", "fixed_fpr": "s", "fixed_no_fpr": "^"}


def load_results(results_root: str | Path) -> pd.DataFrame:
    results_root = Path(results_root)
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results_root.glob("seed_*/**/summary.json"))
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["data_seed", "method"]).reset_index(drop=True)
    return frame


def method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_gate_entanglement_infidelity_to_truth",
        "mean_spam_vector_l2_error_to_truth",
        "accounted_revealed_shots",
        "reduced_chi_square",
        "n_sigma",
    ]
    rows = []
    for method, group in frame.groupby("method", sort=False):
        row = {"method": method, "num_seeds": int(group["data_seed"].nunique())}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_q25"] = float(values.quantile(0.25))
            row[f"{metric}_q75"] = float(values.quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_seed_distributions(frame: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    specs = [
        ("mean_gate_entanglement_infidelity_to_truth", "Gate infidelity to truth", True),
        ("mean_spam_vector_l2_error_to_truth", "SPAM vector error to truth", True),
        ("accounted_revealed_shots", "Revealed-shot cost", True),
    ]
    methods = [method for method in COLORS if method in set(frame["method"])]
    for axis, (metric, title, log_scale) in zip(axes, specs):
        groups = [frame.loc[frame["method"] == method, metric].to_numpy(float) for method in methods]
        boxes = axis.boxplot(groups, labels=methods, patch_artist=True, showfliers=True)
        for patch, method in zip(boxes["boxes"], methods):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.55)
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(alpha=0.25, which="both")
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Paired data-seed distributions")
    fig.tight_layout()
    return fig


def plot_shot_efficiency(frame: pd.DataFrame):
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    for method, group in frame.groupby("method", sort=False):
        axis.scatter(
            group["accounted_revealed_shots"],
            group["mean_gate_entanglement_infidelity_to_truth"],
            color=COLORS.get(method, "#444444"),
            marker=MARKERS.get(method, "o"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.75,
            label=method,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("accounted revealed shots")
    axis.set_ylabel("mean gate entanglement infidelity to truth")
    axis.set_title("Shot efficiency across data seeds")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    return fig


def load_long_metric(results_root: str | Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(results_root).glob(f"seed_*/**/{filename}")):
        frame = pd.read_csv(path)
        metadata = json.loads((path.parent / "problem_metadata.json").read_text(encoding="utf-8"))
        frame.insert(0, "data_seed", metadata["data_seed"])
        frame.insert(1, "method", metadata["method"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
