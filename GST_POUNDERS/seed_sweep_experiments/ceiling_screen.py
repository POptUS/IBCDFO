"""Screen a scenario for its D-optimal shot-allocation ceiling. No GST fits.

Usage: python ceiling_screen.py <name> [key=value ...]
Config overrides are applied to the sweep's own config, so a scenario differs from the
baseline only in the keys named. Prints one result line; also emits JSON on the last line
so a caller can collect it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

SSE = Path(__file__).resolve().parent
sys.path.insert(0, str(SSE))
sys.path.insert(0, str(SSE.parent))

from gst_seed_experiment import (ExperimentConfig, GSTProblem, _make_parameterized_model,
                                 _dense)
import adaptive_shots as ashots

BASE_CFG = SSE / "all_methods_comparison_6000" / "seed_000101" / "lm" / "config.json"
SEED = 101
BASELINE_FRAC = 0.05
FW_ITERS = int(__import__("os").environ.get("FW_ITERS", 2000))
RELAXED = __import__("os").environ.get("RELAXED", "0") == "1"
PER_CIRCUIT = 2521          # ceiling is budget-independent; this only sets a sane scale


def build(overrides):
    d = json.loads(BASE_CFG.read_text())
    d.update(overrides)
    cfg = ExperimentConfig(**{k: v for k, v in d.items()
                              if k in ExperimentConfig.__dataclass_fields__})
    return cfg, GSTProblem(cfg, SEED)


def pj(model, circuits):
    circuits = list(circuits)
    pr = model.bulk_probabilities(circuits)
    dp = model.sim.bulk_dprobs(circuits)
    pv, rows = [], []
    for c in circuits:
        for o in model.probabilities(c).keys():
            pv.append(float(pr[c].get(o, 0.0)))
            rows.append(np.asarray(dp[c][o], dtype=float))
    return np.asarray(pv, float), np.vstack(rows)


def ceiling(name, overrides):
    t0 = time.time()
    cfg, prob = build(overrides)
    n_circ = len(prob.circuits)
    ri = np.repeat(np.arange(n_circ), prob.outcomes_per_circuit)

    truth = _make_parameterized_model(prob.truth_model, cfg.parameterization,
                                      ideal_model=prob.raw_target_model)
    conv = max(float(np.max(np.abs(_dense(truth.operations[l])
                                   - _dense(prob.truth_model.operations[l]))))
               for l in prob.truth_model.operations.keys())
    assert conv < 1e-10, f"truth not representable: {conv:.2e}"
    p, J = pj(truth, prob.circuits)

    N = int(PER_CIRCUIT * n_circ)
    RIDGE, FLOOR = 1e-9, 1e-9

    def H_of(rho):
        w = np.asarray(rho, float)[ri] / np.maximum(p, FLOOR)
        return (J * w[:, None]).T @ J + RIDGE * np.eye(J.shape[1])

    base = np.full(n_circ, BASELINE_FRAC * N / n_circ)
    N_alloc = int(round((1 - BASELINE_FRAC) * N))

    if RELAXED:
        # Frank-Wolfe only, no integer rounding. Identical iteration to
        # allocate_shots_per_circuit's, but the greedy completion is skipped: it recomputes
        # every circuit's score per assigned shot (O(rows*d^2) each, up to n_circuits steps),
        # which at 2Q sizes costs hours and buys nothing here -- the RELAXED optimum is the
        # right quantity for a ceiling and upper-bounds the integer one.
        H0 = H_of(base)
        inv_p = 1.0 / np.maximum(p, FLOOR)
        rho = np.full(n_circ, N_alloc / n_circ)

        def design(w):
            ww = np.asarray(w, float)[ri] * inv_p
            return (J * ww[:, None]).T @ J

        gap = np.inf
        for it in range(FW_ITERS):
            Hinv = ashots.stable_inverse(H0 + design(rho))
            Y = Hinv @ J.T
            rq = np.einsum("ij,ji->i", J, Y) * inv_p
            sc = np.zeros(n_circ)
            np.add.at(sc, ri, rq)
            i_star = int(np.argmax(sc))
            gap = float(N_alloc * sc[i_star] - sc @ rho)
            if gap <= 1e-8:
                break
            gamma = 2.0 / (it + 2.0)
            rho *= (1.0 - gamma)
            rho[i_star] += gamma * N_alloc
        info = {"gap": gap}
        alloc = rho
    else:
        extra, info = ashots.allocate_shots_per_circuit(
            J, p, N_alloc, ri, n_circuit=base, max_iter=FW_ITERS)
        alloc = extra

    Hu, Ho = H_of(np.full(n_circ, N / n_circ)), H_of(base + alloc)
    ev, V = np.linalg.eigh(0.5 * (Hu + Hu.T))
    V = V[:, ev > ashots.RANK_CUTOFF * ev[-1]]
    r = V.shape[1]
    evu, evo = np.linalg.eigvalsh(V.T @ Hu @ V), np.linalg.eigvalsh(V.T @ Ho @ V)
    d_eff = float(np.exp((np.log(evo).sum() - np.log(evu).sum()) / r))

    # Kiefer-Wolfowitz headroom detector on the uniform design: max_s g_s / r
    Hinv = V @ np.linalg.inv(V.T @ Hu @ V) @ V.T
    rowq = np.einsum("ij,ji->i", J, Hinv @ J.T) / np.maximum(p, FLOOR)
    g = np.zeros(n_circ)
    np.add.at(g, ri, rowq)
    # Hu was built from shot counts summing to N, so Hu = N * H(w) for weights w summing
    # to 1; hence g_s = tr(H(w)^-1 A_s) = N * tr(Hu^-1 A_s). Kiefer-Wolfowitz then gives a
    # free check: the w-weighted mean of g_s equals the rank exactly, for ANY design.
    g *= N
    mean_g = float(g.mean())             # w is uniform here, so the weighted mean is the mean
    assert abs(mean_g - r) < 1e-6 * r, f"KW identity broken: mean g_s={mean_g:.4f} vs r={r}"
    head = float(g.max() / r)

    out = dict(name=name, relaxed=RELAXED, d_eff=round(d_eff, 3), headroom=round(head, 2), rank=r,
               params=prob.n, circuits=n_circ, gap=round(float(info["gap"]), 3),
               secs=round(time.time() - t0, 1), overrides=overrides)
    print(f"{name:<22} ceiling {d_eff:>6.3f}x   headroom {head:>7.2f}   "
          f"rank {r:>4}/{prob.n:<5} circuits {n_circ:>6}   [{out['secs']}s]")
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    ov = {}
    for a in sys.argv[2:]:
        k, v = a.split("=", 1)
        ov[k] = json.loads(v)
    print(json.dumps(ceiling(name, ov)))
