"""Ready-to-paste notebook cell: wire adaptive shots into gradient POUNDERS.

Paste the body below as a NEW cell in GST_model.ipynb, placed AFTER the cell that
defines `gst_pounders_full_function`, `base_model_for_pounders`, `pounders_circuits`,
`OUTCOMES_PER_CIRCUIT`, `simulate_gst_dataset_for_shots`, `gst_model_function`,
and `DATA_SEED` (cell ~23), and BEFORE the cell that calls `pounders.pouders(...)`.

It is written against the REAL notebook API (verified):
  * gst_model_function(x, model_template, circuits, return_labels=True)
        -> (p, dp/dparams, var_p, labels)   [UNWEIGHTED jacobian, rows x params]
  * residual rows are circuit-major with OUTCOMES_PER_CIRCUIT outcomes each
  * gst_pounders_full_function reads the GLOBAL `pounders_dataset` at call time,
    so reassigning that global updates the objective's data
  * x -> model via base_model_for_pounders.from_vector(x)

Requires adaptive_shots.py and adaptive_shot_hook.py in this folder (on sys.path).
This file is not imported; it documents the exact cell to paste.
"""

# ==========================================================================
# ---- BEGIN NOTEBOOK CELL (copy from here) --------------------------------
import numpy as np
import importlib
import adaptive_shots
import adaptive_shot_hook
adaptive_shots = importlib.reload(adaptive_shots)
adaptive_shot_hook = importlib.reload(adaptive_shot_hook)

# ---- Master switch: turn adaptive shots on/off ---------------------------
USE_ADAPTIVE_SHOTS = True            # False -> pass iter_callback=None (plain POUNDERS)
ADAPTIVE_CRITERION = "D"             # "D" (log det) or "A" (Tr H^-1)
ADAPTIVE_BASELINE_SHOTS = 20         # small uniform baseline per circuit to start from
ADAPTIVE_RHO_BAND = 0.05
ADAPTIVE_SCHEDULE = adaptive_shot_hook.rho_gated_geometric_budget(
    base=1.5,
    n0=32,
    rho_band=ADAPTIVE_RHO_BAND,
    initial_budget=0,
)
# Matt's adaptive alternative:
# ADAPTIVE_SCHEDULE = adaptive_shot_hook.adaptive_budget(scale=2000.0, n_min=16)

_n_circuits = len(pounders_circuits)
assert _n_circuits * OUTCOMES_PER_CIRCUIT == m, (
    f"expected m={_n_circuits * OUTCOMES_PER_CIRCUIT}, but pouders was given m={m}"
)
_CIRCUIT_TO_IDX = {c: i for i, c in enumerate(pounders_circuits)}

# ---- Running per-circuit shot ledger + a fresh LOW-baseline dataset -------
# Start from a small uniform baseline so the adaptive scheme can actually save
# shots relative to the full N_SAMPLES run.  Reassigns the global the objective reads.
if USE_ADAPTIVE_SHOTS:
    running_shots = np.full(_n_circuits, int(ADAPTIVE_BASELINE_SHOTS), dtype=int)
    pounders_dataset, _ = simulate_gst_dataset_for_shots(
        shots_per_circuit=running_shots, circuits=pounders_circuits,
        seed=DATA_SEED, label="Adaptive baseline dataset",
    )

# ---- The four injected functions (see adaptive_shot_hook docstring) -------
def _probability_and_jacobian(x, circuits):
    p, J, _var, _labels = gst_model_function(
        x, model_template=base_model_for_pounders, circuits=circuits, return_labels=True,
    )
    return p, J                       # UNWEIGHTED (rows x params) -- what allocate_shots_per_circuit wants

def _circuits_from_mask(mask):
    if mask is None:
        active_idx = list(range(_n_circuits))
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        circ_of_row = np.repeat(np.arange(_n_circuits), OUTCOMES_PER_CIRCUIT)
        active_idx = sorted(set(circ_of_row[mask].tolist()))
    circuits = [pounders_circuits[i] for i in active_idx]
    row_index = np.repeat(np.arange(len(circuits)), OUTCOMES_PER_CIRCUIT)
    return circuits, row_index

def _current_shots(circuits):
    per_circuit = np.array([running_shots[_CIRCUIT_TO_IDX[c]] for c in circuits], dtype=float)
    return np.repeat(per_circuit, OUTCOMES_PER_CIRCUIT)  # per-row (matches J row order)

def _add_shots(circuits, extra_per_circuit):
    global pounders_dataset
    for c, e in zip(circuits, extra_per_circuit):
        running_shots[_CIRCUIT_TO_IDX[c]] += int(e)
    # Regenerate at the new per-circuit totals (distribution-equivalent to
    # accumulating; swap for a count-merge if you want exact accumulation).
    pounders_dataset, _ = simulate_gst_dataset_for_shots(
        shots_per_circuit=running_shots, circuits=pounders_circuits,
        seed=DATA_SEED, label="Adaptive dataset update",
    )

# ---- Build the hook (or None) --------------------------------------------
adaptive_hook = (
    adaptive_shot_hook.make_adaptive_shot_hook(
        probability_and_jacobian=_probability_and_jacobian,
        circuits_from_mask=_circuits_from_mask,
        current_shots=_current_shots,
        add_shots=_add_shots,
        schedule=ADAPTIVE_SCHEDULE,
        criterion=ADAPTIVE_CRITERION,
        logger=print,
    )
    if USE_ADAPTIVE_SHOTS else None
)
print("Adaptive shots:", "ON" if adaptive_hook is not None else "OFF")
# ---- END NOTEBOOK CELL ---------------------------------------------------
# ==========================================================================

# Then, in the existing pounders.pouders(...) call, add ONE keyword argument:
#
#     X, F, J, flag, xkin = pounders.pouders(
#         fun=gst_pounders_full_function,   # reads the global pounders_dataset
#         X0=x0_full.reshape(1, -1),
#         n=n, nfmax=NFMAX, gtol=GTOL, delta=DELTA0, m=m,
#         L=lower_bounds, U=upper_bounds, logger=NotebookPoundersLogger(),
#         hfun=h_leastsquares, combinemodels=combine_leastsquares,
#         fpr_reduction=fpr_reduction,
#         residuals_per_circuit=OUTCOMES_PER_CIRCUIT,
#         shots_per_circuit=N_SAMPLES,
#         fpr_use_union_mask=USE_FPR_UNION_MASK,
#         rho_uses_full_objective=POUNDERS_RHO_USES_FULL_OBJECTIVE,
#         iter_callback=adaptive_hook,      # <-- the only new argument
#     )
