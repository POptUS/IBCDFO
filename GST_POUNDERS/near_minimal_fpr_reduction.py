"""Prototype per-germ fiducial-pair reduction for the 2Q XYI+CNOT GST model.

This is an implementation scaffold inspired by:
    Ostrove et al., "Near-Minimal Gate Set Tomography Experiment Designs"

Important practical note:
    Stage 1 supports the paper's twirled-derivative construction and a faster
    probability-Jacobian proxy for debugging. Stage 2 follows the paper's
    notation closely:

        J_{g,c} = d p_{g,c} / d theta
        D_{g,c} = J_{g,c} W_g

    and then greedily selects fiducial pairs until the selected D matrices span
    the selected directions W_g.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
from pygsti.modelpacks import smq2Q_XYICNOT
from pygsti.protocols import StandardGSTDesign


MAX_LENGTHS = [1, 2, 4, 8, 16]
SVD_REL_TOL = 1e-10
RANK_REL_TOL = 1e-10
COMMUTANT_REL_TOL = 1e-12
STAGE1_METHOD = "twirled_derivative"
STAGE2_METHOD = "paper_greedy"
MAX_STAGE1_RANK_PER_GERM = None  # Set to e.g. 80 for faster debugging.
OUTDIR = Path("near_minimal_fpr_results")


def build_xyicnot_setup(max_lengths=None):
    """Return the model and design objects used in the GST_model notebook."""
    max_lengths = list(MAX_LENGTHS if max_lengths is None else max_lengths)
    model = smq2Q_XYICNOT.target_model()
    processor_spec = smq2Q_XYICNOT.processor_spec()
    prep_fiducials = smq2Q_XYICNOT.prep_fiducials()
    meas_fiducials = smq2Q_XYICNOT.meas_fiducials()
    germs = smq2Q_XYICNOT.germs()
    return {
        "model": model,
        "processor_spec": processor_spec,
        "prep_fiducials": prep_fiducials,
        "meas_fiducials": meas_fiducials,
        "germs": germs,
        "max_lengths": max_lengths,
    }


def copy_model_at_vector(model, parameter_vector=None):
    """Copy a pyGSTi model and optionally overwrite its parameter vector."""
    copied = model.copy()
    if parameter_vector is not None:
        copied.from_vector(np.asarray(parameter_vector, dtype=float).reshape(-1), close=False)
    return copied


def random_parameter_vectors(
    base_vector,
    num_vectors,
    mode="local_uniform",
    perturbation_scale=1e-3,
    lower_bound=None,
    upper_bound=None,
    seed=1234,
    clip=None,
):
    """Create random GST parameter vectors.

    mode="local_uniform":
        Draw each parameter uniformly from
        [base_vector_i - perturbation_scale, base_vector_i + perturbation_scale].

    mode="global_uniform":
        Draw each parameter uniformly from [lower_bound, upper_bound].
        Bounds can be scalars or vectors with the same shape as base_vector.
    """
    rng = np.random.default_rng(seed)
    base_vector = np.asarray(base_vector, dtype=float).reshape(-1)

    vectors = []
    for _ in range(int(num_vectors)):
        if mode == "local_uniform":
            if perturbation_scale < 0:
                raise ValueError("perturbation_scale must be nonnegative.")
            low = base_vector - float(perturbation_scale)
            high = base_vector + float(perturbation_scale)
        elif mode == "global_uniform":
            if lower_bound is None or upper_bound is None:
                raise ValueError("global_uniform mode requires lower_bound and upper_bound.")
            low = np.asarray(lower_bound, dtype=float)
            high = np.asarray(upper_bound, dtype=float)
            if low.ndim == 0:
                low = np.full_like(base_vector, float(low))
            else:
                low = low.reshape(base_vector.shape)
            if high.ndim == 0:
                high = np.full_like(base_vector, float(high))
            else:
                high = high.reshape(base_vector.shape)
        else:
            raise ValueError("mode must be 'local_uniform' or 'global_uniform'.")

        if np.any(high <= low):
            raise ValueError("Each upper bound must be greater than the corresponding lower bound.")

        trial = rng.uniform(low, high)

        if clip is not None:
            low, high = clip
            trial = np.clip(trial, low, high)

        vectors.append(trial)

    return vectors


def get_outcomes(model, circuit):
    return list(model.probabilities(circuit).keys())


def candidate_circuits_for_germ(prep_fiducials, meas_fiducials, germ):
    circuits = []
    pair_indices = []
    for i_prep, prep in enumerate(prep_fiducials):
        for i_meas, meas in enumerate(meas_fiducials):
            circuits.append(prep + germ + meas)
            pair_indices.append((i_prep, i_meas))
    return circuits, pair_indices


def probability_jacobian_for_circuits(model, circuits):
    dprobs_by_circuit = model.sim.bulk_dprobs(circuits)
    rows = []
    row_circuits = []
    row_outcomes = []
    for circuit in circuits:
        outcomes = get_outcomes(model, circuit)
        dprobs = dprobs_by_circuit[circuit]
        for outcome in outcomes:
            rows.append(np.asarray(dprobs[outcome], dtype=float))
            row_circuits.append(circuit)
            row_outcomes.append(outcome)
    return np.vstack(rows), row_circuits, row_outcomes


def global_parameter_indices(gpindices):
    """Convert a pyGSTi global-parameter index object into an integer array."""
    if isinstance(gpindices, slice):
        start = 0 if gpindices.start is None else int(gpindices.start)
        stop = int(gpindices.stop)
        step = 1 if gpindices.step is None else int(gpindices.step)
        return np.arange(start, stop, step, dtype=int)
    return np.asarray(gpindices, dtype=int).reshape(-1)


def germ_ptm_derivative_matrix(model, germ):
    """Return d tau(g) / d theta as a matrix with shape d^4 by num_params.

    pyGSTi's circuit product convention is
        tau([G0, G1, ..., Gm]) = Gm ... G1 G0.
    This function applies the product rule to the local operation derivatives
    exposed by pyGSTi and maps them into the global model parameter vector.
    """
    labels = list(germ)
    tau = np.asarray(model.sim.product(germ), dtype=float)
    dim = int(tau.shape[0])
    deriv = np.zeros((dim * dim, int(model.num_params)), dtype=float)

    if len(labels) == 0:
        return tau, deriv

    matrices = [np.asarray(model.operations[label].to_dense(), dtype=float) for label in labels]

    prefix_before = []
    running = np.eye(dim)
    for matrix in matrices:
        prefix_before.append(running)
        running = matrix @ running

    suffix_after = [None] * len(matrices)
    running = np.eye(dim)
    for index in range(len(matrices) - 1, -1, -1):
        suffix_after[index] = running
        running = running @ matrices[index]

    for index, label in enumerate(labels):
        operation = model.operations[label]
        local_derivs = np.asarray(operation.deriv_wrt_params(), dtype=float)
        global_indices = global_parameter_indices(operation.gpindices)
        left = suffix_after[index]
        right = prefix_before[index]

        for local_index, global_index in enumerate(global_indices):
            d_operation = local_derivs[:, local_index].reshape((dim, dim), order="C")
            d_product = left @ d_operation @ right
            deriv[:, int(global_index)] += d_product.reshape(-1, order="C")

    return tau, deriv


def commutant_basis(matrix, rel_tol=COMMUTANT_REL_TOL):
    """Return an orthonormal basis for matrices X satisfying X A = A X.

    The basis is represented in row-major vectorized form so that projecting
    vectorized PTM derivatives onto it gives the Frobenius projection onto the
    commutant of ``matrix``.
    """
    matrix = np.asarray(matrix, dtype=float)
    dim = int(matrix.shape[0])
    commutator_map = np.empty((dim * dim, dim * dim), dtype=float)

    for basis_index in range(dim * dim):
        basis_matrix = np.zeros((dim, dim), dtype=float)
        basis_matrix.reshape(-1, order="C")[basis_index] = 1.0
        commutator = basis_matrix @ matrix - matrix @ basis_matrix
        commutator_map[:, basis_index] = commutator.reshape(-1, order="C")

    return sla.null_space(commutator_map, rcond=rel_tol)


def twirled_derivative_for_germ(model, germ, commutant_rel_tol=COMMUTANT_REL_TOL):
    """Compute the paper's twirled derivative matrix for one germ.

    This implements the projection described in the paper: compute the
    derivative of the germ PTM and project each derivative slice onto the
    commutant of the germ PTM. The result has shape d^4 by num_params.
    """
    tau, derivative_matrix = germ_ptm_derivative_matrix(model, germ)
    basis = commutant_basis(tau, rel_tol=commutant_rel_tol)
    twirled = basis @ (basis.T @ derivative_matrix)
    return twirled, int(basis.shape[1])


def numerical_rank_from_singular_values(singular_values, rel_tol):
    if singular_values.size == 0:
        return 0
    threshold = rel_tol * max(1.0, float(singular_values[0]))
    return int(np.sum(singular_values > threshold))


def right_singular_vectors(matrix, rel_tol=SVD_REL_TOL, max_rank=None):
    _, singular_values, vt = sla.svd(matrix, full_matrices=False, check_finite=False)
    rank = numerical_rank_from_singular_values(singular_values, rel_tol)
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return vt[:rank].T.copy(), singular_values, rank


def column_subset_selection_qr(j_matrix, rel_tol=RANK_REL_TOL):
    # Rank-revealing QR selects columns of J. This is the paper's CSSP idea,
    # using the RRQR heuristic rather than an exhaustive subset search.
    _, r_matrix, pivots = sla.qr(j_matrix, pivoting=True, mode="economic", check_finite=False)
    diag = np.abs(np.diag(r_matrix))
    rank = numerical_rank_from_singular_values(diag, rel_tol)
    return np.asarray(pivots[:rank], dtype=int), rank, diag


def row_space_basis(rows, rel_tol=RANK_REL_TOL):
    if rows.size == 0 or rows.shape[0] == 0:
        return np.empty((rows.shape[1], 0)), 0, np.asarray([])
    _, singular_values, vt = sla.svd(rows, full_matrices=False, check_finite=False)
    rank = numerical_rank_from_singular_values(singular_values, rel_tol)
    return vt[:rank].T.copy(), rank, singular_values


def greedy_select_fiducial_pairs(d_blocks, pair_indices, rel_tol=RANK_REL_TOL):
    """Select row blocks D_{g,c} until their row span has full column rank."""
    if not d_blocks:
        return [], np.empty((0, 0)), 0

    target_rank = int(d_blocks[0].shape[1])
    selected = []
    selected_rows = np.empty((0, target_rank))
    basis = np.empty((target_rank, 0))
    current_rank = 0
    remaining = set(range(len(d_blocks)))

    while current_rank < target_rank and remaining:
        best_index = None
        best_score = -np.inf

        for index in remaining:
            block = d_blocks[index]
            if basis.shape[1] > 0:
                residual = block - (block @ basis) @ basis.T
            else:
                residual = block
            score = float(np.linalg.norm(residual, ord="fro"))
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None or best_score <= 1e-14:
            break

        selected.append(best_index)
        remaining.remove(best_index)
        selected_rows = np.vstack([selected_rows, d_blocks[best_index]])
        basis, current_rank, _ = row_space_basis(selected_rows, rel_tol)

    selected_pairs = [pair_indices[index] for index in selected]
    return selected_pairs, selected_rows, current_rank


def _gramian_rank_and_trace_pinv(rows, rel_tol=RANK_REL_TOL):
    """Return rank(rows) and trace((rows.T @ rows)^+).

    This is the paper-style A-optimality score used by the greedy CSSP
    heuristic. Smaller trace means the selected directional derivative matrix
    is better conditioned in the directions it spans.
    """
    rows = np.asarray(rows, dtype=float)
    if rows.size == 0 or rows.shape[0] == 0:
        return 0, np.inf

    gramian = rows.T @ rows
    evals = np.linalg.eigvalsh(gramian)
    evals = np.asarray(evals, dtype=float)
    max_eval = float(np.max(np.abs(evals))) if evals.size else 0.0
    threshold = float(rel_tol) * max(1.0, max_eval)
    kept = evals[evals > threshold]
    rank = int(kept.size)
    trace_pinv = float(np.sum(1.0 / kept)) if rank > 0 else np.inf
    return rank, trace_pinv


def paper_greedy_select_fiducial_pairs(d_blocks, pair_indices, rel_tol=RANK_REL_TOL):
    """Paper-style greedy FPR Stage 2 selection.

    The paper's Stage 2 greedy heuristic prioritizes increasing the rank of the
    composite directional-derivative matrix and uses an A-optimality score,
    trace((D.T @ D)^+), to break ties among candidates with the same rank. The
    paper uses low-rank update formulas to evaluate this efficiently; here we
    compute the same score directly, which is simpler and fine for these tests.
    """
    if not d_blocks:
        return [], np.empty((0, 0)), 0

    target_rank = int(d_blocks[0].shape[1])
    selected = []
    selected_rows = np.empty((0, target_rank))
    current_rank = 0
    remaining = set(range(len(d_blocks)))

    while current_rank < target_rank and remaining:
        best_index = None
        best_rank = -1
        best_trace = np.inf

        for index in remaining:
            trial_rows = np.vstack([selected_rows, d_blocks[index]])
            trial_rank, trial_trace = _gramian_rank_and_trace_pinv(trial_rows, rel_tol)

            if (
                trial_rank > best_rank
                or (trial_rank == best_rank and trial_trace < best_trace)
                or (
                    trial_rank == best_rank
                    and np.isclose(trial_trace, best_trace, rtol=1e-12, atol=1e-15)
                    and (best_index is None or index < best_index)
                )
            ):
                best_index = index
                best_rank = trial_rank
                best_trace = trial_trace

        if best_index is None or best_rank <= current_rank:
            break

        selected.append(best_index)
        remaining.remove(best_index)
        selected_rows = np.vstack([selected_rows, d_blocks[best_index]])
        current_rank, _ = _gramian_rank_and_trace_pinv(selected_rows, rel_tol)

    selected_pairs = [pair_indices[index] for index in selected]
    return selected_pairs, selected_rows, current_rank


def select_fiducial_pairs_for_stage2(
    d_blocks,
    pair_indices,
    rel_tol=RANK_REL_TOL,
    stage2_method=STAGE2_METHOD,
):
    """Dispatch Stage 2 fiducial-pair selection."""
    if stage2_method == "paper_greedy":
        return paper_greedy_select_fiducial_pairs(d_blocks, pair_indices, rel_tol)
    if stage2_method == "frobenius_residual":
        return greedy_select_fiducial_pairs(d_blocks, pair_indices, rel_tol)
    raise ValueError("stage2_method must be 'paper_greedy' or 'frobenius_residual'.")


def make_reduced_design(processor_spec, prep_fiducials, meas_fiducials, germs, max_lengths, fiducial_pairs):
    return StandardGSTDesign(
        processor_spec,
        prep_fiducials,
        meas_fiducials,
        germs,
        max_lengths,
        fiducial_pairs=fiducial_pairs,
    )


def selected_pair_records(reduced_fiducial_pairs, germs):
    records = []
    for germ_index, germ in enumerate(germs):
        for prep_index, meas_index in reduced_fiducial_pairs.get(germ, []):
            records.append(
                {
                    "germ_index": int(germ_index),
                    "prep_fiducial_index": int(prep_index),
                    "meas_fiducial_index": int(meas_index),
                }
            )
    return pd.DataFrame(records)


def selected_pair_set(pair_records):
    if pair_records is None or len(pair_records) == 0:
        return set()
    return {
        (int(row.germ_index), int(row.prep_fiducial_index), int(row.meas_fiducial_index))
        for row in pair_records.itertuples(index=False)
    }

def residual_mask_from_circuits(selected_circuits, all_circuits, outcomes_per_circuit=4):
    """Return a residual-entry mask for the selected GST circuits.

    The optimizer residual vector is ordered by circuit and then by outcome.
    Selecting one circuit keeps ``outcomes_per_circuit`` consecutive residual
    entries.
    """
    all_circuits = list(all_circuits)
    selected_keys = {str(circuit) for circuit in selected_circuits}
    outcomes_per_circuit = int(outcomes_per_circuit)
    if outcomes_per_circuit <= 0:
        raise ValueError("outcomes_per_circuit must be positive.")

    keep = np.zeros(outcomes_per_circuit * len(all_circuits), dtype=bool)
    for circuit_index, circuit in enumerate(all_circuits):
        if str(circuit) in selected_keys:
            start = outcomes_per_circuit * circuit_index
            keep[start : start + outcomes_per_circuit] = True
    return keep


def make_fpr_reduction_mask_function(
    base_model,
    all_circuits,
    processor_spec,
    prep_fiducials,
    meas_fiducials,
    germs,
    max_lengths,
    outcomes_per_circuit=4,
    stage1_method=STAGE1_METHOD,
    stage2_method=STAGE2_METHOD,
    svd_rel_tol=SVD_REL_TOL,
    rank_rel_tol=RANK_REL_TOL,
    commutant_rel_tol=COMMUTANT_REL_TOL,
    max_stage1_rank_per_germ=MAX_STAGE1_RANK_PER_GERM,
    modelpack_name=None,
    verbose=False,
):
    """Build a POUNDERS-compatible FPR reduction callback.

    The returned function has signature ``fpr_reduction(x)`` and returns a
    boolean mask with one entry per residual component. It intentionally knows
    nothing about POUNDERS internals beyond this mask contract.
    """
    all_circuits = list(all_circuits)
    history = []

    def fpr_reduction(parameter_vector):
        result = run_fpr_reduction(
            parameter_vector=parameter_vector,
            label=f"fpr_call_{len(history)}",
            modelpack_name=modelpack_name,
            base_model=base_model,
            processor_spec=processor_spec,
            prep_fiducials=prep_fiducials,
            meas_fiducials=meas_fiducials,
            germs=germs,
            max_lengths=max_lengths,
            stage1_method=stage1_method,
            stage2_method=stage2_method,
            svd_rel_tol=svd_rel_tol,
            rank_rel_tol=rank_rel_tol,
            commutant_rel_tol=commutant_rel_tol,
            max_stage1_rank_per_germ=max_stage1_rank_per_germ,
            outcomes_per_circuit=outcomes_per_circuit,
            save_results=False,
            verbose=verbose,
        )
        reduced_circuits = list(result["reduced_design"].all_circuits_needing_data)
        selected_circuit_keys = {str(circuit) for circuit in reduced_circuits}
        selected_circuit_indices = [
            int(circuit_index)
            for circuit_index, circuit in enumerate(all_circuits)
            if str(circuit) in selected_circuit_keys
        ]
        mask = residual_mask_from_circuits(
            reduced_circuits,
            all_circuits,
            outcomes_per_circuit=outcomes_per_circuit,
        )
        history.append(
            {
                "call_index": len(history),
                "selected_residuals": int(np.sum(mask)),
                "total_residuals": int(mask.size),
                "selected_circuits": int(np.sum(mask) // outcomes_per_circuit),
                "total_circuits": int(len(all_circuits)),
                "selected_circuit_indices": selected_circuit_indices,
                "selected_residual_indices": np.flatnonzero(mask).astype(int).tolist(),
                "summary": result["summary"],
            }
        )
        fpr_reduction.last_result = result
        return mask

    fpr_reduction.history = history
    fpr_reduction.last_result = None
    return fpr_reduction


def jaccard_overlap(pair_records_a, pair_records_b):
    set_a = selected_pair_set(pair_records_a)
    set_b = selected_pair_set(pair_records_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def run_fpr_reduction(
    parameter_vector=None,
    label="target",
    modelpack_name=None,
    base_model=None,
    processor_spec=None,
    prep_fiducials=None,
    meas_fiducials=None,
    germs=None,
    max_lengths=None,
    stage1_method=STAGE1_METHOD,
    stage2_method=STAGE2_METHOD,
    svd_rel_tol=SVD_REL_TOL,
    rank_rel_tol=RANK_REL_TOL,
    commutant_rel_tol=COMMUTANT_REL_TOL,
    max_stage1_rank_per_germ=MAX_STAGE1_RANK_PER_GERM,
    outcomes_per_circuit=4,
    outdir=None,
    save_results=False,
    verbose=True,
):
    """Run the two-stage FPR prototype for one parameter vector.

    Returns a dictionary containing the summary, per-germ tables, selected
    fiducial pairs, and selected-pair records. Use this from notebooks to
    stress-test different parameter vectors.
    """
    setup = None
    if base_model is None or processor_spec is None or prep_fiducials is None or meas_fiducials is None or germs is None:
        setup = build_xyicnot_setup(max_lengths=max_lengths)
        base_model = setup["model"] if base_model is None else base_model
        processor_spec = setup["processor_spec"] if processor_spec is None else processor_spec
        prep_fiducials = setup["prep_fiducials"] if prep_fiducials is None else prep_fiducials
        meas_fiducials = setup["meas_fiducials"] if meas_fiducials is None else meas_fiducials
        germs = setup["germs"] if germs is None else germs
        max_lengths = setup["max_lengths"] if max_lengths is None else max_lengths

    max_lengths = list(MAX_LENGTHS if max_lengths is None else max_lengths)
    if stage1_method not in {"twirled_derivative", "probability_jacobian_proxy"}:
        raise ValueError("stage1_method must be 'twirled_derivative' or 'probability_jacobian_proxy'.")
    if stage2_method not in {"paper_greedy", "frobenius_residual"}:
        raise ValueError("stage2_method must be 'paper_greedy' or 'frobenius_residual'.")

    model = copy_model_at_vector(base_model, parameter_vector)
    n_params = int(model.num_params)

    standard_design = StandardGSTDesign(
        processor_spec,
        prep_fiducials,
        meas_fiducials,
        germs,
        max_lengths,
    )
    standard_circuits = list(standard_design.all_circuits_needing_data)

    if verbose:
        print(f"FPR reduction run: {label}")
        print("Model:", type(model).__name__)
        print("parameters:", n_params)
        print("prep fiducials:", len(prep_fiducials))
        print("meas fiducials:", len(meas_fiducials))
        print("germs:", len(germs))
        print("standard pyGSTi circuits:", len(standard_circuits))
        print("Stage 1 method:", stage1_method)
        print("Stage 2 method:", stage2_method)

    # Stage 1: compute per-germ right singular vector matrices V_g.
    v_by_germ = {}
    jacobian_blocks_by_germ = {}
    candidate_pairs_by_germ = {}
    stage1_rows = []
    j_columns = []
    j_column_metadata = []

    start = time.perf_counter()
    for germ_index, germ in enumerate(germs):
        germ_start = time.perf_counter()
        circuits, pair_indices = candidate_circuits_for_germ(prep_fiducials, meas_fiducials, germ)

        probability_jacobian = None
        twirled_commutant_dimension = None
        if stage1_method == "twirled_derivative":
            stage1_matrix, twirled_commutant_dimension = twirled_derivative_for_germ(
                model,
                germ,
                commutant_rel_tol=commutant_rel_tol,
            )
        else:
            probability_jacobian, _, _ = probability_jacobian_for_circuits(model, circuits)
            stage1_matrix = probability_jacobian

        v_g, singular_values, rank_g = right_singular_vectors(
            stage1_matrix,
            rel_tol=svd_rel_tol,
            max_rank=max_stage1_rank_per_germ,
        )

        if probability_jacobian is None:
            probability_jacobian, _, _ = probability_jacobian_for_circuits(model, circuits)

        v_by_germ[germ] = v_g
        jacobian_blocks_by_germ[germ] = [
            probability_jacobian[i : i + len(get_outcomes(model, circuits[0])), :]
            for i in range(0, probability_jacobian.shape[0], len(get_outcomes(model, circuits[0])))
        ]
        candidate_pairs_by_germ[germ] = pair_indices

        for local_index in range(v_g.shape[1]):
            j_columns.append(v_g[:, local_index])
            j_column_metadata.append((germ, germ_index, local_index))

        stage1_rows.append(
            {
                "germ_index": germ_index,
                "germ": str(germ),
                "stage1_method": stage1_method,
                "candidate_pairs": len(pair_indices),
                "candidate_probability_rows": int(probability_jacobian.shape[0]),
                "stage1_matrix_rows": int(stage1_matrix.shape[0]),
                "stage1_rank_before_global_cssp": int(rank_g),
                "proxy_rank_before_global_cssp": int(rank_g),
                "twirled_commutant_dimension": twirled_commutant_dimension,
                "elapsed_seconds": time.perf_counter() - germ_start,
            }
        )
        if verbose:
            print(
                f"Stage 1 {stage1_method} germ {germ_index + 1}/{len(germs)}:",
                f"rank={rank_g},",
                f"elapsed={stage1_rows[-1]['elapsed_seconds']:.2f}s",
            )

    j_matrix = np.column_stack(j_columns) if j_columns else np.empty((n_params, 0))
    selected_column_indices, global_rank, css_diag = column_subset_selection_qr(j_matrix, rel_tol=rank_rel_tol)

    if verbose:
        print("Stage 1 concatenated J shape:", j_matrix.shape)
        print("Stage 1 selected global rank:", global_rank)

    selected_vectors_by_germ = defaultdict(list)
    for selected_index in selected_column_indices:
        germ, _, _ = j_column_metadata[int(selected_index)]
        selected_vectors_by_germ[germ].append(j_matrix[:, int(selected_index)])

    w_by_germ = {}
    for germ in germs:
        vectors = selected_vectors_by_germ.get(germ, [])
        if vectors:
            w_by_germ[germ] = np.column_stack(vectors)
        else:
            w_by_germ[germ] = np.empty((n_params, 0))

    # Stage 2: for each germ, select fiducial pairs that span W_g.
    reduced_fiducial_pairs = {}
    stage2_rows = []
    for germ_index, germ in enumerate(germs):
        stage2_start = time.perf_counter()
        w_g = w_by_germ[germ]
        r_g = int(w_g.shape[1])

        if r_g == 0:
            selected_pairs = []
            achieved_rank = 0
        else:
            d_blocks = [j_block @ w_g for j_block in jacobian_blocks_by_germ[germ]]
            selected_pairs, _, achieved_rank = select_fiducial_pairs_for_stage2(
                d_blocks,
                candidate_pairs_by_germ[germ],
                rel_tol=rank_rel_tol,
                stage2_method=stage2_method,
            )

        reduced_fiducial_pairs[germ] = selected_pairs
        stage2_rows.append(
            {
                "germ_index": germ_index,
                "germ": str(germ),
                "stage2_method": stage2_method,
                "selected_directions_r_g": r_g,
                "selected_fiducial_pairs": len(selected_pairs),
                "candidate_fiducial_pairs": len(candidate_pairs_by_germ[germ]),
                "achieved_rank": int(achieved_rank),
                "rank_complete": bool(achieved_rank == r_g),
                "elapsed_seconds": time.perf_counter() - stage2_start,
            }
        )
        if verbose:
            print(
                f"Stage 2 germ {germ_index + 1}/{len(germs)}:",
                f"r_g={r_g},",
                f"pairs={len(selected_pairs)}/{len(candidate_pairs_by_germ[germ])},",
                f"rank={achieved_rank}/{r_g}",
            )

    reduced_design = make_reduced_design(
        processor_spec,
        prep_fiducials,
        meas_fiducials,
        germs,
        max_lengths,
        dict(reduced_fiducial_pairs),
    )
    reduced_circuits = list(reduced_design.all_circuits_needing_data)

    stage1_df = pd.DataFrame(stage1_rows)
    stage2_df = pd.DataFrame(stage2_rows)
    pair_records_df = selected_pair_records(reduced_fiducial_pairs, germs)

    summary = {
        "label": label,
        "modelpack": (
            modelpack_name
            if modelpack_name is not None
            else ("pygsti.modelpacks.smq2Q_XYICNOT" if setup is not None else "unknown")
        ),
        "max_lengths": max_lengths,
        "num_parameters": n_params,
        "num_prep_fiducials": len(prep_fiducials),
        "num_meas_fiducials": len(meas_fiducials),
        "num_germs": len(germs),
        "stage1_method": stage1_method,
        "stage2_method": stage2_method,
        "outcomes_per_circuit": int(outcomes_per_circuit),
        "standard_circuits": len(standard_circuits),
        "reduced_circuits": len(reduced_circuits),
        "circuits_removed": len(standard_circuits) - len(reduced_circuits),
        "reduction_fraction": 1.0 - len(reduced_circuits) / len(standard_circuits),
        "standard_probability_entries": int(outcomes_per_circuit) * len(standard_circuits),
        "reduced_probability_entries": int(outcomes_per_circuit) * len(reduced_circuits),
        "stage1_global_J_shape": list(j_matrix.shape),
        "stage1_global_rank": int(global_rank),
        "stage1_proxy_global_J_shape": list(j_matrix.shape),
        "stage1_proxy_global_rank": int(global_rank),
        "total_selected_fiducial_pairs_across_germs": int(stage2_df["selected_fiducial_pairs"].sum()),
        "total_candidate_fiducial_pairs_across_germs": int(stage2_df["candidate_fiducial_pairs"].sum()),
        "all_stage2_ranks_complete": bool(stage2_df["rank_complete"].all()),
        "elapsed_seconds": time.perf_counter() - start,
        "stage1_note": (
            "Stage 1 uses twirled germ derivatives when stage1_method='twirled_derivative'. "
            "Stage 2 uses probability Jacobians J_{g,c} and directional derivatives D_{g,c}=J_{g,c}W_g."
        ),
    }

    if save_results:
        result_dir = Path(OUTDIR if outdir is None else outdir)
        result_dir.mkdir(exist_ok=True, parents=True)
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(label))
        stage1_df.to_csv(result_dir / f"{safe_label}_stage1_proxy_germ_ranks.csv", index=False)
        stage2_df.to_csv(result_dir / f"{safe_label}_stage2_selected_fiducial_pairs.csv", index=False)
        pair_records_df.to_csv(result_dir / f"{safe_label}_selected_pair_records.csv", index=False)
        with open(result_dir / f"{safe_label}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if verbose:
        print("\nReduction summary")
        for key, value in summary.items():
            print(f"{key}: {value}")

    return {
        "summary": summary,
        "stage1": stage1_df,
        "stage2": stage2_df,
        "pair_records": pair_records_df,
        "reduced_fiducial_pairs": reduced_fiducial_pairs,
        "reduced_design": reduced_design,
    }


def stress_test_random_parameter_vectors(
    num_random_vectors=3,
    mode="local_uniform",
    perturbation_scale=1e-3,
    lower_bound=None,
    upper_bound=None,
    seed=1234,
    clip=None,
    include_target=True,
    stage1_method=STAGE1_METHOD,
    stage2_method=STAGE2_METHOD,
    commutant_rel_tol=COMMUTANT_REL_TOL,
    max_stage1_rank_per_germ=MAX_STAGE1_RANK_PER_GERM,
    save_results=False,
    outdir=None,
    verbose=True,
):
    """Run FPR at the target and several random parameter vectors.

    Use mode="local_uniform" for small perturbations around the target vector,
    or mode="global_uniform" to draw every parameter inside lower/upper bounds.
    """
    setup = build_xyicnot_setup()
    base_model = setup["model"]
    base_vector = np.asarray(base_model.to_vector(), dtype=float)
    run_setup = {
        "base_model": setup["model"],
        "processor_spec": setup["processor_spec"],
        "prep_fiducials": setup["prep_fiducials"],
        "meas_fiducials": setup["meas_fiducials"],
        "germs": setup["germs"],
        "max_lengths": setup["max_lengths"],
    }
    trial_vectors = random_parameter_vectors(
        base_vector,
        num_vectors=num_random_vectors,
        mode=mode,
        perturbation_scale=perturbation_scale,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        seed=seed,
        clip=clip,
    )

    runs = []
    if include_target:
        runs.append(
            run_fpr_reduction(
                parameter_vector=base_vector,
                label="target",
                **run_setup,
                stage1_method=stage1_method,
                stage2_method=stage2_method,
                commutant_rel_tol=commutant_rel_tol,
                max_stage1_rank_per_germ=max_stage1_rank_per_germ,
                save_results=save_results,
                outdir=outdir,
                verbose=verbose,
            )
        )

    for index, vector in enumerate(trial_vectors, start=1):
        runs.append(
            run_fpr_reduction(
                parameter_vector=vector,
                label=f"random_{index}",
                **run_setup,
                stage1_method=stage1_method,
                stage2_method=stage2_method,
                commutant_rel_tol=commutant_rel_tol,
                max_stage1_rank_per_germ=max_stage1_rank_per_germ,
                save_results=save_results,
                outdir=outdir,
                verbose=verbose,
            )
        )

    baseline_records = runs[0]["pair_records"]
    summary_rows = []
    for run in runs:
        row = dict(run["summary"])
        row["pair_jaccard_vs_first_run"] = jaccard_overlap(baseline_records, run["pair_records"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["sampling_mode"] = mode
    summary_df["perturbation_scale"] = float(perturbation_scale)
    summary_df["global_lower_bound"] = (
        float(lower_bound) if lower_bound is not None and np.isscalar(lower_bound) else None
    )
    summary_df["global_upper_bound"] = (
        float(upper_bound) if upper_bound is not None and np.isscalar(upper_bound) else None
    )
    return runs, summary_df


def main():
    run_fpr_reduction(label="target", save_results=True, outdir=OUTDIR, verbose=True)


if __name__ == "__main__":
    main()
