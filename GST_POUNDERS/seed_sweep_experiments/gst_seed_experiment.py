"""Reusable GST/POUNDERS experiment used by the multi-seed runner.

This module is a clean extraction of the working paths in ``GST_model.ipynb``.
It deliberately keeps synthetic truth out of every optimization decision.  Truth
is used only after an iterate has been produced, for diagnostic gate/SPAM error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable
import contextlib
import importlib
import importlib.util
import io
import json
import math
import sys
import warnings

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
GST_POUNDERS_DIR = PACKAGE_DIR.parent
IBCDFO_ROOT = GST_POUNDERS_DIR.parent
POUNDERS_PY = IBCDFO_ROOT / "pounders" / "py"

for _path in (POUNDERS_PY, GST_POUNDERS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# "adaptive_no_fpr" is the isolation arm: adaptive D/A/L shot allocation over ALL circuits,
# with the FPR circuit reduction switched off. It exists because a paired comparison over the
# poisson_logl runs showed the two effects pulling in opposite directions --
#     adaptive_D / fixed_fpr    = 0.565x  (adaptive allocation genuinely helps)
#     fixed_fpr  / fixed_no_fpr = 1.712x  (FPR genuinely hurts)
#     fixed_no_fpr / lm         = 1.005x  (POUNDERS matches LM given the same design)
# whose product is ~0.97, i.e. the observed tie against LM. Every adaptive arm so far has FPR
# bundled in, so the allocation gain has been spent undoing the reduction loss rather than
# banked. This arm unbundles them.  NOTE: with no FPR every circuit is measured, so
# accounted_revealed_shots == physical shots and the budget is directly comparable to lm.
METHODS = ("adaptive_fpr", "adaptive_no_fpr", "fixed_fpr", "fixed_no_fpr")

# Methods whose shot budget grows online, as opposed to a uniform up-front allocation.
ADAPTIVE_METHODS = ("adaptive_fpr", "adaptive_no_fpr")

# Optional per-iteration hook, off by default. When set to a callable it is invoked from
# _run_pounders' iteration_callback with keyword arguments:
#     state, problem, config, method, iteration, cumulative_shots, running_shots
# `problem.dataset` is live at that moment, which is the entire point -- it lets a caller fit
# a comparison estimator (e.g. pyGSTi LM) on exactly the shots and circuits POUNDERS has
# consumed so far, with no redraw and no reconstruction from disk. The return value is ignored.
# Exceptions are caught and printed rather than propagated, so a broken hook cannot kill an
# expensive POUNDERS run. Leave as None for the original behaviour.
LM_CHECKPOINT_HOOK = None


@dataclass(frozen=True)
class ExperimentConfig:
    """All numerical choices required to reproduce a seed-sweep run."""

    model_kind: str = "1Q"
    modelpack_1q: str = "smq1Q_XYZI"
    modelpack_2q: str = "smq2Q_XYICNOT"
    max_lengths: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    parameterization: str = "CPTPLND"

    truth_seed: int = 100
    vary_truth_with_data_seed: bool = False
    op_noise: float = 0.01
    spam_noise: float = 0.005
    coherent_max_rotate: float = 0.0   # >0 adds random per-gate coherent over-rotations (radians)
    noise_model: str = "depolarize_rotate"  # "depolarize_rotate" | "lindblad_hard"
    lindblad_h_max: float = 0.03       # lindblad_hard: coherent (H) coeff range +/-
    lindblad_s_max: float = 0.004      # lindblad_hard: stochastic (S) coeff range [0, max]
    lindblad_a_max: float = 0.004      # lindblad_hard: T1 damping rate range [0, max] (decay toward |0>)
    lindblad_rho_max: float = 0.3      # lindblad_hard: correlated-stochastic (C) range as a
                                       #   correlation coeff rho = C_PQ / sqrt(S_P*S_Q); 0 disables
    sample_error: str = "multinomial"

    objective: str = "weighted_least_squares"
    variance_source: str = "data"
    variance_floor: float = 1e-12
    # Add-s smoothing for the WLS variance ONLY (the residual keeps the raw
    # observed frequency). 0.0 reproduces the historical behaviour exactly.
    #
    # Why it matters: var = f(1-f)/N is exactly 0 whenever a zero (or full) count
    # comes back, so variance_floor takes over and that single residual enters the
    # sum of squares with weight 1/variance_floor = 1e12. With spam_noise=0.005 the
    # smallest truth probability is 0.0025, which returns a zero count 29% of the
    # time at 500 shots -- about 5 outcomes per seed, a DIFFERENT set each seed.
    # The objective being minimised is therefore randomly perturbed per seed.
    #
    # 0.5 is the Jeffreys/Krichevsky-Trofimov estimate (count+0.5)/(N+1), which is
    # bounded away from 0 and 1, so the floor never binds.
    variance_smoothing: float = 0.0
    # Poisson-picture delta-logl, used when objective == "poisson_logl".  These
    # defaults are pyGSTi's own (RawPoissonPicDeltaLogLFunction), so the residual
    # POUNDERS minimises is the identical vector pyGSTi's LM minimises.
    # min_prob_clip must sit below the smallest nonzero frequency or pyGSTi's
    # regularization drives a term negative and raises; it is auto-lowered at run
    # time when adaptive shots make some frequency smaller than this.
    logl_min_prob_clip: float = 1e-4
    logl_radius: float = 1e-4
    nfmax: int = 200
    gtol: float = 1e-4
    initial_delta: float = 0.1
    lower_bound: float = -math.pi
    upper_bound: float = math.pi
    pyrol_max_iters: int = 10
    require_pyrol: bool = True

    # Forward simulator used ONLY by base_model, i.e. only by the POUNDERS oracle.
    #
    # Why you might want "matrix": pyGSTi's default here is MapForwardSimulator, whose dp/dx
    # is a FORWARD FINITE DIFFERENCE with a hardcoded eps=1e-7 (mapforwardsim.py:163). On this
    # problem that Jacobian is accurate to 2e-6 relative far from the optimum, but it carries
    # a roughly CONSTANT ABSOLUTE error of ~0.2 in the gradient. Near a solution the true
    # gradient is also ~0.2, so the reported gradient is ~92% wrong and only 0.47-aligned with
    # the truth. That is the floor POUNDERS' ng stalls on, and why gtol=1e-4 is never reached
    # (0 of 882 archived runs exited on gtol; the best ng ever seen is 0.056).
    #
    # Why it is OFF by default: setting "matrix" on target_model made pyGSTi's OWN LM protocol
    # converge to a far worse point -- 2*deltaLogL 6508 vs 767 at max_lengths [1,2,4,8], and a
    # gauge-optimised infidelity that came out NEGATIVE at the full 7-stage ladder. The fitted
    # model stayed CP (min Choi eigenvalue -3e-16), and the two simulators agree on
    # probabilities to 2e-15 and score a fixed estimate bit-identically, so this is a defect in
    # the LM fit path, not in the metrics. Unexplained -- so the safe scope is base_model only,
    # and the safe default is None (byte-identical to every previous run).
    #
    # Enabling it changes only what POUNDERS sees. Validate on ONE seed before a full sweep:
    # the ng trace should keep descending past ~0.2 instead of flattening there.
    forward_simulator: str | None = None

    # Relative weight the gauge optimisation puts on SPAM alignment versus gate alignment
    # when mapping an estimate onto the truth for scoring. pyGSTi's gaugeopt_to_target
    # defaults to 1.0 for both, which forces the gauge to trade gate error against SPAM
    # error; an arm with worse SPAM then has that error pushed into its gates, where the
    # diamond norm sees it linearly. Measured over 40 budget-matched pairs, adaptive_D's
    # diamond penalty against LM runs 1.095x (p=0.003) at weight 1.0 and disappears by 0.5
    # (1.001x), sitting at 0.95-0.96 (p>0.4) for anything smaller. pyGSTi's own stdgaugeopt
    # is 3-stage and its varySpam suites sweep 1e-4..1e-1, so 1.0 is the outlier, not this.
    # Set to None to use pyGSTi's default (equal weighting).
    gaugeopt_spam_weight: float | None = 0.1
    # Distance the gauge optimisation minimises. "frobenius" is pyGSTi's default; "tracedist"
    # is closer in spirit to the diamond norm (which IS the stabilised trace distance).
    gaugeopt_metric: str = "frobenius"

    use_fpr_union_mask: bool = True
    rho_uses_full_objective: bool = False
    fpr_stage1_method: str = "twirled_derivative"
    fpr_stage2_method: str = "paper_greedy"
    fpr_verbose: bool = False

    fixed_no_fpr_shots: int = 250
    fixed_fpr_shots: int = 800
    adaptive_baseline_shots: int = 500
    adaptive_criterion: str = "D"
    adaptive_schedule: str = "lazy_delta_inverse_square"
    adaptive_schedule_base: float = 1.5
    adaptive_schedule_n0: int = 300
    adaptive_schedule_n_max: int = 8192
    adaptive_total_shot_budget: int = 1_000_000
    adaptive_delta_inverse_square_constant: float = 0.03
    adaptive_delta_inverse_square_n_min: int = 0
    adaptive_delta_inverse_fourth_constant: float = 3e-4
    adaptive_delta_inverse_fourth_n_min: int = 0
    adaptive_delta_floor: float = 1e-12
    adaptive_rho_band: float = 0.5
    # Re-solve the D/A/L design only every k-th POUNDERS iteration. The design solve
    # (Frank-Wolfe up to 200 iterations, then greedy integer rounding over every circuit)
    # dominates the runtime once FPR is off and the design spans all 1918 circuits. The Fisher
    # information barely moves between consecutive steps, so k>1 costs little accuracy and
    # divides the allocation cost by k. The same total budget is still spent -- the schedule
    # keeps accruing across skipped iterations, so shots arrive in k-times larger batches.
    # 1 = solve every iteration (previous behaviour).
    adaptive_allocate_every: int = 1
    # True: one decision variable per CIRCUIT, per-shot info = sum_beta J J^T / p_beta.
    # False: the older per-ROW path weighted by 1/(p(1-p)), max-aggregated to circuits.
    # Identical rankings for a 2-outcome measurement; only the per-circuit form is correct
    # for more outcomes. See the note in adaptive_shot_hook.make_adaptive_shot_hook.
    adaptive_per_circuit_allocation: bool = True
    adaptive_initial_topup: int = 0

    # L-optimality (criterion="L") infidelity-metric (M = infidelity Hessian) settings.
    l_metric_diagonal: bool = True        # diagonal-only Hessian (fast: ~2n gauge-opts) vs full (~2n^2)
    l_metric_eps: float = 1e-3            # finite-difference step for the infidelity Hessian
    l_metric_rebuild_every: int = 25      # rebuild M from the current estimate every K iterations

    probability_tolerance: float = 1e-8
    save_iteration_models: bool = True
    report_diamond_distance: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown experiment config fields: {unknown}")
        if "max_lengths" in payload:
            payload["max_lengths"] = tuple(int(v) for v in payload["max_lengths"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["max_lengths"] = list(self.max_lengths)
        return payload


class PoundersLogger:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

    def log(self, source, message, level=None):
        if self.enabled:
            print(f"[{source}] {message}")


def _model_num_params(model) -> int:
    value = model.num_params
    return int(value() if callable(value) else value)


def _dense(member) -> np.ndarray:
    method = getattr(member, "to_dense", None)
    if callable(method):
        try:
            return np.asarray(method(), dtype=float)
        except TypeError:
            return np.asarray(method("minimal"), dtype=float)
    return np.asarray(member, dtype=float)


def _num_nongauge(model, fallback: int) -> int:
    for name in (
        "num_nongauge_params",
        "num_non_gauge_params",
        "num_nongauge_parameters",
        "num_non_gauge_parameters",
    ):
        value = getattr(model, name, None)
        if value is not None:
            return int(value() if callable(value) else value)
    return int(fallback)


def _make_parameterized_model(model, parameterization: str | None, ideal_model=None):
    result = model.copy()
    if parameterization is None:
        return result
    errors = []
    for name in ("set_all_parameterizations", "set_all_parameterizations_to"):
        method = getattr(result, name, None)
        if not callable(method):
            continue
        if ideal_model is not None:
            try:
                method(parameterization, ideal_model=ideal_model)
                return result
            except Exception as exc:
                errors.append(f"{name}(ideal_model=...): {exc!r}")
        try:
            method(parameterization)
            return result
        except Exception as exc:
            errors.append(f"{name}: {exc!r}")
    raise RuntimeError(
        f"Could not set model parameterization to {parameterization!r}: "
        + " | ".join(errors)
    )


_PAULI_AXES = ("X", "Y", "Z")


def _stochastic_block(s: dict, c: dict, a: dict, scale: float = 1.0) -> np.ndarray:
    """The 1-qubit stochastic block: 3x3 Hermitian over (X,Y,Z), diag S, off-diag C - iA.

    Complete positivity of the resulting channel is exactly positive-semidefiniteness of
    this matrix, so it is the cheapest CP test available.  `scale` multiplies the C terms
    only (A carries the physical T1 rate and is CP-safe on its own).
    """
    block = np.zeros((3, 3), dtype=complex)
    for i, p in enumerate(_PAULI_AXES):
        block[i, i] = s[p]
    for i, p in enumerate(_PAULI_AXES):
        for j, q in enumerate(_PAULI_AXES[i + 1:], start=i + 1):
            value = scale * c.get((p, q), 0.0) - 1j * a.get((p, q), 0.0)
            block[i, j] = value
            block[j, i] = np.conj(value)
    return block


def _cp_safe_c_scale(s: dict, c: dict, a: dict, tol: float = 1e-12) -> float:
    """Largest factor in [0, 1] on the C terms keeping the stochastic block PSD (i.e. CP).

    Drawing C as rho * sqrt(S_P * S_Q) respects the pairwise bound C^2 + A^2 <= S_P * S_Q,
    but the full 3x3 can still fail for correlated signs, so shrink until it holds.
    """
    if float(np.linalg.eigvalsh(_stochastic_block(s, c, a, 1.0)).min()) >= -tol:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(np.linalg.eigvalsh(_stochastic_block(s, c, a, mid)).min()) >= -tol:
            lo = mid
        else:
            hi = mid
    return lo


def _safe_json(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_safe_json(payload), indent=2, sort_keys=True), encoding="utf-8")


def _csv_safe_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda value: json.dumps(_safe_json(value), separators=(",", ":"))
                if isinstance(value, (dict, list, tuple, np.ndarray))
                else value
            )
    return frame


class GSTProblem:
    """One synthetic GST problem shared by all methods for one data seed."""

    def __init__(self, config: ExperimentConfig, data_seed: int):
        import pkgutil
        import pygsti
        import pygsti.modelpacks as modelpacks
        from pygsti.protocols import StandardGSTDesign

        self.pygsti = pygsti
        self.config = config
        self.data_seed = int(data_seed)
        self.truth_seed = int(data_seed if config.vary_truth_with_data_seed else config.truth_seed)

        name = config.modelpack_1q if config.model_kind.upper() == "1Q" else config.modelpack_2q
        try:
            self.modelpack = importlib.import_module(f"pygsti.modelpacks.{name}")
        except Exception:
            self.modelpack = getattr(modelpacks, name, None)
        if self.modelpack is None:
            available = sorted(
                module.name for module in pkgutil.iter_modules(modelpacks.__path__)
                if module.name.startswith(("smq1Q", "smq2Q"))
            )
            raise ImportError(f"Could not load modelpack {name!r}; available={available}")

        self.raw_target_model = self.modelpack.target_model()
        self.target_model = _make_parameterized_model(
            self.raw_target_model, config.parameterization, ideal_model=self.raw_target_model
        )
        self.processor_spec = (
            self.modelpack.processor_spec() if hasattr(self.modelpack, "processor_spec") else None
        )
        self.prep_fiducials = self.modelpack.prep_fiducials()
        self.meas_fiducials = self.modelpack.meas_fiducials()
        self.germs = self.modelpack.germs()

        self.truth_model = self._build_truth_model(config)
        design_source = self.processor_spec if self.processor_spec is not None else self.target_model
        self.design = StandardGSTDesign(
            design_source,
            self.prep_fiducials,
            self.meas_fiducials,
            self.germs,
            list(config.max_lengths),
        )
        self.circuits = list(self.design.all_circuits_needing_data)
        self.outcomes_per_circuit = len(self.truth_model.probabilities(self.circuits[0]))
        self.base_model = _make_parameterized_model(
            self.target_model, config.parameterization, ideal_model=self.raw_target_model
        )
        # base_model ONLY -- it is what the POUNDERS oracle differentiates. truth_model and
        # target_model keep pyGSTi's default on purpose: target_model seeds the LM protocol,
        # and switching its simulator wrecks that fit (see forward_simulator above). Scoring is
        # unaffected either way -- the simulators agree on probabilities to 2e-15 and score a
        # fixed estimate bit-identically.
        if config.forward_simulator:
            try:
                self.base_model.sim = config.forward_simulator
                print(f"[SIM] base_model.sim = {config.forward_simulator!r} "
                      f"(POUNDERS oracle only; truth/target keep the pyGSTi default)")
            except Exception as exc:  # a model kind that rejects it -- keep the default
                print(f"[SIM] could not set sim={config.forward_simulator!r}: {exc!r}")

        self.x0 = np.asarray(self.base_model.to_vector(), dtype=float).reshape(-1)
        self.n = int(self.x0.size)
        self.m = int(len(self.circuits) * self.outcomes_per_circuit)
        self.dataset = None
        self.shots_per_circuit = None

    def _build_truth_model(self, config: ExperimentConfig):
        """Construct the synthetic 'device' (data-generation) model per config.noise_model.

        "depolarize_rotate" (default): depolarizing + optional heterogeneous coherent
            over-rotations (config.coherent_max_rotate).
        "lindblad_hard": a rich per-gate Lindblad error generator spanning all four
            elementary sectors -- coherent (H), stochastic (S), correlated stochastic (C,
            i.e. noise axes tilted out of the computational frame), and T1 amplitude
            damping (S_X + S_Y + A_(X,Y), decay toward |0>) -- drawn randomly per driven
            gate (heterogeneous), plus depolarizing SPAM.  All terms are Lindblad, so the
            truth stays in-model for a CPTPLND fit.
        """
        if config.noise_model == "lindblad_hard":
            import pygsti.baseobjs as _bo
            from pygsti.models import modelconstruction as _mc
            if self.processor_spec is None:
                raise RuntimeError("noise_model='lindblad_hard' requires a processor_spec.")
            rng = np.random.default_rng(self.truth_seed)
            # Driven gates only (the idle does not accept Lindblad coeffs via this API).
            gate_names = [n for n in self.processor_spec.gate_names if "idle" not in n.lower()]
            coeffs = {}
            for name in gate_names:
                terms = {}
                for pauli in ("X", "Y", "Z"):
                    terms[("H", pauli)] = float(rng.uniform(-config.lindblad_h_max, config.lindblad_h_max))
                    terms[("S", pauli)] = float(rng.uniform(0.0, config.lindblad_s_max))
                # T1 amplitude damping toward |0> (Bloch offset along +Z).  Damping at
                # rate `a` is not a lone A term: it is S_X = S_Y = a/4 (the transverse
                # decay that accompanies relaxation) plus A_(X,Y) = -a/4 (the non-unital
                # slide; the minus sign puts the ground state at +Z).  Adding to the
                # existing S draws rather than overwriting keeps CP headroom, since the
                # pair condition is |A_(P,Q)| <= sqrt(S_P * S_Q).
                # NB: pyGSTi A labels take TWO basis elements and only ascending pairs
                # register -- a single-index ("A", "Y") is silently discarded (no error).
                a = float(rng.uniform(0.0, config.lindblad_a_max))
                terms[("S", "X")] += a / 4.0
                terms[("S", "Y")] += a / 4.0
                terms[("A", "X", "Y")] = -a / 4.0

                # Correlated stochastic (C) errors: the off-diagonal of the stochastic
                # block -- Pauli error channels that fluctuate together rather than
                # independently.  Physically this is what a tilted noise axis (e.g. drive
                # phase miscalibration) looks like in the X/Y/Z frame, and it enters at
                # FIRST order in the tilt while the spurious S term is only second order.
                # Drawn as a correlation coefficient rho = C_PQ / sqrt(S_P * S_Q) so it
                # scales to whatever CP headroom each gate has: an absolute range would
                # violate CP for ~50% of gates, since the pair bound is
                # C^2 + A^2 <= S_P * S_Q and the S draws reach 0.
                if config.lindblad_rho_max > 0.0:
                    s_diag = {p: terms[("S", p)] for p in _PAULI_AXES}
                    a_off = {("X", "Y"): -a / 4.0}
                    c_off = {}
                    for p, q in (("X", "Y"), ("X", "Z"), ("Y", "Z")):
                        rho = float(rng.uniform(-config.lindblad_rho_max, config.lindblad_rho_max))
                        c_off[(p, q)] = rho * math.sqrt(s_diag[p] * s_diag[q])
                    scale = _cp_safe_c_scale(s_diag, c_off, a_off)
                    for (p, q), value in c_off.items():
                        terms[("C", p, q)] = scale * value

                coeffs[_bo.Label(name, 0)] = terms
            model = _mc.create_explicit_model(self.processor_spec, lindblad_error_coeffs=coeffs)
            model.set_all_parameterizations("full")   # make SPAM mutable
            if config.spam_noise:
                # NB: depolarize() RETURNS a new model -- the return value must be kept.
                # Discarding it silently leaves SPAM ideal, which also makes several
                # circuits have truth probability exactly 0; those can never produce a
                # nonzero count, so they sit at the 1/variance_floor weight forever.
                model = model.depolarize(spam_noise=config.spam_noise)
            return model

        # default: depolarizing + optional heterogeneous coherent over-rotations
        model = self.raw_target_model.depolarize(
            op_noise=config.op_noise,
            spam_noise=config.spam_noise,
            seed=self.truth_seed,
        )
        if float(getattr(config, "coherent_max_rotate", 0.0) or 0.0) > 0.0:
            model = model.rotate(
                max_rotate=float(config.coherent_max_rotate),
                seed=self.truth_seed + 777,
            )
        return model

    def normalize_shots(self, shots) -> np.ndarray:
        if np.isscalar(shots):
            values = np.full(len(self.circuits), float(shots), dtype=float)
        else:
            values = np.asarray(shots, dtype=float).reshape(-1)
        if values.size != len(self.circuits):
            raise ValueError(f"Expected {len(self.circuits)} shot counts, got {values.size}.")
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Shot counts must be positive finite values.")
        if not np.allclose(values, np.round(values)):
            raise ValueError("Shot counts must be integer-valued.")
        return np.round(values).astype(int)

    def simulate_dataset(self, shots, seed: int | None = None, circuits=None):
        circuits = self.circuits if circuits is None else list(circuits)
        if np.isscalar(shots):
            shot_vector = np.full(len(circuits), int(shots), dtype=int)
        else:
            shot_vector = np.asarray(shots, dtype=int).reshape(-1)
        if shot_vector.size != len(circuits):
            raise ValueError("The shot vector must have one entry per supplied circuit.")
        return self.pygsti.data.simulate_data(
            self.truth_model,
            circuits,
            num_samples=[int(v) for v in shot_vector],
            sample_error=self.config.sample_error,
            seed=int(self.data_seed if seed is None else seed),
        )

    def set_uniform_dataset(self, shots: int) -> None:
        self.shots_per_circuit = self.normalize_shots(shots)
        self.dataset = self.simulate_dataset(self.shots_per_circuit)

    def copy_model_at_x(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n:
            raise ValueError(f"Expected parameter vector length {self.n}, got {x.size}.")
        model = self.base_model.copy()
        model.from_vector(x, close=False)
        return model

    def model_vector_and_jacobian(self, x, circuits=None, data_for_outcomes=None):
        circuits = self.circuits if circuits is None else list(circuits)
        model = self.copy_model_at_x(x)
        probabilities_by_circuit = model.bulk_probabilities(circuits)
        derivatives_by_circuit = model.sim.bulk_dprobs(circuits)
        p_values, jacobian_rows, labels = [], [], []
        for circuit in circuits:
            outcomes = (
                list(data_for_outcomes[circuit].outcomes)
                if data_for_outcomes is not None
                else list(model.probabilities(circuit).keys())
            )
            for outcome in outcomes:
                p_values.append(float(probabilities_by_circuit[circuit].get(outcome, 0.0)))
                jacobian_rows.append(
                    np.asarray(derivatives_by_circuit[circuit][outcome], dtype=float)
                )
                labels.append((circuit, outcome))
        p = np.asarray(p_values, dtype=float)
        jac = np.vstack(jacobian_rows) if jacobian_rows else np.empty((0, self.n))
        return p, jac, labels

    def model_probability_vector(self, x, circuits=None, data_for_outcomes=None):
        """Return model probabilities without paying for an unused Jacobian."""
        circuits = self.circuits if circuits is None else list(circuits)
        data_for_outcomes = self.dataset if data_for_outcomes is None else data_for_outcomes
        model = self.copy_model_at_x(x)
        probabilities_by_circuit = model.bulk_probabilities(circuits)
        p_values, labels = [], []
        for circuit in circuits:
            outcomes = (
                list(data_for_outcomes[circuit].outcomes)
                if data_for_outcomes is not None
                else list(probabilities_by_circuit[circuit].keys())
            )
            for outcome in outcomes:
                p_values.append(float(probabilities_by_circuit[circuit].get(outcome, 0.0)))
                labels.append((circuit, outcome))
        return np.asarray(p_values, dtype=float), labels

    def _model_evaluation_for_residual_mask(
        self,
        x,
        residual_mask=None,
        model_cache=None,
    ):
        """Evaluate all probabilities but only the requested probability derivatives.

        POUNDERS keeps the global residual numbering, while FPR supplies the rows
        needed by the local model.  The probability vector remains full length so
        the full objective can still be reported.  The Jacobian is compact and its
        rows are ordered by ``active_residual_indices``.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if residual_mask is None:
            keep = np.ones(self.m, dtype=bool)
        else:
            keep = np.asarray(residual_mask, dtype=bool).reshape(-1)
            if keep.size != self.m:
                raise ValueError(
                    f"residual_mask has length {keep.size}, expected {self.m}."
                )
        active_rows = np.flatnonzero(keep)
        if active_rows.size == 0:
            raise ValueError("The GST Jacobian cannot be evaluated on an empty mask.")

        _, _, _, _, labels = self.data_vector()
        if len(labels) != self.m:
            raise ValueError(
                f"GST data produced {len(labels)} residual labels, expected {self.m}."
            )

        cache_reusable = (
            isinstance(model_cache, dict)
            and np.array_equal(np.asarray(model_cache.get("x", []), dtype=float), x)
            and model_cache.get("labels") == labels
        )
        cached_rows = (
            np.asarray(model_cache.get("active_residual_indices", []), dtype=int)
            if cache_reusable
            else np.empty(0, dtype=int)
        )
        cached_jac = (
            np.asarray(model_cache.get("probability_jacobian", []), dtype=float)
            if cache_reusable
            else np.empty((0, self.n), dtype=float)
        )
        if cached_jac.shape != (cached_rows.size, self.n):
            cached_rows = np.empty(0, dtype=int)
            cached_jac = np.empty((0, self.n), dtype=float)

        model = None
        if cache_reusable:
            p = np.asarray(model_cache["p"], dtype=float)
            if p.shape != (self.m,):
                cache_reusable = False

        if not cache_reusable:
            model = self.copy_model_at_x(x)
            probabilities_by_circuit = model.bulk_probabilities(self.circuits)
            p = np.asarray(
                [
                    float(probabilities_by_circuit[circuit].get(outcome, 0.0))
                    for circuit, outcome in labels
                ],
                dtype=float,
            )
            cached_rows = np.empty(0, dtype=int)
            cached_jac = np.empty((0, self.n), dtype=float)

        cached_row_position = {
            int(row): position for position, row in enumerate(cached_rows.tolist())
        }
        missing_rows = np.asarray(
            [row for row in active_rows if int(row) not in cached_row_position],
            dtype=int,
        )
        derivatives_by_circuit = {}
        if missing_rows.size:
            if model is None:
                model = self.copy_model_at_x(x)
            missing_circuit_indices = np.unique(
                missing_rows // int(self.outcomes_per_circuit)
            )
            missing_circuits = [
                self.circuits[int(index)] for index in missing_circuit_indices
            ]
            derivatives_by_circuit = model.sim.bulk_dprobs(missing_circuits)

        jacobian_rows = []
        for row in active_rows:
            row = int(row)
            cached_position = cached_row_position.get(row)
            if cached_position is not None:
                jacobian_rows.append(cached_jac[cached_position])
                continue
            circuit, outcome = labels[row]
            jacobian_rows.append(
                np.asarray(derivatives_by_circuit[circuit][outcome], dtype=float)
            )
        probability_jacobian = np.vstack(jacobian_rows)

        cache = {
            "x": x.copy(),
            "p": p,
            "labels": labels,
            "active_residual_indices": active_rows.copy(),
            "probability_jacobian": probability_jacobian,
        }
        cache_stats = {
            "probabilities_reused": bool(cache_reusable),
            "jacobian_rows_reused": int(active_rows.size - missing_rows.size),
            "jacobian_rows_computed": int(missing_rows.size),
        }
        return p, probability_jacobian, labels, active_rows, cache, cache_stats

    def data_vector(self, data=None, circuits=None):
        data = self.dataset if data is None else data
        circuits = self.circuits if circuits is None else list(circuits)
        f_values, var_values, counts, totals, labels = [], [], [], [], []
        smoothing = float(getattr(self.config, "variance_smoothing", 0.0) or 0.0)
        for circuit in circuits:
            row = data[circuit]
            total = float(row.total)
            for outcome in row.outcomes:
                f = float(row.fractions.get(outcome, 0.0))
                f_clip = min(max(f, 0.0), 1.0)
                count = float(row.counts.get(outcome, 0.0))
                if total <= 0:
                    var = np.nan
                elif smoothing > 0.0:
                    # smooth the VARIANCE only; the residual still uses the raw f
                    f_var = (count + smoothing) / (total + 2.0 * smoothing)
                    var = f_var * (1.0 - f_var) / total
                else:
                    var = f_clip * (1.0 - f_clip) / total
                f_values.append(f)
                var_values.append(var)
                counts.append(count)
                totals.append(total)
                labels.append((circuit, outcome))
        return (
            np.asarray(f_values, dtype=float),
            np.asarray(var_values, dtype=float),
            np.asarray(counts, dtype=float),
            np.asarray(totals, dtype=float),
            labels,
        )

    def _logl_objfn(self, freqs):
        """pyGSTi's Poisson-picture delta-logl -- the exact objective its LM minimises.

        Returns a ``RawPoissonPicDeltaLogLFunction`` whose ``lsvec``/``dlsvec`` give
        the residual and d(residual)/dp.  ``min_prob_clip`` has to stay below the
        smallest nonzero frequency, and adaptive shots can shrink that bound as the
        run proceeds, so rebuild whenever it tightens.  The objective already moves
        under POUNDERS every time shots are added, so this is not a new discontinuity.
        """
        freqs = np.asarray(freqs, dtype=float)
        clip = float(self.config.logl_min_prob_clip)
        nonzero = freqs[freqs > 0.0]
        if nonzero.size:
            clip = min(clip, 0.5 * float(nonzero.min()))
        cached = getattr(self, "_logl_raw", None)
        if cached is None or cached[0] != clip:
            from pygsti.objectivefns.objectivefns import RawPoissonPicDeltaLogLFunction

            if cached is not None:
                print(f"[LOGL] min_prob_clip {cached[0]:.3g} -> {clip:.3g}")
            cached = (
                clip,
                RawPoissonPicDeltaLogLFunction(
                    regularization={
                        "min_prob_clip": clip,
                        "radius": float(self.config.logl_radius),
                    }
                ),
            )
            self._logl_raw = cached
        return cached[1]

    def oracle(
        self,
        x,
        return_info: bool = False,
        residual_mask=None,
        model_cache=None,
    ):
        if self.dataset is None:
            raise RuntimeError("Set a dataset before calling the GST oracle.")
        (
            p,
            jac,
            p_labels,
            active_rows,
            evaluation_cache,
            cache_stats,
        ) = self._model_evaluation_for_residual_mask(
            x,
            residual_mask=residual_mask,
            model_cache=model_cache,
        )
        f, var_f, counts, totals, f_labels = self.data_vector()
        if p_labels != f_labels:
            raise ValueError("Model and data outcome order differs.")
        if self.config.objective == "poisson_logl":
            # residual = sqrt(terms), so sum(residual**2) == deltaLogL.  dlsvec is
            # d(residual)/dp elementwise; chain it onto dp/dx for the active rows.
            raw = self._logl_objfn(f)
            residual = raw.lsvec(p, counts, totals, f)
            jac = raw.dlsvec(p, counts, totals, f)[active_rows, None] * jac
        else:
            residual = p - f
            if self.config.objective == "weighted_least_squares":
                if self.config.variance_source != "data":
                    raise NotImplementedError(
                        "The standalone seed runner currently supports data-weighted WLS, "
                        "which is the objective used by the notebook experiments."
                    )
                sigma = np.sqrt(np.maximum(var_f, self.config.variance_floor))
                residual = residual / sigma
                jac = jac / sigma[active_rows, None]
            elif self.config.objective != "least_squares":
                raise NotImplementedError(
                    "Use least_squares, weighted_least_squares, or poisson_logl."
                )
        info = {
            "p": p,
            "f": f,
            "counts": counts,
            "totals": totals,
            "var_f": var_f,
            "labels": p_labels,
            "circuits": self.circuits,
            "shots_per_circuit": self.shots_per_circuit,
            "jacobian_residual_indices": active_rows,
            "probability_jacobian": evaluation_cache["probability_jacobian"],
            "_model_cache": evaluation_cache,
            "model_cache_stats": cache_stats,
        }
        output = (residual, jac.T)
        return (*output, info) if return_info else output

    def probability_and_jacobian(self, x, circuits):
        p, jac, _ = self.model_vector_and_jacobian(x, circuits=circuits)
        return p, jac

    def residual_vector(self, x, data=None):
        """Return the configured full residual without calculating derivatives."""
        data = self.dataset if data is None else data
        p, p_labels = self.model_probability_vector(
            x,
            data_for_outcomes=data,
        )
        f, var_f, counts, totals, f_labels = self.data_vector(data=data)
        if p_labels != f_labels:
            raise ValueError("Model and data outcome order differs.")
        if self.config.objective == "poisson_logl":
            return self._logl_objfn(f).lsvec(p, counts, totals, f)
        residual = p - f
        if self.config.objective == "weighted_least_squares":
            sigma = np.sqrt(np.maximum(var_f, self.config.variance_floor))
            residual = residual / sigma
        elif self.config.objective != "least_squares":
            raise NotImplementedError(
                "Use least_squares, weighted_least_squares, or poisson_logl."
            )
        return residual

    def probability_diagnostics(self, p: np.ndarray) -> dict[str, Any]:
        p = np.asarray(p, dtype=float)
        tol = float(self.config.probability_tolerance)
        return {
            "min_probability": float(np.min(p)),
            "max_probability": float(np.max(p)),
            "num_probability_below_zero": int(np.sum(p < -tol)),
            "num_probability_above_one": int(np.sum(p > 1.0 + tol)),
        }

    def likelihood_diagnostics(self, x, data=None) -> dict[str, Any]:
        data = self.dataset if data is None else data
        p, p_labels = self.model_probability_vector(x, data_for_outcomes=data)
        f, _, counts, _, f_labels = self.data_vector(data=data)
        if p_labels != f_labels:
            raise ValueError("Model and data labels differ in likelihood diagnostic.")
        prob_diag = self.probability_diagnostics(p)
        if prob_diag["num_probability_below_zero"] or prob_diag["num_probability_above_one"]:
            raise ValueError(f"Nonphysical probabilities from a CPTP model: {prob_diag}")
        p_safe = np.clip(p, 1e-15, 1.0)
        f_safe = np.clip(f, 1e-15, 1.0)
        logl_model = float(np.sum(counts * np.log(p_safe)))
        logl_max = float(np.sum(np.where(counts > 0, counts * np.log(f_safe), 0.0)))
        two_delta_logl = float(2.0 * (logl_max - logl_model))
        independent_data = len(self.circuits) * (self.outcomes_per_circuit - 1)
        nongauge = _num_nongauge(self.base_model, self.n)
        dof = int(independent_data - nongauge)
        n_sigma = (
            float((two_delta_logl - dof) / np.sqrt(2.0 * dof))
            if dof > 0
            else float("nan")
        )
        return {
            **prob_diag,
            "logl_model": logl_model,
            "logl_max": logl_max,
            "two_delta_logl": two_delta_logl,
            "num_independent_data": int(independent_data),
            "num_nongauge_params": int(nongauge),
            "dof": dof,
            "n_sigma": n_sigma,
            "n_sigma_less_than_1": bool(n_sigma < 1.0) if np.isfinite(n_sigma) else False,
        }

    def _gaugeopt_kwargs(self):
        """Weighting/metric for gaugeopt_to_target, from config.

        item_weights scales the terms of the gauge objective. Passing 'gates' and 'spam' sets
        the default for each kind; individual labels could override those but are not used
        here. Weighting SPAM below the gates stops a poorly-calibrated SPAM estimate from
        dragging the gauge and inflating the gate error that the diamond norm reports."""
        kw = {}
        w = self.config.gaugeopt_spam_weight
        if w is not None:
            kw["item_weights"] = {"gates": 1.0, "spam": float(w)}
        if self.config.gaugeopt_metric != "frobenius":
            kw["gates_metric"] = self.config.gaugeopt_metric
            kw["spam_metric"] = self.config.gaugeopt_metric
        return kw

    def aligned_error_metrics(self, estimate_model, reference_model, reference_name: str):
        from pygsti.tools import entanglement_infidelity

        reference = reference_model.copy()
        estimate = estimate_model.copy()
        reference.set_all_parameterizations("full")
        estimate.set_all_parameterizations("full")
        with contextlib.redirect_stdout(io.StringIO()):
            aligned = self.pygsti.gaugeopt_to_target(estimate, reference,
                                                     **self._gaugeopt_kwargs())

        d_hilbert = int(round(math.sqrt(reference.dim)))
        gate_rows = []
        for label in reference.operations.keys():
            ent = float(
                entanglement_infidelity(
                    aligned.operations[label], reference.operations[label], reference.basis
                )
            )
            row = {
                "reference": reference_name,
                "gate": str(label),
                "entanglement_infidelity": ent,
                "average_gate_infidelity": float(d_hilbert * ent / (d_hilbert + 1.0)),
            }
            if self.config.report_diamond_distance:
                try:
                    from pygsti.tools import diamonddist

                    row["diamond_distance"] = float(
                        diamonddist(
                            aligned.operations[label].to_dense(),
                            reference.operations[label].to_dense(),
                            reference.basis,
                        )
                    )
                except Exception:
                    row["diamond_distance"] = float("nan")

            # Gauge-INVARIANT companions, computed on `estimate` -- the raw fit, before
            # gaugeopt_to_target. Gauge acts on gates by conjugation (G -> S G S^-1), so a
            # gate's eigenvalues do not move; these two compare eigenvalue spectra and so
            # need no gauge fixing at all. They are what to quote when the gauge weighting
            # is itself in question, since entanglement_infidelity and diamonddist above
            # both depend on the item_weights passed to gaugeopt.
            #
            # WHICH OF THESE IS AN ERROR METRIC:
            #   eigenvalue_diamondnorm            -- yes. max|evA - evB| over matched pairs,
            #                                        so >= 0, zero for a perfect estimate,
            #                                        and sensitive to both rotation-angle
            #                                        and decay error. Prefer this one.
            #   eigenvalue_nonunitary_diamondnorm -- yes, but partial: it matches on
            #                                        | |evA| - |evB| |, i.e. decay only,
            #                                        blind to rotation-angle error.
            #   eigenvalue_nonunitarity_excess    -- NO. This is
            #                                        (d2-1)/d2 * (1 - sqrt(U)), which is
            #                                        SIGNED: negative whenever the estimate
            #                                        is LESS decoherent than the truth. It
            #                                        says which way the estimate misses, not
            #                                        how far. Never average it as an error or
            #                                        take a ratio of it -- both are
            #                                        meaningless once it crosses zero, and it
            #                                        does cross zero on this data.
            #
            # eigenvalue_entanglement_infidelity is deliberately NOT used. It sums
            # conj(evB_j)*evA_i, which with A == B gives sum|ev|^2 < d2 for any non-unitary
            # gate -- so it is NONZERO for a perfect estimate. On this truth model its floor
            # is 7.6e-03 while the actual estimation error is ~1e-05, i.e. the reported
            # number is >99% floor and it ranks every arm identically. The nonunitary_
            # variant divides that floor out and reads 0 for a perfect estimate; verified by
            # evaluating truth against itself. Same trap applies to
            # eigenvalue_avg_gate_infidelity (floor 5.1e-03).
            #
            # They are NOT drop-in replacements. Each is a per-gate spectral comparison, so
            # it misses error in how gates relate to EACH OTHER: a gauge must be one common
            # S for the whole gate set, and matching every gate's spectrum individually does
            # not imply such an S exists. Read them as a gauge-free lower bound on the error,
            # never as the error itself.
            try:
                from pygsti.report.reportables import (
                    eigenvalue_nonunitary_entanglement_infidelity as _ev_inf,
                    eigenvalue_diamondnorm as _ev_dn,
                    eigenvalue_nonunitary_diamondnorm as _ev_dn_nu,
                )

                a = np.asarray(estimate.operations[label].to_dense())
                b = np.asarray(reference.operations[label].to_dense())
                row["eigenvalue_nonunitarity_excess"] = float(_ev_inf(a, b, reference.basis))
                row["eigenvalue_diamondnorm"] = float(_ev_dn(a, b, reference.basis))
                row["eigenvalue_nonunitary_diamondnorm"] = float(_ev_dn_nu(a, b, reference.basis))
            except Exception:
                for _k in ("eigenvalue_nonunitarity_excess", "eigenvalue_diamondnorm",
                           "eigenvalue_nonunitary_diamondnorm"):
                    row[_k] = float("nan")
            gate_rows.append(row)

        # SPAM error, two ways.
        #
        # vector_l2_error is the Frobenius distance between the superket coefficient vectors.
        # It is kept because older runs report it and the notebooks key on it, but it is a
        # weak metric: an L2 norm on Pauli-basis coefficients is basis-dependent and bounds
        # no experiment's ability to tell the two apart.
        #
        # The trace-distance columns are the operational ones, and follow pyGSTi's own report
        # conventions (report.reportables.vec_trace_diff / tools.povm_jtracedist):
        #   prep  -- trace distance between the density matrices, = the largest probability
        #            difference any measurement could produce.
        #   POVM  -- Jamiolkowski trace distance of the POVM viewed as a MAP from states to
        #            classical outcomes. A single effect is not a density matrix, so trace
        #            distance does not apply to one; the POVM as a whole is the right object,
        #            which is why this is one row per POVM rather than per outcome.
        # Both are still computed on the gauge-optimised model, so they remain gauge-dependent
        # -- SPAM has no gauge-invariant analogue (gauge acts one-sidedly on preps/effects).
        from pygsti.report.reportables import vec_trace_diff as _vec_td
        from pygsti.tools.optools import povm_jtracedist as _povm_jtd

        spam_rows = []
        for label in reference.preps.keys():
            try:
                td = float(_vec_td(_dense(aligned.preps[label]),
                                   _dense(reference.preps[label]), reference.basis))
            except Exception:
                td = float("nan")
            spam_rows.append(
                {
                    "reference": reference_name,
                    "member_type": "prep",
                    "member": str(label),
                    "vector_l2_error": float(
                        np.linalg.norm(_dense(aligned.preps[label]) - _dense(reference.preps[label]))
                    ),
                    "tracedist": td,
                }
            )
        for povm_label in reference.povms.keys():
            try:
                jtd = float(_povm_jtd(aligned, reference, povm_label))
            except Exception:
                jtd = float("nan")
            spam_rows.append(
                {
                    "reference": reference_name,
                    "member_type": "povm",
                    "member": str(povm_label),
                    "vector_l2_error": float("nan"),   # per-outcome rows carry the L2
                    "tracedist": jtd,
                }
            )
        for povm_label in reference.povms.keys():
            for outcome in reference.povms[povm_label].keys():
                spam_rows.append(
                    {
                        "reference": reference_name,
                        "member_type": "povm_effect",
                        "member": f"{povm_label}:{outcome}",
                        "vector_l2_error": float(
                            np.linalg.norm(
                                _dense(aligned.povms[povm_label][outcome])
                                - _dense(reference.povms[povm_label][outcome])
                            )
                        ),
                    }
                )

        summary = {
            f"mean_gate_entanglement_infidelity_to_{reference_name}": float(
                np.mean([row["entanglement_infidelity"] for row in gate_rows])
            ),
            f"mean_gate_average_infidelity_to_{reference_name}": float(
                np.mean([row["average_gate_infidelity"] for row in gate_rows])
            ),
            f"mean_spam_vector_l2_error_to_{reference_name}": float(
                np.nanmean([row["vector_l2_error"] for row in spam_rows])
            ),
            # mean over the prep trace distances and the POVM Jamiolkowski trace distance;
            # nanmean because the per-outcome effect rows carry no trace distance
            f"mean_spam_tracedist_to_{reference_name}": float(
                np.nanmean([row.get("tracedist", float("nan")) for row in spam_rows])
            ),
        }

        # nanmean, matching how the diamond aggregate below treats a failed gate
        for key in ("eigenvalue_nonunitarity_excess", "eigenvalue_diamondnorm",
                    "eigenvalue_nonunitary_diamondnorm"):
            vals = np.asarray([row.get(key, float("nan")) for row in gate_rows], dtype=float)
            summary[f"mean_gate_{key}_to_{reference_name}"] = (
                float(np.nanmean(vals)) if vals.size and not np.all(np.isnan(vals))
                else float("nan")
            )

        # Aggregate diamond distance here so every caller gets it in summary.json rather
        # than having to re-read final_gate_errors.csv.  nanmean matches how the analysis
        # notebooks average the CSV column (pandas skips NaN); all-NaN stays NaN.
        if self.config.report_diamond_distance:
            dd = np.asarray([row.get("diamond_distance", float("nan")) for row in gate_rows],
                            dtype=float)
            summary[f"mean_gate_diamond_distance_to_{reference_name}"] = (
                float(np.nanmean(dd)) if dd.size and not np.all(np.isnan(dd)) else float("nan")
            )
        return summary, gate_rows, spam_rows


def _build_fpr(problem: GSTProblem, config: ExperimentConfig):
    from near_minimal_fpr_reduction import make_fpr_reduction_mask_function

    return make_fpr_reduction_mask_function(
        base_model=problem.base_model,
        all_circuits=problem.circuits,
        processor_spec=problem.processor_spec,
        prep_fiducials=problem.prep_fiducials,
        meas_fiducials=problem.meas_fiducials,
        germs=problem.germs,
        max_lengths=list(config.max_lengths),
        outcomes_per_circuit=problem.outcomes_per_circuit,
        stage1_method=config.fpr_stage1_method,
        stage2_method=config.fpr_stage2_method,
        verbose=config.fpr_verbose,
    )


def _build_infidelity_metric(problem, config):
    """Return a callable(state) -> M for L-optimality: M = Hessian of mean-gate infidelity
    around the CURRENT ESTIMATE (state["x"]).  Uses only the estimate + fit template -- NO
    ground truth -> realizable on hardware.  Cached; rebuilt every l_metric_rebuild_every iters
    (each rebuild is ~2n gauge-opts with l_metric_diagonal=True, so it is not free)."""
    import adaptive_shots as ashots
    from pygsti.tools import entanglement_infidelity as _ent_infid

    cache = {"M": None, "built_at": -(10 ** 9)}

    def metric_M(state):
        iteration = int(state.get("iteration", 0))
        if cache["M"] is not None and iteration - cache["built_at"] < int(config.l_metric_rebuild_every):
            return cache["M"]
        x_ref = np.asarray(state["x"], dtype=float).reshape(-1)
        reference = problem.base_model.copy()
        reference.from_vector(x_ref, close=False)
        reference.set_all_parameterizations("full")
        ref_ops, ref_basis = reference.operations, reference.basis

        def infid_to_reference(theta):
            model = problem.base_model.copy()
            model.from_vector(np.asarray(theta, dtype=float), close=False)
            model.set_all_parameterizations("full")
            with contextlib.redirect_stdout(io.StringIO()):
                aligned = problem.pygsti.gaugeopt_to_target(model, reference,
                                                            **problem._gaugeopt_kwargs())
            return float(np.mean([
                float(_ent_infid(aligned.operations[l], ref_ops[l], ref_basis))
                for l in ref_ops.keys()
            ]))

        M = ashots.infidelity_metric_hessian(
            infid_to_reference, x_ref,
            eps=float(config.l_metric_eps),
            diagonal_only=bool(config.l_metric_diagonal),
            psd_floor=0.0,
        )
        cache.update(M=M, built_at=iteration)
        print(f"  [L] rebuilt infidelity metric at estimate (iter {iteration}): "
              f"trace {np.trace(M):.3e}")
        return M

    return metric_M


def _build_adaptive_hook(problem, config, fpr_reduction, running_shots, event_history):
    import adaptive_shots
    import adaptive_shot_hook

    # Reload adaptive_shots BEFORE the hook.  adaptive_shot_hook does
    # `import adaptive_shots as ashots` at module level, so reloading only the hook
    # re-runs that import against the module already cached in sys.modules and picks
    # up nothing.  In a long-running kernel that silently keeps stale allocator code
    # while the run otherwise looks completely normal.  Same order as
    # notebook_adaptive_shot_cell.py.
    adaptive_shots = importlib.reload(adaptive_shots)
    adaptive_shot_hook = importlib.reload(adaptive_shot_hook)
    circuit_to_index = {circuit: i for i, circuit in enumerate(problem.circuits)}
    outcome_labels = list(problem.dataset.outcome_labels)
    running_counts = {
        circuit: {
            outcome: int(round(value))
            for outcome, value in problem.dataset[circuit].counts.items()
        }
        for circuit in problem.circuits
    }
    increment_index = [0]
    revealed_indices: set[int] = set()

    def circuits_from_mask(mask):
        if mask is None:
            active = np.arange(len(problem.circuits), dtype=int)
        else:
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            row_circuit = np.repeat(
                np.arange(len(problem.circuits)), problem.outcomes_per_circuit
            )
            active = np.unique(row_circuit[mask])
        circuits = [problem.circuits[int(i)] for i in active]
        row_index = np.repeat(np.arange(len(circuits)), problem.outcomes_per_circuit)
        return circuits, row_index

    def current_shots(circuits):
        per_circuit = np.asarray(
            [running_shots[circuit_to_index[circuit]] for circuit in circuits], dtype=float
        )
        return np.repeat(per_circuit, problem.outcomes_per_circuit)

    def active_indices_from_state(state):
        circuits, _ = circuits_from_mask(state.get("fpr_mask"))
        return np.asarray([circuit_to_index[circuit] for circuit in circuits], dtype=int)

    budget_cap = int(config.adaptive_total_shot_budget)
    _accounting_frozen = {"on": False, "value": 0}

    def accounted_shots(state):
        """Revealed-shot cost through this iteration, HARD-CAPPED at the budget.

        The per-batch clip in ``make_adaptive_shot_hook`` already stops *requested*
        shots from exceeding the budget. But the online FPR union keeps growing even
        after the budget is spent, and each newly revealed circuit drags its
        pre-measured baseline shots into the tally -- pushing ``accounted`` past the
        budget. To guarantee adaptive never *reports* more than the budget, once the
        cost reaches it we FREEZE the charged circuit set and clamp the value to the
        budget. Later union growth then cannot drag it up.
        """
        if _accounting_frozen["on"]:
            return _accounting_frozen["value"]
        active_indices = active_indices_from_state(state)
        revealed_indices.update(int(index) for index in active_indices)
        if not revealed_indices:
            return 0
        revealed = np.fromiter(sorted(revealed_indices), dtype=int)
        total = int(np.sum(running_shots[revealed]))
        if total >= budget_cap:
            total = budget_cap
            _accounting_frozen["on"] = True
            _accounting_frozen["value"] = total
        return total

    def rebuild_dataset():
        dataset = problem.pygsti.data.DataSet(outcome_labels=outcome_labels)
        for circuit, count_dict in running_counts.items():
            dataset.add_count_dict(circuit, count_dict)
        dataset.done_adding_data()
        problem.dataset = dataset

    def add_shots(circuits, extra_per_circuit):
        changed = [
            (circuit, int(extra))
            for circuit, extra in zip(circuits, extra_per_circuit)
            if int(extra) > 0
        ]
        if not changed:
            return
        increment_index[0] += 1
        incremental = problem.simulate_dataset(
            [extra for _, extra in changed],
            seed=problem.data_seed + 1000 + increment_index[0],
            circuits=[circuit for circuit, _ in changed],
        )
        for circuit, _ in changed:
            for outcome, count in incremental[circuit].counts.items():
                running_counts[circuit][outcome] = (
                    running_counts[circuit].get(outcome, 0) + int(round(count))
                )
            running_shots[circuit_to_index[circuit]] += int(round(incremental[circuit].total))
        rebuild_dataset()

    if config.adaptive_schedule == "rho_gated_geometric":
        schedule = adaptive_shot_hook.rho_gated_geometric_budget(
            base=config.adaptive_schedule_base,
            n0=config.adaptive_schedule_n0,
            n_max=config.adaptive_schedule_n_max,
            rho_band=config.adaptive_rho_band,
            initial_budget=config.adaptive_initial_topup,
        )
    elif config.adaptive_schedule == "rho_gated_delta_inverse_square":
        schedule = adaptive_shot_hook.rho_gated_delta_inverse_square_budget(
            constant=config.adaptive_delta_inverse_square_constant,
            n_min=config.adaptive_delta_inverse_square_n_min,
            delta_floor=config.adaptive_delta_floor,
            rho_band=config.adaptive_rho_band,
            initial_budget=config.adaptive_initial_topup,
        )
    elif config.adaptive_schedule == "lazy_delta_inverse_square":
        schedule = adaptive_shot_hook.lazy_delta_inverse_square_budget(
            base=config.adaptive_schedule_base,
            n0=config.adaptive_schedule_n0,
            constant=config.adaptive_delta_inverse_square_constant,
            n_min=config.adaptive_delta_inverse_square_n_min,
            delta_floor=config.adaptive_delta_floor,
            initial_budget=config.adaptive_initial_topup,
        )
    elif config.adaptive_schedule == "lazy_delta_inverse_fourth":
        schedule = adaptive_shot_hook.lazy_delta_inverse_fourth_budget(
            base=config.adaptive_schedule_base,
            n0=config.adaptive_schedule_n0,
            constant=config.adaptive_delta_inverse_fourth_constant,
            n_min=config.adaptive_delta_inverse_fourth_n_min,
            delta_floor=config.adaptive_delta_floor,
            initial_budget=config.adaptive_initial_topup,
        )
    elif config.adaptive_schedule == "geometric":
        schedule = adaptive_shot_hook.geometric_budget(
            base=config.adaptive_schedule_base,
            n0=config.adaptive_schedule_n0,
            n_max=config.adaptive_schedule_n_max,
        )
    else:
        raise ValueError(
            "adaptive_schedule must be 'rho_gated_geometric', "
            "'rho_gated_delta_inverse_square', 'lazy_delta_inverse_square', "
            "'lazy_delta_inverse_fourth', "
            "or 'geometric'."
        )

    # criterion="L" needs the infidelity metric M (Hessian of mean-gate infidelity around the
    # current estimate -- NO ground truth). Build it lazily/cached; ignored for D and A.
    metric_M = _build_infidelity_metric(problem, config) if config.adaptive_criterion == "L" else None

    inner_hook = adaptive_shot_hook.make_adaptive_shot_hook(
        probability_and_jacobian=problem.probability_and_jacobian,
        circuits_from_mask=circuits_from_mask,
        current_shots=current_shots,
        add_shots=add_shots,
        schedule=schedule,
        criterion=config.adaptive_criterion,
        metric_M=metric_M,
        logger=print,
        total_shot_budget=config.adaptive_total_shot_budget,
        accounted_shots=accounted_shots,
        allocate_every=int(getattr(config, "adaptive_allocate_every", 1) or 1),
        per_circuit_allocation=bool(
            getattr(config, "adaptive_per_circuit_allocation", True)),
    )

    def hook(state):
        result = inner_hook(state)
        active_indices = active_indices_from_state(state)
        revealed_total = accounted_shots(state)
        result_dict = result or {}
        event_history.append(
            {
                "iteration": int(state.get("iteration", 0)),
                "nf": int(state.get("nf", 0)),
                "previous_rho": state.get("previous_rho"),
                "trust_region_delta": state.get("delta"),
                "adaptive_schedule": config.adaptive_schedule,
                "requested_budget": int(result_dict.get("requested_budget", 0)),
                "scheduled_budget": int(result_dict.get("scheduled_budget", 0)),
                "shots_added": int(result_dict.get("shots_added", 0)),
                "active_circuits": int(active_indices.size),
                "revealed_union_circuits": int(len(revealed_indices)),
                "accounted_revealed_shots": int(revealed_total),
                "total_shot_budget": int(config.adaptive_total_shot_budget),
                "remaining_shot_budget": max(
                    int(config.adaptive_total_shot_budget) - int(revealed_total), 0
                ),
                "shot_budget_exhausted": bool(
                    result_dict.get("shot_budget_exhausted", False)
                    or revealed_total >= int(config.adaptive_total_shot_budget)
                ),
                "physical_precomputed_shots": int(np.sum(running_shots)),
            }
        )
        return result

    return hook


def _run_pounders(problem: GSTProblem, config: ExperimentConfig, method: str):
    import gradient_pounders
    import general_h_funs

    pounders = importlib.reload(gradient_pounders)
    general_h_funs = importlib.reload(general_h_funs)
    pounders.PYROL_INNER_ITERATION_LIMIT = int(config.pyrol_max_iters)

    if config.require_pyrol and (
        importlib.util.find_spec("pyrol") is None and importlib.util.find_spec("ROL") is None
    ):
        raise ModuleNotFoundError(
            "PyROL/ROL is not importable. Run the seed sweep in the existing Docker image."
        )

    use_fpr = method in ("adaptive_fpr", "fixed_fpr")
    fpr_reduction = _build_fpr(problem, config) if use_fpr else None
    adaptive_events: list[dict[str, Any]] = []

    if method == "fixed_no_fpr":
        shots = int(config.fixed_no_fpr_shots)
    elif method == "fixed_fpr":
        shots = int(config.fixed_fpr_shots)
    elif method in ADAPTIVE_METHODS:
        shots = int(config.adaptive_baseline_shots)
    else:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")

    problem.set_uniform_dataset(shots)
    running_shots = problem.shots_per_circuit.copy()
    adaptive_hook = (
        _build_adaptive_hook(
            problem, config, fpr_reduction, running_shots, adaptive_events
        )
        if method in ADAPTIVE_METHODS
        else None
    )

    revealed_circuit_indices: set[int] = set()

    def iteration_callback(state):
        """Run adaptive acquisition, then report revealed shots for every method."""
        result = adaptive_hook(state) if adaptive_hook is not None else None

        if method == "fixed_no_fpr":
            cumulative_shots = int(len(problem.circuits) * shots)
        elif method in ADAPTIVE_METHODS:
            cumulative_shots = (
                int(adaptive_events[-1]["accounted_revealed_shots"])
                if adaptive_events
                else 0
            )
        else:
            mask = state.get("fpr_mask")
            if mask is not None:
                selected_rows = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
                revealed_circuit_indices.update(
                    (selected_rows // int(problem.outcomes_per_circuit)).tolist()
                )
            cumulative_shots = int(len(revealed_circuit_indices) * shots)

        iteration = int(state.get("iteration", 0))
        if method in ADAPTIVE_METHODS:
            # The global shot budget only governs the adaptive run; show progress
            # against it here.
            total_budget = int(config.adaptive_total_shot_budget)
            print(
                f"[SHOTS] {method} iter {iteration}: "
                f"{cumulative_shots}/{total_budget} cumulative revealed shots"
            )
        else:
            # Fixed methods have no adaptive budget; reporting one is misleading.
            print(
                f"[SHOTS] {method} iter {iteration}: "
                f"{cumulative_shots} cumulative revealed shots"
            )

        # Opt-in comparison hook (see LM_CHECKPOINT_HOOK at the top of this module). Off by
        # default. problem.dataset is live here, so a caller can fit another estimator on
        # exactly the data POUNDERS has right now. Never allowed to break the run.
        if LM_CHECKPOINT_HOOK is not None:
            try:
                LM_CHECKPOINT_HOOK(
                    state=state,
                    problem=problem,
                    config=config,
                    method=method,
                    iteration=iteration,
                    cumulative_shots=cumulative_shots,
                    running_shots=running_shots,
                )
            except Exception as exc:  # noqa: BLE001 - a hook must never kill the run
                print(f"[LM-CKPT] hook failed at iter {iteration}: {exc!r}")

        return result

    lower = np.full(problem.n, float(config.lower_bound))
    upper = np.full(problem.n, float(config.upper_bound))
    shot_argument = (
        (lambda: int(np.sum(running_shots)))
        if method in ADAPTIVE_METHODS
        else int(shots)
    )

    X, F, J, flag, xkin = pounders.pouders(
        problem.oracle,
        problem.x0.reshape(1, -1),
        problem.n,
        int(config.nfmax),
        float(config.gtol),
        float(config.initial_delta),
        problem.m,
        lower,
        upper,
        PoundersLogger(enabled=True),
        spsolver=3,
        hfun=general_h_funs.h_leastsquares,
        combinemodels=general_h_funs.combine_leastsquares,
        fpr_reduction=fpr_reduction,
        residuals_per_circuit=problem.outcomes_per_circuit,
        shots_per_circuit=shot_argument,
        fpr_use_union_mask=bool(config.use_fpr_union_mask),
        rho_uses_full_objective=bool(config.rho_uses_full_objective),
        iter_callback=iteration_callback,
    )

    best_index = int(xkin)
    x_best = np.asarray(X[best_index], dtype=float).reshape(-1)
    fit_model = problem.copy_model_at_x(x_best)
    progress = list(getattr(pounders, "last_progress_history", []))
    fpr_history = list(getattr(fpr_reduction, "history", [])) if fpr_reduction else []
    return {
        "X": np.asarray(X),
        "F": np.asarray(F),
        "J": J,
        "flag": flag,
        "xkin": best_index,
        "x_best": x_best,
        "fit_model": fit_model,
        "progress": progress,
        "fpr_history": fpr_history,
        "adaptive_events": adaptive_events,
        "running_shots": running_shots,
        "uniform_shots": shots,
    }


def _iteration_truth_tables(problem: GSTProblem, run: dict[str, Any]):
    if not problem.config.save_iteration_models:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    progress = [
        row for row in run["progress"] if row.get("phase") in ("initial", "trial")
    ]
    cache: dict[int, tuple[dict, list, list]] = {}
    summary_rows, gate_rows, spam_rows = [], [], []
    for sequence_index, row in enumerate(progress):
        incumbent_index = int(row.get("incumbent_index", run["xkin"]))
        if incumbent_index not in cache:
            model = problem.copy_model_at_x(run["X"][incumbent_index])
            cache[incumbent_index] = problem.aligned_error_metrics(
                model, problem.truth_model, "truth"
            )
        summary, gates, spam = cache[incumbent_index]
        base = {
            "sequence_index": sequence_index,
            "iteration": int(row.get("iteration", sequence_index)),
            "nf": int(row.get("nf", sequence_index)),
            "incumbent_index": incumbent_index,
            "incumbent_full_objective": row.get("incumbent_full_objective"),
            **summary,
        }
        summary_rows.append(base)
        gate_rows.extend({**base, **item} for item in gates)
        spam_rows.extend({**base, **item} for item in spam)
    return pd.DataFrame(summary_rows), pd.DataFrame(gate_rows), pd.DataFrame(spam_rows)


def _final_shot_accounting(problem: GSTProblem, run: dict[str, Any], method: str):
    progress = run["progress"]
    final_union = (
        int(progress[-1].get("union_circuits_revealed", len(problem.circuits)))
        if progress
        else len(problem.circuits)
    )
    if method in ("fixed_no_fpr", "adaptive_no_fpr"):
        # no FPR -> every circuit is measured, so the revealed set is the whole design
        final_union = len(problem.circuits)
    physical = int(np.sum(run["running_shots"]))
    if method in ADAPTIVE_METHODS:
        if run["adaptive_events"]:
            revealed = int(run["adaptive_events"][-1]["accounted_revealed_shots"])
        else:
            revealed = int(final_union * run["uniform_shots"])
    else:
        revealed = int(final_union * run["uniform_shots"])
    return {
        "revealed_circuits": int(final_union),
        "total_circuits": int(len(problem.circuits)),
        "accounted_revealed_shots": revealed,
        "physical_precomputed_shots": physical,
        "min_shots_per_circuit": int(np.min(run["running_shots"])),
        "mean_shots_per_circuit": float(np.mean(run["running_shots"])),
        "max_shots_per_circuit": int(np.max(run["running_shots"])),
    }


def run_one_experiment(
    *,
    config: ExperimentConfig,
    data_seed: int,
    method: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run and persist one ``(data_seed, method)`` experiment."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = GSTProblem(config, data_seed)
    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(
        output_dir / "problem_metadata.json",
        {
            "data_seed": int(data_seed),
            "truth_seed": int(problem.truth_seed),
            "method": method,
            "model_kind": config.model_kind,
            "num_parameters": problem.n,
            "num_circuits": len(problem.circuits),
            "outcomes_per_circuit": problem.outcomes_per_circuit,
            "num_residuals": problem.m,
            "operation_labels": [str(label) for label in problem.base_model.operations.keys()],
        },
    )

    run = _run_pounders(problem, config, method)
    np.save(output_dir / "x_best.npy", run["x_best"])
    np.save(output_dir / "x_evaluations.npy", run["X"])
    np.save(output_dir / "final_shots_per_circuit.npy", run["running_shots"])

    progress_frame = _csv_safe_frame(run["progress"])
    progress_frame.to_csv(output_dir / "optimizer_progress.csv", index=False)
    _csv_safe_frame(run["fpr_history"]).to_csv(
        output_dir / "fpr_selection_history.csv", index=False
    )
    _csv_safe_frame(run["adaptive_events"]).to_csv(
        output_dir / "adaptive_shot_events.csv", index=False
    )

    fit = run["fit_model"]
    truth_summary, truth_gates, truth_spam = problem.aligned_error_metrics(
        fit, problem.truth_model, "truth"
    )
    ideal_summary, ideal_gates, ideal_spam = problem.aligned_error_metrics(
        fit, problem.raw_target_model, "ideal"
    )
    gate_frame = pd.DataFrame(truth_gates + ideal_gates)
    spam_frame = pd.DataFrame(truth_spam + ideal_spam)
    gate_frame.to_csv(output_dir / "final_gate_errors.csv", index=False)
    spam_frame.to_csv(output_dir / "final_spam_errors.csv", index=False)

    iteration_summary, iteration_gates, iteration_spam = _iteration_truth_tables(problem, run)
    iteration_summary.to_csv(output_dir / "iteration_accuracy.csv", index=False)
    iteration_gates.to_csv(output_dir / "iteration_gate_errors.csv", index=False)
    iteration_spam.to_csv(output_dir / "iteration_spam_errors.csv", index=False)

    residual = problem.residual_vector(run["x_best"])
    objective = float(np.sum(np.asarray(residual, dtype=float) ** 2))
    likelihood = problem.likelihood_diagnostics(run["x_best"])
    shot_summary = _final_shot_accounting(problem, run, method)
    if config.objective == "poisson_logl":
        # sum(lsvec**2) == deltaLogL already, and 2*deltaLogL is the chi^2_dof
        # quantity -- no outcome over-count correction to undo here.
        reduced_chi2 = (
            2.0 * objective / likelihood["dof"] if likelihood["dof"] > 0 else float("nan")
        )
    else:
        reduced_chi2 = (
            objective
            * (problem.outcomes_per_circuit - 1)
            / problem.outcomes_per_circuit
            / likelihood["dof"]
            if likelihood["dof"] > 0
            else float("nan")
        )
    summary = {
        "data_seed": int(data_seed),
        "truth_seed": int(problem.truth_seed),
        "method": method,
        "flag": _safe_json(run["flag"]),
        "xkin": int(run["xkin"]),
        "num_parameters": problem.n,
        "num_circuits": len(problem.circuits),
        "num_residuals": problem.m,
        "objective_mode": str(config.objective),
        "weighted_least_squares_objective": objective,
        "reduced_chi_square": float(reduced_chi2),
        **likelihood,
        **truth_summary,
        **ideal_summary,
        **shot_summary,
    }
    _write_json(output_dir / "summary.json", summary)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    _write_json(output_dir / "completed.json", {"complete": True, **summary})
    return summary
