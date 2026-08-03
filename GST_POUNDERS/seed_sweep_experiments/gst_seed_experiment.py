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


METHODS = ("adaptive_fpr", "fixed_fpr", "fixed_no_fpr")


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
    lindblad_a_max: float = 0.004      # lindblad_hard: amplitude-damping (A) coeff range [0, max]
    sample_error: str = "multinomial"

    objective: str = "weighted_least_squares"
    variance_source: str = "data"
    variance_floor: float = 1e-12
    nfmax: int = 200
    gtol: float = 1e-4
    initial_delta: float = 0.1
    lower_bound: float = -math.pi
    upper_bound: float = math.pi
    pyrol_max_iters: int = 10
    require_pyrol: bool = True

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
        self.x0 = np.asarray(self.base_model.to_vector(), dtype=float).reshape(-1)
        self.n = int(self.x0.size)
        self.m = int(len(self.circuits) * self.outcomes_per_circuit)
        self.dataset = None
        self.shots_per_circuit = None

    def _build_truth_model(self, config: ExperimentConfig):
        """Construct the synthetic 'device' (data-generation) model per config.noise_model.

        "depolarize_rotate" (default): depolarizing + optional heterogeneous coherent
            over-rotations (config.coherent_max_rotate).
        "lindblad_hard": a rich per-gate Lindblad error generator -- coherent (H),
            stochastic (S), and non-unital amplitude-damping (A) terms, drawn randomly
            per driven gate (heterogeneous), plus depolarizing SPAM.  All terms are
            Lindblad, so the truth stays in-model for a CPTPLND fit.
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
                terms[("A", "Y")] = float(rng.uniform(0.0, config.lindblad_a_max))
                coeffs[_bo.Label(name, 0)] = terms
            model = _mc.create_explicit_model(self.processor_spec, lindblad_error_coeffs=coeffs)
            model.set_all_parameterizations("full")   # make SPAM mutable
            if config.spam_noise:
                model.depolarize(spam_noise=config.spam_noise)
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
        for circuit in circuits:
            row = data[circuit]
            total = float(row.total)
            for outcome in row.outcomes:
                f = float(row.fractions.get(outcome, 0.0))
                f_clip = min(max(f, 0.0), 1.0)
                f_values.append(f)
                var_values.append(f_clip * (1.0 - f_clip) / total if total > 0 else np.nan)
                counts.append(float(row.counts.get(outcome, 0.0)))
                totals.append(total)
                labels.append((circuit, outcome))
        return (
            np.asarray(f_values, dtype=float),
            np.asarray(var_values, dtype=float),
            np.asarray(counts, dtype=float),
            np.asarray(totals, dtype=float),
            labels,
        )

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
                "Use least_squares or weighted_least_squares for the seed sweep."
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
        f, var_f, _, _, f_labels = self.data_vector(data=data)
        if p_labels != f_labels:
            raise ValueError("Model and data outcome order differs.")
        residual = p - f
        if self.config.objective == "weighted_least_squares":
            sigma = np.sqrt(np.maximum(var_f, self.config.variance_floor))
            residual = residual / sigma
        elif self.config.objective != "least_squares":
            raise NotImplementedError(
                "Use least_squares or weighted_least_squares for the seed sweep."
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

    def aligned_error_metrics(self, estimate_model, reference_model, reference_name: str):
        from pygsti.tools import entanglement_infidelity

        reference = reference_model.copy()
        estimate = estimate_model.copy()
        reference.set_all_parameterizations("full")
        estimate.set_all_parameterizations("full")
        with contextlib.redirect_stdout(io.StringIO()):
            aligned = self.pygsti.gaugeopt_to_target(estimate, reference)

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
            gate_rows.append(row)

        spam_rows = []
        for label in reference.preps.keys():
            spam_rows.append(
                {
                    "reference": reference_name,
                    "member_type": "prep",
                    "member": str(label),
                    "vector_l2_error": float(
                        np.linalg.norm(_dense(aligned.preps[label]) - _dense(reference.preps[label]))
                    ),
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
                np.mean([row["vector_l2_error"] for row in spam_rows])
            ),
        }
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
                aligned = problem.pygsti.gaugeopt_to_target(model, reference)
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
    import adaptive_shot_hook

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
    elif method == "adaptive_fpr":
        shots = int(config.adaptive_baseline_shots)
    else:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")

    problem.set_uniform_dataset(shots)
    running_shots = problem.shots_per_circuit.copy()
    adaptive_hook = (
        _build_adaptive_hook(
            problem, config, fpr_reduction, running_shots, adaptive_events
        )
        if method == "adaptive_fpr"
        else None
    )

    revealed_circuit_indices: set[int] = set()

    def iteration_callback(state):
        """Run adaptive acquisition, then report revealed shots for every method."""
        result = adaptive_hook(state) if adaptive_hook is not None else None

        if method == "fixed_no_fpr":
            cumulative_shots = int(len(problem.circuits) * shots)
        elif method == "adaptive_fpr":
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
        if method == "adaptive_fpr":
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
        return result

    lower = np.full(problem.n, float(config.lower_bound))
    upper = np.full(problem.n, float(config.upper_bound))
    shot_argument = (
        (lambda: int(np.sum(running_shots)))
        if method == "adaptive_fpr"
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
    if method == "fixed_no_fpr":
        final_union = len(problem.circuits)
    physical = int(np.sum(run["running_shots"]))
    if method == "adaptive_fpr":
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
