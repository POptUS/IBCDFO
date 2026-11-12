import numpy as np
from scipy.optimize import minimize


def minimize_quadratics_lhs(Hres, Gres, Cres, Low, Upp, val_at_center, hfun, rng=None):
    r"""
    A GAMS-free solver for the GOOMBAH trust-region subproblem (TRSP).

    Direct port of ``goombah/m/subproblems/minimize_quadratics_lhs.m``. This
    function

    * forms the composite objective :math:`\hfun(\zvec(\svec))`, where
      :math:`\zvec(\svec)` are the :math:`p` quadratic models of the step
      :math:`\svec`;
    * evaluates that objective on a Latin hypercube sample of
      ``[Low, Upp]``, together with the zero step;
    * polishes the best of those points with ``scipy.optimize.minimize``
      (``method="L-BFGS-B"``), the Python analog of MATLAB's ``fmincon``;
    * returns the resulting step and its predicted decrease.

    This method does not exploit the composite structure of the TRSP and is
    intended for testing GOOMBAH where GAMS is unavailable.

    Unlike the MATLAB version, this omits the ``x_center``/``x1`` arguments:
    at GOOMBAH's only call site, ``x1`` always equals ``x_center``, so the
    candidate they would add (the clipped ``x1 - x_center``) is always the
    zero step, which is already included in the sample.

    :param Hres: (n, n, p) array of p model quadratic terms.
    :param Gres: (n, p) array of p model linear terms.
    :param Cres: length-p array of p model constant terms.
    :param Low: length-n array, lower bounds on the step.
    :param Upp: length-n array, upper bounds on the step.
    :param val_at_center: Objective value at the trust-region center (the
        zero step).
    :param hfun: Function evaluating :math:`\hfun`, as described in
        :py:func:`ibcdfo.run_MSP`.
    :param rng: (optional) a ``numpy.random.Generator`` used for the Latin
        hypercube sample. Defaults to ``numpy.random.default_rng()``.

    :return: Tuple ``(s_k, pred_dec)`` where ``s_k`` is the length-n step and
        ``pred_dec`` is the (nonnegative) predicted decrease from
        ``val_at_center`` to the objective value at ``s_k``.
    """
    Hres = np.asarray(Hres, dtype=float)
    Gres = np.asarray(Gres, dtype=float)
    Cres = np.asarray(Cres, dtype=float)
    Low = np.asarray(Low, dtype=float)
    Upp = np.asarray(Upp, dtype=float)

    n, p = Gres.shape
    assert Hres.shape == (n, n, p), "Hres must be n-by-n-by-p"
    assert Cres.shape == (p,), "Cres must have length p"
    assert Low.shape == (n,) and Upp.shape == (n,), "Low and Upp must have length n"
    assert np.all(Low <= Upp), "Low must not exceed Upp"

    if rng is None:
        rng = np.random.default_rng()

    def obj(s):
        return hfun(_eval_quadratics(Hres, Gres, Cres, s))[0]

    # Latin hypercube sample: each coordinate's num_samples values fall in
    # distinct equal-width slices of [Low[j], Upp[j]].
    num_samples = 300 * n
    S = np.zeros((num_samples, n))
    for j in range(n):
        S[:, j] = Low[j] + (Upp[j] - Low[j]) * (rng.permutation(num_samples) - rng.random(num_samples)) / num_samples
    S = np.vstack((np.zeros(n), S))

    vals = np.array([obj(s) for s in S])
    ind = int(np.argmin(vals))

    res = minimize(obj, S[ind], method="L-BFGS-B", bounds=list(zip(Low, Upp)))
    s_k = np.clip(res.x, Low, Upp)

    # Safeguard in case the polish returns a worse solution than its starting point
    if obj(s_k) > vals[ind]:
        s_k = S[ind]

    pred_dec = max(0.0, val_at_center - obj(s_k))

    return s_k, pred_dec


def _eval_quadratics(Hres, Gres, Cres, s):
    p = Gres.shape[1]
    z = np.empty(p)
    for i in range(p):
        z[i] = 0.5 * s @ Hres[:, :, i] @ s + s @ Gres[:, i] + Cres[i]
    return z
