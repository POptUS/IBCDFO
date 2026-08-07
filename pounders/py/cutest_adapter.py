"""
Adapter: load a CUTEst problem, present it to run(...) in its convention.

run(...) wants:
    f            : callable x -> float
    constraints  : list of callables, each x -> float, with convention c(x) <= 0
    x0           : strictly feasible start
    low, upp     : box bounds on the variables

CUTEst gives:
    p.obj(x)     -> objective value
    p.cons(x)    -> vector of general constraint values c(x)
    p.cl, p.cu   -> per-constraint bounds: cl <= c(x) <= cu
    p.bl, p.bu   -> box bounds on x
    p.x0         -> default start (may be infeasible; we box-clip)

Each general constraint becomes one or two 'g(x) <= 0' rows, but ONLY for sides
that are genuinely bounded:
    cu finite:  c_i(x) - cu <= 0      (upper side)
    cl finite:  cl - c_i(x) <= 0      (lower side)

IMPORTANT: CUTEst encodes "no bound on this side" with the sentinel +/-1e20,
NOT with numpy inf. So np.isfinite(1e20) is True and would wrongly create a
phantom constraint row like  c(x) - 1e20 <= 0  (always satisfied, meaningless,
and it pollutes model-building). We therefore treat anything past +/-1e19 as
infinity and skip that side.

Equality rows (cl == cu) are not handled here -- we filtered them out upstream,
and load raises if any slipped through.
"""
import numpy as np
import pycutest

INF = 1e19  # CUTEst uses +/-1e20 as "no bound"; treat anything past this as inf


def load_cutest(name):
    p = pycutest.import_problem(name)
    if bool(p.is_eq_cons.any()):
        raise ValueError(f"{name} has equality constraints; not supported")

    # objective
    def f(x):
        return float(p.obj(np.asarray(x, float)))

    # one callable per genuinely-bounded side of each constraint.
    # the (lambda ii, b: lambda x: ...)(i, bound) pattern captures the current
    # i and bound value NOW, avoiding the late-binding-in-a-loop closure bug.
    constraints = []
    for i in range(p.m):
        cu, cl = p.cu[i], p.cl[i]
        if cu < INF:                                  # real upper bound
            constraints.append(
                (lambda ii, b: lambda x: float(p.cons(np.asarray(x, float))[ii]) - b)(i, cu))
        if cl > -INF:                                 # real lower bound
            constraints.append(
                (lambda ii, b: lambda x: b - float(p.cons(np.asarray(x, float))[ii]))(i, cl))

    x0 = np.clip(p.x0, p.bl, p.bu)
    return dict(f=f, constraints=constraints, x0=x0,
                low=np.array(p.bl, float), upp=np.array(p.bu, float),
                name=name, n=p.n, m_raw=p.m, m_rows=len(constraints))