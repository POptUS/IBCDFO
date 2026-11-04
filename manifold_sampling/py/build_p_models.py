from ibcdfo.pounders import _formquad as formquad

from .evaluate_points_to_force_valid_model import evaluate_points_to_force_valid_model
from .update_models import update_models


def build_p_models(nf, nf_max, xk_in, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, Low, Upp):
    n = X.shape[1]
    p = F.shape[1]

    valid, Gres, Hres, X, F, h, nf, Hash = update_models(hfun, Ffun, n, p, nf, nf_max, xk_in, delta, F, X, h, Hres, fq_pars, 1, Hash, tol, Low, Upp)
    # Evaluate model-improving points if necessary
    if not valid and not len(Gres) == 0:
        Mdir, np, valid, *_ = formquad(X[: nf + 1], F[: nf + 1], delta, xk_in, fq_pars["npmax"], fq_pars["Par"], 1)
        if valid:
            raise Exception("what to do here")
        X, F, h, nf, Hash = evaluate_points_to_force_valid_model(n, nf, xk_in, delta, X, F, h, Mdir, np, hfun, Ffun, Hash, fq_pars, tol, nf_max, Low, Upp)
        __, Gres, Hres, X, F, h, nf, Hash = update_models(hfun, Ffun, n, p, nf, nf_max, xk_in, delta, F, X, h, Hres, fq_pars, 1, Hash, tol, Low, Upp)

    return Gres, Hres, X, F, h, nf, Hash
