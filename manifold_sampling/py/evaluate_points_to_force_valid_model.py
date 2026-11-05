import numpy as np
from ibcdfo.pounders import _bmpts as bmpts
from ibcdfo.pounders import _formquad as formquad

from .call_user_scripts import call_user_scripts


def evaluate_points_to_force_valid_model(n, nf, xk_in, delta, X, F, hF, Mdir, mp, hfun, Ffun, Hash, fq_pars, tol, nf_max, Low, Upp):
    # Evaluate model-improving points to pick best one
    # ! May eventually want to normalize Mdir first for infty norm
    # bmpts respects the bounds Low and Upp: for each direction it picks the sign (+/-)
    # with the larger feasible step and scales the step to stay inside the box (or
    # falls back to coordinate directions), so every X[xk_in, :] + Xsp below is
    # bound-feasible. The sign is chosen geometrically, not by predicted model decrease.
    # (Maybe a model-informed sign choice could be added here?)
    Mdir1, np1 = bmpts(X[xk_in, :], Mdir[: n - mp + 1], Low, Upp, delta, fq_pars["Par"][3])
    for i in range(n - np1):
        Xsp = Mdir1[i, :]
        # Only do this evaluation if the point is new and nf < nf_max
        if not np.any(np.all(X[xk_in, :] + Xsp == X[: nf + 1], axis=1)) and nf + 1 < nf_max:
            nf, X, F, hF, Hash, _ = call_user_scripts(nf, X, F, hF, Hash, Ffun, hfun, X[xk_in, :] + Xsp, tol, Low, Upp, 1)

    valid = formquad(X[: nf + 1], F[: nf + 1], delta, xk_in, fq_pars["npmax"], fq_pars["Par"], 1)[2]
    if not valid and nf + 1 < nf_max:
        print("Proceeding with nonvalid model! Report this to Stefan in Alg1", nf)
        # uuid = char(java.util.UUID.randomUUID);
        # save(['first_failure_for_row_in_dfo_dat=' int2str(mw_prob_num) '_hfun=' func2str(hfun{1}) ], 'A');
        # error('a')

    return X, F, hF, nf, Hash
