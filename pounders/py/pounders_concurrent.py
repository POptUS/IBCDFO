import sys

import numpy as np

from .bmpts import bmpts
from .bqmin import bqmin
from .checkinputss import checkinputss
from .formquad import formquad
from .prepare_outputs_before_return import prepare_outputs_before_return


def _default_model_par_values(n):
    par = np.zeros(5)
    par[0] = np.sqrt(n)
    par[1] = np.maximum(10, np.sqrt(n))
    par[2] = 10**-3
    par[3] = 0.001
    par[4] = 0

    return par


def _default_model_np_max(n):
    return 2 * n + 1


def _default_prior():
    Prior = {}
    Prior["nfs"] = 0
    Prior["X_init"] = []
    Prior["F_init"] = []
    Prior["xk_in"] = 0

    return Prior


def _rand_unit_rows(k, n):
    R = np.random.randn(k, n)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return R


def _ensure_rows_with_random_unit(A, n_rows, n):
    if n_rows > A.shape[0]:
        A = np.vstack([A, _rand_unit_rows(n_rows - A.shape[0], n)])
    return A


def _overwrite_rows_with_random_unit(A, start, count, n):
    if count > 0:
        A[start : start + count, :] = _rand_unit_rows(count, n)
    return A


def pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior=None, Options=None, Model=None):
    r"""
    This version of |pounders| parallelizes the evaluation of ``Ffun`` across
    all model-building points, which can lead to significantly smaller walltimes
    for many problems.  This is especially true for ``Ffun`` models with large
    input dimension :math:`\np`.

    Otherwise, this implementation and its interface are identical to those of
    the standard |pounders| implementation.  Please refer to
    :py:func:`ibcdfo.run_pounders` for more information.

    .. todo::

        * Mention what parallelization technique is used and what
          responsibilities are placed on users to enable/facilitate this?
    """
    if Options is None:
        Options = {}

    if Model is None:
        Model = {}
        Model["Par"] = _default_model_par_values(n)
        Model["np_max"] = _default_model_np_max(n)
    else:
        if "Par" not in Model:
            Model["Par"] = _default_model_par_values(n)
        if "np_max" not in Model:
            Model["np_max"] = _default_model_np_max(n)

    if Prior is None:
        Prior = _default_prior()
    else:
        key_list = ["nfs", "X_init", "F_init", "xk_in"]
        assert set(Prior.keys()) == set(key_list), f"Prior keys must be {key_list}"
        Prior["X_init"] = np.atleast_2d(Prior["X_init"])
        if Prior["X_init"].ndim == 2 and Prior["X_init"].shape[1] == 1:
            Prior["X_init"] = Prior["X_init"].T

    nfs = Prior["nfs"]
    delta = delta_0
    spsolver = Options.get("spsolver", 2)
    delta_max = Options.get("delta_max", np.minimum(0.5 * np.min(Upp - Low), (10**3) * delta))
    delta_min = Options.get("delta_min", np.minimum(delta * (10**-13), g_tol / 10))
    gamma_dec = Options.get("gamma_dec", 0.5)
    gamma_inc = Options.get("gamma_inc", 2)
    eta_1 = Options.get("eta1", 0.05)
    printf = Options.get("printf", 0)
    delta_inact = Options.get("delta_inact", 0.75)
    batch = Options.get("batch", 1)

    if batch <= 0:
        raise ValueError(f"Batch size must be positive; got batch={batch}")

    assert (nf_max % batch) == 0, f"Assumed nf_max is a multiple of batch, but nf_max={nf_max}, batch={batch}"
    if "hfun" in Options:
        hfun = Options["hfun"]
        combinemodels = Options["combinemodels"]
    else:
        from .general_h_funs import h_leastsquares as hfun
        from .general_h_funs import combine_leastsquares as combinemodels

    # choose your spsolver
    if spsolver == 2:
        try:
            from minqsw import minqsw
        except ModuleNotFoundError as e:
            print(e)
            sys.exit("Ensure a python implementation of MINQ is available. For example, clone https://github.com/POptUS/minq and add minq/py/minq5 to the PYTHONPATH environment variable")

    [flag, X_0, _, F_init, Low, Upp, xk_in] = checkinputss(Ffun, X_0, n, Model["np_max"], nf_max, g_tol, delta_0, Prior["nfs"], m, Prior["X_init"], Prior["F_init"], Prior["xk_in"], Low, Upp)
    if flag == -1:
        X = []
        F = []
        hF = []
        return X, F, hF, flag, xk_in
    eps = np.finfo(float).eps  # Define machine epsilon
    if printf:
        print("  nf   delta    fl  np       f0           g0       ierror")
        progstr = "%4i %9.2e %2i %3i  %11.5e %12.4e %11.3e\n"  # Line-by-line
    if Prior["nfs"] == 0:
        X = np.vstack((X_0, np.zeros((nf_max - 1, n))))
        F = np.zeros((nf_max, m))
        hF = np.zeros(nf_max)
        nf = 0  # in Matlab this is 1
        F_0 = np.atleast_2d(Ffun(X[nf]))
        if F_0.shape[1] != m:
            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -1)
            return X, F, hF, flag, xk_in
        F[nf] = F_0
        if np.any(np.isnan(F[nf])):
            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
            return X, F, hF, flag, xk_in
        if printf:
            print("%4i    Initial point  %11.5e\n" % (nf, hfun(F[nf, :])))
    else:
        X = np.vstack((Prior["X_init"], np.zeros((nf_max, n))))
        F = np.vstack((Prior["F_init"], np.zeros((nf_max, m))))
        hF = np.zeros(nf_max + nfs)
        nf = nfs - 1
        nf_max = nf_max + nfs
    for i in range(nf + 1):
        hF[i] = hfun(F[i])
    Res = np.zeros(np.shape(F))
    Cres = F[xk_in]
    Hres = np.zeros((n, n, m))
    ng = np.nan  # Needed for early termination, e.g., if a model is never built
    while nf + 1 < nf_max:
        #  1a. Compute the interpolation set.
        D = X[: nf + 1] - X[xk_in]
        Res[: nf + 1, :] = (F[: nf + 1, :] - Cres) - np.diagonal(0.5 * D @ (np.tensordot(D, Hres, axes=1))).T
        [Mdir, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], False)

        if mp < n:
            [Mdir, mp] = bmpts(X[xk_in], Mdir[0 : n - mp, :], Low, Upp, delta, Model["Par"][2])
            k_new = int(min(n - mp, nf_max - (nf + 1)))  # new geometry points to send to Ffun (while respecting nfmax)
            if k_new > 0:
                # Pad to next multiple of batch so every call has exactly batch points
                k_pad = (-k_new) % batch
                k_total = k_new + k_pad  # multiple of batch

                # Ensure Mdir has at least k_total directions; if not, append random unit directions
                Mdir = _ensure_rows_with_random_unit(Mdir, k_total, n)

                # Fill padded rows with fresh random unit directions
                Mdir = _overwrite_rows_with_random_unit(Mdir, k_new, k_pad, n)

                # Absolute indices for the new evaluations
                idx_new = (nf + 1) + np.arange(k_total, dtype=int)

                # New points, projected to bounds
                X[idx_new] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdir[:k_total, :]))

                # Evaluate F in fixed-size batches
                for s in range(0, k_total, batch):
                    idx_batch = idx_new[s:s + batch]  # always length batch
                    F[idx_batch] = Ffun(X[idx_batch])

                    if np.any(np.isnan(F[idx_batch])):
                        X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                        return X, F, hF, flag, xk_in

                for i in range(k_total):
                    nf += 1
                    hF[nf] = hfun(F[nf])

                    if printf:
                        print("%4i   Geometry point  %11.5e\n" % (nf, hF[nf]))

                    D = Mdir[i, :]
                    Res[nf, :] = (F[nf, :] - Cres) - 0.5 * D @ np.tensordot(D.T, Hres, 1)

            if nf + 1 >= nf_max:
                break
            # Rebuild quadratic model with the *expanded* interpolation set
            [_, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], False)
            if mp < n:
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -5)
                return X, F, hF, flag, xk_in


        #  1b. Update the quadratic model
        Cres = F[xk_in]
        Hres = Hres + Hresdel
        c = hF[xk_in]
        G, H = combinemodels(Cres, Gres, Hres)
        ind_Lownotbinding = (X[xk_in] > Low) * (G.T > 0)
        ind_Uppnotbinding = (X[xk_in] < Upp) * (G.T < 0)
        ng = np.linalg.norm(G * (ind_Lownotbinding + ind_Uppnotbinding).T, 2)
        if printf:
            IERR = np.zeros(len(Mind))
            for i in range(len(Mind)):
                D = X[Mind[i]] - X[xk_in]
                IERR[i] = (c - hF[Mind[i]]) + (D @ (G + 0.5 * H @ D))
            if np.any(hF[Mind] == 0.0):
                ierror = np.nan
            else:
                ierror = np.linalg.norm(IERR / np.abs(hF[Mind]), np.inf)
            print(progstr % (nf, delta, valid, mp, hF[xk_in], ng, ierror))
            if printf >= 2:
                jerr = np.zeros((len(Mind), m))
                for i in range(len(Mind)):
                    D = X[Mind[i]] - X[xk_in]
                    for j in range(m):
                        jerr[i, j] = (Cres[j] - F[Mind[i], j]) + D @ (Gres[:, j] + 0.5 * Hres[:, :, j] @ D)
                print(jerr)
            # input("Enter a key and press Enter to continue\n") - Don't uncomment when using Pytest with test_pounders.py

        # 2. Critically test invoked if the projected model gradient is small
        if ng < g_tol:
            delta = np.maximum(g_tol, np.max(np.abs(X[xk_in])) * eps)
            [Mdir, _, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], True)
            if not valid:
                [Mdir, mp] = bmpts(X[xk_in], Mdir, Low, Upp, delta, Model["Par"][2])
                k_new = int(min(n - mp, nf_max - (nf + 1)))
                if k_new > 0:
                    k_pad   = (-k_new) % batch
                    k_total = k_new + k_pad  # multiple of batch

                    # Ensure Mdir has >= k_total directions; append random unit directions if needed
                    Mdir = _ensure_rows_with_random_unit(Mdir, k_total, n)

                    # Overwrite padded rows with fresh random unit directions
                    Mdir = _overwrite_rows_with_random_unit(Mdir, k_new, k_pad, n)

                    # Absolute indices for ALL new points (including padding)
                    idx_new = (nf + 1) + np.arange(k_total, dtype=int)

                    # Build all new X points
                    X[idx_new] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdir[:k_total, :]))

                    # Evaluate F in fixed-size batches
                    for s in range(0, k_total, batch):
                        idx_batch = idx_new[s:s + batch]  # always length batch
                        F[idx_batch] = Ffun(X[idx_batch])

                        if np.any(np.isnan(F[idx_batch])):
                            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                            return X, F, hF, flag, xk_in

                    # Now advance nf and compute hF / prints for all k_total points
                    for i in range(k_total):
                        nf += 1
                        hF[nf] = hfun(F[nf])
                        if printf:
                            print("%4i   Critical point  %11.5e\n" % (nf, hF[nf]))

                if nf + 1 >= nf_max:
                    break
                # Recalculate gradient based on a MFN model
                [_, _, valid, Gres, Hres, Mind] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], False)
                G, H = combinemodels(Cres, Gres, Hres)
                ind_Lownotbinding = (X[xk_in] > Low) * (G.T > 0)
                ind_Uppnotbinding = (X[xk_in] < Upp) * (G.T < 0)
                ng = np.linalg.norm(G * (ind_Lownotbinding + ind_Uppnotbinding).T, 2)
            if ng < g_tol:
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, 0)
                return X, F, hF, flag, xk_in

        # 3. Solve a batch of TRSPs with the SAME (G,H) but radii: Delta, Delta/2, 2Delta, Delta/4, 4Delta, ...
        # Build radii sequence
        radii = np.empty(batch, dtype=float)
        radii[0] = float(delta)
        for k in range(1, batch):
            j = (k + 1) // 2  # 1,1,2,2,3,3,...
            if k % 2 == 1:
                radii[k] = delta / (2.0 ** j)     # k=1 -> /2, k=3 -> /4, ...
            else:
                radii[k] = delta * (2.0 ** j)     # k=2 -> *2, k=4 -> *4, ...

        # Arrays for steps / predicted decreases
        Xsp_steps = np.zeros((batch, n), dtype=float)
        mdec_arr  = np.zeros(batch, dtype=float)
        valid_step = np.zeros(batch, dtype=bool)

        # Solve TRSP for each radius
        for k in range(batch):
            delt_k = radii[k]
            Lows = np.maximum(Low - X[xk_in], -delt_k * np.ones(np.shape(Low)))
            Upps = np.minimum(Upp - X[xk_in],  delt_k * np.ones(np.shape(Upp)))

            if spsolver == 1:
                Xsp_k, mdec_k = bqmin(H, G, Lows, Upps)
            elif spsolver == 2:
                Xsp_k, mdec_k, minq_err, _ = minqsw(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
                if minq_err < 0:
                    X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -4)
                    return X, F, hF, flag, xk_in
            else:
                raise ValueError(f"Unsupported spsolver={spsolver}")


            Xsp_k = Xsp_k.squeeze()
            Xsp_steps[k, :] = Xsp_k
            mdec_arr[k] = float(np.asarray(mdec_k).squeeze())

            step_norm_k = np.linalg.norm(Xsp_k, np.inf) if n > 1 else np.abs(Xsp_k)
            valid_step[k] = bool(valid or (step_norm_k >= 0.01 * delt_k and mdec_arr[k] != 0.0))

        if not np.any(valid_step):
            rho = -1  # Force yourself to do a model-improving point
            if printf:
                print("Warning: skipping sp soln!-----------")
        else:
            # Evaluate ALL batch candidates (same model, different radii), regardless of which passed the gate
            cand_idx = np.arange(batch, dtype=int)

            # Form candidate points (Xsp is now a point, not a step)
            Xcand = X[xk_in][None, :] + Xsp_steps[cand_idx, :]
            Xcand = np.minimum(Upp, np.maximum(Low, Xcand))

            # Project if we're within machine precision (apply per-candidate)
            for ii in range(n):  # This will need to be cleaned up eventually
                upp_i = Upp[ii]
                low_i = Low[ii]
                for rr in range(batch):
                    xval = Xcand[rr, ii]
                    if (upp_i - xval < eps * abs(upp_i)) and (upp_i > xval and G[ii] >= 0):
                        Xcand[rr, ii] = upp_i
                        if printf:
                            print("eps project!")
                    elif (xval - low_i < eps * abs(low_i)) and (low_i < xval and G[ii] >= 0):
                        Xcand[rr, ii] = low_i
                        if printf:
                            print("eps project!")

            # Evaluate ALL batch candidates in ONE batch call to Ffun
            idx_store = (nf + 1) + np.arange(batch, dtype=int)
            X[idx_store] = Xcand
            F[idx_store] = Ffun(X[idx_store])

            if np.any(np.isnan(F[idx_store])):
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                return X, F, hF, flag, xk_in

            # Compute hF for all candidates and advance nf (these points consume budget and stay in interpolation set)
            for t in range(batch):
                nf += 1
                hF[nf] = hfun(F[nf])

            # Compute rho for each evaluated candidate (aligned with original definition)
            mdec_c = mdec_arr[cand_idx]
            hf_c   = hF[idx_store]
            hf0    = hF[xk_in]

            rho_c = np.empty(batch, dtype=float)
            for t in range(batch):
                if mdec_c[t] != 0.0:
                    rho_c[t] = (hf_c[t] - hf0) / mdec_c[t]
                else:
                    # mimic original branch (model valid implied for mdec==0 usage)
                    if hf_c[t] == hf0:
                        if delta < np.sqrt(eps):
                            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -2)
                            return X, F, hF, flag, xk_in
                        rho_c[t] = -np.inf
                    else:
                        rho_c[t] = np.inf * np.sign(hf_c[t] - hf0)

            # Pick the best candidate (max rho). Ties broken by first occurrence.
            best_t = int(np.argmax(rho_c))
            # best_t = int(np.argmin(hf_c))
            # best_t = int(np.nanargmin(h_c))
            best_nf = int(idx_store[best_t])
            best_k  = int(cand_idx[best_t])      # which radius/step index this was
            best_rho = float(rho_c[best_t])
            rho = best_rho
            best_mdec = float(mdec_c[best_t])
            best_step = X[best_nf] - X[xk_in]
            best_step_norm = np.linalg.norm(best_step, np.inf) if n > 1 else np.abs(best_step)

            # Original early-exit condition, applied to the chosen candidate
            if (best_mdec == 0.0 and valid and np.array_equiv(X[best_nf], X[xk_in]) and delta < np.sqrt(eps)):
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -2)
                return X, F, hF, flag, xk_in

            # 4a. Update the center based on chosen candidate
            if (best_rho >= eta_1) or (best_rho > 0 and valid):
                Cres = F[xk_in]
                xk_in = best_nf

            # 4b. Update trust-region radius based on chosen candidate's radius and step norm
            delt_best = radii[best_k]
            if (best_rho >= eta_1) and (best_step_norm > delta_inact * delt_best):
                delta = np.minimum(delt_best * gamma_inc, delta_max)
            elif valid:
                delta = delt_best * gamma_dec
                if delta <= delta_min:
                    X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -6)

        # 5. Evaluate a model-improving point if necessary
        if not valid and (nf + 1 < nf_max) and (rho < eta_1):  # Implies xk_in, delta unchanged
            # Need to check because model may be valid after Xsp evaluation
            [Mdir, mp, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], True)
            if not valid:  # ! One strategy for choosing model-improving point:
                # Update model (exists because delta & xk_in unchanged)
                D = X[: nf + 1] - X[xk_in]
                Res[: nf + 1, :] = (F[: nf + 1, :] - Cres) - np.diagonal(0.5 * D @ (np.tensordot(D, Hres, axes=1))).T
                [_, _, valid, Gres, Hresdel, Mind] = formquad(X[: nf + 1, :], Res[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], False)
                if len(Mind) < n + 1:
                    # This is almost never triggered but is a safeguard for
                    # pathological cases where one needs to recover from
                    # unusual conditioning of recent interpolation sets
                    Model["Par"][4] = 1
                    [_, _, valid, Gres, Hresdel, Mind] = formquad(X[: nf + 1, :], Res[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], False)
                    Model["Par"][4] = 0
                Hres = Hres + Hresdel
                # Update for modelimp; Cres unchanged b/c xk_in unchanged
                G, H = combinemodels(Cres, Gres, Hres)

                # evaluate ONE BATCH of model-improving points
                remaining = int(nf_max - (nf + 1))
                k_new = batch

                cand_dirs = []

                # Plus directions
                [Mdir1, mp1] = bmpts(X[xk_in], Mdir[0 : n - mp, :], Low, Upp, delta, Model["Par"][2])
                for i in range(n - mp1):
                    Ddir = Mdir1[i, :]
                    cand_dirs.append((Ddir @ (G + 0.5 * H @ Ddir.T), Ddir))

                # Minus directions
                [Mdir1m, mp2] = bmpts(X[xk_in], -Mdir[0 : n - mp, :], Low, Upp, delta, Model["Par"][2])
                for i in range(n - mp2):
                    Ddir = Mdir1m[i, :]
                    cand_dirs.append((Ddir @ (G + 0.5 * H @ Ddir.T), Ddir))

                cand_dirs.sort(key=lambda t: t[0])  # best predicted first
                Mdir_batch = np.array([d for (_, d) in cand_dirs[:k_new]], dtype=float)
                if Mdir_batch.ndim == 1:  # happens if k_new==1
                    Mdir_batch = Mdir_batch.reshape(1, -1)

                Mdir_batch = _ensure_rows_with_random_unit(Mdir_batch, k_new, n)

                idx_new = (nf + 1) + np.arange(k_new, dtype=int)

                X[idx_new] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdir_batch[:k_new, :]))

                F[idx_new] = Ffun(X[idx_new])
                if np.any(np.isnan(F[idx_new])):
                    X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                    return X, F, hF, flag, xk_in

                for t in range(k_new):
                    nf += 1
                    hF[nf] = hfun(F[nf])

                    if printf:
                        print("%4i   Model batch pt %11.5e\n" % (nf, hF[nf]))

                    Ddir = Mdir_batch[t, :]
                    Res[nf, :] = (F[nf, :] - Cres) - 0.5 * Ddir @ np.tensordot(Ddir.T, Hres, 1)

                    [Mdir, mp, valid, _, _, _] = formquad(
                        X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], True
                    )

                start = idx_new[0]
                stop = nf  # inclusive
                idx_eval = np.arange(start, stop + 1, dtype=int)
                best_idx = int(idx_eval[np.argmin(hF[idx_eval])])

                if hF[best_idx] < hF[xk_in]:  # improvement from model batch point
                    if printf:
                        print("**improvement from model batch point****")

                    # Update model to reflect new base point
                    D = X[best_idx] - X[xk_in]
                    xk_in = best_idx  # Change current center
                    Cres = F[xk_in]

                    # Don't actually use
                    for j in range(m):
                        Gres[:, j] = Gres[:, j] + Hres[:, :, j] @ D.T
    if printf:
        print("Number of function evals exceeded")
    flag = ng
    return X, F, hF, flag, xk_in
