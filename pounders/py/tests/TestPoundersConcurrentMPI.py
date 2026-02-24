"""
Unit test of compute function (MPI-parallelized across benchmark problems)

Run, e.g.:
    mpiexec -n 8 python -m unittest -q test_pounders_mpi.py
"""

import os
import unittest

import ibcdfo
import numpy as np
import scipy as sp
from calfun import calfun
from dfoxs import dfoxs

try:
    from mpi4py import MPI
except ImportError as e:
    raise ImportError("mpi4py is required to run this test under MPI. " "Install via `pip install mpi4py` (and ensure an MPI runtime is available).") from e


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


class TestPoundersMPI(unittest.TestCase):
    def test_benchmark_pounders_mpi(self):
        # Separate output locations for serial vs concurrent results
        out_root = "benchmark_results"
        out_conc = os.path.join(out_root, "concurrent")

        # Safe to do on all ranks
        os.makedirs(out_conc, exist_ok=True)
        COMM.Barrier()

        dfo = np.loadtxt("dfo.dat")

        spsolver = 2
        g_tol = 1e-13
        factor = 10

        # Iterate all problems, but each MPI rank only runs its assigned rows
        for row, (nprob, n, m, factor_power) in enumerate(dfo):
            if row % SIZE != RANK:
                continue  # not my row

            n = int(n)
            m = int(m)
            nprob = int(nprob)

            nf_max = 50 * (n + 1)
            # nf_max = 100

            def Ffun_batch(Y):
                Y = np.atleast_2d(Y)
                out = np.zeros((Y.shape[0], m))
                for i, y in enumerate(Y):
                    out[i] = calfun(y, m, nprob, "smooth", 0, num_outs=2)[1]
                return np.squeeze(out)

            X_0 = dfoxs(n, nprob, int(factor**factor_power))
            Low = -np.inf * np.ones((1, n))
            Upp = np.inf * np.ones((1, n))
            nfs = 1
            F_init = np.zeros((1, m))
            F_init[0] = Ffun_batch(X_0)
            xind = 0
            delta = 0.1
            printf = 0

            hfun = ibcdfo.pounders.h_leastsquares
            combinemodels = ibcdfo.pounders.combine_leastsquares
            hfun_name = combinemodels.__name__

            for batch in [5,10]:
                # ---------------- Save outputs (separate files) ----------------
                # Include rank in filename to avoid collisions if you later change the partitioning
                base = f"nf_max={nf_max}_prob={row}_spsolver={spsolver}"
                fname_conc = os.path.join(out_conc, f"pounders4py_conc={batch}_{base}.mat")

                # ---------------- Skip if already computed ----------------
                if os.path.exists(fname_conc):
                    print(f"[rank {RANK}] Skipping row={row}, hfun={hfun_name} (files exist)")
                    continue

                Opts = {
                    "printf": printf,
                    "spsolver": spsolver,
                    "hfun": hfun,
                    "batch": batch,
                    "combinemodels": combinemodels,
                }
                Prior = {"nfs": 1, "F_init": F_init, "X_init": X_0, "xk_in": xind}

                # --- Concurrent run ---
                X, F, hF, flag, xk_best = ibcdfo.run_pounders_concurrent(
                    Ffun_batch,
                    X_0,
                    n,
                    nf_max,
                    g_tol,
                    delta,
                    m,
                    Low,
                    Upp,
                    Prior=Prior,
                    Options=Opts,
                    Model={},
                )
                import ipdb; ipdb.set_trace(context=21)

                # ---------------- Assertions ----------------
                evals = F.shape[0]
                self.assertNotEqual(flag, 1, f"[rank {RANK}] pounders failed. (flag={flag})")
                self.assertTrue(
                    hfun(F[0]) > hfun(F[xk_best]),
                    f"[rank {RANK}] No improvement found: " f"hfun(F[0])={hfun(F[0])}, hfun(F[xk_best])={hfun(F[xk_best])}",
                )
                self.assertTrue(
                    X.shape[0] <= nf_max + nfs,
                    f"[rank {RANK}] POUNDERs grew the size of X: X.shape[0]={X.shape[0]}, " f"limit={nf_max + nfs}",
                )

                if flag == 0:
                    self.assertTrue(
                        evals <= nf_max + nfs,
                        f"[rank {RANK}] POUNDERs evaluated more than nf_max evaluations: " f"evals={evals}, limit={nf_max + nfs}",
                    )
                elif flag not in (-6, -4):
                    self.assertTrue(
                        evals == nf_max + nfs,
                        f"[rank {RANK}] POUNDERs didn't use nf_max evaluations: " f"evals={evals}, expected={nf_max + nfs}, flag={flag}",
                    )

                Results_conc = {
                    f"pounders4py_concurrent_{row}": {
                        "alg": "pounders4py_concurrent",
                        "problem": f"problem {row} from More/Wild",
                        "Fvec": F,
                        "H": hF,
                        "X": X,
                        "flag": flag,
                        "xk_best": xk_best,
                    }
                }

                sp.io.savemat(fname_conc, Results_conc)


if __name__ == "__main__":
    unittest.main()
