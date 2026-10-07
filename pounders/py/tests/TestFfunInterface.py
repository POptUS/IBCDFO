"""
Confirm that |pounders| handles user-provided Ffun flexibility as desired.

While this is checking the |pounders| interface, these tests are quite involved
and are, therefore, executed in a dedicated TestCase rather than including them
in TestPoundersInterface.
"""

import unittest

import numpy as np

from ibcdfo.pounders._run_user_friendly import run_user_friendly


class TestFfunInterface(unittest.TestCase):
    def testVector(self):
        # ----- HARDCODED VALUES
        EPS = np.finfo(float).eps

        THETA_STAR = np.array([2.1, -0.4, 3.4])
        X_ALL = np.array([-1.1, 0.2, 1.3, 2.1])
        N = len(THETA_STAR)
        M = len(X_ALL)
        A = np.array([(1.0, x, x**2) for x in X_ALL])
        LOW = np.full(N, -np.inf, float)
        UPP = np.full(N, np.inf, float)
        X_0 = np.full(N, 0.5, float)

        # ----- DETERMINE BENCHMARK
        def Ffun(theta):
            return A @ (theta - THETA_STAR)

        X_good, F_good, hF_good, flag, xk_in_good = run_user_friendly(
            Ffun=Ffun,
            n=N,
            m=M,
            Low=LOW,
            Upp=UPP,
            X_0=X_0,
            nf_max=50,
            g_tol=1.0e-13,
            delta_0=1.0,
            batched_Ffun=False,
        )
        self.assertEqual(flag, 0)
        max_rel_err = np.max(np.fabs(1.0 - X_good[xk_in_good, :] / THETA_STAR))
        self.assertTrue(max_rel_err <= 4.0 * EPS)
        self.assertTrue(np.max(np.fabs(F_good[xk_in_good, :])) <= 4.0 * EPS)
        self.assertTrue(np.fabs(hF_good[xk_in_good]) <= 4.0 * EPS)

        # ----- CONFIRM IDENTICAL RESULTS
        # We expect all results produced with acceptable, compatible Ffuns to
        # have identical results.

        # -- Single-evaluation Ffun
        def single_eval_tuple(theta):
            return tuple(Ffun(theta))

        def single_eval_list(theta):
            return list(Ffun(theta))

        for good in [single_eval_tuple, single_eval_list]:
            X, F, hF, flag, xk_in = run_user_friendly(
                Ffun=good,
                n=N,
                m=M,
                Low=LOW,
                Upp=UPP,
                X_0=X_0,
                nf_max=50,
                g_tol=1.0e-13,
                delta_0=1.0,
                batched_Ffun=False,
            )
            self.assertEqual(flag, 0)
            self.assertEqual(xk_in, xk_in_good)
            self.assertTrue(np.array_equal(X, X_good))
            self.assertTrue(np.array_equal(F, F_good))
            self.assertTrue(np.array_equal(hF, hF_good))

        # -- Batched-evaluation Ffun
        # TODO: Should this be setup so that it keeps track of the largest batch
        # that was executed so that we can also confirm that batch execution
        # actually happens?
        def Ffun_batched(theta):
            k = theta.shape[0]
            F_batch = np.full((k, M), np.nan, float)
            for i in range(k):
                F_batch[i, :] = Ffun(theta[i, :])
            return F_batch

        def batched_list(theta):
            k = theta.shape[0]
            F_batch = []
            for i in range(k):
                F_batch.append(single_eval_list(theta[i, :]))
            return F_batch

        def batched_tuple(theta):
            k = theta.shape[0]
            F_batch = []
            for i in range(k):
                F_batch.append(single_eval_tuple(theta[i, :]))
            return tuple(F_batch)

        for good in [Ffun_batched, batched_list, batched_tuple]:
            X, F, hF, flag, xk_in = run_user_friendly(
                Ffun=good,
                n=N,
                m=M,
                Low=LOW,
                Upp=UPP,
                X_0=X_0,
                nf_max=50,
                g_tol=1.0e-13,
                delta_0=1.0,
                batched_Ffun=True,
            )
            self.assertEqual(flag, 0)
            self.assertEqual(xk_in, xk_in_good)
            self.assertTrue(np.array_equal(X, X_good))
            self.assertTrue(np.array_equal(F, F_good))
            self.assertTrue(np.array_equal(hF, hF_good))

        # ----- CONFIRM ERRORS CAUGHT
        # -- Single-evaluation Ffun
        def bad_scalar(theta):
            return 1.1

        def bad_too_few(theta):
            return np.ones(M - 1)

        def bad_too_many(theta):
            return np.ones(M + 1)

        def bad_2d(theta):
            return np.ones((2, M))

        for bad in [bad_scalar, bad_too_few, bad_too_many, bad_2d]:
            with self.assertRaises(ValueError):
                run_user_friendly(
                    Ffun=bad,
                    n=N,
                    m=M,
                    Low=LOW,
                    Upp=UPP,
                    X_0=X_0,
                    nf_max=50,
                    g_tol=1.0e-13,
                    delta_0=1.0,
                    batched_Ffun=False,
                )

        # -- Batched-evaluation Ffun
        def bad_1d(theta):
            k = theta.shape[0]
            return np.zeros(k)

        def bad_too_few_m(theta):
            k = theta.shape[0]
            return np.ones((k, M - 1))

        def bad_too_many_m(theta):
            k = theta.shape[0]
            return np.ones((k, M + 1))

        def bad_too_many_k(theta):
            k = theta.shape[0]
            return np.ones((k + 1, M))

        def bad_3d(theta):
            k = theta.shape[0]
            return np.ones((k, M, 1))

        for bad in [bad_1d, bad_too_few_m, bad_too_many_m, bad_too_many_k, bad_3d]:
            with self.assertRaises(ValueError):
                run_user_friendly(
                    Ffun=bad,
                    n=N,
                    m=M,
                    Low=LOW,
                    Upp=UPP,
                    X_0=X_0,
                    nf_max=50,
                    g_tol=1.0e-13,
                    delta_0=1.0,
                    batched_Ffun=True,
                )

    def testScalar(self):
        # ----- HARDCODED VALUES
        THETA_STAR = np.array([2.1, -0.4, 3.4])
        N = len(THETA_STAR)
        M = 1
        C = np.diag([1.1, 2.2, 3.3])
        LOW = np.full(N, -np.inf, float)
        UPP = np.full(N, np.inf, float)
        X_0 = np.full(N, 0.5, float)

        # ----- DETERMINE BENCHMARK
        # This returns scalars
        def Ffun(theta):
            return (theta - THETA_STAR) @ C @ (theta - THETA_STAR)

        X_good, F_good, hF_good, flag, xk_in_good = run_user_friendly(
            Ffun=Ffun,
            n=N,
            m=M,
            Low=LOW,
            Upp=UPP,
            X_0=X_0,
            nf_max=50,
            g_tol=1.0e-13,
            delta_0=1.0,
            batched_Ffun=False,
        )
        self.assertEqual(flag, 0)
        max_rel_err = np.max(np.fabs(1.0 - X_good[xk_in_good, :] / THETA_STAR))
        self.assertTrue(max_rel_err <= 5.0e-5)
        self.assertTrue(np.squeeze(F_good[xk_in_good, :]) <= 7.5e-10)
        self.assertTrue(np.fabs(hF_good[xk_in_good]) <= 5.625e-19)

        # ----- CONFIRM IDENTICAL RESULTS
        # We expect all results produced with acceptable, compatible Ffuns to
        # have identical results.

        # -- Single-evaluation Ffun
        def single_eval_np_1d(theta):
            return np.array([Ffun(theta)])

        def single_eval_np_2d(theta):
            return np.array([Ffun(theta)]).reshape((1, 1))

        def single_eval_list_1d(theta):
            return [Ffun(theta)]

        def single_eval_list_2d(theta):
            return [[Ffun(theta)]]

        def single_eval_tuple(theta):
            return tuple([Ffun(theta)])

        single_eval_all = [
            single_eval_np_1d,
            single_eval_np_2d,
            single_eval_list_1d,
            single_eval_list_2d,
            single_eval_tuple,
        ]
        for good in single_eval_all:
            X, F, hF, flag, xk_in = run_user_friendly(
                Ffun=good,
                n=N,
                m=M,
                Low=LOW,
                Upp=UPP,
                X_0=X_0,
                nf_max=50,
                g_tol=1.0e-13,
                delta_0=1.0,
                batched_Ffun=False,
            )
            self.assertEqual(flag, 0)
            self.assertEqual(xk_in, xk_in_good)
            self.assertTrue(np.array_equal(X, X_good))
            self.assertTrue(np.array_equal(F, F_good))
            self.assertTrue(np.array_equal(hF, hF_good))

        # -- Batched-evaluation Ffun
        def Ffun_batched(theta):
            k = theta.shape[0]
            F_batch = np.full((k), np.nan, float)
            for i in range(k):
                F_batch[i] = Ffun(theta[i, :])
            return F_batch

        def batched_np_row(theta):
            k = theta.shape[0]
            return Ffun_batched(theta).reshape((1, k))

        def batched_np_col(theta):
            k = theta.shape[0]
            return Ffun_batched(theta).reshape((k, 1))

        def batched_list_1d(theta):
            return list(Ffun_batched(theta))

        def batched_list_2d(theta):
            return [list(Ffun_batched(theta))]

        def batched_tuple(theta):
            return tuple(Ffun_batched(theta))

        batched_all = [
            Ffun_batched,
            batched_np_row,
            batched_np_col,
            batched_list_1d,
            batched_list_2d,
            batched_tuple,
        ]
        for good in batched_all:
            # print(good(THETA_STAR.reshape((N, 1))))
            X, F, hF, flag, xk_in = run_user_friendly(
                Ffun=good,
                n=N,
                m=M,
                Low=LOW,
                Upp=UPP,
                X_0=X_0,
                nf_max=50,
                g_tol=1.0e-13,
                delta_0=1.0,
                batched_Ffun=True,
            )
            self.assertEqual(flag, 0)
            self.assertEqual(xk_in, xk_in_good)
            self.assertTrue(np.array_equal(X, X_good))
            self.assertTrue(np.array_equal(F, F_good))
            self.assertTrue(np.array_equal(hF, hF_good))

        # ----- CONFIRM ERRORS CAUGHT
        # -- Single-evaluation Ffun
        def bad_too_few(theta):
            return np.array([])

        def bad_too_many(theta):
            return np.ones(2)

        def bad_2d(theta):
            return np.ones((2, 1))

        for bad in [bad_too_few, bad_too_many, bad_2d]:
            with self.assertRaises(ValueError):
                run_user_friendly(
                    Ffun=bad,
                    n=N,
                    m=M,
                    Low=LOW,
                    Upp=UPP,
                    X_0=X_0,
                    nf_max=50,
                    g_tol=1.0e-13,
                    delta_0=1.0,
                    batched_Ffun=False,
                )

        # -- Batched-evaluation Ffun
        def bad_too_many_m(theta):
            k = theta.shape[0]
            return np.ones((k, 2))

        def bad_too_few_k(theta):
            k = theta.shape[0]
            return np.ones((k - 1, 1))

        def bad_too_many_k(theta):
            k = theta.shape[0]
            return np.ones((k + 1, 1))

        for bad in [bad_too_many_m, bad_too_few_k, bad_too_many_k]:
            with self.assertRaises(ValueError):
                run_user_friendly(
                    Ffun=bad,
                    n=N,
                    m=M,
                    Low=LOW,
                    Upp=UPP,
                    X_0=X_0,
                    nf_max=50,
                    g_tol=1.0e-13,
                    delta_0=1.0,
                    batched_Ffun=True,
                )
