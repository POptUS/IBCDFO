"""
Unit test of POUNDERS interface.
"""

import io
import copy
import numbers
import unittest

import numpy as np

from contextlib import redirect_stdout

import ibcdfo


class TestPoundersInterface(unittest.TestCase):
    def setUp(self):
        self.__THETA_STAR = np.array([2.1, -0.4, 3.4])
        x_all = np.array([-1.1, 0.2, 1.3, 2.1])
        M = np.array([(1.0, x, x**2) for x in x_all])

        def Ffun(theta):
            return M @ (theta - self.__THETA_STAR)

        n = len(self.__THETA_STAR)
        xk_in = 1
        X_init = np.zeros((2, n))
        nfs = X_init.shape[0]
        X_init[xk_in, :] = np.full(n, 0.5, float)
        F_init = np.array([Ffun(X_init[i, :]) for i in range(nfs)])

        # These should yield a good solution
        self.__kwargs = {
            "Ffun": Ffun,
            "X_0": X_init[xk_in, :],
            "n": n,
            "nf_max": 15,
            "g_tol": 1.0e-13,
            "delta_0": 0.1,
            "m": len(x_all),
            "Low": np.full(n, -2.2, float),
            "Upp": np.full(n, 4.4, float),
            "Prior": {
                "nfs": nfs,
                "X_init": X_init,
                "F_init": F_init,
                "xk_in": xk_in,
            },
            "Options": None,
            "Model": {
                "np_max": 2 * n + 1,
            },
        }

        self.__NOT_REAL = (None, "", {}, [1.1], {1.1})
        self.__NOT_INT = (None, "", 1.0, 1.1, {}, [1], {1})
        self.__NOT_NUMPY_ARRAY = (None, "", 1, 1.1, {}, [1], {1})
        self.__NOT_FUNCTION = (None, "", 1, 1.1, {}, [1], {1})

    def __test(self, new_args, expected_exception):
        # ----- HARDCODED VALUE
        EPS = np.finfo(float).eps
        SUCCESS_MSG = "g is sufficiently small."
        ERROR_HDR = "Error: "

        # ----- ALTER CONFIGURATION
        kwargs = copy.deepcopy(self.__kwargs)
        for key, value in new_args.items():
            if key in ["nfs", "X_init", "F_init", "xk_in"]:
                kwargs["Prior"][key] = value
            elif key in ["np_max"]:
                kwargs["Model"][key] = value
            else:
                kwargs[key] = value

        # ----- OPTIMIZE & ERROR CHECK
        for solver in [ibcdfo.run_pounders]:
            if expected_exception is None:
                with redirect_stdout(io.StringIO()) as buffer:
                    X, F, hF, flag, xk_in = solver(**kwargs)
                self.assertEqual(buffer.getvalue().strip(), SUCCESS_MSG)

                self.assertTrue(isinstance(X, np.ndarray))
                self.assertEqual(X.ndim, 2)
                # self.assertEqual(X.shape, (self.nf_max, self.n))

                self.assertTrue(isinstance(F, np.ndarray))
                self.assertEqual(F.ndim, 2)
                # self.assertEqual(F.shape, (self.nf_max, self.m))

                self.assertTrue(isinstance(hF, np.ndarray))
                self.assertEqual(hF.ndim, 1)
                # self.assertEqual(len(hF), self.nf_max)

                self.assertTrue(isinstance(flag, numbers.Integral))
                self.assertEqual(flag, 0)

                self.assertTrue(isinstance(xk_in, numbers.Integral))
                # self.assertTrue(0 <= xk_in < self.nf_max)

                max_rel_err = np.max(np.fabs(1.0 - X[xk_in, :] / self.__THETA_STAR))
                self.assertTrue(max_rel_err <= 4.0 * EPS)
                self.assertTrue(np.max(np.fabs(F[xk_in, :])) <= 4.0 * EPS)
                self.assertTrue(np.fabs(hF[xk_in]) <= 4.0 * EPS)
            else:
                with self.assertRaises(expected_exception) as err:
                    solver(**kwargs)
                err_msg = str(err.exception)
                # print(err_msg)
                self.assertTrue(err_msg.startswith(ERROR_HDR))

    def testConfirmGoodArguments(self):
        self.__test({}, None)

    def testFfun(self):
        for bad in self.__NOT_FUNCTION:
            self.__test({"Ffun": bad}, TypeError)

    def testN(self):
        for bad in self.__NOT_INT:
            self.__test({"n": bad}, TypeError)
        for bad in [-10, -1, 0]:
            self.__test({"n": bad}, ValueError)

    def testM(self):
        for bad in self.__NOT_INT:
            self.__test({"m": bad}, TypeError)
        for bad in [-10, -1, 0]:
            self.__test({"m": bad}, ValueError)

    def testNpMax(self):
        self.assertEqual(self.__kwargs["n"], 3)
        MIN, MAX = 4, 10

        for bad in self.__NOT_INT:
            self.__test({"np_max": bad}, TypeError)
        for good in range(MIN, MAX + 1):
            self.__test({"np_max": good}, None)
        for bad in [MIN - 1, MAX + 1]:
            self.__test({"np_max": bad}, ValueError)

    def testNfMax(self):
        n = self.__kwargs["n"]
        m = self.__kwargs["m"]

        self.assertEqual(n, 3)
        self.assertEqual(self.__kwargs["Prior"]["nfs"], 2)

        for bad in self.__NOT_INT:
            self.__test({"nf_max": bad}, TypeError)

        # No prior evaluations given
        min_required = 5
        test_case = {"nfs": 0, "X_init": np.full((0, n), np.nan, float), "F_init": np.full((0, m), np.nan, float), "xk_in": 0}
        # TODO: What to do for this test?!
        # for good in [0, 1, 10, 10_000]:
        #     test_case["nf_max"] = min_required + good
        #     print(test_case["nf_max"])
        #     self.__test(test_case, None)
        for bad in range(min_required):
            test_case["nf_max"] = bad
            self.__test(test_case, ValueError)

        # Two prior evaluation
        min_required = 3
        # TODO: What to do for this test?!
        # for good in [0, 1, 10, 10_000]:
        #     self.__test({"nf_max": min_required + good}, None)
        for bad in range(min_required):
            self.__test({"nf_max": bad}, ValueError)

        # n+2 or more prior evaluations
        min_required = 1
        for nfs in [5, 6, 10, 10_000]:
            test_case = {"nfs": nfs, "X_init": np.full((nfs, n), 0.5, float), "F_init": np.zeros((nfs, m)), "xk_in": 0}
            # TODO: What to do for this test?!
            # for good in [1, 2, 10, 10_000]:
            #     test_case["nf_max"] = good
            #     self.__test(test_case, None)
            test_case["nf_max"] = 0
            self.__test(test_case, ValueError)

    def testX0Errors(self):
        EPS = np.finfo(float).eps

        n = self.__kwargs["n"]
        X_0 = self.__kwargs["X_0"].copy()

        # Expects 1D array ...
        for bad in self.__NOT_NUMPY_ARRAY:
            self.__test({"X_0": bad}, TypeError)
        self.__test({"X_0": np.atleast_2d(X_0)}, ValueError)

        # of correct length
        for bad in [n - 1, n + 1]:
            self.__test({"X_0": np.full(bad, 0.5, float)}, ValueError)

        # X_0 on bounds acceptable ...
        #
        # Since the prior evaluations are already correctly set
        # relative to X_0 it's easy and cleaner to change the bounds.
        # TODO: What to do about these?!
        # self.__test({"Low": X_0}, None)
        # self.__test({"Upp": X_0}, None)

        # outside is not
        for i in range(n):
            Low_to_fail = X_0.copy()
            Low_to_fail[i] = (1.0 + EPS) * Low_to_fail[i]
            self.__test({"Low": Low_to_fail}, ValueError)

            Upp_to_fail = X_0.copy()
            Upp_to_fail[i] = (1.0 - EPS) * Upp_to_fail[i]
            self.__test({"Upp": Upp_to_fail}, ValueError)

    def testGTol(self):
        for bad in self.__NOT_REAL:
            self.__test({"g_tol": bad}, TypeError)
        for bad in [-np.inf, -1.0, -np.finfo(float).smallest_normal, 0.0, np.nan, np.inf]:
            self.__test({"g_tol": bad}, ValueError)

    def testDelta(self):
        for bad in self.__NOT_REAL:
            self.__test({"delta_0": bad}, TypeError)
        for bad in [-np.inf, -1.0, -np.finfo(float).smallest_normal, 0.0, np.nan, np.inf]:
            self.__test({"delta_0": bad}, ValueError)

    def testNfs(self):
        for bad in self.__NOT_INT:
            self.__test({"nfs": bad}, TypeError)
        for bad in [-10, -1]:
            self.__test({"nfs": bad}, ValueError)

    def testXinitErrors(self):
        # MIN = np.finfo(float).min
        # MAX = np.finfo(float).max
        EPS = np.finfo(float).eps

        NFS_NEW = 3

        n = self.__kwargs["n"]
        m = self.__kwargs["m"]
        nfs = self.__kwargs["Prior"]["nfs"]
        X_init = self.__kwargs["Prior"]["X_init"].copy()

        for bad in self.__NOT_NUMPY_ARRAY:
            self.__test({"X_init": bad}, TypeError)

        self.__test({"X_init": np.zeros(n)}, ValueError)
        self.__test({"X_init": np.zeros((nfs, n, 1))}, ValueError)

        for bad in [nfs - 1, nfs + 1]:
            self.__test({"X_init": np.zeros((bad, n))}, ValueError)

        for bad in [n - 1, n + 1]:
            self.__test({"X_init": np.zeros((nfs, bad))}, ValueError)

        for bad in [np.nan, np.inf, -np.inf]:
            self.__test({"X_init": np.full(X_init.shape, bad, float)}, ValueError)

        # X_init[xk_in, :] does not match given X_0
        #
        # Intentionally set other points outside bounds, which should be
        # acceptable.
        test_case = {"nfs": NFS_NEW, "X_init": None, "F_init": np.zeros((NFS_NEW, m)), "xk_in": -1}
        for xk_in in range(NFS_NEW):
            i, j = sorted(set(range(NFS_NEW)).difference({xk_in}))

            X_init = np.full((NFS_NEW, n), np.nan, float)
            test_case["xk_in"] = xk_in
            test_case["X_init"] = X_init

            # See valid initial points pass ...
            # TODO: What to do about this?
            # X_init[xk_in, :] = self.__kwargs["X_0"].copy()
            # X_init[i, :] = np.full(n, MIN, float)
            # X_init[j, :] = np.full(n, MAX, float)
            # self.__test(test_case, None)

            # and then let's make 'em fail.
            for k in range(n):
                X_init[xk_in, :] = self.__kwargs["X_0"].copy()
                X_init[xk_in, k] = (1.0 + EPS) * X_init[xk_in, k]
                self.__test(test_case, ValueError)

    def testFinitErrors(self):
        m = self.__kwargs["m"]
        nfs = self.__kwargs["Prior"]["nfs"]
        F_init = self.__kwargs["Prior"]["F_init"].copy()

        for bad in self.__NOT_NUMPY_ARRAY:
            self.__test({"F_init": bad}, TypeError)

        self.__test({"F_init": np.zeros(m)}, ValueError)
        self.__test({"F_init": np.zeros((nfs, m, 1))}, ValueError)

        for bad in [nfs - 1, nfs + 1]:
            self.__test({"F_init": np.zeros((bad, m))}, ValueError)

        for bad in [m - 1, m + 1]:
            self.__test({"F_init": np.zeros((nfs, bad))}, ValueError)

        for bad in [np.nan, np.inf, -np.inf]:
            self.__test({"F_init": np.full(F_init.shape, bad, float)}, ValueError)

    def testXKinErrors(self):
        n = self.__kwargs["n"]
        m = self.__kwargs["m"]

        for bad in self.__NOT_INT:
            self.__test({"xk_in": bad}, TypeError)

        # No priors provided
        test_case = {"nfs": 0, "X_init": np.full((0, n), np.nan, float), "F_init": np.full((0, m), np.nan, float), "xk_in": 0}
        self.__test(test_case, None)
        for bad in [-1, 1, 2, 5]:
            test_case["xk_in"] = bad
            self.__test(test_case, ValueError)

        # Invalid integer index with priors provided
        for bad in [-1, self.__kwargs["Prior"]["nfs"]]:
            self.__test({"xk_in": bad}, ValueError)

        bad_no_priors = {"nfs": -0, "X_init": np.full((0, n), 0.0, float), "F_init": np.full((0, m), 0.0, float), "xk_in": -1}
        for bad in [-1, 1]:
            bad_no_priors["xk_in"] = bad
            self.__test(bad_no_priors, ValueError)

    def testBounds(self):
        n = self.__kwargs["n"]
        Low = self.__kwargs["Low"]
        Upp = self.__kwargs["Upp"]

        for bad in self.__NOT_NUMPY_ARRAY:
            self.__test({"Low": bad}, TypeError)
            self.__test({"Upp": bad}, TypeError)

        # Need to be 1D arrays
        self.__test({"Low": np.atleast_2d(Low)}, ValueError)
        self.__test({"Upp": np.atleast_2d(Upp)}, ValueError)

        # Incorrect lengths
        for bad in [np.zeros(n - 1), np.zeros(n + 1)]:
            self.__test({"Low": bad}, ValueError)
            self.__test({"Upp": bad}, ValueError)

        # Sensible +/-Inf values are acceptable, ...
        self.__test({"Low": np.full(n, -np.inf, float)}, None)
        self.__test({"Upp": np.full(n, np.inf, float)}, None)

        # NaN values, not so much.
        for i in range(n):
            Low_to_fail = Low.copy()
            Low_to_fail[i] = np.nan
            self.__test({"Low": Low_to_fail}, ValueError)

            Upp_to_fail = Upp.copy()
            Upp_to_fail[i] = np.nan
            self.__test({"Upp": Upp_to_fail}, ValueError)

        # Low >= Upp
        for i in range(n):
            Low_to_fail = Low.copy()
            Low_to_fail[i] = Upp[i]
            self.__test({"Low": Low_to_fail}, ValueError)
