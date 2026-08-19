"""
Unit test of POUNDERS interface.
"""

import io
import copy
import numbers
import unittest

import numpy as np
import itertools as it

from contextlib import redirect_stdout

import ibcdfo
from ibcdfo.pounders.defaults import (
    ALL_MODEL_KEYS,
    ALL_OPTIONS_KEYS,
    EXPECTED_PRIOR_KEYS,
    compute_default_model,
    compute_default_options,
)


class TestPoundersInterface(unittest.TestCase):
    def setUp(self):
        self.__SUCCESS_MSG = "g is sufficiently small."
        self.__ERROR_HDR = "Error: "

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
        #
        # Tests that need to change values should make a deepcopy and alter that
        # dictionary.
        self.__KWARGS = {
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
        }
        self.__KWARGS["Model"] = compute_default_model(self.__KWARGS["n"])
        self.__KWARGS["Options"] = compute_default_options(
            self.__KWARGS["delta_0"],
            self.__KWARGS["g_tol"],
            self.__KWARGS["Low"],
            self.__KWARGS["Upp"],
        )
        self.assertEqual(set(self.__KWARGS["Prior"]), EXPECTED_PRIOR_KEYS)
        self.assertEqual(set(self.__KWARGS["Model"]), ALL_MODEL_KEYS)
        self.assertEqual(set(self.__KWARGS["Options"]), ALL_OPTIONS_KEYS)

        self.__NOT_REAL = (None, "", {}, [1.1], {1.1})
        self.__NOT_INT = (None, "", 1.0, 1.1, {}, [1], {1})
        self.__NOT_NUMPY_ARRAY = (None, "", 1, 1.1, {}, [1], {1})
        self.__NOT_FUNCTION = (None, "", 1, 1.1, {}, [1], {1})

    def __test(self, new_args, expected_exception):
        # For `expected_exception`, pass
        # - a Python Exception object to confirm that an expectation of that
        #   type was raised
        # - self.__SUCCESS_MSG to confirm that the optimization ran successfully
        #   to completion (flag = 0)
        # - a different string (including "") to confirm that the optimization
        #   terminated with different (or no) output and with flag >= 0.0

        # ----- HARDCODED VALUE
        EPS = np.finfo(float).eps

        # ----- ALTER CONFIGURATION
        kwargs = copy.deepcopy(self.__KWARGS)
        for key, value in new_args.items():
            if key in EXPECTED_PRIOR_KEYS:
                kwargs["Prior"][key] = value
            elif key in ALL_MODEL_KEYS:
                kwargs["Model"][key] = value
            elif key in ALL_OPTIONS_KEYS:
                kwargs["Options"][key] = value
            else:
                kwargs[key] = value

        # ----- OPTIMIZE & ERROR CHECK
        for solver in [ibcdfo.run_pounders]:
            if isinstance(expected_exception, str):
                with redirect_stdout(io.StringIO()) as buffer:
                    X, F, hF, flag, xk_in = solver(**kwargs)
                self.assertEqual(buffer.getvalue().strip(), expected_exception)

                self.assertTrue(isinstance(X, np.ndarray))
                self.assertEqual(X.ndim, 2)
                # X.shape[0] checked below
                self.assertEqual(X.shape[1], kwargs["n"])

                self.assertTrue(isinstance(F, np.ndarray))
                self.assertEqual(F.ndim, 2)
                self.assertEqual(F.shape[0], X.shape[0])
                self.assertEqual(F.shape[1], kwargs["m"])

                self.assertTrue(isinstance(hF, np.ndarray))
                self.assertEqual(hF.ndim, 1)
                self.assertEqual(len(hF), X.shape[0])

                self.assertTrue(isinstance(xk_in, numbers.Integral))
                self.assertTrue(0 <= xk_in < X.shape[0])

                nfs = 0 if kwargs["Prior"] is None else kwargs["Prior"]["nfs"]
                budget = kwargs["nf_max"] + nfs
                if expected_exception == self.__SUCCESS_MSG:
                    self.assertTrue(isinstance(flag, numbers.Integral))
                    self.assertEqual(flag, 0)

                    self.assertTrue(nfs < X.shape[0] <= budget)

                    max_rel_err = np.max(np.fabs(1.0 - X[xk_in, :] / self.__THETA_STAR))
                    self.assertTrue(max_rel_err <= 4.0 * EPS)
                    self.assertTrue(np.max(np.fabs(F[xk_in, :])) <= 4.0 * EPS)
                    self.assertTrue(np.fabs(hF[xk_in]) <= 4.0 * EPS)
                else:
                    self.assertTrue(isinstance(flag, numbers.Real))
                    if flag > 0.0:
                        self.assertTrue(flag < 165.0)
                        self.assertEqual(X.shape[0], budget)
                    elif flag == 0.0:
                        self.assertTrue(nfs < X.shape[0] <= budget)
                    else:
                        # Hopefully tests never hit this.
                        self.assertTrue(flag >= 0.0)
            else:
                with self.assertRaises(expected_exception) as err:
                    solver(**kwargs)
                err_msg = str(err.exception)
                # print(err_msg)
                self.assertTrue(err_msg.startswith(self.__ERROR_HDR))

    def testConfirmGoodArguments(self):
        self.__test({}, self.__SUCCESS_MSG)

    def testPriorKeys(self):
        n = self.__KWARGS["n"]
        X_0 = self.__KWARGS["X_0"].copy()
        Ffun = self.__KWARGS["Ffun"]

        # None is fine, ...
        self.__test({"Prior": None}, self.__SUCCESS_MSG)
        # but not empty containers
        for bad in [set(), []]:
            self.__test({"Prior": bad}, TypeError)

        self.__test({"Prior": {}}, ValueError)

        # Can't take any away
        for key in EXPECTED_PRIOR_KEYS:
            bad = copy.deepcopy(self.__KWARGS["Prior"])
            del bad[key]
            self.assertTrue(set(bad).issubset(EXPECTED_PRIOR_KEYS))
            self.assertNotEqual(set(bad), EXPECTED_PRIOR_KEYS)
            self.__test({"Prior": bad}, ValueError)

        # Can't add any extras
        bad = copy.deepcopy(self.__KWARGS["Prior"])
        bad["!NOT A KEY!"] = 1.1
        self.assertNotEqual(set(bad), EXPECTED_PRIOR_KEYS)
        self.__test({"Prior": bad}, ValueError)

        # Confirm that providing starting point value doesn't change outputs.
        #
        # NOTE: Unfortunately this doesn't confirm that POUNDERS used the
        # starting point since it could have just ignored the provided value and
        # continued as usual.  This test could potentially be useful during
        # manual testing if the POUNDERS log output makes it clear that it
        # didn't evaluate the starting point and, therefore, performed one less
        # evaluation.
        kwargs = copy.deepcopy(self.__KWARGS)
        kwargs["Prior"] = None
        kwargs["nf_max"] = n + 2
        X, F, hF, flag, xk_in = ibcdfo.run_pounders(**kwargs)
        self.assertTrue(flag > 0.0)
        self.assertEqual(len(hF), kwargs["nf_max"])

        kwargs["Prior"] = {
            "nfs": 1,
            "X_init": np.atleast_2d(X_0),
            "F_init": np.atleast_2d(Ffun(X_0)),
            "xk_in": 0,
        }
        kwargs["nf_max"] = n + 1
        X_1, F_1, hF_1, flag_1, xk_in_1 = ibcdfo.run_pounders(**kwargs)
        self.assertEqual(flag, flag_1)
        self.assertEqual(len(hF), kwargs["nf_max"] + kwargs["Prior"]["nfs"])
        self.assertTrue(np.array_equal(X, X_1))
        self.assertTrue(np.array_equal(F, F_1))
        self.assertTrue(np.array_equal(hF, hF_1))
        self.assertEqual(xk_in, xk_in_1)

    def testModelKeys(self):
        for good in [None, {}]:
            self.__test({"Model": good}, self.__SUCCESS_MSG)

        # Get benchmark result with all arguments passed with default values
        kwargs = copy.deepcopy(self.__KWARGS)
        with redirect_stdout(io.StringIO()) as buffer:
            X_full, F_full, hF_full, flag, xk_in_full = ibcdfo.run_pounders(**kwargs)
        self.assertEqual(buffer.getvalue().strip(), self.__SUCCESS_MSG)
        self.assertEqual(flag, 0)

        # Confirm identical results when all proper subsets passed.
        params_fixed = sorted(set(self.__KWARGS["Model"]))
        for i in range(1, len(params_fixed)):
            for subset in it.combinations(params_fixed, i):
                # print(i, subset)
                kwargs["Model"] = {key: self.__KWARGS["Model"][key] for key in subset}
                with redirect_stdout(io.StringIO()) as buffer:
                    X, F, hF, flag, xk_in = ibcdfo.run_pounders(**kwargs)
                self.assertEqual(buffer.getvalue().strip(), self.__SUCCESS_MSG)
                self.assertEqual(flag, 0)
                self.assertTrue(np.array_equal(X, X_full))
                self.assertTrue(np.array_equal(F, F_full))
                self.assertTrue(np.array_equal(hF, hF_full))
                self.assertEqual(xk_in, xk_in_full)

        for bad in [set(), []]:
            self.__test({"Model": bad}, TypeError)

        # Can't add any extras
        bad = copy.deepcopy(self.__KWARGS["Model"])
        bad["!NOT A KEY!"] = 1.1
        self.assertNotEqual(set(bad), ALL_MODEL_KEYS)
        self.__test({"Model": bad}, ValueError)

    def testOptionsKeys(self):
        for good in [None, {}]:
            self.__test({"Options": good}, self.__SUCCESS_MSG)

        # Get benchmark result with all arguments passed with default values
        kwargs = copy.deepcopy(self.__KWARGS)
        with redirect_stdout(io.StringIO()) as buffer:
            X_full, F_full, hF_full, flag, xk_in_full = ibcdfo.run_pounders(**kwargs)
        self.assertEqual(buffer.getvalue().strip(), self.__SUCCESS_MSG)
        self.assertEqual(flag, 0)

        # Confirm identical results when all proper subsets passed.
        defaults = copy.deepcopy(self.__KWARGS["Options"])
        params_fixed = sorted(set(defaults).difference({"hfun", "combinemodels"}))
        for i in range(1, len(params_fixed)):
            for subset in it.combinations(params_fixed, i):
                # print(i, subset)
                kwargs["Options"] = {key: defaults[key] for key in subset}
                with redirect_stdout(io.StringIO()) as buffer:
                    X, F, hF, flag, xk_in = ibcdfo.run_pounders(**kwargs)
                self.assertEqual(buffer.getvalue().strip(), self.__SUCCESS_MSG)
                self.assertEqual(flag, 0)
                self.assertTrue(np.array_equal(X, X_full))
                self.assertTrue(np.array_equal(F, F_full))
                self.assertTrue(np.array_equal(hF, hF_full))
                self.assertEqual(xk_in, xk_in_full)

                kwargs["Options"]["hfun"] = defaults["hfun"]
                kwargs["Options"]["combinemodels"] = defaults["combinemodels"]
                with redirect_stdout(io.StringIO()) as buffer:
                    X, F, hF, flag, xk_in = ibcdfo.run_pounders(**kwargs)
                self.assertEqual(buffer.getvalue().strip(), self.__SUCCESS_MSG)
                self.assertEqual(flag, 0)
                self.assertTrue(np.array_equal(X, X_full))
                self.assertTrue(np.array_equal(F, F_full))
                self.assertTrue(np.array_equal(hF, hF_full))
                self.assertEqual(xk_in, xk_in_full)

        for key in ALL_OPTIONS_KEYS:
            self.__test({key: self.__KWARGS["Options"][key]}, self.__SUCCESS_MSG)

        for bad in [set(), []]:
            self.__test({"Options": bad}, TypeError)

        # Can't add any extras
        bad = copy.deepcopy(self.__KWARGS["Options"])
        bad["!NOT A KEY!"] = 1.1
        self.assertNotEqual(set(bad), ALL_OPTIONS_KEYS)
        self.__test({"Options": bad}, ValueError)

        # We either don't pass hfun/combinemodels or we pass both
        bad = copy.deepcopy(self.__KWARGS["Options"])
        del bad["hfun"]
        self.__test({"Options": bad}, ValueError)

        bad = copy.deepcopy(self.__KWARGS["Options"])
        del bad["combinemodels"]
        self.__test({"Options": bad}, ValueError)

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
        self.assertEqual(self.__KWARGS["n"], 3)
        MIN, MAX = 4, 10

        for bad in self.__NOT_INT:
            self.__test({"np_max": bad}, TypeError)
        for good in range(MIN, MAX + 1):
            self.__test({"np_max": good}, self.__SUCCESS_MSG)
        for bad in [MIN - 1, MAX + 1]:
            self.__test({"np_max": bad}, ValueError)

    def testNfMax(self):
        n = self.__KWARGS["n"]
        m = self.__KWARGS["m"]
        X_0 = self.__KWARGS["X_0"].copy()
        Ffun = self.__KWARGS["Ffun"]

        self.assertEqual(n, 3)
        self.assertEqual(self.__KWARGS["Prior"]["nfs"], 2)

        for bad in self.__NOT_INT:
            self.__test({"nf_max": bad}, TypeError)

        # No prior evaluations given
        min_required = n + 2
        test_case = {
            "nfs": 0,
            "X_init": np.full((0, n), np.nan, float),
            "F_init": np.full((0, m), np.nan, float),
            "xk_in": 0,
        }
        for good in [0, 1, 2]:
            # Too few evaluations to converge to solution
            test_case["nf_max"] = min_required + good
            self.__test(test_case, "")
        for good in [10, 10_000]:
            test_case["nf_max"] = min_required + good
            self.__test(test_case, self.__SUCCESS_MSG)
        for bad in range(min_required):
            test_case["nf_max"] = bad
            self.__test(test_case, ValueError)

        # Starting point value provided
        min_required = n + 1
        test_case = {
            "nfs": 1,
            "X_init": np.atleast_2d(X_0),
            "F_init": np.atleast_2d(Ffun(X_0)),
            "xk_in": 0,
        }
        for good in [0, 1, 2]:
            # Too few evaluations to converge to solution
            test_case["nf_max"] = min_required + good
            self.__test(test_case, "")
        for good in [10, 10_000]:
            test_case["nf_max"] = min_required + good
            self.__test(test_case, self.__SUCCESS_MSG)
        for bad in range(min_required):
            test_case["nf_max"] = bad
            self.__test(test_case, ValueError)

        # Two prior evaluation
        min_required = n + 1
        for good in [0, 1, 2]:
            self.__test({"nf_max": min_required + good}, "")
        for good in [10, 10_000]:
            self.__test({"nf_max": min_required + good}, self.__SUCCESS_MSG)
        for bad in range(min_required):
            self.__test({"nf_max": bad}, ValueError)

    def testX0Errors(self):
        EPS = np.finfo(float).eps
        WARNING = "Note: Geometry points need to be coordinate directions!"

        n = self.__KWARGS["n"]
        X_0 = self.__KWARGS["X_0"].copy()

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
        self.__test({"Low": X_0}, WARNING)
        self.__test({"Upp": X_0}, f"{WARNING}\n{self.__SUCCESS_MSG}")

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
        EPS = np.finfo(float).eps
        NFS_NEW = 3

        Ffun = self.__KWARGS["Ffun"]
        n = self.__KWARGS["n"]
        nfs = self.__KWARGS["Prior"]["nfs"]
        X_init = self.__KWARGS["Prior"]["X_init"].copy()

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
        test_case = {
            "nfs": NFS_NEW,
            "X_init": None,
            "F_init": None,
            "xk_in": -1,
        }
        for xk_in in range(NFS_NEW):
            i, j = sorted(set(range(NFS_NEW)).difference({xk_in}))

            X_init = np.full((NFS_NEW, n), np.nan, float)
            test_case["xk_in"] = xk_in
            test_case["X_init"] = X_init

            # See valid initial points pass ...
            X_init[xk_in, :] = self.__KWARGS["X_0"].copy()
            X_init[i, :] = self.__KWARGS["Low"] - 0.1
            X_init[j, :] = self.__KWARGS["Upp"] + 0.1
            test_case["F_init"] = np.array([Ffun(X_init[i, :]) for i in range(X_init.shape[0])])
            self.__test(test_case, self.__SUCCESS_MSG)

            # and then let's make 'em fail.
            for k in range(n):
                X_init[xk_in, :] = self.__KWARGS["X_0"].copy()
                X_init[xk_in, k] = (1.0 + EPS) * X_init[xk_in, k]
                self.__test(test_case, ValueError)

            # Confirm no repeated points even if they have the same F values
            X_init[xk_in, :] = self.__KWARGS["X_0"].copy()
            X_init[j, :] = X_init[i, :]
            test_case["F_init"] = np.array([Ffun(X_init[i, :]) for i in range(X_init.shape[0])])
            self.__test(test_case, ValueError)

            X_init[j, :] = X_init[xk_in, :]
            test_case["F_init"] = np.array([Ffun(X_init[i, :]) for i in range(X_init.shape[0])])
            self.__test(test_case, ValueError)

    def testFinitErrors(self):
        m = self.__KWARGS["m"]
        nfs = self.__KWARGS["Prior"]["nfs"]
        F_init = self.__KWARGS["Prior"]["F_init"].copy()

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
        n = self.__KWARGS["n"]
        m = self.__KWARGS["m"]

        for bad in self.__NOT_INT:
            self.__test({"xk_in": bad}, TypeError)

        # No priors provided
        test_case = {
            "nfs": 0,
            "X_init": np.full((0, n), np.nan, float),
            "F_init": np.full((0, m), np.nan, float),
            "xk_in": 0,
        }
        self.__test(test_case, self.__SUCCESS_MSG)
        for bad in [-1, 1, 2, 5]:
            test_case["xk_in"] = bad
            self.__test(test_case, ValueError)

        # Invalid integer index with priors provided
        for bad in [-1, self.__KWARGS["Prior"]["nfs"]]:
            self.__test({"xk_in": bad}, ValueError)

    def testBounds(self):
        n = self.__KWARGS["n"]
        Low = self.__KWARGS["Low"].copy()
        Upp = self.__KWARGS["Upp"].copy()

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
        self.__test({"Low": np.full(n, -np.inf, float)}, self.__SUCCESS_MSG)
        self.__test({"Upp": np.full(n, np.inf, float)}, self.__SUCCESS_MSG)

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
