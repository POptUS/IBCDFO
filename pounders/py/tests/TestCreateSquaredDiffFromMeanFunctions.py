"""
Unit test of create_squared_diff_from_mean_functions()
"""

import unittest

import numpy as np

from ibcdfo.pounders import create_squared_diff_from_mean_functions


class TestCreateSquaredDiffFromMeanFunctions(unittest.TestCase):
    def setUp(self):
        # Test problem 1 setup
        #   - Here and in tests *_bar refers to an average of *
        self.__n = 2
        self.__m = 3
        self.__F = np.array([3.3, -1.1, 4.4])
        # F_bar = 2.2
        # F_bar_sqr = 4.84
        # Delta_F = F - F_bar = np.array([1.1, -3.3, 2.2])
        # F_sum = 16.94

        self.__Cres = self.__F

        self.__Gres = np.array([[1.1, 0.3, 2.2], [2.2, 0.3, 1.4]])
        # G_bar = np.array([1.2, 1.3])

        self.__Hres = np.full([self.__n, self.__n, self.__m], np.nan, float)
        self.__Hres[:, :, 0] = np.array([[1.2, -2.3], [-2.3, 3.4]])
        self.__Hres[:, :, 1] = np.array([[-1.4, 3.2], [3.2, -2.1]])
        self.__Hres[:, :, 2] = np.array([[3.5, 1.8], [1.8, -2.5]])

    def testErrors(self):
        bad_all = [None, "", "bad", [1.1], (1.1,), {1.1}, {}, 1j, 1.0 - 2.0 * 1j]
        for bad in bad_all:
            with self.assertRaises(TypeError):
                create_squared_diff_from_mean_functions(bad)

        for bad in [np.nan, -np.inf, np.inf]:
            with self.assertRaises(ValueError):
                create_squared_diff_from_mean_functions(bad)

    def testConfirmImmutable(self):
        # Construct using variable declared in this scope & collect results
        alpha = 1.2
        hfun, combinemodels = create_squared_diff_from_mean_functions(alpha)
        hF = hfun(self.__F)
        G, H = combinemodels(self.__Cres, self.__Gres, self.__Hres)

        # Alter same construction variable & confirm that it yields different
        # results
        alpha *= -2.3
        hfun_2, combinemodels_2 = create_squared_diff_from_mean_functions(alpha)
        hF_2 = hfun_2(self.__F)
        G_2, H_2 = combinemodels_2(self.__Cres, self.__Gres, self.__Hres)
        self.assertNotEqual(hF, hF_2)
        self.assertEqual(G.shape, G_2.shape)
        self.assertFalse(np.array_equal(G, G_2))
        self.assertEqual(H.shape, H_2.shape)
        self.assertFalse(np.array_equal(H, H_2))

        # Confirm that changing the construction variable didn't alter the
        # original functions
        hF_new = hfun(self.__F)
        G_new, H_new = combinemodels(self.__Cres, self.__Gres, self.__Hres)
        self.assertEqual(hF, hF_new)
        self.assertTrue(np.array_equal(G, G_new))
        self.assertTrue(np.array_equal(H, H_new))

    def testFunctions(self):
        EPS = np.finfo(float).eps

        # Handworked intermediate results for test problem 1
        H_bar = np.array([[1.1, 0.9], [0.9, -0.4]])
        H_JmGJmG = np.array([[3.64, 1.82], [1.82, 3.64]])
        H_GG = np.array([[2.88, 3.12], [3.12, 3.38]])
        H_FH = np.array([[27.28, -18.26], [-18.26, 10.34]])

        # Include
        # - Negative and positive
        # - Integer and float literals
        # - The special case of alpha=0.0
        for alpha in [-1.1, -5, 0.0, 2.3, 4]:
            hfun, combinemodels = create_squared_diff_from_mean_functions(alpha)
            self.assertTrue(callable(hfun))
            self.assertTrue(callable(combinemodels))

            # Check F = F_bar = 0.0 super-duper special case
            F = np.zeros(10)
            G, H = combinemodels(F, self.__Gres, self.__Hres)
            self.assertEqual(len(G), self.__n)
            self.assertEqual(H.shape, (self.__n, self.__n))
            self.assertTrue(np.array_equal(H, H.T))

            self.assertEqual(0.0, hfun(F))

            self.assertTrue(all(G == 0.0))

            H_f = H_JmGJmG - alpha * H_GG
            max_abs_err = np.max(np.fabs(H.flatten() - H_f.flatten()))
            self.assertTrue(max_abs_err <= 35.0 * EPS)

            # Check F - F_bar = 0 with F_bar != 0 special cases
            for F_bar in [-10.1, 3.3]:
                F = np.full(6, F_bar, float)
                hF = hfun(F)
                G, H = combinemodels(F, self.__Gres, self.__Hres)
                self.assertEqual(len(G), self.__n)
                self.assertEqual(H.shape, (self.__n, self.__n))
                self.assertTrue(np.array_equal(H, H.T))

                # The MATLAB version of this test confirms results identical to
                # correct result.
                abs_err = np.fabs(hF + alpha * F_bar**2)
                self.assertTrue(abs_err <= 70.0 * EPS)

                grad_f = -2.0 * alpha * F_bar * np.array([1.2, 1.3])
                max_abs_err = np.max(np.fabs(G - grad_f))
                self.assertTrue(max_abs_err <= 130.0 * EPS)

                H_f = H_JmGJmG - alpha * (H_GG + 2.0 * F_bar * H_bar)
                max_abs_err = np.max(np.fabs(H.flatten() - H_f.flatten()))
                self.assertTrue(max_abs_err <= 195.0 * EPS)

            # Check generic test problem
            hF = hfun(self.__F)
            G, H = combinemodels(self.__Cres, self.__Gres, self.__Hres)
            self.assertEqual(len(G), self.__n)
            self.assertEqual(H.shape, (self.__n, self.__n))
            self.assertTrue(np.array_equal(H, H.T))

            hF_expected = 16.94 - alpha * 4.84
            abs_err = np.fabs(hF - hF_expected)
            self.assertTrue(abs_err <= 20.0 * EPS)

            grad_f = np.array([10.12, 9.02]) - alpha * np.array([5.28, 5.72])
            max_abs_err = np.max(np.fabs(G - grad_f))
            self.assertTrue(max_abs_err <= 35.0 * EPS)

            H_f = H_JmGJmG + H_FH - alpha * (H_GG + 2.0 * 2.2 * H_bar)
            max_abs_err = np.max(np.fabs(H.flatten() - H_f.flatten()))
            self.assertTrue(max_abs_err <= 70.0 * EPS)
