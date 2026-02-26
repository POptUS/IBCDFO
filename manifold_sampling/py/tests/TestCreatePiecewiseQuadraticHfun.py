"""
Unit test of create_piecewise_quadratic_hfun()

Claude Sonnet 4.5 via Argo was used to generate a starting point for this test.
It was provided with the current working versions of

1. create_piecewise_quadratic_hfun.m, and
2. create_censored_L1_loss_hfun.m,
3. TestCreateCensoredL1LossHfun.m

at that time and asked to build up this test of (1) so that it tests (1) in a
way analogous to how (3) tests (2).

That code was then reviewed, cleaned, and evolved with no further contributions
or alterations by a generative AI tool.  Developers are, as expected, taking
responsibility for the correctness of this content and its ability to test the
related code.
"""

import unittest

import numpy as np

from ibcdfo.manifold_sampling import create_piecewise_quadratic_hfun


class TestCreatePiecewiseQuadraticHfun(unittest.TestCase):
    def setUp(self):
        EPS = np.finfo(float).eps
        M, L = (3, 4)

        # Confirm that we have good arguments for use in testing
        #
        # Use Fortran ordering so that this creates the same problem as defined
        # in the MATLAB test suite.
        Qs = np.arange(1, M * M * L + 1).astype(float)
        self.__Qs = Qs.reshape((M, M, L), order="F")
        zs = np.arange(1, M * L + 1).astype(float)
        self.__zs = zs.reshape((M, L), order="F")
        self.__cs = np.arange(1, L + 1).astype(float)

        self.__hfun = create_piecewise_quadratic_hfun(self.__Qs, self.__zs, self.__cs)

        # The test values were not determined by hand and are, therefore, not
        # known to be correct.  Rather, they were gathered from test output at
        # some time and are used here to catch regressions and to confirm that
        # the Python and MATLAB code are returning the same values for the same
        # problems.
        self.__Z = 1.1 * np.arange(1, M + 1)
        self.__hF, self.__grads, self.__Hash = self.__hfun(self.__Z)
        self.assertEqual(self.__hF, 22285.6)
        expected = np.array([-1635.6, -1688.4, -1741.2])
        rel_diff = np.max(np.fabs(1.0 - np.squeeze(self.__grads) / expected))
        self.assertTrue(rel_diff <= 5.0 * EPS)
        self.assertEqual(len(self.__Hash), 1)
        self.assertEqual(self.__grads.shape, (M, 1))
        # In the MATLAB version of this test, the hash is "4".  This is due to
        # the fact that for this problem the hash result is the index to the
        # active quadratic piece and indices in MATLAB are 1-based instead of
        # 0-based.
        self.assertEqual(self.__Hash, ["3"])

        # Sanity check hash result/H0
        hF_H0, grads_H0 = self.__hfun(self.__Z, self.__Hash)
        self.assertEqual(hF_H0, self.__hF)
        self.assertTrue(np.array_equal(grads_H0, self.__grads))

        # Good but different size for testing size incompatibilities
        M_long = M + 1
        L_long = L - 1
        Qs = np.arange(1, M_long * M_long * L_long + 1).astype(float)
        self.__Qs_long = Qs.reshape((M_long, M_long, L_long), order="F")
        zs = np.arange(1, M_long * L_long + 1).astype(float)
        self.__zs_long = zs.reshape((M_long, L_long), order="F")
        self.__cs_long = np.arange(1, L_long + 1).astype(float)

        hfun = create_piecewise_quadratic_hfun(self.__Qs_long, self.__zs_long, self.__cs_long)

        Z_long = 2.1 * np.arange(1, M_long + 1)
        hF, grads, Hash = hfun(Z_long)
        self.assertEqual(hF, 17286.0)
        expected = np.array([-1594, -1636, -1678, -1720])
        rel_diff = np.max(np.fabs(1.0 - np.squeeze(grads) / expected))
        self.assertTrue(rel_diff <= 5.0 * EPS)
        self.assertEqual(len(Hash), 1)
        self.assertEqual(grads.shape, (M_long, 1))
        # Similar to above comment, the hash here is one less than the result
        # in the MATLAB version of this test.
        self.assertEqual(Hash, ["2"])

        # Sanity check hash result/H0
        hF_H0, grads_H0 = hfun(Z_long, Hash)
        self.assertEqual(hF_H0, hF)
        self.assertTrue(np.array_equal(grads_H0, grads))

    def testErrors(self):
        M, L = self.__zs.shape

        # Qs, zs & cs must be numpy arrays ...
        bad_all = [None, "bad", 1.1, [1.1], (1.1,), {1.1}]
        for bad in bad_all:
            with self.assertRaises(TypeError):
                create_piecewise_quadratic_hfun(bad, self.__zs, self.__cs)
            with self.assertRaises(TypeError):
                create_piecewise_quadratic_hfun(self.__Qs, bad, self.__cs)
            with self.assertRaises(TypeError):
                create_piecewise_quadratic_hfun(self.__Qs, self.__zs, bad)

        # Qs must be 3D ...
        bad_all = [
            np.array([]),
            np.array([[]]),
            np.array(1.1),
            np.array([1.1]),
            np.array([[1.1]]),
            self.__Qs[:, :, 0],
            np.stack((self.__Qs, self.__Qs), axis=3),
        ]
        for bad in bad_all:
            self.assertTrue(bad.ndim != 3)
            with self.assertRaises(ValueError):
                create_piecewise_quadratic_hfun(bad, self.__zs, self.__cs)

        # zs must be 2D
        bad_all = [
            np.array([]),
            np.array(1.1),
            np.array([1.1]),
            self.__zs[:, 0],
            np.stack((self.__zs, self.__zs), axis=2),
            np.atleast_3d(self.__zs.copy()),
        ]
        for bad in bad_all:
            self.assertTrue(bad.ndim != 2)
            with self.assertRaises(ValueError):
                create_piecewise_quadratic_hfun(self.__Qs, bad, self.__cs)

        # cs must be effectively 1D (i.e., allow for 2D row/column vectors)
        for bad in [np.array([]), np.stack((self.__cs, self.__cs), axis=1)]:
            with self.assertRaises(ValueError):
                create_piecewise_quadratic_hfun(self.__Qs, self.__zs, bad)

        cs_row = np.atleast_2d(self.__cs.copy()).T
        self.assertEqual(2, cs_row.ndim)
        self.assertNotEqual(1, cs_row.shape[0])
        hfun_r = create_piecewise_quadratic_hfun(self.__Qs, self.__zs, cs_row)
        hF_r, grads_r, Hash_r = hfun_r(self.__Z)
        self.assertEqual(hF_r, self.__hF)
        self.assertTrue(np.array_equal(grads_r, self.__grads))
        self.assertEqual(Hash_r, self.__Hash)

        cs_column = cs_row.T
        self.assertEqual(2, cs_column.ndim)
        self.assertNotEqual(1, cs_column.shape[1])
        hfun_c = create_piecewise_quadratic_hfun(self.__Qs, self.__zs, cs_column)
        hF_c, grads_c, Hash_c = hfun_c(self.__Z)
        self.assertEqual(hF_c, self.__hF)
        self.assertTrue(np.array_equal(grads_c, self.__grads))
        self.assertEqual(Hash_c, self.__Hash)

        # Finite reals please for Qs, zs, and cs
        for bad_value in [np.nan, np.inf, -np.inf, 1j, 1.0 - 2.0 * 1j]:
            for k in range(L):
                bad = self.__cs.copy()
                if isinstance(bad_value, complex):
                    bad = bad.astype(complex)
                bad[k] = bad_value
                with self.assertRaises(ValueError):
                    create_piecewise_quadratic_hfun(self.__Qs, self.__zs, bad)

                for i in range(M):
                    bad = self.__zs.copy()
                    if isinstance(bad_value, complex):
                        bad = bad.astype(complex)
                    bad[i, k] = bad_value
                    with self.assertRaises(ValueError):
                        create_piecewise_quadratic_hfun(self.__Qs, bad, self.__cs)

                    for j in range(M):
                        bad = self.__Qs.copy()
                        if isinstance(bad_value, complex):
                            bad = bad.astype(complex)
                        bad[i, j, k] = bad_value
                        with self.assertRaises(ValueError):
                            create_piecewise_quadratic_hfun(bad, self.__zs, self.__cs)

        # Qs, zs & cs must have compatible shapes
        # Qs must be square in first two dimensions
        bad_Qs = np.ones((M, M + 1, L))
        with self.assertRaises(ValueError):
            create_piecewise_quadratic_hfun(bad_Qs, self.__zs, self.__cs)

        # zs must match Qs dimensions
        bad_zs = np.ones((M + 1, L))
        with self.assertRaises(ValueError):
            create_piecewise_quadratic_hfun(self.__Qs, bad_zs, self.__cs)

        bad_zs = np.ones((M, L + 1))
        with self.assertRaises(ValueError):
            create_piecewise_quadratic_hfun(self.__Qs, bad_zs, self.__cs)

        # cs must match Qs third dimension
        bad_cs = np.ones(L + 1)
        with self.assertRaises(ValueError):
            create_piecewise_quadratic_hfun(self.__Qs, self.__zs, bad_cs)

        # Test incompatibility between different good arrays
        with self.assertRaises(ValueError):
            create_piecewise_quadratic_hfun(self.__Qs_long, self.__zs, self.__cs)
        with self.assertRaises(ValueError):
            create_piecewise_quadratic_hfun(self.__Qs, self.__zs_long, self.__cs)
        with self.assertRaises(ValueError):
            create_piecewise_quadratic_hfun(self.__Qs, self.__zs, self.__cs_long)

        # Confirm that l must be >= 2
        Qs_bad = np.atleast_3d(self.__Qs[:, :, 0])
        zs_bad = np.atleast_2d(self.__zs[:, 0]).T
        cs_bad = np.atleast_1d(self.__cs[0])
        # Confirm that they are mutually compatible so that error is just due to
        # l=1.
        self.assertTrue(Qs_bad.ndim == 3)
        self.assertEqual(Qs_bad.shape[0], Qs_bad.shape[1])
        self.assertEqual(1, Qs_bad.shape[2])
        self.assertTrue(zs_bad.ndim == 2)
        self.assertEqual(zs_bad.shape[0], Qs_bad.shape[0])
        self.assertEqual(1, zs_bad.shape[1])
        self.assertEqual(1, len(cs_bad))
        with self.assertRaises(ValueError):
            create_piecewise_quadratic_hfun(Qs_bad, zs_bad, cs_bad)

    def testBadZArguments(self):
        M = self.__zs.shape[0]

        # z must be numpy array
        bad_all = [None, "bad", 1.1, list(self.__Z), tuple(self.__Z), set(self.__Z)]
        for bad in bad_all:
            with self.assertRaises(AssertionError):
                self.__hfun(bad)

        # z must be 1D
        bad_all = [np.array([[]]), np.atleast_2d(self.__Z), np.stack((self.__Z, self.__Z), axis=1)]
        for bad in bad_all:
            with self.assertRaises(AssertionError):
                self.__hfun(bad)

        # z must have correct length
        for bad in [np.array([]), np.ones(M - 1), np.ones(M + 1)]:
            with self.assertRaises(ValueError):
                self.__hfun(bad)

        # z must contain finite real values
        for bad_value in [np.nan, np.inf, -np.inf, 1j, 1.0 - 2.0 * 1j]:
            bad_orig = np.random.randn(M)
            for i in range(M):
                bad = bad_orig.copy()
                if isinstance(bad_value, complex):
                    bad = bad.astype(complex)
                bad[i] = bad_value
                with self.assertRaises(ValueError):
                    self.__hfun(bad)

    def testConfirmReadonly(self):
        # NOTE: This test is only useful for developing and manually testing the
        # code since the function under test doesn't actually alter Qs, zs, or cs
        # and likely should never do so.
        #
        # Therefore, developers can, if so desired, temporarily inject changes
        # to Qs, zs, or cs in the function and run this test to confirm that the
        # function is written such that it is incapable of inadvertently altering
        # the Qs, zs, cs arrays created and managed here.
        Qs = self.__Qs.copy()
        zs = self.__zs.copy()
        cs = self.__cs.copy()
        hfun = create_piecewise_quadratic_hfun(Qs, zs, cs)
        hfun(self.__Z)
        self.assertTrue(np.array_equal(Qs, self.__Qs))
        self.assertTrue(np.array_equal(zs, self.__zs))
        self.assertTrue(np.array_equal(cs, self.__cs))

    def testConfirmImmutable(self):
        # Construct using variables declared in this scope & collect results
        Qs = self.__Qs.copy()
        zs = self.__zs.copy()
        cs = self.__cs.copy()
        hfun = create_piecewise_quadratic_hfun(Qs, zs, cs)
        hF, grads, Hash = hfun(self.__Z)

        # Alter same construction variables & confirm that they yield different
        # results
        Qs *= -2.3
        hfun_2 = create_piecewise_quadratic_hfun(Qs, zs, cs)
        hF_2, grads_2, Hash_2 = hfun_2(self.__Z)
        self.assertNotEqual(hF_2, hF)
        self.assertFalse(np.array_equal(grads_2, grads))
        self.assertFalse(np.array_equal(Hash_2, Hash))

        zs *= -1.1
        hfun_3 = create_piecewise_quadratic_hfun(Qs, zs, cs)
        hF_3, grads_3, Hash_3 = hfun_3(self.__Z)
        self.assertNotEqual(hF_3, hF)
        self.assertNotEqual(hF_3, hF_2)
        self.assertFalse(np.array_equal(grads_3, grads))
        self.assertFalse(np.array_equal(grads_3, grads_2))
        self.assertFalse(np.array_equal(Hash_3, Hash))
        # self.assertFalse(np.array_equal(Hash_3, Hash_2))

        cs *= -1.5
        hfun_4 = create_piecewise_quadratic_hfun(Qs, zs, cs)
        hF_4, grads_4, Hash_4 = hfun_4(self.__Z)
        self.assertNotEqual(hF_4, hF)
        self.assertNotEqual(hF_4, hF_2)
        self.assertNotEqual(hF_4, hF_3)
        self.assertFalse(np.array_equal(grads_4, grads))
        self.assertFalse(np.array_equal(grads_4, grads_2))
        # self.assertFalse(np.array_equal(grads_4, grads_3))
        self.assertFalse(np.array_equal(Hash_4, Hash))
        # self.assertFalse(np.array_equal(Hash_4, Hash_2))
        # self.assertFalse(np.array_equal(Hash_4, Hash_3))

        # Confirm that changing the construction variables didn't alter the
        # original function
        hF_new, grads_new, Hash_new = hfun(self.__Z)
        self.assertEqual(hF_new, hF)
        self.assertTrue(np.array_equal(grads_new, grads))
        self.assertTrue(np.array_equal(Hash_new, Hash))
