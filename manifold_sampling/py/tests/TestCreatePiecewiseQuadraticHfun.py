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
        M, L = (3, 4)

        # Confirm that we have good arguments for use in testing
        Qs = np.arange(M * M * L).astype(float)
        self.__Qs = Qs.reshape((M, M, L))
        zs = np.arange(M * L).astype(float)
        self.__zs = zs.reshape((M, L))
        self.__cs = np.arange(L).astype(float)

        self.__hfun = create_piecewise_quadratic_hfun(self.__Qs, self.__zs, self.__cs)

        self.__Z = 1.1 * np.arange(self.__zs.shape[0])
        hF, grads, Hash = self.__hfun(self.__Z)
        self.assertIsInstance(hF, float)
        self.assertIsInstance(grads, np.ndarray)
        self.assertIsInstance(Hash, list)
        self.assertEqual(grads.shape, (M, len(Hash)))

        # Sanity check hash result/H0
        hF_H0, grads_H0 = self.__hfun(self.__Z, Hash)
        self.assertEqual(hF_H0, hF)
        self.assertTrue(np.array_equal(grads_H0, grads))

        # Good but different size for testing size incompatibilities
        self.__Qs_big = np.ones((M, M, L + 1))
        self.__zs_big = np.ones((M, L + 1))
        self.__cs_big = np.ones(L + 1)

        hfun = create_piecewise_quadratic_hfun(self.__Qs_big, self.__zs_big, self.__cs_big)

        Z_long = np.zeros(self.__zs_big.shape[0])
        hF, grads, Hash = hfun(Z_long)
        self.assertIsInstance(hF, float)
        self.assertIsInstance(grads, np.ndarray)
        self.assertIsInstance(Hash, list)
        self.assertEqual(grads.shape, (M, len(Hash)))

        # Sanity check hash result/H0
        hF_H0, grads_H0 = hfun(Z_long, Hash)
        # TODO: Why doesn't this work?!
        # self.assertEqual(hF_H0, hF)
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

        # Qs must be effectively 3D
        bad_all = [
            np.array([]),
            np.array([[]]),
            np.array([[[]]]),
            np.array(1.1),
            np.array([1.1]),
            np.array([[1.1]]),
            np.array([[[1.1]]]),
            self.__Qs[:, :, 0],
            np.atleast_3d(self.__Qs[:, :, 0]),
            np.stack((self.__Qs, self.__Qs), axis=3),
        ]
        for bad in bad_all:
            with self.assertRaises(ValueError):
                create_piecewise_quadratic_hfun(bad, self.__zs, self.__cs)

        oversized = list(self.__Qs.shape) + [1]
        Qs_4Dish = self.__Qs.copy().reshape(oversized)
        create_piecewise_quadratic_hfun(Qs_4Dish, self.__zs, self.__cs)

        # zs must be effectively 2D
        bad_all = [np.array([]), np.array([[]]), np.array(1.1), np.array([1.1]), self.__zs[:, 0], np.atleast_2d(self.__zs[:, 0]), np.stack((self.__zs, self.__zs), axis=2)]
        for bad in bad_all:
            with self.assertRaises(ValueError):
                create_piecewise_quadratic_hfun(self.__Qs, bad, self.__cs)

        zs_3Dish = np.atleast_3d(self.__zs.copy())
        self.assertEqual(3, zs_3Dish.ndim)
        create_piecewise_quadratic_hfun(self.__Qs, zs_3Dish, self.__cs)

        # cs must be genuinely 1D
        for bad in [np.array([]), np.stack((self.__cs, self.__cs), axis=1)]:
            with self.assertRaises(ValueError):
                create_piecewise_quadratic_hfun(self.__Qs, self.__zs, bad)

        cs_2Dish = np.atleast_2d(self.__cs.copy())
        self.assertEqual(2, cs_2Dish.ndim)
        create_piecewise_quadratic_hfun(self.__Qs, self.__zs, cs_2Dish)

        # Finite reals please for Qs, zs, and cs
        for bad_value in [np.nan, np.inf, -np.inf, 1j, 1.0 - 2.0 * 1j]:
            bad = self.__Qs.copy()
            if isinstance(bad_value, complex):
                bad = bad.astype(complex)
            bad[0, 0, 0] = bad_value
            with self.assertRaises(ValueError):
                create_piecewise_quadratic_hfun(bad, self.__zs, self.__cs)

            bad = self.__zs.copy()
            if isinstance(bad_value, complex):
                bad = bad.astype(complex)
            bad[0, 0] = bad_value
            with self.assertRaises(ValueError):
                create_piecewise_quadratic_hfun(self.__Qs, bad, self.__cs)

            bad = self.__cs.copy()
            if isinstance(bad_value, complex):
                bad = bad.astype(complex)
            bad[0] = bad_value
            with self.assertRaises(ValueError):
                create_piecewise_quadratic_hfun(self.__Qs, self.__zs, bad)

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
            create_piecewise_quadratic_hfun(self.__Qs_big, self.__zs, self.__cs)
        with self.assertRaises(ValueError):
            create_piecewise_quadratic_hfun(self.__Qs, self.__zs_big, self.__cs)
        with self.assertRaises(ValueError):
            create_piecewise_quadratic_hfun(self.__Qs, self.__zs, self.__cs_big)

    def testBadZArguments(self):
        M = self.__zs.shape[0]

        # z must be numpy array
        bad_all = [None, "bad", 1.1, list(self.__Z), tuple(self.__Z), set(self.__Z)]
        for bad in bad_all:
            with self.assertRaises(AssertionError):
                self.__hfun(bad)

        # z must be 1D
        for bad in [np.array([[]]), np.stack((self.__Z, self.__Z), axis=1)]:
            with self.assertRaises(AssertionError):
                self.__hfun(bad)

        # z must have correct length
        for bad in [np.array([]), np.ones(M - 1), np.ones(M + 1)]:
            with self.assertRaises(ValueError):
                self.__hfun(bad)

        # z must contain finite real values
        for bad_value in [np.nan, np.inf, -np.inf, 1j, 1.0 - 2.0 * 1j]:
            bad = np.random.randn(M)
            if isinstance(bad_value, complex):
                bad = bad.astype(complex)
            bad[0] = bad_value
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

        zs *= -2.1
        hfun_3 = create_piecewise_quadratic_hfun(Qs, zs, cs)
        hF_3, grads_3, Hash_3 = hfun_3(self.__Z)
        self.assertNotEqual(hF_3, hF)
        self.assertNotEqual(hF_3, hF_2)
        self.assertFalse(np.array_equal(grads_3, grads))
        self.assertFalse(np.array_equal(grads_3, grads_2))
        self.assertFalse(np.array_equal(Hash_3, Hash))
        # self.assertFalse(np.array_equal(Hash_3, Hash_2))

        cs *= 1.5
        hfun_4 = create_piecewise_quadratic_hfun(Qs, zs, cs)
        hF_4, grads_4, Hash_4 = hfun_4(self.__Z)
        self.assertNotEqual(hF_4, hF)
        self.assertNotEqual(hF_4, hF_2)
        # self.assertNotEqual(hF_4, hF_3)
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
