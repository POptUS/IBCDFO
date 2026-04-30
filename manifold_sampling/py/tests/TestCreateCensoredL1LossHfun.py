"""
Unit test of create_censored_L1_loss_hfun()
"""

import unittest

import numpy as np

from ibcdfo.manifold_sampling import create_censored_L1_loss_hfun


class TestCreateCensoredL1LossHfun(unittest.TestCase):
    def setUp(self):
        M = 10
        self.__SHAPE_2D = (5, 2)

        # Confirm that we have good arguments for use in testing
        self.__C = np.arange(1, M + 1).astype(float)
        self.__D = -1.1 * np.arange(1, M + 1).astype(float)

        self.__hfun = create_censored_L1_loss_hfun(self.__C, self.__D)

        # The test values were not determined by hand and are, therefore, not
        # known to be correct.  Rather, they were gathered from test output at
        # some time and are used here to catch regressions and to confirm that
        # the Python and MATLAB code are returning the same values for the same
        # problems.
        self.__Z = 2.1 * np.arange(self.__C.shape[0])
        self.__hF, self.__grads, self.__Hash = self.__hfun(self.__Z)
        self.assertEqual(156.0, self.__hF)
        expected = np.ones((M, 1))
        expected[0] = 0
        self.assertTrue(np.array_equal(self.__grads, expected))
        self.assertEqual(self.__grads.shape, (M, len(self.__Hash)))
        self.assertEqual(self.__Hash, ["2111111111"])

        # Sanity check hash result/H0
        hF_H0, grads_H0 = self.__hfun(self.__Z, self.__Hash)
        self.assertEqual(hF_H0, self.__hF)
        self.assertTrue(np.array_equal(grads_H0, self.__grads))

        # Good but different size for testing size incompatibilities
        self.__C_short = np.array([1.1, 2.2, -3.3])
        self.__D_short = np.array([3.3, -1.1, -2.2])
        M_short = len(self.__C_short)
        self.assertNotEqual(len(self.__C_short), len(self.__C))
        self.assertNotEqual(len(self.__D_short), len(self.__D))

        hfun = create_censored_L1_loss_hfun(self.__C_short, self.__D_short)

        Z_short = 2.1 * np.arange(1, self.__C_short.shape[0] + 1)
        hF, grads, Hash = hfun(Z_short)
        self.assertEqual(15.0, hF)
        expected = np.ones((M_short, 1))
        expected[0] = -1
        self.assertTrue(np.array_equal(grads, expected))
        self.assertEqual(grads.shape, (M_short, len(Hash)))
        self.assertEqual(Hash, ["311"])

        # Sanity check hash result/H0
        hF_H0, grads_H0 = hfun(Z_short, Hash)
        self.assertEqual(hF_H0, hF)
        self.assertTrue(np.array_equal(grads_H0, grads))

    def testErrors(self):
        # C & D must be NumPy arrays ...
        bad_all = [None, "bad", 1.1, [1.1], (1.1,), {1.1}, [1.1, 2.2], (1.1, 2.2), {1.1, 2.2}]
        for bad in bad_all:
            with self.assertRaises(TypeError):
                create_censored_L1_loss_hfun(bad, self.__D)
            with self.assertRaises(TypeError):
                create_censored_L1_loss_hfun(self.__C, bad)

        # but have at least 2 elements
        bad_all = [np.array([]), np.array([[]]), np.array(1.1), np.array([1.1]), np.array([[1.1]])]
        for bad in bad_all:
            with self.assertRaises(NotImplementedError):
                create_censored_L1_loss_hfun(bad, self.__D)
            with self.assertRaises(NotImplementedError):
                create_censored_L1_loss_hfun(self.__C, bad)

        # Finite reals please
        for bad_value in [np.nan, np.inf, -np.inf, 1j, 1.0 - 2.0 * 1j]:
            for i in range(len(self.__C)):
                bad = self.__C.copy()
                if isinstance(bad_value, complex):
                    bad = bad.astype(complex)
                bad[i] = bad_value
                with self.assertRaises(ValueError):
                    create_censored_L1_loss_hfun(bad, self.__D)
                with self.assertRaises(ValueError):
                    create_censored_L1_loss_hfun(self.__C, bad)

        # C & D must be *effectively* 1D ...
        C_2D = np.atleast_2d(self.__C.copy())
        D_2D = np.atleast_2d(self.__D.copy())
        self.assertEqual(2, C_2D.ndim)
        self.assertEqual(2, D_2D.ndim)
        hfun_2D = create_censored_L1_loss_hfun(C_2D, D_2D)
        hF_2D, grads_2D, Hash_2D = hfun_2D(self.__Z)
        self.assertEqual(hF_2D, self.__hF)
        self.assertTrue(np.array_equal(grads_2D, self.__grads))
        self.assertEqual(Hash_2D, self.__Hash)

        # but not actually >= 2D
        C_bad = self.__C.copy().reshape(self.__SHAPE_2D)
        D_bad = self.__D.copy().reshape(self.__SHAPE_2D)
        with self.assertRaises(ValueError):
            create_censored_L1_loss_hfun(C_bad, self.__D)
        with self.assertRaises(ValueError):
            create_censored_L1_loss_hfun(self.__C, D_bad)
        with self.assertRaises(ValueError):
            create_censored_L1_loss_hfun(C_bad, D_bad)

        # C & D must have the same effective shape
        with self.assertRaises(ValueError):
            create_censored_L1_loss_hfun(self.__C, self.__D_short)
        with self.assertRaises(ValueError):
            create_censored_L1_loss_hfun(self.__C_short, self.__D)

    def testBadZArguments(self):
        M = len(self.__C)

        # z must be NumPy array
        bad_all = [None, "bad", 1.1, (1.1,), {1.1}, [1.1, 2.2], (1.1, 2.2), {1.1, 2.2}]
        for bad in bad_all:
            with self.assertRaises(AssertionError):
                self.__hfun(bad)

        # z must be 1D
        for bad in [np.array([[]]), np.atleast_2d(self.__Z), np.zeros((M, 2))]:
            with self.assertRaises(AssertionError):
                self.__hfun(bad)

        # z must have correct length
        for bad in [np.array([]), np.ones(M - 1), np.ones(M + 1)]:
            with self.assertRaises(ValueError):
                self.__hfun(bad)

        # z must contain finite real values
        for bad_value in [np.nan, np.inf, -np.inf, 1j, 1.0 - 2.0 * 1j]:
            for i in range(len(self.__Z)):
                bad = self.__Z.copy()
                if isinstance(bad_value, complex):
                    bad = bad.astype(complex)
                bad[i] = bad_value
                with self.assertRaises(ValueError):
                    self.__hfun(bad)

    def testConfirmReadonly(self):
        # NOTE: This test is only useful for developing and manually testing the
        # code since the function under test doesn't actually alter C or D and
        # likely should never do so.
        #
        # Therefore, developers can, if so desired, temporarily inject changes
        # to C or D in the function and run this test to confirm that the
        # function is written such that it incapable of inadvertently altering
        # the C, D arrays created and managed here.
        C = self.__C.copy()
        D = self.__D.copy()
        hfun = create_censored_L1_loss_hfun(C, D)
        hfun(self.__Z)
        self.assertTrue(np.array_equal(C, self.__C))
        self.assertTrue(np.array_equal(D, self.__D))

    def testConfirmImmutable(self):
        # Construct using variable declared in this scope & collect results
        C = self.__C.copy()
        D = self.__D.copy()
        hfun = create_censored_L1_loss_hfun(C, D)
        hF, grads, Hash = hfun(self.__Z)

        # Alter same construction variables & confirm that they yield different
        # results
        C *= -2.3
        hfun_2 = create_censored_L1_loss_hfun(C, D)
        hF_2, grads_2, Hash_2 = hfun_2(self.__Z)
        self.assertNotEqual(hF_2, hF)
        self.assertFalse(np.array_equal(grads_2, grads))
        self.assertNotEqual(Hash_2, Hash)

        D *= -4.1
        hfun_3 = create_censored_L1_loss_hfun(C, D)
        hF_3, grads_3, Hash_3 = hfun_3(self.__Z)
        self.assertNotEqual(hF_3, hF)
        self.assertNotEqual(hF_3, hF_2)
        self.assertFalse(np.array_equal(grads_3, grads))
        self.assertFalse(np.array_equal(grads_3, grads_2))
        self.assertNotEqual(Hash_3, Hash)
        self.assertNotEqual(Hash_3, Hash_2)

        # Confirm that changing the construction variables didn't alter the
        # original functions
        hF_new, grads_new, Hash_new = hfun(self.__Z)
        self.assertEqual(hF_new, hF)
        self.assertTrue(np.array_equal(grads_new, grads))
        self.assertEqual(Hash_new, Hash)
