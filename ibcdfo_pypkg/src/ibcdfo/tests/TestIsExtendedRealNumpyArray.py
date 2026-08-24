import unittest

import numpy as np

from ibcdfo._variable_checks import is_extended_real_numpy_array


class TestIsExtendedRealNumpyArray(unittest.TestCase):

    def setUp(self):
        self.__NOT_NUMPY_ARRAY = (
            None,
            "",
            "not_array",
            True,
            False,
            1,
            1.1,
            (1.1 + 2.2 * 1j),
            (0.0 * 1j),
            {},
            (1),
            [1],
            {1},
        )
        self.__COMPLEX = {1j, (1.1 - 2.2 * 1j), (0.0 * 1j)}
        self.__NOT_EXTENDED = {np.nan}

        self.__GOOD_1D = np.full(5, 1.1, float)
        self.__GOOD_2D = np.full((5, 3), 1.1, float)
        self.__GOOD_3D = np.full((5, 3, 4), 1.1, float)

        # 0D arrays are strange, so check this more carefully
        self.__GOOD_0D = np.array(0.0)
        self.assertTrue(isinstance(self.__GOOD_0D, np.ndarray))
        self.assertEqual(self.__GOOD_0D.ndim, 0)
        self.assertTrue(np.isreal(self.__GOOD_0D))
        self.assertTrue(np.isfinite(self.__GOOD_0D))

    def test0dArrays(self):
        with self.assertRaises(AssertionError):
            self.assertFalse(is_extended_real_numpy_array(self.__GOOD_0D, ndim=0))

    def test1dArrays(self):
        NDIM = 1
        WRONG_NDIM = (self.__GOOD_0D, self.__GOOD_2D, self.__GOOD_3D)
        GOOD = self.__GOOD_1D.copy()

        self.assertTrue(is_extended_real_numpy_array(GOOD, ndim=NDIM))
        self.assertFalse(is_extended_real_numpy_array(GOOD.astype(int), ndim=NDIM))

        for bad in self.__NOT_NUMPY_ARRAY:
            self.assertFalse(is_extended_real_numpy_array(bad, ndim=NDIM))

        for bad in [None, "", "a"]:
            bad_array = np.full(GOOD.shape, bad)
            self.assertEqual(bad_array.ndim, NDIM)
            self.assertFalse(is_extended_real_numpy_array(bad_array, ndim=NDIM))

        for bad in WRONG_NDIM:
            self.assertNotEqual(bad.ndim, NDIM)
            self.assertFalse(is_extended_real_numpy_array(bad, ndim=NDIM))

        for i in range(len(GOOD)):
            bad_array = GOOD.copy()
            for bad in self.__NOT_EXTENDED:
                bad_array[i] = bad
                self.assertFalse(is_extended_real_numpy_array(bad_array, ndim=NDIM))

            bad_array = bad_array.astype(complex)
            for bad in self.__COMPLEX:
                bad_array[i] = bad
                self.assertFalse(is_extended_real_numpy_array(bad_array, ndim=NDIM))

        test = np.full(GOOD.shape, np.inf, float)
        self.assertTrue(is_extended_real_numpy_array(test, ndim=NDIM))
        test = np.full(GOOD.shape, -np.inf, float)
        self.assertTrue(is_extended_real_numpy_array(test, ndim=NDIM))

        test = GOOD.copy()
        test[0] = -np.inf
        test[-1] = np.inf
        self.assertTrue(is_extended_real_numpy_array(test, ndim=NDIM))

    def test2dArrays(self):
        NDIM = 2
        WRONG_NDIM = (self.__GOOD_0D, self.__GOOD_1D, self.__GOOD_3D)
        GOOD = self.__GOOD_2D.copy()

        self.assertTrue(is_extended_real_numpy_array(GOOD, ndim=NDIM))
        self.assertFalse(is_extended_real_numpy_array(GOOD.astype(int), ndim=NDIM))

        for bad in self.__NOT_NUMPY_ARRAY:
            self.assertFalse(is_extended_real_numpy_array(bad, ndim=NDIM))

        for bad in [None, "", "a"]:
            bad_array = np.full(GOOD.shape, bad)
            self.assertEqual(bad_array.ndim, NDIM)
            self.assertFalse(is_extended_real_numpy_array(bad_array, ndim=NDIM))

        for bad in WRONG_NDIM:
            self.assertNotEqual(bad.ndim, NDIM)
            self.assertFalse(is_extended_real_numpy_array(bad, ndim=NDIM))

        for i in range(GOOD.shape[0]):
            for j in range(GOOD.shape[1]):
                bad_array = GOOD.copy()
                for bad in self.__NOT_EXTENDED:
                    bad_array[i, j] = bad
                    self.assertFalse(is_extended_real_numpy_array(bad_array, ndim=NDIM))

                bad_array = bad_array.astype(complex)
                for bad in self.__COMPLEX:
                    bad_array[i, j] = bad
                    self.assertFalse(is_extended_real_numpy_array(bad_array, ndim=NDIM))

        test = np.full(GOOD.shape, np.inf, float)
        self.assertTrue(is_extended_real_numpy_array(test, ndim=NDIM))
        test = np.full(GOOD.shape, -np.inf, float)
        self.assertTrue(is_extended_real_numpy_array(test, ndim=NDIM))

        test = GOOD.copy()
        test[0, -1] = -np.inf
        test[-1, 1] = np.inf
        self.assertTrue(is_extended_real_numpy_array(test, ndim=NDIM))

    def test3dArrays(self):
        NDIM = 3
        WRONG_NDIM = (self.__GOOD_0D, self.__GOOD_1D, self.__GOOD_2D)
        GOOD = self.__GOOD_3D.copy()

        self.assertTrue(is_extended_real_numpy_array(GOOD, ndim=NDIM))
        self.assertFalse(is_extended_real_numpy_array(GOOD.astype(int), ndim=NDIM))

        for bad in self.__NOT_NUMPY_ARRAY:
            self.assertFalse(is_extended_real_numpy_array(bad, ndim=NDIM))

        for bad in [None, "", "a"]:
            bad_array = np.full(GOOD.shape, bad)
            self.assertEqual(bad_array.ndim, NDIM)
            self.assertFalse(is_extended_real_numpy_array(bad_array, ndim=NDIM))

        for bad in WRONG_NDIM:
            self.assertNotEqual(bad.ndim, NDIM)
            self.assertFalse(is_extended_real_numpy_array(bad, ndim=NDIM))

        for i in range(GOOD.shape[0]):
            for j in range(GOOD.shape[1]):
                for k in range(GOOD.shape[2]):
                    bad_array = GOOD.copy()
                    for bad in self.__NOT_EXTENDED:
                        bad_array[i, j, k] = bad
                        self.assertFalse(is_extended_real_numpy_array(
                            bad_array, ndim=NDIM)
                        )  # fmt: skip

                    bad_array = bad_array.astype(complex)
                    for bad in self.__COMPLEX:
                        bad_array[i, j, k] = bad
                        self.assertFalse(is_extended_real_numpy_array(
                            bad_array, ndim=NDIM)
                        )  # fmt: skip

        test = np.full(GOOD.shape, np.inf, float)
        self.assertTrue(is_extended_real_numpy_array(test, ndim=NDIM))
        test = np.full(GOOD.shape, -np.inf, float)
        self.assertTrue(is_extended_real_numpy_array(test, ndim=NDIM))

        test = GOOD.copy()
        test[0, -1, 2] = -np.inf
        test[-1, 1, 3] = np.inf
        self.assertTrue(is_extended_real_numpy_array(test, ndim=NDIM))
