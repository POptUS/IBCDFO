"""
Unit test of checkinputss()
"""

import copy
import unittest

import numpy as np

from ibcdfo.pounders.checkinputss import checkinputss


class TestPoundersInterface(unittest.TestCase):
    def setUp(self):
        self.__ERROR_HDR = "Error: "
        self.__NOT_INT = (None, "", 1.0, 1.1, {}, [1], {1})

        N, M = (3, 1)

        # These should pass all checks.
        #
        # Tests that need to change values should make a deepcopy and alter that
        # dictionary.
        self.__KWARGS = {
            "n": N,
            "m": M,
            "Ffun": np.linalg.norm,
            "Low": np.full(N, -np.inf, float),
            "Upp": np.full(N, np.inf, float),
            "X_0": np.full(N, 0.5, float),
            "np_max": 2 * N + 1,
            "nf_max": 10,
            "g_tol": 1.0e-13,
            "delta_0": 0.1,
            "nfs": 0,
            "X_init": np.full((0, N), 0.0, float),
            "F_init": np.full((0, M), 0.0, float),
            "xk_in": 0,
        }
        checkinputss(**self.__KWARGS)

    def testN(self):
        # Prefer testing checkinputss() by testing direct calls to pounders.py.
        # POUNDERS error checks n before calling checkinputss() because n is
        # used to set other local variables to be checked.  Therefore, we test
        # here in case this functionality is actually used at some point.
        for bad in self.__NOT_INT:
            kwargs = copy.deepcopy(self.__KWARGS)
            kwargs["n"] = bad
            with self.assertRaises(TypeError) as err:
                checkinputss(**kwargs)
            err_msg = str(err.exception)
            # print(err_msg)
            self.assertTrue(err_msg.startswith(self.__ERROR_HDR))
        for bad in [-10, -1, 0]:
            kwargs = copy.deepcopy(self.__KWARGS)
            kwargs["n"] = bad
            with self.assertRaises(ValueError) as err:
                checkinputss(**kwargs)
            err_msg = str(err.exception)
            # print(err_msg)
            self.assertTrue(err_msg.startswith(self.__ERROR_HDR))

    def testM(self):
        # See notes for testN().
        for bad in self.__NOT_INT:
            kwargs = copy.deepcopy(self.__KWARGS)
            kwargs["m"] = bad
            with self.assertRaises(TypeError) as err:
                checkinputss(**kwargs)
            err_msg = str(err.exception)
            # print(err_msg)
            self.assertTrue(err_msg.startswith(self.__ERROR_HDR))
        for bad in [-10, -1, 0]:
            kwargs = copy.deepcopy(self.__KWARGS)
            kwargs["m"] = bad
            with self.assertRaises(ValueError) as err:
                checkinputss(**kwargs)
            err_msg = str(err.exception)
            # print(err_msg)
            self.assertTrue(err_msg.startswith(self.__ERROR_HDR))
