import sys
import unittest

import numpy as np

from ibcdfo._variable_checks import is_integer


class TestIsInteger(unittest.TestCase):

    def testFunction(self):
        GOOD_INT = [-sys.maxsize, -1001, -1, 0, 1, 27, 25_234, sys.maxsize]
        NOT_INT = (
            None,
            "",
            "not_an_int",
            False,
            True,
            0.0,
            1.0,
            1.1,
            {},
            [1],
            {1},
            np.array(1),
            np.array([1]),
        )

        for good in GOOD_INT:
            self.assertTrue(isinstance(good, int))
            self.assertTrue(is_integer(good))

            self.assertTrue(is_integer(np.int64(good)))
            try:
                self.assertTrue(is_integer(np.int32(good)))
            except OverflowError:
                pass
            try:
                self.assertTrue(is_integer(np.int8(good)))
            except OverflowError:
                pass

            if good >= 0:
                self.assertTrue(is_integer(np.uint64(good)))
                try:
                    self.assertTrue(is_integer(np.uint32(good)))
                except OverflowError:
                    pass
                try:
                    self.assertTrue(is_integer(np.uint8(good)))
                except OverflowError:
                    pass

        for bad in NOT_INT:
            self.assertFalse(is_integer(bad))
