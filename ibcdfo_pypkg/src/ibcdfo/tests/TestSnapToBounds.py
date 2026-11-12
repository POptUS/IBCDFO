import unittest
import warnings

import numpy as np

from ibcdfo._bound_checks import snap_to_bounds

LOW = np.array([-1.0, -1.0, -1.0])
UPP = np.array([1.0, 1.0, 1.0])
SNAP_TOL = 1.0e-6


def snap(x, Low=LOW, Upp=UPP, snap_tol=SNAP_TOL):
    return snap_to_bounds(x, Low, Upp, snap_tol, "x")


class TestSnapToBounds(unittest.TestCase):

    def snapQuietly(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return snap(*args, **kwargs)

    def testFeasiblePointIsUnchanged(self):
        x = np.array([-1.0, 0.25, 1.0])
        self.assertTrue(np.array_equal(self.snapQuietly(x), x))

    def testViolationBelowTheNoiseFloorIsSilent(self):
        x = np.array([-1.0 - 1.0e-15, 0.0, 1.0 + 1.0e-15])
        snapped = self.snapQuietly(x)
        self.assertTrue(np.array_equal(snapped, np.array([-1.0, 0.0, 1.0])))

    def testViolationWithinSnapTolWarnsAndSnaps(self):
        x = np.array([0.0, 0.0, 1.0 + 1.0e-10])
        with self.assertWarns(UserWarning):
            snapped = snap(x)
        self.assertTrue(np.array_equal(snapped, np.array([0.0, 0.0, 1.0])))

    def testViolationBeyondSnapTolRaises(self):
        for x in [np.array([0.0, 0.0, 1.1]), np.array([-1.1, 0.0, 0.0])]:
            with self.assertRaises(ValueError):
                snap(x)

    def testEachPointOfAMatrixIsChecked(self):
        X = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
        with self.assertRaises(ValueError):
            snap(X)

    def testInfiniteBoundsAreNeverViolated(self):
        x = np.array([-1.0e300, 0.0, 1.0e300])
        Low = np.full(3, -np.inf)
        Upp = np.full(3, np.inf)
        self.assertTrue(np.array_equal(self.snapQuietly(x, Low, Upp), x))

    def testLargeXTripsOnItsOwnRounding(self):
        # The check is absolute, so it does not scale with x. One unit in the
        # last place of 1e10 is ~1.9e-6, past the default snap_tol, and a
        # solver that lands exactly on the bound still trips the check. This
        # pins that documented limitation; raising snap_tol is the remedy.
        Upp = np.full(3, 1.0e10)
        x = np.array([0.0, 0.0, np.nextafter(1.0e10, np.inf)])
        self.assertGreater(x[2] - Upp[2], SNAP_TOL)
        with self.assertRaises(ValueError):
            snap(x, -Upp, Upp)
        with self.assertWarns(UserWarning):
            snap(x, -Upp, Upp, snap_tol=1.0e-5)
