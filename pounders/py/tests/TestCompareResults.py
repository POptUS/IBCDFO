"""
Unit tests of the benchmark-result comparison helper (compare_results).  The
.mat files are synthesized here so that no external data is needed.
"""

import contextlib
import io
import pathlib
import tempfile
import unittest

import numpy as np
import scipy.io

from ibcdfo.pounders.tests import compare_results


def _write(directory, name="problem_1.mat", problem="problem 1 from More/Wild", H=None, Fvec=None, X=None, flag=0, xk_best=1):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "alg": "POUNDERS_Py",
        "problem": problem,
        "H": np.array([3.0, 2.0, 1.0]) if H is None else H,
        "Fvec": np.arange(6, dtype=float).reshape(3, 2) if Fvec is None else Fvec,
        "X": np.arange(6, dtype=float).reshape(3, 2) if X is None else X,
        "flag": flag,
        "xk_best": xk_best,
    }
    path = directory / name
    scipy.io.savemat(path, data)
    return path


class TestCompareResults(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ref_dir = pathlib.Path(self._tmp.name) / "ref"
        self.new_dir = pathlib.Path(self._tmp.name) / "new"

    def _compare(self, ref, new):
        with contextlib.redirect_stdout(io.StringIO()):
            return compare_results(ref, new)

    def test_identical(self):
        """Identical files pass."""
        ref = _write(self.ref_dir)
        new = _write(self.new_dir)
        self.assertTrue(self._compare(ref, new))

    def test_different_filename(self):
        """Mismatched filenames fail."""
        ref = _write(self.ref_dir)
        new = _write(self.new_dir, name="other.mat")
        self.assertFalse(self._compare(ref, new))

    def test_different_problem(self):
        """Mismatched problems fail."""
        ref = _write(self.ref_dir)
        new = _write(self.new_dir, problem="problem 2 from More/Wild")
        self.assertFalse(self._compare(ref, new))

    def test_nonfinite_values(self):
        """NaN/Inf in benchmark or new result fails."""
        good = _write(self.ref_dir)
        bad_H = _write(self.new_dir / "a", H=np.array([3.0, np.nan, 1.0]))
        bad_F = _write(self.new_dir / "b", Fvec=np.full((3, 2), np.inf))
        # Non-finite in the new result, then in the benchmark
        self.assertFalse(self._compare(good, bad_H))
        self.assertFalse(self._compare(good, bad_F))
        self.assertFalse(self._compare(bad_H, good))
        self.assertFalse(self._compare(bad_F, good))

    def test_differences_detected(self):
        """Any difference in best index, flag, H, F, or X fails."""
        ref = _write(self.ref_dir)
        self.assertFalse(self._compare(ref, _write(self.new_dir / "idx", xk_best=2)))
        self.assertFalse(self._compare(ref, _write(self.new_dir / "flag", flag=1.0)))
        self.assertFalse(self._compare(ref, _write(self.new_dir / "H", H=np.array([3.0, 2.5, 1.0]))))
        F = np.arange(6, dtype=float).reshape(3, 2)
        F[1, 0] += 1.0
        self.assertFalse(self._compare(ref, _write(self.new_dir / "F", Fvec=F)))
        X = np.arange(6, dtype=float).reshape(3, 2)
        X[1, 1] += 1.0
        self.assertFalse(self._compare(ref, _write(self.new_dir / "X", X=X)))

    def test_delta_min_is_warning_not_failure(self):
        """Matching delta_min flag (-6) passes with a warning."""
        ref = _write(self.ref_dir, flag=-6)
        new = _write(self.new_dir, flag=-6)
        self.assertTrue(self._compare(ref, new))

    def test_identical_hard_failure_is_warning_not_failure(self):
        """Matching hard-failure flag (-3) passes with a warning."""
        ref = _write(self.ref_dir, flag=-3)
        new = _write(self.new_dir, flag=-3)
        self.assertTrue(self._compare(ref, new))


if __name__ == "__main__":
    unittest.main()
