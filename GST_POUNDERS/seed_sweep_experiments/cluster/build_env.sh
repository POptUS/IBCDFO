#!/bin/bash
# Build the portable Python environment ONCE on a CHTC submit node -> rolenv.tar.gz (~300 MB).
#
# No Docker, no Apptainer: PyROL ships a prebuilt manylinux_2_28 wheel, so a plain venv runs
# on any EL8+ execute node. Everything is version-pinned so a rerun months later is the same
# environment -- an unpinned pygsti or pyrol would silently change the optimizer.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

PY=${PY:-python3}
"$PY" -V

rm -rf rolenv rolenv.tar.gz
"$PY" -m venv rolenv
./rolenv/bin/pip install --upgrade pip wheel >/dev/null

./rolenv/bin/pip install \
    "numpy==2.1.3" \
    "scipy==1.14.1" \
    "pandas==2.2.3" \
    "pygsti==0.9.14.3" \
    "pyrol==0.5.3" \
    "cvxpy==1.6.0" \
    "cvxopt==1.3.2"

# cvxopt matters: diamond distance is an SDP and cvxpy's default CLARABEL cannot solve it.
# Without cvxopt every diamond value silently comes back NaN.
./rolenv/bin/python - <<'PYCHK'
import numpy, scipy, pandas, pygsti, cvxpy
print("numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| pandas", pandas.__version__, "| pygsti", pygsti.__version__,
      "| cvxpy", cvxpy.__version__)
import pyrol; print("pyrol OK")
from pygsti.tools import optools as _ot
v = float(_ot.diamonddist(numpy.eye(4), numpy.diag([1.0,.999,.999,.999]), mx_basis="pp"))
assert numpy.isfinite(v), "diamonddist returned non-finite -- cvxopt missing?"
print("diamonddist OK ->", v)
PYCHK

tar czf rolenv.tar.gz rolenv
echo "-> rolenv.tar.gz ($(( $(wc -c < rolenv.tar.gz) / 1048576 )) MB)"
