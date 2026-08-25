#!/bin/bash
# Build the portable Python environment ONCE on a CHTC submit node -> rolenv.tar.gz (~300 MB).
#
# No Docker, no Apptainer: PyROL ships prebuilt manylinux_2_28 wheels (cp38..cp313), so a plain
# venv runs on any EL8+ execute node.
#
#   bash build_env.sh                       # auto-detect an interpreter
#   PYVER=python3.12 bash build_env.sh      # force one
#
# CHTC access points ship python3 but the default is often 3.9, and numpy 2.1+ needs 3.10+.
# The pins below therefore depend on which interpreter is found. Everything is version-pinned
# so a rebuild months later is the same environment -- an unpinned pygsti or pyrol would
# silently change the optimizer.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

# ---------------------------------------------------------------- pick an interpreter
pick_python() {
    if [ -n "${PYVER:-}" ]; then
        command -v "$PYVER" >/dev/null 2>&1 && { PY=$(command -v "$PYVER"); return 0; }
        echo "ERROR: PYVER=$PYVER is not on PATH" >&2; return 1
    fi
    # newest first: numpy/scipy pins are better on 3.11+, and 3.9 is the floor
    for c in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
        command -v "$c" >/dev/null 2>&1 || continue
        v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
        case "$v" in
            3.9|3.10|3.11|3.12|3.13)
                # venv+ensurepip are split out of some distro pythons
                "$c" -c 'import venv, ensurepip' >/dev/null 2>&1 || continue
                PY=$(command -v "$c"); return 0 ;;
        esac
    done
    echo "ERROR: no usable python found (need 3.9-3.13 with venv + ensurepip)." >&2
    echo "  On CHTC try:   module avail python     then    module load <one>" >&2
    echo "  Or force one:  PYVER=/path/to/python3.12 bash build_env.sh" >&2
    return 1
}
pick_python || exit 1
PYMM=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "== python: $PY  ($("$PY" --version 2>&1))"

# ------------------------------------------------------------------------ version pins
# numpy 2.1+ requires python >= 3.10, so 3.9 gets the last 2.0.x. Both are ABI-compatible
# with the pygsti and pyrol wheels; the choice does not change any result.
case "$PYMM" in
    3.9)  NUMPY=2.0.2 ; SCIPY=1.13.1 ;;
    *)    NUMPY=2.1.3 ; SCIPY=1.14.1 ;;
esac
PANDAS=2.2.3
PYGSTI=0.9.14.3
PYROL=${ROLVER:-0.5.3}
echo "== pins: numpy==$NUMPY scipy==$SCIPY pandas==$PANDAS pygsti==$PYGSTI pyrol==$PYROL"

# ------------------------------------------------------------------------- build it
rm -rf rolenv rolenv.tar.gz
# --copies, not symlinks: the venv is tarred and unpacked on an execute node that will not
# have this interpreter at the same path.
"$PY" -m venv --copies rolenv
./rolenv/bin/pip install --upgrade pip wheel >/dev/null

./rolenv/bin/pip install \
    "numpy==$NUMPY" "scipy==$SCIPY" "pandas==$PANDAS" \
    "pygsti==$PYGSTI" "pyrol==$PYROL" \
    "cvxpy" "cvxopt"

# cvxopt is not optional: diamond distance is an SDP and cvxpy's default CLARABEL cannot solve
# it. Without cvxopt every diamond value silently comes back NaN.
./rolenv/bin/python - <<'PYCHK'
import numpy, scipy, pandas, pygsti, cvxpy
print("numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| pandas", pandas.__version__, "| pygsti", pygsti.__version__,
      "| cvxpy", cvxpy.__version__)
import pyrol; print("pyrol OK")
from pygsti.tools import optools as _ot
v = float(_ot.diamonddist(numpy.eye(4), numpy.diag([1.0,.999,.999,.999]), mx_basis="pp"))
assert numpy.isfinite(v), "diamonddist returned non-finite -- is cvxopt installed?"
print("diamonddist OK ->", v)
PYCHK

tar czf rolenv.tar.gz rolenv
echo "-> rolenv.tar.gz ($(( $(wc -c < rolenv.tar.gz) / 1048576 )) MB)"
