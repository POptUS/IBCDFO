# Sourced by setup.sh and build_env.sh. Sets $PY to a usable interpreter.
#
# CHTC access points are EL8/EL9: they ship python3 but usually NOT a bare
# `python`, and not necessarily python3.11. Anything that assumes either name
# dies immediately under `set -e`.
#
# PyROL (pyroltrilinos) publishes wheels for cp38..cp313, so any python3.8+
# works. Override with:  PYVER=python3.12 bash setup.sh

pick_python() {
    if [ -n "${PYVER:-}" ]; then
        if command -v "$PYVER" >/dev/null 2>&1; then
            PY=$(command -v "$PYVER")
            return 0
        fi
        echo "ERROR: PYVER=$PYVER was requested but is not on PATH" >&2
        return 1
    fi
    for c in python3.12 python3.11 python3.10 python3.13 python3.9 python3.8 python3 python; do
        if command -v "$c" >/dev/null 2>&1; then
            v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
            case "$v" in
                3.8|3.9|3.10|3.11|3.12|3.13)
                    # venv must be available too -- some distro pythons split it out
                    if "$c" -c 'import venv, ensurepip' >/dev/null 2>&1; then
                        PY=$(command -v "$c")
                        return 0
                    fi
                    ;;
            esac
        fi
    done
    echo "ERROR: no suitable python found (need 3.8-3.13 with venv + ensurepip)." >&2
    echo "  tried: python3.12 python3.11 python3.10 python3.13 python3.9 python3.8 python3 python" >&2
    echo "  on CHTC try:  module avail python   then   module load <one>" >&2
    echo "  or pass one explicitly:  PYVER=/path/to/python3.11 bash setup.sh" >&2
    return 1
}
