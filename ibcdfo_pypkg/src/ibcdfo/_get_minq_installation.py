import os
import sys

import subprocess as sbp

from pathlib import Path


def get_minq_installation():
    """
    This function attempts to load ``minqsw``, which is assumed to be located in
    a valid MINQ clone.  If found, it gathers information about the clone.
    Otherwise, it instructs users to install MINQ and inform Python of its
    location.

    :return: ``(hash, installation)`` where
        * ``hash`` is the git hash of the MINQ commit required by this version
          of IBCDFO
        * ``installation`` is a ``dict`` containing useful information regarding
          the MINQ installation including whether it is deemed valid for use
          with this package
    """
    # ----- HARDCODED VALUES
    # This must be the full git commit hash
    VALID_MINQ_HASH = "7749b83645ea21e303a94e1200542f7028499bb8"

    GIT_HASH_CMD = ['git', 'rev-parse', 'HEAD']
    CLONE_STATE_CMD = ["git", "diff-index", "--quiet", "HEAD", "--"]
    CLEAN_GIT_CLONE = 0
    DIRTY_GIT_CLONE = 1

    # If we put this import statement at the top of the module, which would get
    # run every time this module is imported, then the import might incorrectly
    # fail for users who have not installed MINQ because they are only
    # interested in using IBCDFO functionality that does not use MINQ.
    # Therefore, we only import MINQ if IBCDFO code explicitly calls this
    # function, which it should only do if it needs MINQ.
    #
    # This presently assumes that IBCDFO code that uses MINQ expects users to
    # inform Python of the location of MINQ5.  It doesn't account for the
    # situation that the the code might need to use a different version of MINQ.
    # This is presently acceptable since MINQ5 is the only version in Python.
    try:
        import minqsw
        minq_path = Path(minqsw.__file__).resolve().parents[2]
    except ModuleNotFoundError:
        sys.exit(
            "Ensure a Python implementation of MINQ is available. For example, "
            "clone https://github.com/POptUS/minq and add minq/py/minq5 to the "
            "PYTHONPATH environment variable"
        )

    cwd_original = os.getcwd()

    try:
        os.chdir(minq_path)
        result = sbp.run(GIT_HASH_CMD, capture_output=True, check=True)
        assert result.stderr.decode() == ""
        assert result.returncode == 0
        git_hash = result.stdout.decode().strip()

        result = sbp.run(CLONE_STATE_CMD, capture_output=True, check=True)
        assert result.stdout.decode() == ""
        assert result.stderr.decode() == ""
        exit_code = result.returncode
    except sbp.CalledProcessError as err:
        stdout = err.stdout.decode()
        stderr = err.stderr.decode()
        print()
        msg = "Unable to determine MINQ git information (Return code {})"
        print(msg.format(err.returncode))
        print(" ".join(err.cmd))
        if stdout != "":
            print("stdout")
            for line in stdout.split("\n"):
                print(f"\t{line}")
        if stderr != "":
            print("stderr")
            for line in stderr.split("\n"):
                print(f"\t{line}")
        raise
    finally:
        os.chdir(cwd_original)

    assert exit_code in {CLEAN_GIT_CLONE, DIRTY_GIT_CLONE}

    installation = {
        "path": minq_path,
        "hash": git_hash,
        "is_clean": (exit_code == CLEAN_GIT_CLONE),
        "is_valid": (git_hash == VALID_MINQ_HASH)
    }

    return VALID_MINQ_HASH, installation
