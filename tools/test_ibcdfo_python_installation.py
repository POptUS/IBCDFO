#!/usr/bin/env python

"""
Run the script with -h to obtain more information regarding the script.
"""

import sys
import inspect
import argparse
import importlib

from pathlib import Path

# ----- HARDCODED VALUES
# Exit codes so that this can be used in CI build server
_FAILURE = 1
_SUCCESS = 0


try:
    import ibcdfo
except ImportError as error:
    print()
    print(f"ERROR: {error.name} Python package not installed")
    print()
    exit(_FAILURE)


def main():
    DEFAULT_VERBOSITY = 1
    VALID_VERBOSITY = [0, 1, 2]

    # ----- SPECIFY COMMAND LINE USAGE
    DESCRIPTION = (
        "- Print useful IBCDFO package information\n"
        "- Run the package's test suite\n"
        "- Return pass/fail status of testing as CI-compatible exit code"
    )
    VERBOSE_HELP = "Verbosity level of unittest logging"
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--verbose", "-v",
        type=int, choices=VALID_VERBOSITY, default=DEFAULT_VERBOSITY,
        help=VERBOSE_HELP
    )

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()
    verbosity_level = args.verbose

    # ----- PRINT VERSION INFORMATION
    pkg = importlib.metadata.distribution("ibcdfo")
    location = Path(inspect.getfile(ibcdfo)).parents[0]

    print()
    print("Name: {}".format(pkg.metadata["Name"]))
    print("Version: {}".format(pkg.metadata["Version"]))
    print("Summary: {}".format(pkg.metadata["Summary"]))
    print("Homepage: {}".format(pkg.metadata["Home-page"]))
    print("License: {}".format(pkg.metadata["License"]))
    print("Python requirements: {}".format(pkg.metadata["Requires-Python"]))
    print("Package dependencies:")
    for dependence in pkg.metadata.get_all("Requires-Dist"):
        print(f"\t{dependence}")
    print("Location: {}".format(location))
    print()
    sys.stdout.flush()

    # ----- RUN FULL TEST SUITE
    return _SUCCESS if ibcdfo.test(verbosity_level) else _FAILURE


if __name__ == "__main__":
    exit(main())
