#!/usr/bin/env python

import sys
import argparse
import traceback

from pathlib import Path

from ibcdfo.pounders.tests import compare_results


def main():
    # ----- HARDCODED VALUES
    # CI-compatible exit codes
    SUCCESS = 0
    FAILURE = 1

    RED = "\033[0;91;1m"   # Bright Red/bold
    BLUE = "\033[0;34;1m"  # Blue/bold
    NC = "\033[0m"         # No Color/Not bold

    # ----- SPECIFY COMMAND LINE USAGE
    DESCRIPTION = "Compare new POUNDERS results against benchmarks"
    REFERENCE_HELP = "Path to folder containing benchmarks"
    NEW_HELP = "Path to folder containing new results to compare"
    DEBUG_HELP = "Print information to help debug this script"
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("reference", nargs=1, type=str, help=REFERENCE_HELP)
    parser.add_argument("new", nargs=1, type=str, help=NEW_HELP)
    parser.add_argument("--debug", "-d", action='store_true', help=DEBUG_HELP)

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()
    ref_path = Path(args.reference[0]).resolve()
    new_path = Path(args.new[0]).resolve()

    def log_and_abort(msg):
        print()
        print(f"ERROR: {msg}")
        print()
        if args.debug:
            traceback.format_exc()
        sys.exit(FAILURE)

    # ----- IDENTIFY ALL RESULTS
    benchmarks = [fname.name for fname in ref_path.glob("*.mat")]
    results = [fname.name for fname in new_path.glob("*.mat")]

    print()
    print("POUNDERS Regression Testing")
    print("-" * 80)
    print(f"Benchmarks\t{ref_path}")
    print(f"New Results\t{new_path}")
    print()

    # ----- COMPARE RESULTS
    # Allow for the benchmarks to have results that we don't have new results
    # for.  A missing benchmark is considered a failure.
    n_tests = 0
    n_skip = 0
    n_failed = 0
    for filename in results:
        if filename not in benchmarks:
            print(f"{filename} ... {RED}SKIP{NC}")
            n_skip += 1
            continue

        ref_fname = ref_path.joinpath(filename)
        new_fname = new_path.joinpath(filename)

        if not compare_results(ref_fname, new_fname):
            n_failed += 1
        n_tests += 1

    print()
    print(f"N Failed\t\t{n_failed}")
    print(f"N Skipped\t\t{n_skip}")
    print(f"N Total Tests\t\t{n_tests}")

    if (n_skip != 0) or (n_failed != 0):
        print(f"\n{RED}FAILURE{NC}\n")
        return FAILURE

    print(f"\n{BLUE}SUCCESS{NC}\n")
    return SUCCESS


if __name__ == "__main__":
    sys.exit(main())
