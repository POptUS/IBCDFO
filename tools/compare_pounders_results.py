#!/usr/bin/env python

import sys
import argparse

from pathlib import Path

from ibcdfo.pounders.tests import compare_results


def main():
    # ----- HARDCODED VALUES
    # CI-compatible exit codes
    SUCCESS = 0
    FAILURE = 1

    RED = "\033[0;91;1m"  # Bright Red/bold
    BLUE = "\033[0;34;1m"  # Blue/bold
    NC = "\033[0m"  # No Color/Not bold

    # ----- SPECIFY COMMAND LINE USAGE
    DESCRIPTION = "Compare new POUNDERS results against benchmarks"
    REFERENCE_HELP = "Path to folder containing benchmarks"
    NEW_HELP = "Path to folder containing new results to compare"
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("reference", nargs=1, type=str, help=REFERENCE_HELP)
    parser.add_argument("new", nargs=1, type=str, help=NEW_HELP)

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()
    ref_path = Path(args.reference[0]).resolve()
    new_path = Path(args.new[0]).resolve()

    print()
    print("POUNDERS Regression Testing")
    print("-" * 80)
    print(f"Benchmarks\t{ref_path}")
    print(f"New Results\t{new_path}")
    print()

    # ----- IDENTIFY ALL NEW RESULTS
    new_results = [fname.name for fname in new_path.glob("*.mat")]

    # ----- COMPARE NEW RESULTS AGAINST BENCHMARKS
    # Allow for the benchmarks folder to contain results that we don't have new
    # results for.  A new result without a matching benchmark is considered a
    # failure.
    #
    # A new result and a benchmark are considered to be matching if they have
    # the same filename.
    n_tests = 0
    n_skip = 0
    n_failed = 0
    for filename in new_results:
        new_fname = new_path.joinpath(filename)
        ref_fname = ref_path.joinpath(filename)
        if not ref_fname.exists():
            print(f"{filename} ... {RED}SKIP{NC}")
            n_skip += 1
            continue

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
