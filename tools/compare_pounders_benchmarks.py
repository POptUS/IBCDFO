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

    # ----- COMPARE RESULTS
    result_not_found = False

    # Allow for the new results to have results that we don't have benchmarks
    # for.  A missing new result is considered a failure.
    for filename in benchmarks:
        if filename not in results:
            result_not_found = True
            print(f"ERROR: Did not find new result {filename}")

        ref_fname = ref_path.joinpath(filename)
        new_fname = new_path.joinpath(filename)

        if not compare_results(ref_fname, new_fname):
            return FAILURE

    if result_not_found:
        return FAILURE

    return SUCCESS


if __name__ == "__main__":
    sys.exit(main())
