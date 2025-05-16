#!/bin/bash

#
# Delete files and folders that are not part of the repository but are created
# as part of building and testing the package.
#

SCRIPT_PATH=$(dirname -- "${BASH_SOURCE[0]}")
IBCDFO_PYPKG=$(realpath $SCRIPT_PATH/../ibcdfo_pypkg)

echo -n "Clean-up clone at $IBCDFO_PYPKG (y/n): "
read do_cleaning
if [[ "$do_cleaning" == "y" ]]; then
    rm -rf $IBCDFO_PYPKG/.tox
    rm -rf $IBCDFO_PYPKG/dist
    rm -rf $IBCDFO_PYPKG/build
    rm -f  $IBCDFO_PYPKG/.coverage_ibcdfo
    rm -f  $IBCDFO_PYPKG/cobertura_ibcdfo.xml
    rm -rf $IBCDFO_PYPKG/htmlcov_ibcdfo
    rm -f  $IBCDFO_PYPKG/dfo.dat
    rm -f  $IBCDFO_PYPKG/mpc_test_files_smaller_Q.zip
    rm -rf $IBCDFO_PYPKG/mpc_test_files_smaller_Q
    rm -rf $IBCDFO_PYPKG/benchmark_results
    rm -rf $IBCDFO_PYPKG/msp_benchmark_results
fi
