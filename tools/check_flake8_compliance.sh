#!/bin/bash

#
# Check that all Python code in the repository adheres to the PEP8 coding
# standard.
#
# This returns exit codes that are compatible with the use of this
# script in CI jobs.
#

SCRIPT_PATH=$(dirname -- "${BASH_SOURCE[0]}")
CLONE_PATH=$SCRIPT_PATH/..

# Note that the tox job doesn't check the files included in its package by
# symlink.
declare -a FOLDERS=("tools" \
                    "pounders/py" \
                    "manifold_sampling/py")

pushd $CLONE_PATH &> /dev/null  || exit 1

pushd ibcdfo_pypkg &> /dev/null || exit 1
# Let Python package determine if its code is acceptable
tox -r -e check                 || exit $?

# Load virtual env so that flake8 is available and ...
. ./.tox/check/bin/activate     || exit $?
popd &> /dev/null

# manually check Python code *not* included in a package directly.
for dir in "${FOLDERS[@]}"; do
    echo " "
    echo "Check Python code in $dir/* ..."
    pushd $dir &> /dev/null     || exit 1
    flake8 --config=./.flake8   || exit $?
    popd &> /dev/null
done
echo " "

popd &> /dev/null
