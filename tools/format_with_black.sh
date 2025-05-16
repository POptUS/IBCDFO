#!/bin/bash

#
# Format all code to match black's uncompromising tastes.
#
# This returns exit codes that are compatible with the use of this
# script in CI jobs.
#

SCRIPT_PATH=$(dirname -- "${BASH_SOURCE[0]}")
CLONE_PATH=$SCRIPT_PATH/..

# Note that the tox job does follow the symlinks and will, therefore, reformat
# all source code that is ultimately included in the package.
declare -a FOLDERS=("tools")

pushd $CLONE_PATH

pushd ibcdfo_pypkg &> /dev/null || exit 1
# Let Python package format its code.
tox -r -e format                || exit $?

# Load virtual env so that black is available and ...
. ./.tox/format/bin/activate    || exit $?
popd &> /dev/null

# manually format Python code *not* included in a package.
for dir in "${FOLDERS[@]}"; do
    echo " "
    echo "Format Python code in $dir/* ..."
    pushd $dir &> /dev/null     || exit 1
    black --config=./.black .   || exit $?
    popd &> /dev/null
done
echo " "

popd &> /dev/null
