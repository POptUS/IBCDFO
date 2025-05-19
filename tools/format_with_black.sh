#!/bin/bash

#
# Format all code to match black's uncompromising tastes.
#
# NOTE: This will potentially alter Python code in your local clone.  Avoid
# using this if you have uncommitted changes or untracked files.
#
# This returns exit codes that are compatible with the use of this
# script in CI jobs.
#

SCRIPT_PATH=$(dirname -- "${BASH_SOURCE[0]}")
CLONE_PATH=$SCRIPT_PATH/..

# Note that the tox job does follow the symlinks and will, therefore, reformat
# all source code that is ultimately included in the package.
declare -a FOLDERS=("tools")

if   [[ $# -eq 0 ]]; then
    tox_task=format
    flags=
elif [[ $# -eq 1 ]]; then
    if [[ "$1" != "--diff" ]]; then
        echo
        echo "Only optional argument allowed is --diff"
        echo
        exit 1
    fi
    tox_task=format_safe
    flags="--check --diff"
else
    echo
    echo "No or one command line arguments please"
    echo
    exit 1
fi

pushd $CLONE_PATH &> /dev/null       || exit 1

pushd ibcdfo_pypkg &> /dev/null      || exit 1
# Let Python package format its code.
tox -r -e $tox_task                  || exit $?

# Load virtual env so that black is available and ...
. ./.tox/${tox_task}/bin/activate    || exit $?
popd &> /dev/null

# manually format Python code *not* included in a package.
for dir in "${FOLDERS[@]}"; do
    echo " "
    echo "Format Python code in $dir/* ..."
    pushd $dir &> /dev/null          || exit 1
    black --config=./.black . $flags || exit $?
    popd &> /dev/null
done
echo " "

popd &> /dev/null
