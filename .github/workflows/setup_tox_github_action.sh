#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo
    echo "Please pass GitHub action runner OS (e.g., Linux or macOS)"
    echo
    exit 1
fi
runner_os=$1

which python
which pip
echo " "
python -c "import platform ; print(platform.machine())"
python -c "import platform ; print(platform.system())"
python -c "import platform ; print(platform.release())"
python -c "import platform ; print(platform.platform())"
python -c "import platform ; print(platform.version())"
if [ "$runner_os" = "macOS" ]; then
    python -c "import platform ; print(platform.mac_ver())"
fi
echo " "
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools
python -m pip install build
python -m pip install tox
echo " "
python --version
tox --version
echo " "
pip list
echo " "

# The following two packages are used for testing. One way to get them inside
# tox is to use the PYTHONPATH environment variable.
#
# TODO: It would be better to use the MINQ submodule in the IBCDFO clone to
# confirm that the submodule is working properly and to use the official minq
# hash assigned in the submodule.
git clone https://github.com/POptUS/BenDFO.git
git clone https://github.com/POptUS/MINQ.git
pushd BenDFO/py/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
popd
pushd MINQ/py/minq5/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
popd
echo "PYTHONPATH=$PYTHONPATH" >> $GITHUB_ENV
