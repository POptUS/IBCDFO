#!/bin/bash

python -m pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel
pip install --upgrade tox
echo
which python
which pip
which tox
echo
python --version
pip --version
tox --version
echo
pip list

# The following two packages are used for testing. One way to get them inside
# tox is to use the PYTHONPATH environment variable.
git clone -b py_calfun https://github.com/POptUS/BenDFO.git
git clone https://github.com/POptUS/MINQ.git
pushd BenDFO/py/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
popd
pushd MINQ/py/minq5/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "PYTHONPATH=$PYTHONPATH" >> $GITHUB_ENV
