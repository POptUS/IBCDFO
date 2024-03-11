# Developer's Guide

## Python Implementations
The Python packages in this repository and the management of coverage reports
for the full repository are managed with
[tox](https://tox.wiki/en/latest/index.html), which can be used for CI work
among other work.  However, the same `tox` setups can be
[used by developers](https://github.com/POptUS/IBCDFO/blob/main/ibcdfo_pypkg/DeveloperInfo.md)
if so desired, which can be useful since `tox` will automatically setup and
manage dedicated virtual environments for the developer.  The following guide
can be used to setup `tox` as a command line tool on an individual platform in
a dedicated, minimal virtual environment and is based on a
[webinar](https://www.youtube.com/watch?v=PrAyvH-tm8E) by Oliver Bestwalter.

Developers that would like to use `tox` should learn about the tool so that, at
the very least, they understand the difference between running `tox` and `tox
-r`.

To create a Python virtual environment based on a desired Python dedicated to
hosting `tox`, execute some variation of
```
$ cd
$ deactivate (to deactivate the current virtual environment if you are in one)
$ /path/to/desired/python --version
$ /path/to/desired/python -m venv $HOME/.toxbase
$ ./.toxbase/bin/pip list
$ ./.toxbase/bin/python -m pip install --upgrade pip
$ ./.toxbase/bin/pip install --upgrade setuptools
$ ./.toxbase/bin/pip install tox
$ ./.toxbase/bin/tox --version
$ ./.toxbase/bin/pip list
```

To avoid the need to activate `.toxbase`, we setup `tox` in `PATH` for use
across all development environments that we might have on our system. In the
following, please replace `.bash_profile` with the appropriate shell
configuration file and tailor to your needs.
```
$ mkdir $HOME/local/bin
$ ln -s $HOME/.toxbase/bin/tox $HOME/local/bin/tox
$ vi $HOME/.bash_profile (add $HOME/local/bin to PATH)
$ . $HOME/.bash_profile
$ which tox
$ tox --version
```

For information on using `tox` with a particular Python package refer to the
`README.md` in the root folder of each package.

## Using `tox` for Global Coverage Tasks
The Python environments setup and managed at the root level of this repository
are for working globally with all coverage results generated independently by
testing individual code units in the repository.  In particular, it can be used
to combine these into a single file for generating global coverage reports.  As
such, this is a `tox` tool layer that requires advanced manual effort.  Its
primary use is with CI for
[automated report generation](https://github.com/POptUS/IBCDFO/blob/main/.github/workflows/github-ci-action.yml).

To use this layer, learn about and setup `tox` as described above.

No work will be carried out by default with the calls `tox` and `tox -r`.

The following commands can be run from the directory that contains this
`tox.ini`.
* `tox -r -e aggregate -- <coverage files>`
  * Combine all given `coverage.py` coverage files into the file `.coverage`
    located in the same directory as `tox.ini`
  * For best results, none of the given files should be named `.coverage`
  * Preserve the original coverage files
* `tox -r -e report`
  * It is intended that this be run after or with `aggregate`
  * Output a report and generate an HTML report for the aggregated coverage results
* `tox -r -e coveralls`
  * This is likely only useful for CI solutions
  * It is intended that this be run after or with `aggregate`
  * Send the aggegrated coverage report to Coveralls

Additionally, you can run any combination of the above such as
```
tox -r -e report,coveralls,aggregate -- <coverage files>
```
