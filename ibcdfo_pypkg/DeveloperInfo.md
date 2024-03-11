## Python Developer Environments
Developers that prefer to work with a self-managed Python environment (virtual
or otherwise), can install a fixed version of IBCDFO by executing
```
$ cd /path/to/clone/ibcdfo_pypkg
$ python setup.py sdist
$ python -m pip install dist/ibcdfo-<version>.tar.gz
```
where `<version>` can be determined by reading the output of the setup command.
An installation of IBCDFO that always uses the current state of the code in the
developer's IBCDFO clone (i.e., install in editable or developer mode) can be
made using
```
$ python -m pip install -e /path/to/clone/ibcdfo_pypkg
```

Developer's can use `tox` to manage their Python environments automatically.
In particular, `tox` is  setup within IBCDFO to manage independent Python
virtual environments for many individuals based on the efforts of the limited
few that setup and maintain `tox`.  Assuming that users have already executed
`tox -e coverage` (see below), developers can activate a Python virtual
environment that always uses the current state of the code in the developer's
ICBDFO clone by executing
```
$ . /path/to/clone/ibcdfo_pypkg/.tox/coverage/bin/activate
```

## Development with `tox`
Developer's interested in using `tox` should first learn about and setup `tox`
as described in the [main DeveloperInfo](https://github.com/POptUS/IBCDFO/blob/main/DeveloperInfo.md).

No work will be carried out by default with the calls `tox` and `tox -r`.

The following commands can be run from the directory that contains this file.
* `tox -r -e coverage`
  * Execute the full test suite for the package and save coverage results to the
    coverage file
  * The test runs the package code in the local clone rather than code installed
    into python so that coverage results accessed through web services such as
    Coveralls are clean and straightforward
* `tox -r -e nocoverage`
  * Execute the full test suite for the package using the code installed into
    python
* `tox -r -e pounders`
  * Execute the test suite for the pounders subpackage only using the code
    installed into python
* `tox -r -e manifold_sampling`
  * Execute the test suite for the manifold_sampling subpackage only using the
    code installed into python
* `tox -r -e report`
  * It is intended that this be run after or with coverage
  * Display a report and generate an HTML report for the package's full test
    suite
* `tox -r -e check`
  * This is likely only useful for developers working on a local clone
  * This task should never call any tools that automatically __alter__ files
  * Run several checks on the code to report possible issues

Additionally, you can run any combination of the above such as
`tox -r -e report,coverage`.

If the environment variable `COVERAGE_FILE` is set, then this is the coverage
file that will be used with all associated work.  If it is not specified, then
the coverage file is `.coverage_ibcdfo`.

## Manual Developer Testing
It is possible to test manually outside of `tox`, which could be useful for
testing at the level of a single test.

To run the full test suite, execute
```
$ python -m unittest ibcdfo
```
To run only the `TestPoundersSimple` pounders test, execute
```
$ python -m unittest ibcdfo.pounders.tests.TestPoundersSimple
```

## Adding a New Subpackage to `IBCDFO`
* Add new subpackage to the root of the repo in accord with the POptUS
  repository requirements
* Increment `VERSION` appropriately
* Add in the new subpackage implementation as symlinks in the correct
  `ibcdfo_pypkg` subdirectory
* Update `load_tests.py` in the main package so that it builds a suite that
  includes the tests of the subpackage
* Update the `README.md` file if necessary
* Adapt `setup.py`
  * Update or expand all requirements as needed
  * Add test and package data in new subpackage to `package_data` if any
  * Update all other metadata as needed
* Update `tox.ini`
  * Add a new testenv in `tox.ini` dedicated to the new subpackage if so desired
* Do local testing with `tox` if so desired
* Synchronize python version information in GitHub CI actions to version changes
  made in `setup.py` (if any)
* Commit, push, and check associated GitHub CI action logs to see if constructed
  and integrated correctly
