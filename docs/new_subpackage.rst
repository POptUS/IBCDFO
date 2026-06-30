Add a New Subpackage to |ibcdfo|
================================
.. _ibcdfo_pypkg: https://github.com/POptUS/IBCDFO/tree/main/ibcdfo_pypkg/src/ibcdfo
.. _README.md: https://github.com/poptus/IBCDFO/blob/main/README.md
.. _tox.ini: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg/tox.ini
.. _pyproject.toml: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg/pyproject.toml
.. _load_tests.py: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg/src/ibcdfo/load_tests.py

Prior to initiating this work, please discuss this process with the |poptus|
development team to coordinate the successful review and integration of
the subpackage, including how and when it will be included in subsequent
releases.

Working in a dedicated feature branch based on the latest commit in ``main``:

* Add the new subpackage to the root of the repository in accordance with |poptus|
  repository requirements
* Review license and copyright information provided in the |ibcdfo| repository,
  determine the license to be used for distributing the new subpackage, and specify
  the license information correctly in the repository
* Add the new subpackage implementation as symlinks in the appropriate
  ibcdfo_pypkg_ subdirectory
* Update load_tests.py_ in the main package so that it builds a test suite that
  includes the subpackage tests
* Adapt pyproject.toml_

  * Update the authors list if necessary
  * Update or expand all requirements as needed, including supported versions
  * Add test and package data in the new subpackage to the setuptools ``package-data``
    section (if any)
  * Update all other metadata as needed

* Update tox.ini_

  * Add a new testenv dedicated to the new subpackage, if desired
  * Synchronize Python and external dependency version information with
    ``pyproject.toml`` (if applicable)

* Test locally with |tox|
* Synchronize Python and external dependency version information in GitHub CI
  actions with changes made in ``pyproject.toml`` (if applicable)
* Commit, push, and check associated GitHub CI action logs to verify correct
  integration
* Update the README.md_ file if necessary
* Update and add content for the subpackage in the User and Developer Guides
* Add examples to the Jupyter book if necessary

After the feature branch is merged into ``main`` |via| a PR with review, confirm
that all GitHub CI actions are passing successfully on the merge commit.
