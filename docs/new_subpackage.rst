Add New Subpackage to |ibcdfo|
==============================
.. _ibcdfo_pypkg: https://github.com/POptUS/IBCDFO/tree/main/ibcdfo_pypkg/src/ibcdfo
.. _README.md: https://github.com/poptus/IBCDFO/blob/main/README.md
.. _tox.ini: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg/tox.ini
.. _pyproject.toml: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg/pyproject.toml
.. _load_tests.py: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg/src/ibcdfo/load_tests.py

Prior to initiating this work, please discuss this process with the |poptus|
development team to coordinate upfront the successful review and integration of
the subpackage including how and when it will be included in subsequent
releases.

Working in a dedicated feature branch based off of the latest commit in ``main``:

* Add new subpackage to the root of the repo in accord with the |poptus|
  repository requirements
* Study license and copyright information provided in the |ibcdfo| repository,
  determine license to be used for distributing the new subpackage, and specify
  license information correctly in repository
* Add in the new subpackage implementation as symlinks in the correct
  ibcdfo_pypkg_ subdirectory
* Update load_tests.py_ in the main package so that it builds a suite that
  includes the tests of the subpackage
* Adapt pyproject.toml_

  * Update authors list if necessary
  * Update or expand all requirements as needed including supported versions
  * Add test and package data in new subpackage to setuptools ``package-data``
    section (if any)
  * Update all other metadata as needed

* Update tox.ini_

  * Add a new testenv in tox.ini_ dedicated to the new subpackage if so
    desired
  * Synchronize Python and external dependence version information to contents
    of ``pyproject.toml`` (if any)

* Test locally with |tox|
* Synchronize Python and external dependence version information in GitHub CI
  actions to version changes made in ``pyproject.toml`` (if any)
* Commit, push, and check associated GitHub CI action logs to see if constructed
  and integrated correctly
* Update the README.md_ file if necessary
* Update and add content for subpackage in User and Developer Guides
* Add examples to the Jupyter book if necessary

After the feature branch is merged into ``main`` |via| a PR with review, confirm
that all GitHub CI actions are passing successfully on the merge commit.
