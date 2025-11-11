Getting Started
===============
.. _BenDFO: https://github.com/POptUS/BenDFO
.. _repository: https://github.com/poptus/IBCDFO

Both the Python and |matlab| installations require installing a local clone of
the |ibcdfo| `repository`_, which depends on one or more submodules.  After
cloning the repository and before using the code in the clone, from within the
clone please run

.. code:: console

    git submodule update --init --recursive

to fetch all files contained in the submodules.  To maintain this clone
up-to-date, run

.. code:: console

    git pull --recurse-submodules

which not only updates the clone but also updates all submodules, instead of
``git pull``.

Python Installation
-------------------
.. _MINQ: https://github.com/POptUS/minq

|ibcdfo| requires the installation of a local clone of `MINQ`_, which is not
distributed as a Python package but is included in your |ibcdfo| local clone as
a submodule.  Therefore, the above installation procedure should have already
installed it for use.  When using the |ibcdfo| package, specify its location by
adding

.. code:: console

    /path/to/IBCDFO/minq/py/minq5

to your ``PYTHONPATH`` environment variable.

Install from PyPI
^^^^^^^^^^^^^^^^^
.. _PyPI: https://pypi.org/project/ibcdfo/

The |ibcdfo| Python package is available for installation |via| PyPI_.  It can
be installed by setting up a terminal with the desired target Python and
executing

.. code-block:: console

    python -m pip install ibcdfo

.. todo::
    Should users checkout the release commit associated with their installed
    version to ensure that they are using the correct MINQ version?

Install from Clone
^^^^^^^^^^^^^^^^^^
The |ibcdfo| Python package can be installed directly from the contents of the
local clone by setting up a terminal with the desired target Python and
executing

.. code-block:: console

    $ cd /path/to/clone/ibcdfo_pypkg
    $ python -m pip install .

Testing Installation
^^^^^^^^^^^^^^^^^^^^
The |ibcdfo| Python package requires that the `BenDFO`_ Python package be
installed for testing.  It is not distributed as a Python package and users must
clone it manually.  When testing the |ibcdfo| package, specify its location by
adding

.. code:: console

    /path/to/BenDFO/py

to your ``PYTHONPATH`` environment variable.

An installation can be tested by executing from ``/path/to/BenDFO/data``

.. code-block:: console

    $ python
    >>> import ibcdfo
    >>> ibcdfo.__version__
    <version>
    >>> ibcdfo.test()

where the output ``<version>`` should be identical to the value used during
installation.

Setup MATLAB Code
-----------------
The above general procedure for installing and updating a local clone is
sufficient for installing |matlab| code.  To test the clone, users must have an
up-to-date `BenDFO`_ clone installed in the same folder as their |ibcdfo| clone.

To run tests with coverage enabled,

   1. open |matlab| in the ``tools`` folder and
   2. execute ``test_ibcdfo``.

The test output indicates where the HTML-format code coverage report can be
found, should this be useful.

.. todo::
    Move information about running with coverage to Dev Guide?
