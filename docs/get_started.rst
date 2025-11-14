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

to fetch all files contained in the submodules.  Instead of using ``git pull``
to maintain your clone up-to-date, run

.. code:: console

    git pull --recurse-submodules

which not only updates the |ibcdfo| content but also updates all submodules.

Python Installation
-------------------
.. _MINQ: https://github.com/POptUS/minq

|ibcdfo| requires the installation of an up-to-date local clone of `MINQ`_,
which is not distributed as a Python package but is included in your |ibcdfo|
local clone as a submodule.  The above procedures should be sufficient for
installing and maintaining up-to-date the |minq| requirement.  When using the
|ibcdfo| package, specify |minq|'s location by adding

.. code:: console

    /path/to/IBCDFO/minq/py/minq5

to your ``PYTHONPATH`` environment variable.

Install from PyPI
^^^^^^^^^^^^^^^^^
.. _PyPI: https://pypi.org/project/ibcdfo/

The |ibcdfo| Python package is available for installation |via| PyPI_.  It can
be installed by setting up a terminal with the desired target Python and
executing

.. code:: console

    python -m pip install ibcdfo

.. todo::
    Should users checkout the release commit associated with their installed
    version to ensure that they are using the correct MINQ version?

Install from Clone
^^^^^^^^^^^^^^^^^^
A static installation of the |ibcdfo| Python package can be made using the
contents of your local clone by setting up a terminal with the desired target
Python and executing

.. code:: console

    $ cd /path/to/clone/ibcdfo_pypkg
    $ python -m pip install .

Testing Installation
^^^^^^^^^^^^^^^^^^^^
Testing the |ibcdfo| Python package requires that the `BenDFO`_ Python package
be installed and maintained up-to-date.  It is not distributed as a Python
package and users must clone it manually.  When testing the |ibcdfo| package,
specify |bendfo|'s location by adding

.. code:: console

    /path/to/BenDFO/py

to your ``PYTHONPATH`` environment variable.

An installation can be tested by executing from ``/path/to/BenDFO/data``

.. code:: console

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

To run tests,

   1. open |matlab| in the ``tools`` folder,
   2. execute ``test_ibcdfo``, and
   3. inspect output.
