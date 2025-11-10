Getting Started
===============

Python package Installation
---------------------------

Install from PyPI
^^^^^^^^^^^^^^^^^
.. _PyPI: https://pypi.org/project/ibcdfo/

The |ibcdfo| Python pacakge is available for installation |via| PyPI_.  It can
be installbed by setting up a terminal with the target Python and |pip| pair
and executing

.. code-block:: console

    pip install ibcdfo

Install from Clone
^^^^^^^^^^^^^^^^^^

.. _repository: https://github.com/poptus/IBCDFO/blob/main

Note that the code in the |ibcdfo| repository_ depends on one or more
submodules.  After cloning the repository, from within the clone please run

.. code-block:: console

    git submodule update --init --recursive

to fetch all files contained in the submodules.  This must be done before
attempting to use the code in the clone.  Issuing the command ``git pull`` will
update the repository, but not the submodules.  To update the clone and all its
submodules simultaneously, run

.. code-block:: console

    git pull --recurse-submodules

A static, source version of the |ibcdfo| Python package can be installed by
setting up a terminal with the target Python and |pip| pair and executing

.. code-block:: console

    $ cd /path/to/clone/ibcdfo_pypkg
    $ python setup.py sdist
    $ python -m pip install dist/ibcdfo-<version>.tar.gz

where ``<version>`` can be determined by looking at the output of the ``sdist``
command.

The |ibcdfo| Python package can be installed in editable or developer mode by
setting up a terminal with the target Python and |pip| pair and executing

.. code-block:: console

    python -m pip install -e /path/to/clone/ibcdfo_pypkg

Testing Installation
^^^^^^^^^^^^^^^^^^^^

An installation can be tested by executing

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

.. todo::
    Write this
