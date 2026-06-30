.. _getting_started:

Getting Started
===============
.. _MINQ: https://github.com/POptUS/minq
.. _BenDFO: https://github.com/POptUS/BenDFO

Some |ibcdfo| functionality requires the use of the `MINQ`_ code.  Therefore,
both Python and |matlab| users should install a local |minq| clone prior to
testing or using |ibcdfo|.  Each version of |ibcdfo| is linked to a single
commit in the |minq| repository, which was typically the latest commit on the
|minq| ``main`` branch at the time that the |ibcdfo| version was released.  If a
user's |minq| clone is not set to the correct commit, an error message indicates
the commit required by the user's |ibcdfo| installation.

Similarly, testing |ibcdfo| requires that Python and |matlab| users install a
`BenDFO`_ local clone.  Prior to running tests, users should ensure that their
|bendfo| clone is set to the latest commit on ``main``.

Python Installation
-------------------
After installing a |minq| clone and before testing or using a Python |ibcdfo|
installation, provide to Python the location of your |minq| installation by
adding

.. code:: console

    /path/to/MINQ/py/minq5

to your ``PYTHONPATH`` environment variable.

Install from PyPI
^^^^^^^^^^^^^^^^^
.. _PyPI: https://pypi.org/project/ibcdfo/

The |ibcdfo| Python package is available for installation |via| PyPI_.  It can
be installed by setting up a terminal with the desired target Python and
executing

.. code:: console

    python -m pip install ibcdfo

Install from Clone
^^^^^^^^^^^^^^^^^^
A static installation of the |ibcdfo| Python package can be made using the
contents of a local |ibcdfo| clone by setting up a terminal with the desired
target Python and executing

.. code:: console

    $ cd /path/to/IBCDFO/ibcdfo_pypkg
    $ python -m pip install .

Testing the Installation
^^^^^^^^^^^^^^^^^^^^^^^^
Before testing the |ibcdfo| package, specify |bendfo|'s location by adding

.. code:: console

    /path/to/BenDFO/py

to your ``PYTHONPATH`` environment variable.  To test an installation, from the
``/path/to/BenDFO/data`` directory execute

.. code:: console

    $ python
    >>> import ibcdfo
    >>> ibcdfo.__version__
    <version>
    >>> ibcdfo.test()

where the output ``<version>`` should be identical to the value used during
installation.

|matlab| Installation
---------------------
.. _repository: https://github.com/poptus/IBCDFO

A |matlab| installation of |ibcdfo| requires installing a local clone of
the |ibcdfo| `repository`_ and adding

.. code:: console

    /path/to/MINQ/m/minq5
    /path/to/MINQ/m/minq8

to the |matlab| path.  To simplify testing, |matlab| users should prefer
installing |bendfo| in the same folder as their |ibcdfo| clone.

To run tests,

   1. open |matlab| in the ``tools`` folder,
   2. execute ``test_ibcdfo``, and
   3. inspect output.
