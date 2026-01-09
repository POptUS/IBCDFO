Documentation
=============

Tools
-----
.. _`typos`: https://github.com/crate-ci/typos

A GitHub action is run automatically to check for typographic errors in all
documentation and source code in the repository using the typos_ tool with our
``.typos.toml`` configuration file.  An associated ``typos`` command line tool
can also be installed locally by developers for checking eagerly for mistakes:

.. code:: console

    $ cd /path/to/IBCDFO
    $ typos --config=.typos.toml

Other useful tools are described below with regard to the particular
documentation that they serve.

Guides
------
.. _`Sphinx`: https://www.sphinx-doc.org
.. _`reStructuredText`: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _`autodoc`: https://www.sphinx-doc.org/en/master/tutorial/automatic-doc-generation.html
.. _`Read the Docs`: https://about.readthedocs.com

Both the User and Developer Guides are developed in a single Sphinx_ project,
which resides in the ``docs`` folder, for publication |via| `Read the Docs`_.
The guides' contents are assembled from files in ``docs`` and from docstrings of
Python and |matlab| code.  General text is written in reStructuredText_.
Python and |matlab| docstrings should be written using the default autodoc_
formatting.

Manually maintaining the list of what exceptions are raised by a function or
class is difficult at best since, for example, readers might reasonably assume
that all possible exceptions that could be raised by a function will be noted in
the function's documentation.  This would include all exceptions raised by all
functions called by that function.  In addition, exceptions that are raised with
concise, informative error messages are self documenting.  Therefore, we do
**not** document exceptions.

Presently, this package does not require the use of type hints in Python code.
Similarly, there is no requirement to specify argument or return types in the
Python or |matlab| docstrings.  Note that Python docstrings do not need to
specify the default values of optional arguments since Sphinx is able to
identify this information in the code and include it appropriately in the
rendered documentation.

.. todo::
    What about |matlab|?

The guides can be rendered locally in HTML format using |tox|

.. code:: console

    $ cd /path/to/IBCDFO/ibcdfo_pypkg
    $ tox -e html
 
with the rendered output available at ``docs/build_html/index.html``.  The
configuration for that task in ``tox.ini`` can be used as a guide for working
with this documentation outside of |tox|.

Macro Definitions
-----------------
To aid in presenting uniform content not only within each set of documents but
also across all documents, a set of common text macros have been defined in

* ``docs/sphinx_macros.json``

and common LaTeX math mode macros in

* ``docs/latex_macros_base.json``
* ``docs/latex_macros_notation.json``.

Please familiarize yourself with this list of macros before altering
documentation and please use the macros throughout your changes and additions.
