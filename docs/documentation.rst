Documentation
=============

Tools
-----
.. _`typos`: https://github.com/crate-ci/typos

A GitHub action is run automatically to check for typographic errors in all
documentation and source code in the repository using the `typos`_ tool with our
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

Both the User and Developer Guides are developed in a single `Sphinx`_ project,
which resides in the ``docs`` folder, for publication |via| `Read the Docs`_.
The guides' contents are assembled from files in ``docs`` and from docstrings of
Python code in the package.  General text is written in `reStructuredText`_.
Python docstrings should be written using the default `autodoc`_ formatting.

Manually maintaining the list of what exceptions are raised by a function or
class is difficult at best since, for example, readers might reasonably assume
that all possible exceptions that could be raised by a function will be noted in
the function's documentation.  This would include all exceptions raised by all
functions called by that function.  Therefore, we do **not** document
exceptions.

Presently, this package does not require the use of type hints.  Similarly,
there is no requirement to specify argument or return types in the docstring.
Note that docstrings do not need to specify the default values of optional
arguments since Sphinx should be able to identify this information in the code
and include it appropriately in the rendered documentation.

The guides can be rendered locally in HTML format using |tox|

.. code:: console

    $ cd /path/to/IBCDFO/ibcdfo_pypkg
    $ tox -e html
 
with the rendered output available at ``docs/build_html/index.html``.  Similar
commands will generate PDF-format output with the ``pdf`` task.  The
configuration for those two tasks in ``tox.ini`` can be used as a guide for
working with this documentation outside of |tox|.

Macro Definitions
-----------------
To aid in presenting uniform content not only within each set of documents but
also across all documents, a set of common macros have been defined for both
documentation tools

* ``docs/sphinx_macros.json``
* ``book/_config.yml``

Please familiarize yourself with this list of macros before altering
documentation and please use the macros throughout your changes and additions.
