# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import json

# IBCDFO contents can potentially have several different versions, including
# different versions for each language implementation of a single method.
#
# Therefore, we choose to link this version information to the version of the
# documentation, which spans all methods and implementations.
#
# TODO: Is this really what we want?  Will there be a global version of IBCDFO?
# Can we manage this value automatically?
project = 'IBCDFO'
copyright = "2023, The Regents of the University of California, " \
    "through Lawrence Berkeley National Laboratory and UChicago Argonne " \
    "LLC through Argonne National Laboratory (subject to receipt of any " \
    "required approvals from the U.S. Dept. of Energy).  All rights reserved"
author = "Jeffrey Larson, Matt Menickelly, Jared P. O'Neal, and Stefan M. Wild"
version = "0.0.1"
release = version

latex_packages = [
    'xspace',
    'mathtools',
    'amsfonts', 'amsmath', 'amssymb'
]
latex_macro_files = ['base', 'notation']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# General
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.todo',
              'sphinx.ext.mathjax',
              'sphinxcontrib.bibtex']
numfig = True

# https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#substitutions
rst_prolog = ""
with open("sphinx_macros.json", "r") as fptr:
    macro_configs = json.load(fptr)
for key, value in macro_configs.items():
    rst_prolog += f".. |{key}| replace:: {value}\n"

# Extensions
autoclass_content = "init"
autodoc_member_order = "bysource"

autosectionlabel_prefix_document = True

todo_include_todos = True

mathjax3_config = {
    'loader': {},
    'tex': {
        'macros': {}
    }
}
for each in latex_macro_files:
    with open(f"latex_macros_{each}.json", "r") as fptr:
        macro_configs = json.load(fptr)
    for cmd_type, macros_all in macro_configs.items():
        for command, value in macros_all.items():
            assert command not in mathjax3_config['tex']['macros']
            mathjax3_config['tex']['macros'][command] = value

bibtex_bibfiles = ['ibcdfo.bib']

# -- Options for Math --------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-math

math_numfig = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']

# -- LaTeX configuration -----------------------------------------------------
# Some of this configuration is from
# https://stackoverflow.com/questions/9728292/creating-latex-math-macros-within-sphinx

latex_engine = "pdflatex"
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": ""
}
for package in latex_packages:
    latex_elements['preamble'] += f'\\usepackage{{{package}}}\n'
# Setup LaTeX to split author names across more than one line
latex_documents = [
    ('index', 'ibcdfo.tex', 'IBCDFO',
     author.replace(', ', '\\and ').replace(' and ', '\\and and '),
     'manual')
]

# Configure LaTeX with macros
for each in latex_macro_files:
    with open(f"latex_macros_{each}.json", "r") as fptr:
        macro_configs = json.load(fptr)
    for cmd_type, macros_all in macro_configs.items():
        for command, value in macros_all.items():
            if isinstance(value, str):
                macro = f"\{cmd_type}{{\{command}}} {{{value}}}"
            elif len(value) == 2:
                value, n_args = value
                macro = f"\{cmd_type}{{\{command}}}[{n_args}] {{{value}}}"
            else:
                raise NotImplementedError("No use case yet")
            latex_elements['preamble'] += (macro + "\n")
