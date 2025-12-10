# -- Project information -----------------------------------------------------
import json

from pathlib import Path

from ibcdfo import __version__

_ROOT_PATH = Path(__file__).parents[1].resolve()

# In accord with version specifications for the contents of this repository, we
# link the documentation's version to the Python package's version.
project = 'IBCDFO'
copyright = "2023, The Regents of the University of California, " \
    "through Lawrence Berkeley National Laboratory and UChicago Argonne " \
    "LLC through Argonne National Laboratory (subject to receipt of any " \
    "required approvals from the U.S. Dept. of Energy).  All rights reserved"
author = "Jeffrey Larson, Matt Menickelly, and Stefan M. Wild"
version = __version__
release = version

latex_packages = [
    'xspace',
    'mathtools',
    'amsfonts', 'amsmath', 'amssymb'
]
latex_macro_files = ['base', 'notation']

# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.todo',
              'sphinxcontrib.matlab',
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
autosectionlabel_prefix_document = True

todo_include_todos = True

bibtex_bibfiles = ['ibcdfo.bib']

matlab_src_dir = _ROOT_PATH
matlab_short_links = True

mathjax3_config = {
    'loader': {},
    'tex': {
        'macros': {}
    }
}
for each in latex_macro_files:
    with open(f"latex_macros_{each}.json", "r") as fptr:
        macro_configs = json.load(fptr)
    for _, macros_all in macro_configs.items():
        for command, value in macros_all.items():
            assert command not in mathjax3_config['tex']['macros']
            mathjax3_config['tex']['macros'][command] = value

# -- Options for Math --------------------------------------------------------
math_numfig = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

# -- LaTeX configuration -----------------------------------------------------
latex_engine = "pdflatex"
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": ""
}
for package in latex_packages:
    latex_elements['preamble'] += f'\\usepackage{{{package}}}\n'
# Setup LaTeX to split author names across more than one line
#latex_documents = [
#    ('index', 'ibcdfo.tex', 'IBCDFO',
#     author.replace(', ', '\\and ').replace(' and ', '\\and and '),
#     'manual')
#]

# Configure LaTeX with macros
for each in latex_macro_files:
    with open(f"latex_macros_{each}.json", "r") as fptr:
        macro_configs = json.load(fptr)
    for cmd_type, macros_all in macro_configs.items():
        for command, value in macros_all.items():
            if isinstance(value, str):
                macro = rf"\{cmd_type}{{\{command}}} {{{value}}}"
            elif len(value) == 2:
                value, n_args = value
                macro = rf"\{cmd_type}{{\{command}}}[{n_args}] {{{value}}}"
            else:
                raise NotImplementedError("No use case yet")
            latex_elements['preamble'] += (macro + "\n")
