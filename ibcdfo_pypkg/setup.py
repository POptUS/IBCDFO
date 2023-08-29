import codecs
from pathlib import Path

from setuptools import setup

_PKG_ROOT = Path(__file__).resolve().parent


def readme_md():
    fname = _PKG_ROOT.joinpath("README.md")
    with codecs.open(fname, encoding="utf8") as fptr:
        return fptr.read()


def version():
    fname = _PKG_ROOT.joinpath("VERSION")
    with open(fname, "r") as fptr:
        return fptr.read().strip()


# Changes made to python_requires should be propagated to all tox.ini and all
# GitHub Action config files.
python_requires = ">=3.8"
code_requires = ["numpy", "scipy"]
test_requires = []  # "BenDFO" is required, but not yet installable
install_requires = code_requires + test_requires

package_data = {
    "IBCDFO": [
        "pounders/tests/regression_tests/benchmark_resutls/*.txt",
    ]
}

project_urls = {
    "Source": "https://github.com/POptUS/IBCDFO",
    "Documentation": "https://github.com/POptUS/IBCDFO",
    "Tracker": "https://github.com/POptUS/IBCDFO/issues",
}

setup(
    name="ibcdfo",
    version=version(),
    author="Jeffrey Larson, Matt Menickelly, and Stefan M. Wild",
    author_email="jmlarson@anl.gov",
    maintainer="Jeffrey Larson",
    maintainer_email="jmlarson@anl.gov",
    package_dir={"": "src"},
    package_data=package_data,
    url="https://github.com/POptUS/IBCDFO",
    project_urls=project_urls,
    license="MIT",
    description="Interpolation-Based Composite Derivative-Free Optimization",
    long_description=readme_md(),
    long_description_content_type="text/markdown",
    python_requires=python_requires,
    install_requires=install_requires,
    keywords="ibcdfo",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
    ],
)
