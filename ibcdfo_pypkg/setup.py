import codecs

from pathlib import Path
from setuptools import setup, find_packages

_FILE_PATH = Path(__file__).parent

def readme_md():
    fname = _FILE_PATH.joinpath("README.md")
    with codecs.open(fname, encoding="utf8") as fptr:
        return fptr.read()

def version():
    fname = _FILE_PATH.joinpath("VERSION")
    with open(fname, "r") as fptr:
        return fptr.read().strip()

# wheel must be installed before installing this package so that the LICENSE
# file is actually installed as part of the source distribution when it is
# installed with pip.  This implies that the LICENSE is in the sdist tarball for
# those who access the code outside of pip.
python_requires = ">=3.7"
install_requires = ["wheel", "numpy>=1.16.5", "scipy>=1.6"]
# TODO: Can we integrate testing using tox or other solution?
#test_requires = ["mpi4py", "oct2py"]

packages = find_packages(include=['ibcdfo', \
                                  'ibcdfo.pounders'])
# TODO: This is a hack because these are python files rather than package data,
# which could be something like data files needed by tests.  I believe that this
# is unnecessary if test file organization follows the pattern
#                         test/test*.py
package_data = {"ibcdfo.pounders": ["tests/*/*.py"]}

project_urls = {"Source":        "https://github.com/POptUS/IBCDFO", \
                "Documentation": "https://github.com/POptUS/IBCDFO", \
                "Tracker":       "https://github.com/POptUS/IBCDFO/issues"}

setup(
    name="ibcdfo",
    version=version(),
    author="Jeffrey Larson, Matt Menickelly, and Stefan M. Wild",
    author_email="jmlarson@anl.gov",
    packages=packages,
    package_data=package_data,
    url="https://github.com/POptUS/IBCDFO",
    project_urls=project_urls,
    license="MIT",
    description="Interpolation-Based Composite Derivative-Free Optimization",
    long_description=readme_md(),
    long_description_content_type='text/markdown',
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
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering",
    ],
)
