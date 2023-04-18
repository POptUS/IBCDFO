import codecs
from pathlib import Path

from setuptools import setup

_FILE_PATH = Path(__file__).parent


def readme():
    fname = _FILE_PATH.joinpath("README.md")
    with codecs.open(fname, encoding="utf8") as fptr:
        return fptr.read()


def version():
    fname = _FILE_PATH.joinpath("VERSION")
    with open(fname, "r") as fptr:
        return fptr.read().strip()


reqs_list = ["numpy", "scipy"]

pkg_dict = {"ibcdfo": ["pounders/py/*", "pounders/py/tests/regression_tests/*"]}

test_list = ["mpi4py", "oct2py"]

setup(
    name="ibcdfo",
    version=version(),
    author="Jeffrey Larson, Matt Menickelly, and Stefan M. Wild",
    author_email="jmlarson@anl.gov",
    maintainer="Jeffrey Larson, Matt Menickelly, and Stefan M. Wild",
    maintainer_email="jmlarson@anl.gov",
    packages=["pounders"],
    package_data=pkg_dict,
    url="https://github.com/POptUS/IBCDFO/",
    license="MIT License",
    description="Interpolation-Based Composite Derivative-Free Optimization",
    long_description=readme(),
    setup_requires=[],
    install_requires=reqs_list,
    tests_require=test_list,
    # test_suite='ibcdfo.test_suite',
    keywords="ibcdfo",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering",
    ],
)
