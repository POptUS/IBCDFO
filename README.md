# IBCDFO

Interpolation-Based Composite Derivative-Free Optimization

![GitHub](https://img.shields.io/github/license/poptus/IBCDFO)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/POptUS/IBCDFO/badge.svg?branch=main)](https://coveralls.io/github/POptUS/IBCDFO?branch=main)

This page contains source code for interpolation-based optimization methods for
composite derivative-free optimization.

Relevant references include:

  - J. Larson and M. Menickelly. Structure-aware methods for expensive
  derivative-free nonsmooth composite optimization. *arXiv:2207.08264*. 2022.
  [LINK](https://arxiv.org/abs/2207.08264)

  - J. Larson, M. Menickelly, and B. Zhou. Manifold sampling for optimizing
  nonsmooth nonconvex compositions. *SIAM Journal on Optimization*.
  31(4):2638–2664, 2021
  [DOI](https://doi.org/10.1137/20M1378089)

  - K. A. Khan, J. Larson, and S. M. Wild. Manifold sampling for optimization of
  nonconvex functions that are piecewise linear compositions of smooth
  components. *SIAM Journal on Optimization* 28(4):3001--3024, 2018,
  [DOI](https://doi.org/10.1137/17m114741x)

  - S. M. Wild. POUNDERS in TAO: Solving Derivative-Free Nonlinear
  Least-Squares Problems with POUNDERS. *Advances and Trends in Optimization with
  Engineering Applications*. SIAM. 529--539, 2017.
  [DOI](https://doi.org/10.1137%2F1.9781611974683.ch40)

  - J. Larson, M. Menickelly, and S. M. Wild. Manifold sampling for l1 nonconvex
  optimization. *SIAM Journal on Optimization*. 26(4):2540–2563, 2016.
  [DOI](https://doi.org/10.1137/15M1042097)

## Contributing to IBCDFO

Contributions are welcome in a variety of forms; please see [CONTRIBUTING](CONTRIBUTING.rst).

## Installation & Updating
Note that this repository depends on one or more submodules.  After cloning
this repository, from within the clone please run

``git submodule update --init --recursive``

to fetch all files contained in the submodules.  This must be done before
attempting to use the code in the clone.  Issuing the command `git pull` will
update the repository, but not the submodules.  To update the clone and all its
submodules simultaneously, run

``git pull --recurse-submodules``.

The `ibcdfo` python package can be installed by setting up a terminal with the
target python and pip pair and executing
```
> pushd ibcdfo_pypkg
> python setup.py sdist
> pip install dist/ibcdfo-<version>.tar.gz
> popd
```
where `<version>` can be determined by looking at the output of the `sdist`
command.  The installation can be partially tested by executing
```
> python
>>> import ibcdfo
>>> ibcdfo.__version__
<version>
```
where the output `<version>` should be identical to the value used during
installation.

## License

All code included in IBCDFO is open source, with the particular form of license contained in the top-level
subdirectories.  If such a subdirectory does not contain a LICENSE file, then it is automatically licensed
as described in the otherwise encompassing IBCDFO [LICENSE](/LICENSE).

## Resources

To seek support or report issues, e-mail:

 * ``poptus@mcs.anl.gov``
