# Manifold Sampling

## Overview
This code solves the problem
````
    minimize h(F(x))
````
where `x` is an `[n by 1]` vector, `F` is a blackbox function mapping from `R^n` to
`R^p`, and `h` is a nonsmooth function mapping from `R^p` to `R`.

## API
The manifold_sampling_primal API is
````
[X, F, h, xkin, flag] = manifold_sampling_primal(hfun, Ffun, x0, L, U, nfmax, subprob_switch)
````

### Inputs
````
 hfun:    [func]   Given point z, returns
                     - [scalar] the value h(z)
                     - [p x l] gradients for all l limiting gradients at z
                     - [1 x l set of strings] hashes for each manifold active at z
                   Given point z and l hashes H, returns
                     - [1 x l] the value h_i(z) for each hash in H
                     - [p x l] gradients of h_i(z) for each hash in H
 Ffun:    [func]    Evaluates F, the black box simulation, returning a [1 x p] vector.
 x0:      [dbl] [1-by-n] Starting point
 L        [dbl] [1-by-n] Vector of lower bounds
 U        [dbl] [1-by-n] Vector of upper bounds
````

### Outputs
````
Outputs:
  X:      [nfmax x n]   Points evaluated
  F:      [nfmax x p]   Their simulation values
  h:      [nfmax x 1]   The values h(F(x))
  xkin:   [int]         Current trust region center
  flag:   [int]         Inform user why we stopped.
                          -1 if error
                           0 if nfmax function evaluations were performed
                           final model gradient norm otherwise
````

Some other values and intermediate Variables:
````
 n:       [int]     Dimension of the domain of F (deduced from x0)
 p:       [int]     Dimension of the domain of h (deduced from evaluating F(x0))
 delta:   [dbl]     Positive starting trust region radius
 nf    [int]         Counter for the number of function evaluations
 s_k   [dbl]         Step from current iterate to approx. TRSP solution
 norm_g [dbl]        Stationary measure ||g||
 Gres [n x p]        Model gradients for each of the p outputs from Ffun
 Hres [n x n x p]    Model Hessians for each of the p outputs from Ffun
 Hash [cell]         Contains the hashes active at each evaluated point in X
 Act_Z_k [l cell]      Set of hashes for active selection functions in TR
 G_k  [n x l]        Matrix of model gradients composed with gradients of elements in Act_Z_k
 D_k  [p x l_2]      Matrix of gradients of selection functions at different points in p-space
````

## Installation

### MATLAB
Users must download and install the
[allcomb](https://www.mathworks.com/matlabcentral/fileexchange/10064-allcomb-varargin)
function from MathWork's File Exchange and add its location to the MATLAB path.

## Testing

### MATLAB
To run tests of MATLAB-based manifold sampling, in addition to general
installation steps users must have an up-to-date
[BenDFO](https://github.com/POptUS/BenDFO) clone installed and add

    /path/to/BenDFO/data
    /path/to/BenDFO/m

to their MATLAB path.

Note that some code in manifold sampling and its tests automatically alter the
MATLAB path.  While the manifold sampling tests will reset the path to its
original state if all tests pass, the path might remain altered if a test
fails.

The MATLAB implementation of manifold sampling contains a single test case
`Testmanifoldsampling.m`, which calls individual tests such as
`test_one_norm.m`.

To fully test the MATLAB implementation of manifold sampling with
`Testmanifoldsampling` but without coverage:

   1. change to the `manifold_sampling/m/tests` directory
   2. open MATLAB, and
   3. execute `runtests` from the prompt.

To fully test the MATLAB implementation of manifold sampling with
`Testmanifoldsampling` and with coverage:

   1. change to the `manifold_sampling/m` directory
   2. open MATLAB, and
   3. execute `runtests("IncludeSubfolders", true, "ReportCoverageFor", pwd)`

The test output indicates where the HTML-format code coverage report can be found.

Users can also run each test function individually as usual if so desired.
Please refer to the inline documentation of each test or test case for more
information on how to run the test.
