# POUNDerS - Practical Optimization Using No Derivatives for sums of Squares

## Overview

This code minimizes a sum of squares of blackbox (''zeroth-order'') functions, solving

````
min { f(X)=sum_(i=1:m) F_i(X)^2, such that Low_j <= X_j <= Upp_j, j=1,...,n }
````

where the user-provided blackbox `F` is specified in the handle fun. Evaluation
of this `F` must result in the return of a `1-by-m` row vector. Bounds must be
specified in Upp and Low but can be set to `Low=-Inf(1,n)` and `Upp=Inf(1,n)` if the
unconstrained solution is desired. The algorithm will not evaluate `F`
outside of these bounds, but it is possible to take advantage of function
values at infeasible `X` if these are passed initially through `(X0,F0)`.
In each iteration, the algorithm forms a set of quadratic models interpolating the
functions in `F` and minimizes an associated scalar-valued model within an
infinity-norm trust region.

Optionally, a user can specify an outer-function that maps the elements
of `F` to a scalar value to be minimized. Doing this also requires a function
handle (combinemodels) that tells pounders how to map the linear and
quadratic terms from the models of `F` into a single quadratic model.

## API
The POUNDerS API is

````
  [X, F, hF, flag, xk_in] = pounders(fun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp)
````
with optional `Prior`, `Options`, and `Model` inputs.

### Inputs
The inputs to POUNDerS are as follows, with default/recommended values
indicated in parentheses.

Required inputs are:
````
Ffun    [f h] Function handle so that Ffun(x) evaluates F (@calfun)
X_0     [dbl] [1-by-n] Initial point (zeros(1,n))
n       [int] Dimension (number of continuous variables)
nf_max  [int] Maximum number of function evaluations (>n+1) (100)
g_tol   [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
delta_0 [dbl] Positive initial trust region radius (.1)
m       [int] Number of components returned from Ffun
Low     [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
Upp     [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
````
Optional inputs are:
````
Prior   [dict/struct] of past evaluations of Ffun with keys/fields:
    X_init  [dbl] [nfs-by-n] Set of initial points
    F_init  [dbl] [nfs-by-m] Set of values for points in X_init
    xk_in   [int] Index in X_init for initial starting point
    nfs     [int] Number of function values in F_init known in advance

Options [dict/struct] of options to the method with keys/fields:
    printf   [int] 0 No printing to screen (default)
                   1 Debugging level of output to screen
                   2 More verbose screen output
    spsolver       [int] Trust-region subproblem solver flag (2)
    hfun           [f h] Function handle for mapping output from F
    combinemodels  [f h] Function handle for combining models of F

Model   [dict/struct] of options for model building with keys/fields:
    np_max  [int] Maximum number of interpolation points (>n+1) (2*n+1)
    Par     [1-by-4] list for formquad
````


### Outputs
The outputs from POUNDERs are:
````
X       [dbl] [nf_max+nfs-by-n] Locations of evaluated points
F       [dbl] [nf_max+nfs-by-m] Ffun values of evaluated points in X
hF      [dbl] [nf_max+nfs-by-1] Composed values h(Ffun) for evaluated points in X
flag    [dbl] Termination criteria flag:
              = 0 normal termination because of grad,
              > 0 exceeded nf_max evals,   flag = norm of grad at final X
              = -1 if input was fatally incorrect (error message shown)
              = -2 if a valid model produced X[nf] == X[xk_in] or (mdec == 0, hF[nf] == hF[xk_in])
              = -3 error if a NaN was encountered
              = -4 error in TRSP solver
              = -5 unable to get model improvement with current parameters
xk_in    [int] Index of point in X representing approximate minimizer
````

## Testing

### MATLAB
To run tests of MATLAB-based POUNDERs, users must have an up-to-date
[BenDFO](https://github.com/POptUS/BenDFO) clone installed and add

    /path/to/BenDFO/data
    /path/to/BenDFO/m

to their MATLAB path.  They should also ensure that the `minq` submodule in
their IBCDFO clone is at the latest version.

Note that some code in POUNDERs and its tests automatically alter the MATLAB
path.  While the POUNDERs tests will reset the path to its original state if
all tests pass, the path might remain altered if a test fails.

The MATLAB implementation of POUNDERs contains a single test case `Testpounders.m`,
which calls individual tests such as `test_bmpts.m`.

To fully test the MATLAB implementation of POUNDERs with `Testpounders` but without coverage:

   1. change to the `pounders/m/tests` directory
   2. open MATLAB, and
   3. execute `runtests` from the prompt.

To fully test the MATLAB implementation of POUNDERs with `Testpounders` and with coverage:

   1. change to the `pounders/m` directory
   2. open MATLAB, and
   3. execute `runtests("IncludeSubfolders", true, "ReportCoverageFor", pwd)`

The test output indicates where the HTML-format code coverage report can be found.

Users can also run each test function individually as usual if so desired.
Please refer to the inline documentation of each test or test case for more
information on how to run the test.

### Python
To fully test the Python implementation of POUNDERs:

   1. navigate to the `ibcdfo_pypkg` directory
   2. execute `tox -r -e coverage` in the terminal
