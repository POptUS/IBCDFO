# POUNDerS - Practical Optimization Using No Derivatives for sums of Squares

## Overview

This code minimizes a sum of squares of blackbox (''zeroth-order'') functions, solving

````
min { f(X)=sum_(i=1:m) F_i(X)^2, such that L_j <= X_j <= U_j, j=1,...,n }
````

where the user-provided blackbox `F` is specified in the handle fun. Evaluation
of this `F` must result in the return of a `1-by-m` row vector. Bounds must be
specified in U and L but can be set to `L=-Inf(1,n)` and `U=Inf(1,n)` if the
unconstrained solution is desired. The algorithm will not evaluate `F`
outside of these bounds, but it is possible to take advantage of function
values at infeasible `X` if these are passed initially through `(X0,F0)`.
In each iteration, the algorithm forms a set of quadratic models interpolating the
functions in `F` and minimizes an associated scalar-valued model within an
infinity-norm trust region.

## API
The POUNDerS API is

````
  [X,F,flag,xkin] = pounders(fun,X0,n,npmax,nfmax,gtol,delta,nfs,m,F0,xkin,L,U,printf)
````

### Inputs
The inputs to POUNDerS are as follows, with default/recommended values
indicated in parentheses:
````
fun     [f h] Function handle so that fun(x) evaluates F (@calfun)
X0      [dbl] [max(nfs,1)-by-n] Set of initial points (zeros(1,n))
n       [int] Dimension (number of continuous variables)
npmax   [int] Maximum number of interpolation points (>n+1) (2*n+1)
nfmax   [int] Maximum number of function evaluations (>n+1) (100)
gtol    [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
delta   [dbl] Positive trust region radius (.1)
nfs     [int] Number of function values (at X0) known in advance (0)
m       [int] Number of residual components (outputs of F)
F0      [dbl] [nfs-by-m] Set of known function values ([])
xkin    [int] Index of point in X0 at which to start from (1)
L       [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
U       [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
printf  [log] 0 No printing to screen (default)
              1 Debugging level of output to screen
              2 More verbose screen output
spsolver [int] Trust-region subproblem solver flag (2)
````

Optionally, a user can specify an outer-function that maps the the elements
of `F` to a scalar value (to be minimized). Doing this also requires a function
handle (combinemodels) that tells pounders how to map the linear and
quadratic terms from the residual models into a single quadratic TRSP model.

````
hfun           [f h] Function handle for mapping output from F
combinemodels  [f h] Function handle for combine residual models
````

### Outputs
The outputs from POUNDERs are:
````
X       [dbl] [nfmax+nfs-by-n] Locations of evaluated points
F       [dbl] [nfmax+nfs-by-m] Function values of evaluated points
flag    [dbl] Termination criteria flag:
              = 0 normal termination because of model grad,
              > 0 exceeded nfmax evals,   flag = norm of model grad at final X
              = -1 if input was fatally incorrect (error message shown)
              = -2 model failure
              = -3 error if a NaN was encountered
              = -4 error in TRSP Solver
              = -5 unable to get model improvement with current parameters
xkin    [int] Index of point in X representing approximate minimizer
````

## Testing

To fully test the MATLAB implementation of POUNDERs:

   1. change to the `pounders/m/tests` directory
   2. open MATLAB, and
   3. execute `runtests` from the prompt.
