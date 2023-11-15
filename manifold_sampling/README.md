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
