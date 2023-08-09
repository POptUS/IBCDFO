# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
import numpy as np


BenDFO.probtype = "smooth"
addpath("../")
addpath("../../../../BenDFO/m/")
addpath("../../../../BenDFO/data/")
addpath("../h_examples/")
addpath("../../../pounders/m")

mkdir("benchmark_results")
nfmax = 500
factor = 10
subprob_switch = "linprog"
scipy.io.loadmat("dfo.txt")
filename = np.array(["./benchmark_results/manifold_samplingM_nfmax=", num2str(nfmax), ".mat"])
Results = cell(1, 53)
# for row = find(cellfun(@length,Results)==0)
for row in np.array([1, 2, 7, 8, 43, 44, 45]).reshape(-1):
    nprob = dfo(row, 1)
    n = dfo(row, 2)
    m = dfo(row, 3)
    factor_power = dfo(row, 4)
    BenDFO.nprob = nprob
    BenDFO.n = n
    BenDFO.m = m
    LB = -Inf * np.ones((1, n))
    UB = Inf * np.ones((1, n))
    xs = dfoxs(n, nprob, factor**factor_power)
    for jj in np.arange(1, 2 + 1).reshape(-1):
        if jj == 1:
            hfun = pw_maximum_squared
        if jj == 2:
            hfun = pw_minimum_squared
        Ffun = calfun_wrapper
        x0 = np.transpose(xs)
        X, F, h, xkin, flag = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch)
        Results[jj, row].alg = "Manifold sampling"
        Results[jj, row].problem = np.array(["problem ", num2str(row), " from More/Wild with hfun="])
        Results[jj, row].Fvec = F
        Results[jj, row].H = h
        Results[jj, row].X = X

save(filename, "Results")


def calfun_wrapper(x):
    __, fvec, __ = calfun(x)
    fvec = np.transpose(fvec)
    return fvec
