% Declares global variables (nprob,m,probtype) used within calfun.m and
% evaluates calfun at x.

function [fvec] = calfun_wrapper(x, m, nprob, probtype, fvals, nfev, np)

    bendfo_location = "../../../../../BenDFO/";
    addpath([bendfo_location  "m/"]);

    BenDFO.nprob = nprob;
    BenDFO.m = m;
    BenDFO.n = length(x);

    [~, fvec] = calfun(x, BenDFO, probtype);
end
