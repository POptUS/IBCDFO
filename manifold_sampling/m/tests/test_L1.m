n = 2;
m = 2;

nfmax = 1000;
subprob_switch = 'linprog';
LB = -2 * ones(1, n);
UB = 2 * ones(1, n);
x0 = [-1.2, 1.0];


hfun = @one_norm; 

[X, F, h, xkin, flag] = manifold_sampling_primal(hfun, @Ffun, x0, LB, UB, nfmax, subprob_switch);


function F = Ffun(x)
    % Rosenbrock function
    F = [10 * (x(2) - x(1)^2), 1 - x(1)];
end