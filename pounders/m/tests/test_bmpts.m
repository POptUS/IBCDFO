% This tests bmpts.m when it notes that:
% "Geometry points need to be coordinate directions!"

nf_max = 100;
g_tol = 1e-13;

n = 2;
m = n;

xs = [1e-8 -1e-8];

Low = [0 -1]; % Lower bounds
Upp = [1 0]; % Upper bounds

X_0 = zeros(3, n); % Starting points

X_0(1, :) = xs; % Near origin
X_0(2, :) = 10 * xs; % Farther from origin
X_0(3, :) = 100 * xs; % Colinear

objective = @(x) x; % Identity mapping
F_init = zeros(3, n);
for i = 1:3
    F_init(i, :) = objective(X_0(i, :));
end

nfs = 3; % Points that have been evaluated
xk_in = 1; % Best point's index in X0
delta_0 = 1e3; % Starting TR radius

Prior.xk_in = xk_in;
Prior.X_0 = X_0;
Prior.F_init = F_init;
Prior.nfs = nfs;

[X, F, hf, flag, xk_best] = pounders(objective, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior);



