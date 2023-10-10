% This tests bmpts.m when it notes that:
% "Geometry points need to be coordinate directions!"

nfmax = 100;
gtol = 1e-13;

n = 2;
m = n;

xs = [1e-8 -1e-8];

npmax = 2 * n + 1;  % Maximum number of interpolation points [2*n+1]
L = [0 -1]; % Lower bounds
U = [1 0]; % Upper bounds

X0 = zeros(3, n); % Starting points

X0(1, :) = xs; % Near origin
X0(2, :) = 10 * xs; % Farther from origin
X0(3, :) = 100 * xs; % Colinear

objective = @(x) x; % Identity mapping
for i = 1:3
    F0(i, :) = objective(X0(i, :));
end

nfs = 3; % Points that have been evaluated
xkin = 1; % Best point's index in X0
delta = 1e3; % Starting TR radius

[X, F, flag, xk_best] = pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, 0, 1);
