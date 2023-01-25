function [J, fvec] = jacobian(m, n, x, nprob)
%     This subroutine computes the Jacobian of the nonlinear equations
%     defining the benchmark problems in
%
%     Benchmarking Derivative-Free Optimization Algorithms
%     Jorge J. More' and Stefan M. Wild
%     SIAM J. Optimization, Vol. 20 (1), pp.172-191, 2009.
%
%     The dependencies of this function are based on executing the AD
%     software adimat on a modified from of the dfovec function from
%     http://www.mcs.anl.gov/~more/dfo/
%     See the instructions of dfovec.m for additional details on these
%     nonlinear benchmark problems (and appropriate values of m and n).
%
%   J = jacobian(m,n,x,nprob)
%
%       J is an output array of size m-by-n, with J(i,j) denoting the
%         derivative (evaluated at x) of the ith equation with respect to
%         the jth variable.
%       fvec returns the usual dfovec
%       m and n are positive integer input variables. n must not
%         exceed m.
%       x is an input array of length n.
%       nprob is a positive integer input variable which defines the
%         number of the problem. nprob must not exceed 22.
%
%     Argonne National Laboratory
%     Stefan Wild. July 2014.

% global tau
% global probtype nfev Jfev fvecHIST xHIST JHIST
% addpath('adimat_out');
% This directory contains the dependency g_dfovec_1d.m
% if strcmp('nondiff',probtype)

% Initialization for adimat objects
t = 0; g_t = 1; g_x = zeros(size(x));
J = zeros(m, n);
if nprob == 8 || nprob == 9 || nprob == 13 || nprob == 16 || nprob == 17 || nprob == 18
    indvec = find(x > 0)';
    x = max(x, 0);
    for ind = indvec % Do one coordinate direction at a time:
%         [g_fvec, fvec] = g_dfovec_1d(g_t, t, ind, m, n, g_x, x, nprob);
        [g_fvec, fvec] = g_dfovec_1d(g_t, t, ind, m, n, g_x, x, nprob);
        J(:, ind) = g_fvec;
    end
    if isempty(indvec) % Still need to assign fvec
%         [~, fvec] = g_dfovec_1d(g_t, t, ind, m, n, g_x, x, nprob);
        [~, fvec] = g_dfovec_1d(g_t, t, ind, m, n, g_x, x, nprob);
    end
else
    for ind = 1:n % Do one coordinate direction at a time:
%         [g_fvec, fvec] = g_dfovec_1d(g_t, t, ind, m, n, g_x, x, nprob);
        [g_fvec, fvec] = g_dfovec_1d(g_t, t, ind, m, n, g_x, x, nprob);
        J(:, ind) = g_fvec;
    end
end
% for ind = 1:n % Do one coordinate direction at a time:
%     [g_fvec, fvec] = g_dfovec_1d_without_clears(g_t, t, ind, m, n, g_x, x, nprob);
%     J(:,ind) = g_fvec;
% end

for i = 1:m
    if abs(fvec(i)) > 1e64 || isnan(fvec(i)) % This value should match blackbox.m
        J(i, :) = zeros;
    end
end
if max(abs(fvec)) > 1e64 % This value should match blackbox.m
    fvec(abs(fvec) > 1e64) = sign(fvec(abs(fvec) > 1e64)) * 1e64;
end
if any(isnan(fvec))
    fvec(isnan(fvec)) = 1e64;
end

% nfev = nfev + 1;
% Jfev = Jfev + 1;
%
% fvecHIST(:,nfev) = fvec;
% xHIST(:,nfev) = x;
% JHIST{Jfev} = J;
end
