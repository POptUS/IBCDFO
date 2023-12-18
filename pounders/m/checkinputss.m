% checkinputss.m, Version 0.1, Modified 3/3/10
% Stefan Wild and Jorge More', Argonne National Laboratory.
%
% [flag,X0,np_max,F0,Low,Upp] = ...
%          checkinputss(fun,X0,n,np_max,nf_max,g_tol,delta,nfs,m,F0,xk_in,Low,Upp)
%
% Checks the inputs provided to pounders.
% A warning message is produced if a nonfatal input is given (and the
% input is changed accordingly).
% An error message (flag=-1) is produced if the pounders cannot continue.
%
% --INPUTS-----------------------------------------------------------------
% see inputs for pounders
% --OUTPUTS----------------------------------------------------------------
% flag  [int] = 1 if inputs pass the test
%             = 0 if a warning was produced (X0,np_max,F0,Low,Upp are changed)
%             = -1 if a fatal error was produced (pounders terminates)
%
function [flag, X0, np_max, F0, Low, Upp] = ...
    checkinputss(fun, X0, n, np_max, nf_max, g_tol, delta, nfs, m, F0, xk_in, Low, Upp)

flag = 1; % By default, everything is OK

% Verify that fun is a function handle.
if ~isa(fun, 'function_handle')
    disp('  Error: fun is not a function handle');
    flag = -1;
    return
end

% Verify X0 is the appropriate size
[nfs2, n2] = size(X0);
if n ~= n2
    % Attempt to transpose:
    if n2 == 1 && nfs2 == n
        X0 = X0';
        disp('  Warning: X0 is n-by-1 column vector, using row vector X0''');
        flag = 0;
    else
        disp('  Error: size(X0,2)~=n');
        flag = -1;
        return
    end
end

% Check max number of interpolation points
if np_max < n + 1 || np_max > .5 * (n + 1) * (n + 2)
    np_max = max(n + 1, min(np_max, .5 * (n + 1) * (n + 2)));
    disp(['  Warning: np_max not in [n+1,.5(n+1)(n+2)] using ', num2str(np_max)]);
    flag = 0;
end

% Check standard positive quantities
if nf_max < 1
    disp('  Error: max number of evaluations is less than 1');
    flag = -1;
    return
elseif g_tol <= 0
    disp('  Error: g_tol must be positive');
    flag = -1;
    return
elseif delta <= 0
    disp('  Error: delta must be positive');
    flag = -1;
    return
end

% Check number of starting points
if nfs2 ~= max(nfs, 1)
    disp('  Warning: number of starting f values nfs does not match input X0');
    flag = 0;
end

% Check matrix of initial function values
[nfs2, m2] = size(F0);
if nfs2 < nfs
    disp('  Error: fewer than nfs function values in F0');
    flag = -1;
    return
elseif nfs > 1 && m ~= m2
    disp('  Error: F0 does not contain the right number of residuals');
    flag = -1;
    return
elseif nfs2 > nfs
    disp('  Warning: number of starting f values nfs does not match input F0');
    flag = 0;
end
if any(any(isnan(F0)))
    disp("  Error: F0 contains a NaN.");
    flag = -1;
    return
end

% Check starting point
if xk_in > max(nfs, 1) || xk_in < 1 || mod(xk_in, 1) ~= 0
    disp('  Error: starting point index not an integer between 1 and nfs');
    flag = -1;
    return
end

% Check the bounds
[nfs2, n2] = size(Low);
[nfs3, n3] = size(Upp);
if n3 ~= n2 || nfs2 ~= nfs3
    disp('  Error: bound dimensions inconsistent');
    flag = -1;
    return
elseif n2 ~= n && (n2 == 1 && nfs2 == n) % Attempt to transpose
    Low = Low';
    Upp = Upp';
    disp('  Warning: bounds are n-by-1, using transposed row vectors');
    flag = 0;
elseif n2 ~= n || nfs2 ~= 1
    disp('  Error: bounds are not 1-by-n vectors');
    flag = -1;
    return
end

if min(Upp - Low) <= 0
    disp('  Error: must have Upp>Low');
    flag = -1;
    return
end
if min(min(X0(xk_in, :) - Low), min(Upp - X0(xk_in, :))) < 0
    disp('  Error: starting point outside of bounds (Low, Upp)');
    flag = -1;
    return
end
