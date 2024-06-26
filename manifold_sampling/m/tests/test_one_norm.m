% To run this test, please download allcomb from the MathWorks FileExchange and
% add its location to the MATLAB path as well.
% https://www.mathworks.com/matlabcentral/fileexchange/10064-allcomb-varargin
%

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
oldpath = addpath(fullfile(here_path, '..'));
addpath(fullfile(here_path, '..', 'h_examples'));

n = 2;
m = 2;

nfmax = -1;
subprob_switch = 'linprog';
LB = -2 * ones(1, n);
UB = 2 * ones(1, n);
x0 = [-1.2, 1.0];

hfun = @one_norm;

[X, F, h, xkin, flag] = manifold_sampling_primal(hfun, @Ffun, x0, LB, UB, nfmax, subprob_switch);

path(oldpath);

function F = Ffun(x)
    % Rosenbrock function
    F = [10 * (x(2) - x(1)^2), 1 - x(1)];
end
