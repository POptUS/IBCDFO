% Adjust these:
n = 4; % Number of parameters to be optimized
X_0 = rand(1, n); % starting parameters for the optimizer
nf_max = 100; % Max number of evaluations to be used by optimizer
Low = -1 * ones(1, n); % 1-by-n Vector of lower bounds
Upp = ones(1, n); % 1-by-n Vector of upper bounds
printf = true;

% Not as important to adjust:
hfun = @emittance_h; % You need to define or handle external functions differently in MATLAB
combinemodels = @emittance_combine; % Same here for external functions
m = 3; % The number of outputs from the beamline simulation. Should be 3 for emittance minimization
g_tol = 1e-8; % Stopping tolerance
delta_0 = 0.1; % Initial trust-region radius
F_0 = zeros(1, m); % Initial evaluations (parameters with completed simulations)
F_0(1, :) = call_beamline_simulation(X_0);
nfs = 1; % Number of initial evaluations
xk_in = 1; % Index in F_0 for starting the optimization (usually the point with minimal emittance)
spsolver = 2;

Prior.xk_in = xk_in;
Prior.X_0 = X_0;
Prior.F_init = F_0;
Prior.nfs = nfs;

Options.hfun = hfun;
Options.combinemodels = combinemodels;
Options.spsolver = spsolver;
Options.printf = printf;

[Xout, Fouf, hFout, flag, xk_best] = pounders(@call_beamline_simulation, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior, Options);

assert(flag >= 0, 'pounders crashed');

assert(hFout(xk_best) == min(hFout), 'The minimum emittance is not at xk_best');

% Define the call_beamline_simulation function
function out = call_beamline_simulation(x)
    % In here, put your call to your simulation that takes in the
    % parameters x and returns the three values used in the calculation of
    % emittance.
    % out = put_your_sim_call_here(x);

    out = x(1:3); % This is not doing any beamline simulation!

    assert(length(out) == 3, 'Incorrect output dimension');
end
