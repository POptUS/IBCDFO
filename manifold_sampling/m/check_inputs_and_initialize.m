function [n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, trust_rho, xkin, Hres] = check_inputs_and_initialize(x0, F0, nfmax)

    global h_activity_tol

    n = length(x0);
    p = length(F0);
    delta = 0.1;
    printf = 1;

    h_activity_tol = min(1e-8, delta);

    % Internal parameters/tolerances for formquad
    fq_pars.Par(1) = sqrt(n); % [dbl] delta multiplier for checking validity
    fq_pars.Par(2) = max(10, sqrt(n)); % [dbl] delta multiplier for all interp. points
    fq_pars.Par(3) = 1e-3;  % [dbl] Pivot threshold for validity (1e-5)
    fq_pars.Par(4) = .001;  % [dbl] Pivot threshold for additional points (.001)
    % fq_pars.npmax = (n + 1) * (n + 2) / 2;     % [int] number of points in model building
    fq_pars.npmax = 2 * n + 1;     % [int] number of points in model building

    % Internal parameters/tolerances for manifold sampling
    tol.maxdelta = 1e8;
    tol.mindelta = 1e-13;
    tol.gtol = 1e-13;
    tol.norm_g_change =  1e-3;  % Tolerance for the change of norm(g_k)
    tol.kappa_d = 1e-4;  % [dbl] > 0 fraction of Cauchy decrease
    tol.eta1 = 0.01;     % [dbl] in (0, 1) for declaring rho sufficiently large (a successful iteration)
    tol.eta2 = 1e4;      % [dbl] in (1/kappa_mh, inf) for deciding if the model gradient is sufficiently large
    tol.eta3 = 0.5;
    tol.gamma_dec = 0.5; % [dbl] in (0, 1) for shrinking delta
    tol.gamma_inc = 2;   % [dbl] >= 1 for increasing delta
    tol.hfun_test_mode = 1;   % [bool] Run some checks every time the hfun is called to see if it is implemented correctly.
    % kappa_mh = 0;    % [dbl] > 0 that bounds the component model Hessians

    tol.gentype = 2;

    assert(nfmax >= n + 1, "nfmax is less than n+1, exiting");

    X = [x0; zeros(nfmax - 1, n)]; % Stores the point locations
    F = [F0; zeros(nfmax - 1, p)];         % Stores the simulation values
    h = zeros(nfmax, 1);         % Stores the function values
    Hash = cell(nfmax, 1);       % Stores the hashes

    nf = 1;
    trust_rho = 1;

    xkin = 1;
    Hres = zeros(n, n, p);
end
