function [n, delta, printf, fq_pars, tol, X, F, hF, Hash, nf, successful, xk_in, Hres] = check_inputs_and_initialize(x0, F0, nf_max, Options)
    % Validates the inputs shared by |manifold_sampling| and |goombah|.
    % Fills in defaults for any missing fields of Options.
    % Makes/allocates variables used by the solvers.
    %
    % :param Options: [struct] (optional) Solver options. Any field that is
    %     missing takes the default given below:
    %
    %     * **printf**         [int] 0 for no output, 1 for standard output, 2 for additional output in |goombah|, including each GAMS run. Level 2 has no effect in |manifold_sampling| (1)
    %     * **delta_0**        [dbl] Initial trust-region radius (0.1)
    %     * **delta_max**      [dbl] Maximum trust-region radius (1e8)
    %     * **delta_min**      [dbl] Minimum trust-region radius (1e-13)
    %     * **g_tol**          [dbl] Tolerance on the stationary measure (1e-13)
    %     * **eta_1**          [dbl] In (0, 1); rho at or above this is a successful iteration (0.01)
    %     * **eta_3**          [dbl] rho above this allows the radius to grow (0.5)
    %     * **gamma_dec**      [dbl] In (0, 1); factor for shrinking delta (0.5)
    %     * **gamma_inc**      [dbl] >= 1; factor for increasing delta (2)
    %     * **hfun_test_mode** [bool] Check the hfun implementation on every call (1)
    %     * **h_activity_tol** [dbl] Tolerance for declaring an hfun manifold active (min(1e-8, delta_0))

    global h_activity_tol

    if nargin < 4 || isempty(Options)
        Options = struct();
    end
    if ~isstruct(Options)
        error("Options must be a struct");
    end

    if ~isfield(Options, 'printf')
        Options.printf = 1; % [int] 0: silent, 1: standard output, 2: additional output in goombah (1)
    end
    if ~isfield(Options, 'delta_0')
        Options.delta_0 = 0.1; % [dbl] Initial trust-region radius (0.1)
    end
    if ~isfield(Options, 'delta_max')
        Options.delta_max = 1e8; % [dbl] Maximum trust-region radius (1e8)
    end
    if ~isfield(Options, 'delta_min')
        Options.delta_min = 1e-13; % [dbl] Minimum trust-region radius (1e-13)
    end
    if ~isfield(Options, 'g_tol')
        Options.g_tol = 1e-13; % [dbl] Tolerance on the stationary measure (1e-13)
    end
    if ~isfield(Options, 'eta_1')
        Options.eta_1 = 0.01; % [dbl] in (0, 1) for declaring rho sufficiently large (0.01)
    end
    if ~isfield(Options, 'eta_3')
        Options.eta_3 = 0.5; % [dbl] rho above which delta may grow (0.5)
    end
    if ~isfield(Options, 'gamma_dec')
        Options.gamma_dec = 0.5; % [dbl] in (0, 1) for shrinking delta (0.5)
    end
    if ~isfield(Options, 'gamma_inc')
        Options.gamma_inc = 2; % [dbl] >= 1 for increasing delta (2)
    end
    if ~isfield(Options, 'hfun_test_mode')
        Options.hfun_test_mode = 1; % [bool] Check the hfun implementation on every call (1)
    end
    if ~isfield(Options, 'h_activity_tol')
        Options.h_activity_tol = min(1e-8, Options.delta_0); % [dbl] hfun activity tolerance (min(1e-8, delta_0))
    end

    n = length(x0);
    p = length(F0);
    delta = Options.delta_0;
    printf = Options.printf;

    h_activity_tol = Options.h_activity_tol;

    % Internal parameters/tolerances for formquad
    fq_pars.Par(1) = sqrt(n); % [dbl] delta multiplier for checking validity
    fq_pars.Par(2) = max(10, sqrt(n)); % [dbl] delta multiplier for all interp. points
    fq_pars.Par(3) = 1e-3;  % [dbl] Pivot threshold for validity (1e-5)
    fq_pars.Par(4) = .001;  % [dbl] Pivot threshold for additional points (.001)
    fq_pars.Par(5) = 0;  % [log] Flag to find affine points in forward order (0)
    fq_pars.npmax = 2 * n + 1;     % [int] number of points in model building

    % Internal parameters/tolerances for manifold sampling
    tol.maxdelta = Options.delta_max;
    tol.mindelta = Options.delta_min;
    tol.gtol = Options.g_tol;
    tol.norm_g_change =  1e-3;  % Tolerance for the change of norm(g_k)
    tol.kappa_d = 1e-4;  % [dbl] > 0 fraction of Cauchy decrease
    tol.eta1 = Options.eta_1; % [dbl] in (0, 1) for declaring rho sufficiently large (a successful iteration)
    tol.eta2 = 1e4;      % [dbl] in (1/kappa_mh, inf) for deciding if the model gradient is sufficiently large
    tol.eta3 = Options.eta_3;
    tol.gamma_dec = Options.gamma_dec; % [dbl] in (0, 1) for shrinking delta
    tol.gamma_inc = Options.gamma_inc; % [dbl] >= 1 for increasing delta
    tol.hfun_test_mode = Options.hfun_test_mode; % [bool] Run some checks every time the hfun is called to see if it is implemented correctly.
    % kappa_mh = 0;    % [dbl] > 0 that bounds the component model Hessians

    assert(nf_max >= n + 1, "nf_max is less than n+1, exiting");

    X = [x0; zeros(nf_max - 1, n)]; % Stores the point locations
    F = [F0; zeros(nf_max - 1, p)];         % Stores the simulation values
    hF = zeros(nf_max, 1);        % Stores the function values
    Hash = cell(nf_max, 1);       % Stores the hashes

    nf = 1;
    successful = 1;

    xk_in = 1;
    Hres = zeros(n, n, p);
end
