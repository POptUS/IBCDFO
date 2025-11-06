% POUNDerS Version 0.1,    Modified 04/9/2010. Copyright 2010
% Stefan Wild and Jorge More', Argonne National Laboratory.

function [X, F, hF, flag, xk_in] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior, Options, Model)

% Check for missing arguments and initialize if necessary
if nargin < 12 || isempty(Model)
    Model = struct();
end
if nargin < 11 || isempty(Options)
    Options = struct();
end
if nargin < 10 || isempty(Prior)
    Prior = struct();
    Prior.nfs = 0;
    Prior.X_init = [];
    Prior.F_init = [];
    Prior.xk_in = 1;
end

if ~isstruct(Options)
    error("Options must be a struct");
end
if ~isstruct(Prior)
    error("Prior must be a struct");
end
if ~isstruct(Model)
    error("Model must be a struct");
end

% --INTERNAL PARAMETERS [won't be changed elsewhere, defaults in ( ) ]-----
if ~isfield(Options, 'delta_max')
    Options.delta_max = min(.5 * min(Upp - Low), 1e3 * delta_0); % [dbl] Maximum tr radius
end
if ~isfield(Options, 'delta_min')
    Options.delta_min = min(delta_0 * 1e-13, g_tol / 10); % [dbl] Min tr radius (technically 0)
end
if ~isfield(Options, 'gamma_dec')
    Options.gamma_dec = .5; % [dbl] Parameter in (0,1) for shrinking delta  (.5)
end
if ~isfield(Options, 'gamma_inc')
    Options.gamma_inc = 2;  % [dbl] Parameter (>=1) for enlarging delta   (2)
end
if ~isfield(Options, 'eta_1')
    Options.eta_1 = .05;     % [dbl] Parameter for accepting point, 0<eta_1<1 (.2)
end
if ~isfield(Options, 'delta_inact')
    Options.delta_inact = 0.75;
end
if ~isfield(Options, 'spsolver')
    Options.spsolver = 2;
end

if isfield(Options, 'hfun')
    hfun = Options.hfun;
    combinemodels = Options.combinemodels;
else
    % Use least-squares hfun by default
    [here_path, ~, ~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(here_path, 'general_h_funs'));
    hfun = @(F)sum(F.^2);
    combinemodels = @leastsquares;
end
if ~isfield(Options, 'spsolver')
    Options.spsolver = 2; % Use minq5 by default
end
if ~isfield(Options, 'printf')
    Options.printf = 0; % Don't print by default
end

if ~isfield(Model, 'np_max')
    Model.np_max = 2 * n + 1;
end
if ~isfield(Model, 'Par')
    Model.Par = zeros(1, 5);

    Model.Par(1) = sqrt(n); % [dbl] delta multiplier for checking validity
    Model.Par(2) = max(10, sqrt(n)); % [dbl] delta multiplier for all interp. points
    Model.Par(3) = 1e-3;  % [dbl] Pivot threshold for validity (1e-5)
    Model.Par(4) = .001;  % [dbl] Pivot threshold for additional points (.001)
    Model.Par(5) = 0;     % [log] Flag to find affine points in forward order (0)
end

nfs = Prior.nfs;

delta = delta_0;
spsolver = Options.spsolver;
delta_max = Options.delta_max;
delta_min = Options.delta_min;
gamma_dec = Options.gamma_dec;
gamma_inc = Options.gamma_inc;
eta_1 = Options.eta_1;
printf = Options.printf;
delta_inact = Options.delta_inact;

if spsolver == 2 % Arnold Neumaier's minq5
    [here_path, ~, ~] = fileparts(mfilename('fullpath'));
    minq_path = fullfile(here_path, '..', '..', 'minq');
    addpath(fullfile(minq_path, 'm', 'minq5'));
elseif spsolver == 3 % Arnold Neumaier's minq8
    [here_path, ~, ~] = fileparts(mfilename('fullpath'));
    minq_path = fullfile(here_path, '..', '..', 'minq');
    addpath(fullfile(minq_path, 'm', 'minq8'));
end

% 0. Check inputs
[flag, X_0, np_max, F_0, Low, Upp, xk_in] = ...
    checkinputss(Ffun, X_0, n, Model.np_max, nf_max, g_tol, delta, nfs, m, Prior.F_init, Prior.xk_in, Low, Upp);
if flag == -1 % Problem with the input
    X = [];
    F = [];
    hF = [];
    return
end

if printf
    disp('  nf   delta    fl  np       f0           g0       ierror');
    progstr = '%4i %9.2e %2i %3i  %11.5e %12.4e %11.3e\n'; % Line-by-line
end
% -------------------------------------------------------------------------

% --INTERMEDIATE VARIABLES-------------------------------------------------
% D       [dbl] [1-by-n] Generic displacement vector
% G       [dbl] [n-by-1] Model gradient at X(xk_in,:)
% H       [dbl] [n-by-n] Model Hessian at X(xk_in,:)
% Hdel    [dbl] [n-by-n] Change to model Hessian at X(xk_in,:)
% Lows    [dbl] [1-by-n] Vector of subproblem lower bounds
% Upps    [dbl] [1-by-n] Vector of subproblem upper bounds
% Mdir    [dbl] [n-by-n] Unit row directions to improve model/geometry
% Mind    [int] [np_max-by-1] Integer vector of model interpolation indices
% Xsp     [dbl] [1-by-n] Subproblem solution
% c       [dbl] Model value at X(xk_in,:)
% mdec    [dbl] Change predicted by the model, m(nf)-m(xk_in)
% nf      [int] Counter for the number of function evaluations
% ng      [dbl] Norm of (projection of) G
% np      [int] Number of model interpolation points
% rho     [dbl] Ratio of actual decrease to model decrease
% valid   [log] Flag saying if model is fully linear within Par(1)*delta
% -------------------------------------------------------------------------

if nfs == 0 % Need to do the first evaluation
    X = [X_0; zeros(nf_max - 1, n)]; % Stores the point locations
    F = zeros(nf_max, m); % Stores the function values
    hF = zeros(nf_max, 1); % Stores the sum of squares of evaluated points
    nf = 1;
    F_0 = Ffun(X(nf, :));
    if length(F_0) ~= m
        [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -1);
        return
    end
    F(nf, :) = F_0;
    if any(isnan(F(nf, :)))
        [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -3);
        return
    end
    if printf
        fprintf('%4i    Initial point  %11.5e\n', nf, hfun(F(nf, :)));
    end
else % Have other function values around
    X = [X_0(1:nfs, :); zeros(nf_max, n)]; % Stores the point locations
    F = [F_0(1:nfs, :); zeros(nf_max, m)]; % Stores the function values
    hF = zeros(nf_max + nfs, 1); % Stores the sum of squares of evaluated points
    nf = nfs;
    nf_max = nf_max + nfs;
end
for i = 1:nf
    hF(i) = hfun(F(i, :));
end

Res = zeros(size(F)); % Stores the residuals for model updates
Cres = F(xk_in, :);
Hres = zeros(n, n, m);
ng = NaN; % Needed for early termination, e.g., if a model is never built
% H = zeros(n); G = zeros(n,1); c = hF(xk_in);

% ! NOTE: Currently do not move to a geometry point (including in
% the first iteration!) if it has a lower f value

while nf < nf_max
    check_dims_and_Hres(n, m, Hres); % GH Actions debug prints

    % 1a. Compute the interpolation set.
    for i = 1:nf
        D = X(i, :) - X(xk_in, :);
        Res(i, :) = F(i, :) - Cres - .5 * D * reshape(D * reshape(Hres, n, m * n), n, m);
    end
    [Mdir, np, valid, Gres, Hresdel, Mind] = ...
        formquad(X(1:nf, :), Res(1:nf, :), delta, xk_in, np_max, Model.Par, 0);
    if np < n  % Must obtain and evaluate bounded geometry points
        [Mdir, np] = bmpts(X(xk_in, :), Mdir(1:n - np, :), Low, Upp, delta, Model.Par(3));
        for i = 1:min(n - np, nf_max - nf)
            nf = nf + 1;
            X(nf, :) = min(Upp, max(Low, X(xk_in, :) + Mdir(i, :))); % Temp safeguard
            F(nf, :) = Ffun(X(nf, :));
            if any(isnan(F(nf, :)))
                [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -3);
                return
            end
                hF(nf) = hfun(F(nf, :));
            if printf
                fprintf('%4i   Geometry point  %11.5e\n', nf, hF(nf));
            end
            D = Mdir(i, :);
            Res(nf, :) = F(nf, :) - Cres - .5 * D * reshape(D * reshape(Hres, n, m * n), n, m);
        end
        if nf >= nf_max
            break
        end
        [~, np, valid, Gres, Hresdel, Mind] = ...
            formquad(X(1:nf, :), Res(1:nf, :), delta, xk_in, np_max, Model.Par, 0);
        if np < n
            [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -5);
            return
        end
    end

    % 1b. Update the quadratic model
    Cres = F(xk_in, :);
    Hres = Hres + Hresdel;
    c = hF(xk_in);
    [G, H] = combinemodels(Cres, Gres, Hres);
    ind_Lnotbinding = and(X(xk_in, :) > Low, G' > 0);
    ind_Unotbinding = and(X(xk_in, :) < Upp, G' < 0);
    ng = norm(G .* (ind_Lnotbinding + ind_Unotbinding)');

    if printf  % Output stuff: ---------(can be removed later)------------
        IERR = zeros(1, size(Mind, 1));
        for i = 1:size(Mind, 1)
            D = (X(Mind(i), :) - X(xk_in, :));
            IERR(i) = (c - hF(Mind(i))) + D * (G + .5 * H * D');
        end
        ierror = norm(IERR ./ max(abs(hF(Mind, :)'), 0), inf); % Interp. error
        fprintf(progstr, nf, delta, valid, np, hF(xk_in), ng, ierror);
        if printf >= 2
            for i = 1:size(Mind, 1)
                D = (X(Mind(i), :) - X(xk_in, :));
                for j = 1:m
                    jerr(i, j) = (Cres(j) - F(Mind(i), j)) + D * (Gres(:, j) + .5 * Hres(:, :, j) * D');
                end
            end
            disp(jerr);
        end
    end

    % 2. Criticality test invoked if the projected model gradient is small
    if ng < g_tol
        % Check to see if the model is valid within a region of size g_tol
        delta = max(g_tol, max(abs(X(xk_in, :))) * eps); % Safety for tiny g_tol values
        [Mdir, ~, valid] = ...
            formquad(X(1:nf, :), F(1:nf, :), delta, xk_in, np_max, Model.Par, 1);
        if ~valid % Make model valid in this small region
            [Mdir, np] = bmpts(X(xk_in, :), Mdir, Low, Upp, delta, Model.Par(3));
            for i = 1:min(n - np, nf_max - nf)
                nf = nf + 1;
                X(nf, :) = min(Upp, max(Low, X(xk_in, :) + Mdir(i, :))); % Temp safeg.
                F(nf, :) = Ffun(X(nf, :));
                if any(isnan(F(nf, :)))
                    [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -3);
                    return
                end
                hF(nf) = hfun(F(nf, :));
                if printf
                    fprintf('%4i   Critical point  %11.5e\n', nf, hF(nf));
                end
            end
            if nf >= nf_max
                break
            end
            % Recalculate gradient based on a MFN model
            [~, ~, valid, Gres, Hres, Mind] = ...
                formquad(X(1:nf, :), F(1:nf, :), delta, xk_in, np_max, Model.Par, 0);
            [G, H] = combinemodels(Cres, Gres, Hres);
            ind_Lnotbinding = and(X(xk_in, :) > Low, G' > 0);
            ind_Unotbinding = and(X(xk_in, :) < Upp, G' < 0);
            ng = norm(G .* (ind_Lnotbinding + ind_Unotbinding)');
        end
        if ng < g_tol % We trust the small gradient norm and return
            [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, 0);
            return
        end
    end

    % 3. Solve the subproblem min{G'*s+.5*s'*H*s : Lows <= s <= Upps }
    Lows = max(Low - X(xk_in, :), -delta);
    Upps = min(Upp - X(xk_in, :), delta);
    if spsolver == 1 % Stefan's crappy 10line solver
        [Xsp, mdec] = bqmin(H, G, Lows, Upps);
    elseif spsolver == 2 % Arnold Neumaier's minq5
        [Xsp, mdec, minq_err] = minqsw(0, G, H, Lows', Upps', 0, zeros(n, 1));
        if minq_err < 0
            [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -4);
            return
        end

    elseif spsolver == 3 % Arnold Neumaier's minq8

        data.gam = 0;
        data.c = G;
        data.b = zeros(n, 1);
        [tmp1, tmp2] = ldl(H);
        data.D = diag(tmp2);
        data.A = tmp1';

        [Xsp, mdec] = minq8(data, Lows', Upps', zeros(n, 1), 10 * n);
    end
    Xsp = Xsp'; % Solvers currently work with column vectors
    step_norm = norm(Xsp, inf);

    % 4. Evaluate the function at the new point (provided the model is
    % valid, or the step is sufficiently large and mdec isn't zero)
    if valid || (step_norm >= 0.01 * delta && mdec ~= 0)

        Xsp = min(Upp, max(Low, X(xk_in, :) + Xsp));  % Temp safeguard; note Xsp is not a step anymore

        % Project if we're within machine precision
        for i = 1:n % ! This will need to be cleaned up eventually
            if Upp(i) - Xsp(i) < eps * abs(Upp(i)) && Upp(i) > Xsp(i) && G(i) >= 0
                Xsp(i) = Upp(i);
                disp('eps project!');
            elseif Xsp(i) - Low(i) < eps * abs(Low(i)) && Low(i) < Xsp(i) && G(i) >= 0
                Xsp(i) = Low(i);
                disp('eps project!');
            end
        end

        if mdec == 0 && valid && all(Xsp == X(xk_in, :)) && delta < sqrt(eps)
            [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -2);
            return
        end

        nf = nf + 1;
        X(nf, :) = Xsp;
        if all(Xsp == X(xk_in, :))
            % We don't want to do the expensive F eval if Xsp is already in X
            F(nf, :) = F(xk_in, :);
        else
            F(nf, :) = Ffun(X(nf, :));
        end

        if any(isnan(F(nf, :)))
            [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -3);
            return
        end
        hF(nf) = hfun(F(nf, :));

        if mdec ~= 0
            rho = (hF(nf) - hF(xk_in)) / mdec;
        else % Note: this conditional only occurs when model is valid
            if hF(nf) == hF(xk_in)
                if delta < sqrt(eps)
                    [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -2);
                    return
                else
                    rho = -inf;
                end
            else
                rho = inf * sign(hF(nf) - hF(xk_in));
            end
        end

        % 4a. Update the center
        if (rho >= eta_1)  || ((rho > 0) && (valid))
            %  Update model to reflect new center
            Cres = F(xk_in, :);
            xk_in = nf; % Change current center
        end

        % 4b. Update the trust-region radius:
        if (rho >= eta_1)  &&  (step_norm > delta_inact * delta)
            delta = min(delta * gamma_inc, delta_max);
        elseif valid
            delta = delta * gamma_dec;
            if delta <= delta_min
                [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -6);
                return
            end
        end
    else % Don't evaluate f at Xsp
        rho = -1; % Force yourself to do a model-improving point
        if printf
            disp('Warning: skipping sp soln!---------');
        end
    end

    % 5. Evaluate a model-improving point if necessary
    if ~(valid) && (nf < nf_max) && (rho < eta_1) % Implies xk_in & delta unchanged
        % Need to check because model may be valid after Xsp evaluation
        [Mdir, np, valid] = ...
            formquad(X(1:nf, :), F(1:nf, :), delta, xk_in, np_max, Model.Par, 1);
        if ~(valid)  % ! One strategy for choosing model-improving point:
            % Update model (exists because delta & xk_in unchanged)
            for i = 1:nf
                D = (X(i, :) - X(xk_in, :));
                Res(i, :) = F(i, :) - Cres - .5 * D * reshape(D * reshape(Hres, n, m * n), n, m);
            end
            [~, ~, valid, Gres, Hresdel, Mind] = ...
                formquad(X(1:nf, :), Res(1:nf, :), delta, xk_in, np_max, Model.Par, 0);
            if length(Mind) < n + 1
                % This is almost never triggered but is a safeguard for
                % pathological cases where one needs to recover from
                % unusual conditioning of recent interpolation sets
                Model.Par(5) = 1;
                [~, ~, valid, Gres, Hresdel, Mind] = formquad(X(1:nf, :), Res(1:nf, :), delta, xk_in, np_max, Par, 0);
                Model.Par(5) = 0;
            end
            Hres = Hres + Hresdel;
            % Update for modelimp; Cres unchanged b/c xk_in unchanged
            [G, H] = combinemodels(Cres, Gres, Hres);

            % Evaluate model-improving points to pick best one
            % ! May eventually want to normalize Mdir first for infty norm
            % Plus directions
            [Mdir1, np1] = bmpts(X(xk_in, :), Mdir(1:n - np, :), Low, Upp, delta, Model.Par(3));
            for i = 1:n - np1
                D = Mdir1(i, :);
                Res(i, 1) = D * (G + .5 * H * D');
            end
            [a1, b] = min(Res(1:n - np1, 1));
            Xsp = Mdir1(b, :);
            % Minus directions
            [Mdir1, np2] = bmpts(X(xk_in, :), -Mdir(1:n - np, :), Low, Upp, delta, Model.Par(3));
            for i = 1:n - np2
                D = Mdir1(i, :);
                Res(i, 1) = D * (G + .5 * H * D');
            end
            [a2, b] = min(Res(1:n - np2, 1));
            if a2 < a1
                Xsp = Mdir1(b, :);
            end

            nf = nf + 1;
            X(nf, :) = min(Upp, max(Low, X(xk_in, :) + Xsp)); % Temp safeguard
            F(nf, :) = Ffun(X(nf, :));
            if any(isnan(F(nf, :)))
                [X, F, hF, flag] = prepare_outputs_before_return(X, F, hF, nf, -3);
                return
            end
            hF(nf) = hfun(F(nf, :));
            if printf
                fprintf('%4i   Model point     %11.5e\n', nf, hF(nf));
            end
            if hF(nf, :) < hF(xk_in, :)  % ! Eventually check suff decrease here!
                if printf
                    disp('**improvement from model point****');
                end
                %  Update model to reflect new base point
                D = (X(nf, :) - X(xk_in, :));
                xk_in = nf; % Change current center
                Cres = F(xk_in, :);
                % Don't actually use:
                for j = 1:m
                    Gres(:, j) = Gres(:, j) + Hres(:, :, j) * D';
                end
            end
        end
    end
end
if printf
    disp('Number of function evals exceeded');
end
flag = ng;
end

function check_dims_and_Hres(n, m, Hres)
    % CHECK_DIMS_AND_HRES  Verify that n, m, and m*n are positive integer scalars and that Hres has expected shape [n n m].
    %
    % Usage:
    %   check_dims_and_Hres(n, m, Hres)
    %
    % Prints diagnostic info and throws an error if any condition fails.

    % --- Check n ---
    if ~(isscalar(n) && isfinite(n) && n == round(n) && n > 0)
        error('Invalid n: class=%s, value=%g (must be positive finite integer scalar).', class(n), n);
    end

    % --- Check m ---
    if ~(isscalar(m) && isfinite(m) && m == round(m) && m > 0)
        error('Invalid m: class=%s, value=%g (must be positive finite integer scalar).', class(m), m);
    end

    % --- Check m*n ---
    mn = m * n;
    if ~(isfinite(mn) && mn == round(mn) && mn > 0)
        error('Invalid m*n: m=%g, n=%g, m*n=%g (must be positive integer).', m, n, mn);
    end

    % --- Check Hres shape ---
    szH = size(Hres);
    % Pad size vector to length 3 in case m == 1 (MATLAB drops trailing singleton dims)
    if numel(szH) < 3
        szH(end + 1:3) = 1;
    end

    expected = [n, n, m];
    if any(szH(1:3) ~= expected)
        error('Hres has wrong shape. Expected [%d %d %d], got %s.', n, n, m, mat2str(szH));
    end
end
