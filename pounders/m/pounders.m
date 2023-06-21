% POUNDerS Version 0.1,    Modified 04/9/2010. Copyright 2010
% Stefan Wild and Jorge More', Argonne National Laboratory.

function [X, F, flag, xkin] = ...
    pounders(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver, hfun, combinemodels)

if ~exist('hfun', 'var')
    % Use least-squares hfun by default
    addpath('../general_h_funs/');
    hfun = @(F)sum(F.^2);
    combinemodels = @leastsquares;
end
if ~exist('spsolver', 'var')
    spsolver = 2; % Use minq5 by default
end
if ~exist('printf', 'var')
    printf = 0; % Don't print by default
end
% 0. Check inputs
[flag, X0, npmax, F0, L, U] = ...
    checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U);
if flag == -1 % Problem with the input
    X = [];
    F = [];
    return
end

% --INTERNAL PARAMETERS [won't be changed elsewhere, defaults in ( ) ]-----
maxdelta = min(.5 * min(U - L), 1e3 * delta); % [dbl] Maximum tr radius
mindelta = min(delta * 1e-13, gtol / 10); % [dbl] Min tr radius (technically 0)
gam0 = .5;      % [dbl] Parameter in (0,1) for shrinking delta  (.5)
gam1 = 2;       % [dbl] Parameter >1 for enlarging delta   (2)
eta1 = .05;     % [dbl] Parameter 2 for accepting point, 0<eta1<1 (.2)
Par(1) = sqrt(n); % [dbl] delta multiplier for checking validity
Par(2) = max(10, sqrt(n)); % [dbl] delta multiplier for all interp. points
Par(3) = 1e-3;  % [dbl] Pivot threshold for validity (1e-5)
Par(4) = .001;  % [dbl] Pivot threshold for additional points (.001)
if printf
    disp('  nf   delta    fl  np       f0           g0       ierror');
    progstr = '%4i %9.2e %2i %3i  %11.5e %12.4e %11.3e\n'; % Line-by-line
end
% -------------------------------------------------------------------------

% --INTERMEDIATE VARIABLES-------------------------------------------------
% D       [dbl] [1-by-n] Generic displacement vector
% G       [dbl] [n-by-1] Model gradient at X(xkin,:)
% H       [dbl] [n-by-n] Model Hessian at X(xkin,:)
% Hdel    [dbl] [n-by-n] Change to model Hessian at X(xkin,:)
% Lows    [dbl] [1-by-n] Vector of subproblem lower bounds
% Upps    [dbl] [1-by-n] Vector of subproblem upper bounds
% Mdir    [dbl] [n-by-n] Unit row directions to improve model/geometry
% Mind    [int] [npmax-by-1] Integer vector of model interpolation indices
% Xsp     [dbl] [1-by-n] Subproblem solution
% c       [dbl] Model value at X(xkin,:)
% mdec    [dbl] Change predicted by the model, m(nf)-m(xkin)
% nf      [int] Counter for the number of function evaluations
% ng      [dbl] Norm of (projection of) G
% np      [int] Number of model interpolation points
% rho     [dbl] Ratio of actual decrease to model decrease
% valid   [log] Flag saying if model is fully linear within Par(1)*delta
% -------------------------------------------------------------------------

if nfs == 0 % Need to do the first evaluation
    X = [X0; zeros(nfmax - 1, n)]; % Stores the point locations
    F = zeros(nfmax, m); % Stores the function values
    nf = 1;
    F0 = fun(X(nf, :));
    if length(F0) ~= m
        disp('  Error: F0 does not contain the right number of residuals');
        flag = -1;
        return
    end
    F(nf, :) = F0;
    if any(isnan(F(nf, :)))
        [X, F, flag] = prepare_outputs_before_return(X, F, nf, -3);
        return
    end
    if printf
        fprintf('%4i    Initial point  %11.5e\n', nf, sum(F(nf, :).^2));
    end
else % Have other function values around
    X = [X0(1:nfs, :); zeros(nfmax, n)]; % Stores the point locations
    F = [F0(1:nfs, :); zeros(nfmax, m)]; % Stores the function values
    nf = nfs;
    nfmax = nfmax + nfs;
end
Fs = zeros(nfmax + nfs, 1); % Stores the sum of squares of evaluated points
for i = 1:nf
    Fs(i) = hfun(F(i, :));
end

Res = zeros(size(F)); % Stores the residuals for model updates
Cres = F(xkin, :);
Hres = zeros(n, n, m);
ng = NaN; % Needed for early termination, e.g., if a model is never built
% H = zeros(n); G = zeros(n,1); c = Fs(xkin);

% ! NOTE: Currently do not move to a geometry point (including in
% the first iteration!) if it has a lower f value

while nf < nfmax
    % 1a. Compute the interpolation set.
    for i = 1:nf
        D = X(i, :) - X(xkin, :);
        Res(i, :) = F(i, :) - Cres - .5 * D * reshape(D * reshape(Hres, n, m * n), n, m);
    end
    [Mdir, np, valid, Gres, Hresdel, Mind] = ...
        formquad(X(1:nf, :), Res(1:nf, :), delta, xkin, npmax, Par, 0);
    if np < n  % Must obtain and evaluate bounded geometry points
        [Mdir, np] = bmpts(X(xkin, :), Mdir(1:n - np, :), L, U, delta, Par(3));
        for i = 1:min(n - np, nfmax - nf)
            nf = nf + 1;
            X(nf, :) = min(U, max(L, X(xkin, :) + Mdir(i, :))); % Temp safeguard
            F(nf, :) = fun(X(nf, :));
            if any(isnan(F(nf, :)))
                [X, F, flag] = prepare_outputs_before_return(X, F, nf, -3);
                return
            end
                Fs(nf) = hfun(F(nf, :));
            if printf
                fprintf('%4i   Geometry point  %11.5e\n', nf, Fs(nf));
            end
            D = Mdir(i, :);
            Res(nf, :) = F(nf, :) - Cres - .5 * D * reshape(D * reshape(Hres, n, m * n), n, m);
        end
        if nf >= nfmax
            break
        end
        [~, np, valid, Gres, Hresdel, Mind] = ...
            formquad(X(1:nf, :), Res(1:nf, :), delta, xkin, npmax, Par, 0);
        if np < n
            [X, F, flag] = prepare_outputs_before_return(X, F, nf, -5);
            return
        end
    end

    % 1b. Update the quadratic model
    Cres = F(xkin, :);
    Hres = Hres + Hresdel;
    c = Fs(xkin);
    [G, H] = combinemodels(Cres, Gres, Hres);
    ind_Lnotbinding = and(X(xkin, :) > L, G' > 0);
    ind_Unotbinding = and(X(xkin, :) < U, G' < 0);
    ng = norm(G .* (ind_Lnotbinding + ind_Unotbinding)');

    if printf >= 2  % Output stuff: ---------(can be removed later)------------
        IERR = zeros(1, size(Mind, 1));
        for i = 1:size(Mind, 1)
            D = (X(Mind(i), :) - X(xkin, :));
            IERR(i) = (c - Fs(Mind(i))) + D * (G + .5 * H * D');
        end
        for i = 1:size(Mind, 1)
            D = (X(Mind(i), :) - X(xkin, :));
            for j = 1:m
                jerr(i, j) = (Cres(j) - F(Mind(i), j)) + D * (Gres(:, j) + .5 * Hres(:, :, j) * D');
            end
        end
        ierror = norm(IERR ./ max(abs(Fs(Mind, :)'), 0), inf); % Interp. error
        fprintf(progstr, nf, delta, valid, np, Fs(xkin), ng, ierror);
    end % ------------------------------------------------------------------

    % 2. Criticality test invoked if the projected model gradient is small
    if ng < gtol
        % Check to see if the model is valid within a region of size gtol
        delta = max(gtol, max(abs(X(xkin, :))) * eps); % Safety for tiny gtols
        [Mdir, ~, valid] = ...
            formquad(X(1:nf, :), F(1:nf, :), delta, xkin, npmax, Par, 1);
        if ~valid % Make model valid in this small region
            [Mdir, np] = bmpts(X(xkin, :), Mdir, L, U, delta, Par(3));
            for i = 1:min(n - np, nfmax - nf)
                nf = nf + 1;
                X(nf, :) = min(U, max(L, X(xkin, :) + Mdir(i, :))); % Temp safeg.
                F(nf, :) = fun(X(nf, :));
                if any(isnan(F(nf, :)))
                    [X, F, flag] = prepare_outputs_before_return(X, F, nf, -3);
                    return
                end
                Fs(nf) = hfun(F(nf, :));
                if printf
                    fprintf('%4i   Critical point  %11.5e\n', nf, Fs(nf));
                end
            end
            if nf >= nfmax
                break
            end
            % Recalculate gradient based on a MFN model
            [~, ~, valid, Gres, Hres, Mind] = ...
                formquad(X(1:nf, :), F(1:nf, :), delta, xkin, npmax, Par, 0);
            [G, H] = combinemodels(Cres, Gres, Hres);
            ind_Lnotbinding = and(X(xkin, :) > L, G' > 0);
            ind_Unotbinding = and(X(xkin, :) < U, G' < 0);
            ng = norm(G .* (ind_Lnotbinding + ind_Unotbinding)');
        end
        if ng < gtol % We trust the small gradient norm and return
            [X, F, flag] = prepare_outputs_before_return(X, F, nf, 0);
            return
        end
    end

    % 3. Solve the subproblem min{G'*s+.5*s'*H*s : Lows <= s <= Upps }
    Lows = max(L - X(xkin, :), -delta);
    Upps = min(U - X(xkin, :), delta);
    if spsolver == 1 % Stefan's crappy 10line solver
        [Xsp, mdec] = bqmin(H, G, Lows, Upps);
    elseif spsolver == 2 % Arnold Neumaier's minq5
        [Xsp, mdec, minq_err] = minqsw(0, G, H, Lows', Upps', 0, zeros(n, 1));
        if minq_err < 0
            [X, F, flag] = prepare_outputs_before_return(X, F, nf, -4);
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

    % 4. Evaluate the function at the new point (provided mdec isn't zero with an invalid model)
    if (step_norm >= 0.01 * delta || valid) && ~(mdec == 0 && ~valid)

        Xsp = min(U, max(L, X(xkin, :) + Xsp));  % Temp safeguard; note Xsp is not a step anymore

        % Project if we're within machine precision
        for i = 1:n % ! This will need to be cleaned up eventually
            if U(i) - Xsp(i) < eps * abs(U(i)) && U(i) > Xsp(i) && G(i) >= 0
                Xsp(i) = U(i);
                disp('eps project!');
            elseif Xsp(i) - L(i) < eps * abs(L(i)) && L(i) < Xsp(i) && G(i) >= 0
                Xsp(i) = L(i);
                disp('eps project!');
            end
        end

        if mdec == 0 && valid && all(Xsp == X(xkin, :))
            [X, F, flag] = prepare_outputs_before_return(X, F, nf, -2);
            return
        end

        nf = nf + 1;
        X(nf, :) = Xsp;
        F(nf, :) = fun(X(nf, :));
        if any(isnan(F(nf, :)))
            [X, F, flag] = prepare_outputs_before_return(X, F, nf, -3);
            return
        end
        Fs(nf) = hfun(F(nf, :));

        if mdec ~= 0
            rho = (Fs(nf) - Fs(xkin)) / mdec;
        else % Note: this conditional only occurs when model is valid
            if Fs(nf) == Fs(xkin)
                [X, F, flag] = prepare_outputs_before_return(X, F, nf, -2);
                return
            else
                rho = inf * sign(Fs(nf) - Fs(xkin));
            end
        end

        % 4a. Update the center
        if (rho >= eta1)  || ((rho > 0) && (valid))
            %  Update model to reflect new center
            Cres = F(xkin, :);
            xkin = nf; % Change current center
        end

        % 4b. Update the trust-region radius:
        if (rho >= eta1)  &&  (step_norm > .75 * delta)
            delta = min(delta * gam1, maxdelta);
        elseif valid
            delta = max(delta * gam0, mindelta);
        end
    else % Don't evaluate f at Xsp
        rho = -1; % Force yourself to do a model-improving point
        if printf
            disp('Warning: skipping sp soln!---------');
        end
    end

    % 5. Evaluate a model-improving point if necessary
    if ~(valid) && (nf < nfmax) && (rho < eta1) % Implies xkin,delta unchanged
        % Need to check because model may be valid after Xsp evaluation
        [Mdir, np, valid] = ...
            formquad(X(1:nf, :), F(1:nf, :), delta, xkin, npmax, Par, 1);
        if ~(valid)  % ! One strategy for choosing model-improving point:
            % Update model (exists because delta & xkin unchanged)
            for i = 1:nf
                D = (X(i, :) - X(xkin, :));
                Res(i, :) = F(i, :) - Cres - .5 * D * reshape(D * reshape(Hres, n, m * n), n, m);
            end
            [~, ~, valid, Gres, Hresdel, Mind] = ...
                formquad(X(1:nf, :), Res(1:nf, :), delta, xkin, npmax, Par, 0);
            Hres = Hres + Hresdel;
            % Update for modelimp; Cres unchanged b/c xkin unchanged
            [G, H] = combinemodels(Cres, Gres, Hres);

            % Evaluate model-improving points to pick best one
            % ! May eventually want to normalize Mdir first for infty norm
            % Plus directions
            [Mdir1, np1] = bmpts(X(xkin, :), Mdir(1:n - np, :), L, U, delta, Par(3));
            for i = 1:n - np1
                D = Mdir1(i, :);
                Res(i, 1) = D * (G + .5 * H * D');
            end
            [a1, b] = min(Res(1:n - np1, 1));
            Xsp = Mdir1(b, :);
            % Minus directions
            [Mdir1, np2] = bmpts(X(xkin, :), -Mdir(1:n - np, :), L, U, delta, Par(3));
            for i = 1:n - np2
                D = Mdir1(i, :);
                Res(i, 1) = D * (G + .5 * H * D');
            end
            [a2, b] = min(Res(1:n - np2, 1));
            if a2 < a1
                Xsp = Mdir1(b, :);
            end

            nf = nf + 1;
            X(nf, :) = min(U, max(L, X(xkin, :) + Xsp)); % Temp safeguard
            F(nf, :) = fun(X(nf, :));
            if any(isnan(F(nf, :)))
                [X, F, flag] = prepare_outputs_before_return(X, F, nf, -3);
                return
            end
            Fs(nf) = hfun(F(nf, :));
            if printf
                fprintf('%4i   Model point     %11.5e\n', nf, Fs(nf));
            end
            if Fs(nf, :) < Fs(xkin, :)  % ! Eventually check suff decrease here!
                if printf
                    disp('**improvement from model point****');
                end
                %  Update model to reflect new base point
                D = (X(nf, :) - X(xkin, :));
                xkin = nf; % Change current center
                Cres = F(xkin, :);
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
