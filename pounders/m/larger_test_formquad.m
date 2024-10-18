% load('formquad_inputs.mat');
load('top_of_while_loop.mat');
nf = nf + 1; % Translating python indexing to matlab

bendfo_location = '../../../BenDFO';

if ~exist(bendfo_location, 'dir')
    error("These tests depend on the BenDFO repo: https://github.com/POptUS/BenDFO. Make sure BenDFO is on your path in MATLAB");
end

addpath('./tests/');
addpath([bendfo_location, '/m']);
addpath([bendfo_location, '/data']);

load dfo.dat;

nprob = dfo(11, 1);

BenDFO.nprob = nprob;
BenDFO.m = 4;
BenDFO.n = n;

hfun = @(F)F;
combinemodels = @identity_combine;

fun = @(x)calfun_wrapper_y(x, BenDFO, 'smooth');

while nf <= nfmax
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

    if printf  % Output stuff: ---------(can be removed later)------------
        IERR = zeros(1, size(Mind, 1));
        for i = 1:size(Mind, 1)
            D = (X(Mind(i), :) - X(xkin, :));
            IERR(i) = (c - Fs(Mind(i))) + D * (G + .5 * H * D');
        end
        ierror = norm(IERR ./ max(abs(Fs(Mind, :)'), 0), inf); % Interp. error
        fprintf(progstr, nf, delta, valid, np, Fs(xkin), ng, ierror);
        if printf >= 2
            for i = 1:size(Mind, 1)
                D = (X(Mind(i), :) - X(xkin, :));
                for j = 1:m
                    jerr(i, j) = (Cres(j) - F(Mind(i), j)) + D * (Gres(:, j) + .5 * Hres(:, :, j) * D');
                end
            end
            disp(jerr);
        end
    end

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
