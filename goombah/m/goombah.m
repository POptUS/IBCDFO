% This code solves the problem
%     minimize h(F(x))
% where x is an [n by 1] vector, F is a blackbox function mapping from R^n to
% R^p, and h is a nonsmooth function mapping from R^p to R.
%
%
% Inputs:
%  hfun:    [func handle] Evaluates h, returning the [scalar] function
%                         value and [k x m] subgradients for all k limiting
%                         gradients at the point given.
%  Ffun:    [func handle] Evaluates F, the black box simulation, returning
%                         a [1 x m] vector.
%  nfmax:   [int]         Maximum number of function evaluations.
%  x0:      [1 x n dbl]   Starting point.
%  LB:      [1 x n dbl]   Lower bounds.
%  UB:      [1 x n dbl]   Upper bounds.
%  GAMS_options:
%  subprob_switch:
%
% Outputs:
%   X:      [nfmax x n]   Points evaluated
%   F:      [nfmax x p]   Their simulation values
%   h:      [nfmax x 1]   The values h(F(x))
%   xkin:   [int]         Current trust region center

function [X, F, h, xkin] = goombah(hfun, Ffun, nfmax, x0, LB, UB, GAMS_options, subprob_switch)

    % Deduce p from evaluating Ffun at x0
    try
        F0 = Ffun(x0);
    catch
        warning('Problem using Ffun. Exiting');
        X = [];
        F = [];
        h = [];
        xkin = [];
        flag = -1;
        return
    end

    [n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, successful, xkin, Hres] = check_inputs_and_initialize(x0, F0, nfmax);

    [h(nf), ~, hashes_at_nf] = hfun(F(nf, :));
    Hash(nf, 1:length(hashes_at_nf)) = hashes_at_nf;

    H_mm = zeros(n);
    beta_exp = 1.0;

    while nf < nfmax && delta > tol.mindelta
        % ================================
        % Build p component models
        [Gres, Hres, X, F, h, nf, Hash] = build_p_models(nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, LB, UB);

        if isempty(Gres)
            disp(['Empty Gres. Delta = ' num2str(delta)]);
            X = X(1:nf, :);
            F = F(1:nf, :);
            h = h(1:nf, :);
            return
        end

        Low = max(LB - X(xkin, :), -delta);
        Upp = min(UB - X(xkin, :), delta);

        [sk, pred_dec] = save_quadratics_call_GAMS(Hres, Gres, F(xkin, :), Low, Upp, X(xkin, :), X(xkin, :), h(xkin), GAMS_options, hfun);
        if pred_dec == 0
            rho_k = -inf;
        else
            % Evaluate F
            [nf, X, F, h, Hash, hashes_at_nf] = ...
                    call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X(xkin, :) + sk, tol, LB, UB, 1);
            rho_k = (h(xkin) - h(nf)) / min(1.0, delta^(1.0 + beta_exp));
        end

        if rho_k > tol.eta1 % Success with h of M step
            if norm(X(xkin, :) - X(nf, :), 'inf') >= 0.9 * delta
                delta = delta * tol.gamma_inc;
            end
            xkin = nf;
        else
            %% Need to do one primal MS loop

            %% Build set of activities Act_Z_k and gradients D_k
            [D_k, Act_Z_k, f_bar] = choose_generator_set(X, Hash, 2, xkin, nf, delta, F, hfun);

            while nf < nfmax % start MS loop
                % Construct G_k and beta
                G_k = Gres * D_k;
                beta = max(0, f_bar' - h(xkin));

                H_k = zeros(size(G_k, 2), n + 1, n + 1);
                for i = 1:size(G_k, 2) % would like to vectorize this tensor product ...
                    for j = 1:size(Hres, 3) % p
                        H_k(i, 2:end, 2:end) = squeeze(H_k(i, 2:end, 2:end)) + D_k(j, i) * Hres(:, :, j);
                    end
                end

                % Find a candidate s_k by solving QP
                Low = max(LB - X(xkin, :), -delta);
                Upp = min(UB - X(xkin, :), delta);

                [s_k, tau_k] = minimize_affine_envelope(h(xkin), f_bar, beta, G_k, H_mm, delta, Low, Upp, H_k, subprob_switch);

                % Compute stationary measure chi_k
                Low = max(LB - X(xkin, :), -1.0);
                Upp = min(UB - X(xkin, :), 1.0);

                [~, ~, chi_k] = minimize_affine_envelope(h(xkin), f_bar, beta, G_k, zeros(n), delta, Low, Upp, H_k, subprob_switch);

                % Convergence test: tiny master model gradient and tiny delta
                if chi_k <= tol.gtol && delta <= tol.mindelta
                    disp('Convergence satisfied: small stationary measure and small delta');
                    X = X(1:nf, :);
                    F = F(1:nf, :);
                    h = h(1:nf, :);
                    return
                end

                [nf, X, F, h, Hash, hashes_at_nf] = ...
                  call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X(xkin, :) + s_k', tol, LB, UB, 1);

                % Compute rho_k
                ared = h(xkin) - h(nf);
                pred = -tau_k;
                rho_k = ared / pred;

                if rho_k >= tol.eta1 && pred > 0
                    successful = true; % iteration is successful
                    break
                elseif (rho_k < tol.eta1 || pred < 0) && all(ismember(hashes_at_nf, Act_Z_k))
                    successful = false;
                    break
                elseif pred <= 0
                    successful = false;
                    break
                else % stay in the manifold sampling loop
                    % Update models now that F(x+s) has been evaluated
                    [Gres, Hres, X, F, h, nf, Hash] = build_p_models(nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, LB, UB);
                    if isempty(Gres)
                        disp(['Model building failed. Empty Gres. Delta = ' num2str(delta)]);
                        X = X(1:nf, :);
                        F = F(1:nf, :);
                        h = h(1:nf, :);
                        flag = -1;
                        return
                    end

                    % Update activities and gradients
                    new_hashes = setdiff(hashes_at_nf, Act_Z_k);
                    Act_Z_k = [Act_Z_k, new_hashes];
                    [new_fbar, new_grads] = hfun(F(xkin, :), new_hashes);
                    D_k = [D_k, new_grads];
                    f_bar = [f_bar, new_fbar];
                end

            end % manifold sampling loop
            if successful
                xkin = nf; % iteration is successful
                if rho_k > tol.eta1 && norm(s_k) > 0.8 * delta
                    % Update Delta if rho is sufficiently large
                    delta = delta * tol.gamma_inc;
                    % h_activity_tol = min(1e-8, delta);
                end
            else
                % iteration is unsuccessful; shrink Delta
                delta = max(delta * tol.gamma_dec, tol.mindelta);
                % h_activity_tol = min(1e-8, delta);
            end
        end % if-else conditional on success of GOOMBAH step
        fprintf('nf: %8d; fval: %8e; radius: %8e; \n', nf, h(xkin), delta);
    end

    if nf >= nfmax
        flag = 0; % Budget exceeded
    else
        X = X(1:nf, :);
        F = F(1:nf, :);
        h = h(1:nf, :);
        flag = chi_k;
    end

end
