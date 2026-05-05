function [X, F, h, xkin] = goombah(hfun, Ffun, nf_max, x0, L, U, GAMS_options, subprob_switch)
    % Run |goombah| on the optimization problem specified by the given arguments.
    %
    %  :param hfun:    [func handle] Evaluates :math:`\hfun`, returning the
    %       scalar function value and :math:`k \times \nd` subgradients for all
    %       :math:`k` limiting gradients at the point given.
    %  :param Ffun:    [func handle] Evaluates :math:`\Ffun`, the black box
    %       simulation, returning a :math:`1 \times \nd` vector.
    %  :param nf_max:  [int] Maximum number of function evaluations.
    %  :param x0:      [:math:`1 \times \np` dbl] Starting point.
    %  :param L:       [:math:`1 \times \np` dbl] Lower bounds.
    %  :param U:       [:math:`1 \times \np` dbl] Upper bounds.
    %  :param GAMS_options:
    %  :param subprob_switch:
    %
    %  :return:
    %       * **X**    [:math:`\mathrm{nf\_max} \times \np`] Points evaluated
    %       * **F**    [:math:`\mathrm{nf\_max} \times \nd`] Their simulation values
    %       * **h**    [:math:`\mathrm{nf\_max} \times 1`] The values :math:`\hfun(\Ffun(\psp)))`
    %       * **xkin** [int] Current trust region center

    % Deduce p from evaluating Ffun at x0
    try
        F0 = Ffun(x0);
        F0 = F0(:)';
    catch
        warning('Problem using Ffun. Exiting');
        X = [];
        F = [];
        h = [];
        xkin = [];
        flag = -1;
        return
    end

    [n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, successful, xkin, Hres] = check_inputs_and_initialize(x0, F0, nf_max);

    [h(nf), ~, hashes_at_nf] = hfun(F(nf, :));
    Hash(nf, 1:length(hashes_at_nf)) = hashes_at_nf;

    I_n = eye(n);
    for i = 1:n
        [nf, X, F, h, Hash] = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X(xkin, :) + delta * I_n(i, :), tol, L, U, 1);
    end

    H_mm = zeros(n);
    beta_exp = 1.0;

    while nf < nf_max && delta > tol.mindelta
        [~, xkin] = min(h(1:nf));
        % ================================
        % Build p component models
        [Gres, Hres, X, F, h, nf, Hash] = build_p_models(nf, nf_max, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U);

        if isempty(Gres)
            disp(['Empty Gres. Delta = ' num2str(delta)]);
            X = X(1:nf, :);
            F = F(1:nf, :);
            h = h(1:nf, :);
            return
        end
        if nf >= nf_max
            return
        end

        Low = max(L - X(xkin, :), -delta);
        Upp = min(U - X(xkin, :), delta);

        [sk, pred_dec] = save_quadratics_call_GAMS(Hres, Gres, F(xkin, :), Low, Upp, X(xkin, :), X(xkin, :), h(xkin), GAMS_options, hfun);
        if pred_dec == 0
            rho_k = -inf;
        else
            % Evaluate F
            [nf, X, F, h, Hash] = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X(xkin, :) + sk, tol, L, U, 1);
            rho_k = (h(xkin) - h(nf)) / min(1.0, delta^(1.0 + beta_exp));
        end

        if rho_k > tol.eta1 % Success with GOOMBAH step
            if norm(X(xkin, :) - X(nf, :), 'inf') >= 0.8 * delta
                delta = delta * tol.gamma_inc;
            end
            xkin = nf;
        else
            %% Need to do one MS-P loop

            bar_delta = delta;

            % Line 3: manifold sampling while loop
            while nf < nf_max

                % Line 4: build models
                [Gres, Hres, X, F, h, nf, Hash] = build_p_models(nf, nf_max, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U);
                if isempty(Gres)
                    disp(['Model building failed. Empty Gres. Delta = ' num2str(delta)]);
                    X = X(1:nf, :);
                    F = F(1:nf, :);
                    h = h(1:nf, :);
                    flag = -1;
                    return
                end
                if nf >= nf_max
                    return
                end

                % Line 5: Build set of activities Act_Z_k, gradients D_k, G_k, and beta
                [D_k, Act_Z_k, f_bar] = choose_generator_set(X, Hash, 3, xkin, nf, delta, F, hfun);
                G_k = Gres * D_k;
                beta = max(0, f_bar' - h(xkin));

                % Line 6: Choose Hessions
                H_k = zeros(size(G_k, 2), n + 1, n + 1);
                for i = 1:size(G_k, 2) % would like to vectorize this tensor product ...
                    for j = 1:size(Hres, 3) % p
                        H_k(i, 2:end, 2:end) = squeeze(H_k(i, 2:end, 2:end)) + D_k(j, i) * Hres(:, :, j);
                    end
                end

                % Line 7: Find a candidate s_k by solving QP
                Low = max(L - X(xkin, :), -delta);
                Upp = min(U - X(xkin, :), delta);
                [s_k, tau_k] = minimize_affine_envelope(h(xkin), f_bar, beta, G_k, H_mm, delta, Low, Upp, H_k, subprob_switch);

                % Line 8: Compute stationary measure chi_k
                Low = max(L - X(xkin, :), -1.0);
                Upp = min(U - X(xkin, :), 1.0);
                [~, ~, chi_k] = minimize_affine_envelope(h(xkin), f_bar, beta, G_k, zeros(n), delta, Low, Upp, zeros(size(G_k, 2), n + 1, n + 1), subprob_switch);

                % Lines 9-11: Convergence test: tiny master model gradient and tiny delta
                if chi_k <= tol.gtol && delta <= tol.mindelta
                    disp('Convergence satisfied: small stationary measure and small delta');
                    X = X(1:nf, :);
                    F = F(1:nf, :);
                    h = h(1:nf, :);
                    flag = chi_k;
                    return
                end

                if printf
                    trsp_fun = @(x) max_affine(x, h(xkin), f_bar, beta, G_k, H_mm);
        %             plot_again_j(X, xkin, delta, s_k, [], nf, trsp_fun, L, U);
                end

                % Line 12: Evaluate F
                [nf, X, F, h, Hash, hashes_at_nf] = ...
                    call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X(xkin, :) + s_k', tol, L, U, 1);

                % Line 13: Compute rho_k
                ared = h(xkin) - h(nf);
                pred = -tau_k;
                rho_k = ared / pred;

                % Lines 14-16: Check for success
                if rho_k >= tol.eta1 && pred > 0
                    successful = true; % iteration is successful
                    break
                else % Line 17: Stay in the manifold sampling loop

                    % Line 18: Check temporary activities after adding TRSP solution to X
                    [~, tmp_Act_Z_k, ~] = choose_generator_set(X, Hash, 3, xkin, nf, delta, F, hfun);

                    % Lines 19: See if any new activities
                    if all(ismember(tmp_Act_Z_k, Act_Z_k))

                        % Line 20: See if intersection is nonempty
                        if any(ismember(hashes_at_nf, Act_Z_k))
                            successful = false; % iteration is unsuccessful
                            break
                        else
                            % Line 24: Shrink delta
                            delta = tol.gamma_dec * delta;
                        end
                    end
                end
            end

            if successful
                xkin = nf; % Line 15: Update TR center and radius
                if rho_k > tol.eta3 && norm(s_k, "inf") > 0.8 * bar_delta
                    % Update delta if rho is sufficiently large
                    delta = bar_delta * tol.gamma_inc;
                    % h_activity_tol = min(1e-8, delta);
                end
            else
                % Line 21: iteration is unsuccessful; shrink Delta
                delta = max(bar_delta * tol.gamma_dec, tol.mindelta);
                % h_activity_tol = min(1e-8, delta);
            end

        end
        fprintf('nf: %8d; fval: %8e; radius: %8e; \n', nf, h(xkin), delta);
    end

    if nf >= nf_max
        flag = 0; % Budget exceeded
    else
        X = X(1:nf, :);
        F = F(1:nf, :);
        h = h(1:nf, :);
    end
end
