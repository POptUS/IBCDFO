function [X, F, hF, xk_in, flag] = goombah(hfun, Ffun, x0, Low, Upp, nf_max, TRSP_options, subprob_switch, Options)
    % Run |goombah| on the optimization problem specified by the given arguments.
    %
    %  :param hfun:    [func handle] Evaluates :math:`\hfun`, returning the
    %       scalar function value and the :math:`p \times l` matrix whose
    %       columns are the :math:`l` limiting gradients at the given point.
    %  :param Ffun:    [func handle] Evaluates :math:`\Ffun`, the black box
    %       simulation, returning a :math:`1 \times p` vector.
    %  :param x0:      [:math:`1 \times \np` dbl] Starting point.
    %  :param Low:     [:math:`1 \times \np` dbl] Lower bounds; set unbounded entries to ``-Inf``.
    %  :param Upp:     [:math:`1 \times \np` dbl] Upper bounds; set unbounded entries to ``Inf``.
    %  :param nf_max:  [int] Maximum number of function evaluations.
    %  :param TRSP_options: [struct] Options for solving the trust-region
    %       subproblem. Fields:
    %
    %       * **solver** - ``'GAMS'`` to solve the subproblem with
    %         `GAMS <https://www.gams.com>`_
    %         or ``'lhs'`` to use a simple solver that samples the
    %         trust region and refines the best point with ``fmincon``. The
    %         latter is not recommended and is intended for testing and
    %         development. All other fields apply only to ``'GAMS'``.
    %
    %       * **file** - Path to the GAMS model file that corresponds to **hfun**.
    %
    %       * **solvers** - Vector of integers that selects the GAMS solvers to
    %         try. The step from the best successful solve is used as the trial
    %         point. Each solver must be available with your GAMS license. The
    %         solver that each integer selects is set in **TRSP_options.file**.
    %
    %       * **atol**, **rtol** - (optional) Absolute and relative tolerances
    %         (default ``1e-4`` each) for the check that the GAMS and |matlab|
    %         objective values agree. If they do not, the step is rejected.
    %
    %       * **hfun_data** - (optional) Users that pass an **hfun** that
    %         corresponds to one of the |goombah|-provided parameterized
    %         :math:`h` functions must use this field to provide GAMS with the
    %         same parameters used to construct their **hfun**.  See the general
    %         |goombah| documentation for more information.
    %
    %  :param subprob_switch: [str] Solver for the subproblem of the manifold sampling
    %       subproblems that |goombah| solves as a recourse:
    %
    %       * ``'linprog'`` - Solve the linear program with ``linprog``
    %         (requires the |matlab| Optimization Toolbox).
    %       * ``'GAMS_LP'`` - Solve the linear program with GAMS.
    %       * ``'GAMS_QCP'`` - Solve, with GAMS, the quadratically constrained
    %         problem that includes the Hessians of the models.
    %
    %  :param Options: [struct] (optional) Solver options. Omit or pass ``[]``
    %       for the defaults. Any missing field takes its default:
    %
    %       * **printf**         [int] 0 for no output, 1 for standard output, 2 for additional output,
    %         including each GAMS run (1)
    %       * **delta_0**        [dbl] Initial trust-region radius (0.1)
    %       * **delta_max**      [dbl] Maximum trust-region radius (1e8)
    %       * **delta_min**      [dbl] Minimum trust-region radius (1e-13)
    %       * **g_tol**          [dbl] Tolerance on the stationary measure (1e-13)
    %       * **eta_1**          [dbl] In (0, 1); rho above this is a successful iteration (0.01)
    %       * **eta_3**          [dbl] rho above this allows the radius to grow (0.5)
    %       * **gamma_dec**      [dbl] In (0, 1); factor for shrinking delta (0.5)
    %       * **gamma_inc**      [dbl] >= 1; factor for increasing delta (2)
    %       * **beta_exp**       [dbl] Exponent in the denominator of the |goombah| step ratio (1.0)
    %       * **hfun_test_mode** [bool] Check the hfun implementation on every call (1)
    %       * **h_activity_tol** [dbl] Tolerance for declaring an hfun manifold active (min(1e-8, delta_0))
    %
    %  :return:
    %       * **X**     [:math:`k \times \np`] Points evaluated, where
    %         :math:`k \leq \mathrm{nf\_max}`
    %       * **F**     [:math:`k \times p`] Their simulation values
    %       * **hF**    [:math:`k \times 1`] The values :math:`\hfun(\Ffun(\psp)))`
    %       * **xk_in** [int] Current trust region center
    %       * **flag**  [dbl] Termination criteria flag. The values are the same as
    %         for ``manifold_sampling_primal``: 0 for normal termination, 1 if
    %         the budget was exhausted, -1 if model building failed or **Ffun** failed, and -6 if
    %         the trust-region radius reached its minimum without satisfying the
    %         stationarity tolerance. A flag of -6 is a common way for a successful
    %         run to end; check the final ``chi_k`` that is displayed, and ``hF``

    [here_path, ~, ~] = fileparts(mfilename('fullpath'));
    repo_root = fullfile(here_path, '..', '..');
    addpath(fullfile(repo_root, 'pounders', 'm'));
    addpath(fullfile(repo_root, 'manifold_sampling', 'm'));
    addpath(fullfile(repo_root, 'goombah', 'm', 'subproblems'));

    if nargin < 9
        Options = [];
    end

    % Deduce p from evaluating Ffun at x0
    try
        F0 = Ffun(x0);
        F0 = F0(:)';
    catch
        warning('Problem using Ffun. Exiting');
        X = [];
        F = [];
        hF = [];
        xk_in = [];
        flag = -1;
        return
    end

    [n, delta, printf, fq_pars, tol, X, F, hF, Hash, nf, successful, xk_in, Hres] = check_inputs_and_initialize(x0, F0, nf_max, Options);
    if ~isfield(Options, 'beta_exp')
        beta_exp = 1.0; % [dbl] Exponent in the denominator of the step ratio (1.0)
    else
        beta_exp = Options.beta_exp;
    end
    TRSP_options.printf = printf;
    assert(any(strcmp(TRSP_options.solver, {'GAMS', 'lhs'})), 'TRSP_options.solver must be ''GAMS'' or ''lhs''');
    if strcmp(TRSP_options.solver, 'lhs')
        warning("GOOMBAH:lhsTrspSolver", ...
                "The lhs TRSP solver should only be used for testing or debugging");
    end

    [hF(nf), ~, hashes_at_nf] = hfun(F(nf, :));
    Hash(nf, 1:length(hashes_at_nf)) = hashes_at_nf;

    I_n = eye(n);
    for i = 1:n
        [nf, X, F, hF, Hash] = call_user_scripts(nf, X, F, hF, Hash, Ffun, hfun, X(xk_in, :) + delta * I_n(i, :), tol, Low, Upp, 1);
    end

    H_mm = zeros(n);
    chi_k = inf; % Stationary measure; set whenever it is computed
    while nf < nf_max && delta > tol.mindelta
        [~, xk_in] = min(hF(1:nf));
        % ================================
        % Build p component models
        [Gres, Hres, X, F, hF, nf, Hash] = build_p_models(nf, nf_max, xk_in, delta, F, X, hF, Hres, fq_pars, tol, hfun, Ffun, Hash, Low, Upp);

        if isempty(Gres)
            [X, F, hF, flag] = prepare_ms_outputs_before_return(X, F, hF, nf, -1, printf);
            return
        end
        if nf >= nf_max
            [X, F, hF, flag] = prepare_ms_outputs_before_return(X, F, hF, nf, 1, printf); % Budget exceeded
            return
        end

        Lows = max(Low - X(xk_in, :), -delta);
        Upps = min(Upp - X(xk_in, :), delta);

        if strcmp(TRSP_options.solver, 'lhs')
            [s_k, pred_dec] = minimize_quadratics_lhs(Hres, Gres, F(xk_in, :), Lows, Upps, X(xk_in, :), X(xk_in, :), hF(xk_in), hfun);
        else
            [s_k, pred_dec] = save_quadratics_call_GAMS(Hres, Gres, F(xk_in, :), Lows, Upps, X(xk_in, :), X(xk_in, :), hF(xk_in), TRSP_options, hfun);
        end
        if pred_dec == 0
            rho_k = -inf;
        else
            % Evaluate F
            [nf, X, F, hF, Hash] = call_user_scripts(nf, X, F, hF, Hash, Ffun, hfun, X(xk_in, :) + s_k, tol, Low, Upp, 1);
            rho_k = (hF(xk_in) - hF(nf)) / min(1.0, delta^(1.0 + beta_exp));
        end

        if rho_k > tol.eta1 % Success with GOOMBAH step
            if norm(X(xk_in, :) - X(nf, :), 'inf') >= 0.8 * delta
                delta = delta * tol.gamma_inc;
            end
            xk_in = nf;
        else
            %% Need to do one MS-P loop

            bar_delta = delta;

            % Line 3: manifold sampling while loop
            while nf < nf_max

                % Line 4: build models
                [Gres, Hres, X, F, hF, nf, Hash] = build_p_models(nf, nf_max, xk_in, delta, F, X, hF, Hres, fq_pars, tol, hfun, Ffun, Hash, Low, Upp);
                if isempty(Gres)
                    [X, F, hF, flag] = prepare_ms_outputs_before_return(X, F, hF, nf, -1, printf);
                    return
                end
                if nf >= nf_max
                    [X, F, hF, flag] = prepare_ms_outputs_before_return(X, F, hF, nf, 1, printf); % Budget exceeded
                    return
                end

                % Line 5: Build set of activities Act_Z_k, gradients D_k, G_k, and beta
                [D_k, Act_Z_k, f_bar] = choose_generator_set(X, Hash, xk_in, nf, delta, F, hfun);
                G_k = Gres * D_k;
                beta = max(0, f_bar' - hF(xk_in));

                % Line 6: Choose Hessians
                H_k = zeros(size(G_k, 2), n + 1, n + 1);
                for i = 1:size(G_k, 2) % would like to vectorize this tensor product ...
                    for j = 1:size(Hres, 3) % p
                        H_k(i, 2:end, 2:end) = squeeze(H_k(i, 2:end, 2:end)) + D_k(j, i) * Hres(:, :, j);
                    end
                end

                % Line 7: Find a candidate s_k_msp by solving QP
                Lows = max(Low - X(xk_in, :), -delta);
                Upps = min(Upp - X(xk_in, :), delta);
                [s_k_msp, tau_k] = minimize_affine_envelope(hF(xk_in), f_bar, beta, G_k, H_mm, delta, Lows, Upps, H_k, subprob_switch);

                % Line 8: Compute stationary measure chi_k
                Lows = max(Low - X(xk_in, :), -1.0);
                Upps = min(Upp - X(xk_in, :), 1.0);
                [~, ~, chi_k] = minimize_affine_envelope(hF(xk_in), f_bar, beta, G_k, zeros(n), delta, Lows, Upps, zeros(size(G_k, 2), n + 1, n + 1), subprob_switch);

                % Lines 9-11: Convergence test: tiny master model gradient and tiny delta
                if chi_k <= tol.gtol && delta <= tol.mindelta
                    [X, F, hF, flag] = prepare_ms_outputs_before_return(X, F, hF, nf, 0, printf);
                    return
                end

                if printf
                    trsp_fun = @(x) max_affine(x, hF(xk_in), f_bar, beta, G_k, H_mm);
        %             plot_again_j(X, xk_in, delta, s_k_msp, [], nf, trsp_fun, Low, Upp);
                end

                % Line 12: Evaluate F
                [nf, X, F, hF, Hash, hashes_at_nf] = ...
                    call_user_scripts(nf, X, F, hF, Hash, Ffun, hfun, X(xk_in, :) + s_k_msp', tol, Low, Upp, 1);

                % Line 13: Compute rho_k
                ared = hF(xk_in) - hF(nf);
                pred = -tau_k;
                rho_k = ared / pred;

                % Lines 14-16: Check for success
                if rho_k >= tol.eta1 && pred > 0
                    successful = true; % iteration is successful
                    break
                else % Line 17: Stay in the manifold sampling loop

                    % Line 18: Check temporary activities after adding TRSP solution to X
                    [~, tmp_Act_Z_k, ~] = choose_generator_set(X, Hash, xk_in, nf, delta, F, hfun);

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
                xk_in = nf; % Line 15: Update TR center and radius
                if rho_k > tol.eta3 && norm(s_k_msp, "inf") > 0.8 * bar_delta
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
        if printf
            fprintf('nf: %8d; fval: %8e; radius: %8e; \n', nf, hF(xk_in), delta);
        end
    end

    if nf >= nf_max
        exit_flag = 1; % Budget exceeded
    elseif chi_k <= tol.gtol
        exit_flag = 0;
    else
        exit_flag = -6; % delta_min reached without satisfying g_tol
    end
    [X, F, hF, flag] = prepare_ms_outputs_before_return(X, F, hF, nf, exit_flag, printf, chi_k);
end
