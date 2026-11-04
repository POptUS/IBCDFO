% This code solves the problem
%     minimize h(F(x))
% where x is a [1 by n] vector, F is a blackbox function mapping from R^n to
% R^p, and h is a nonsmooth function mapping from R^p to R.
%
%
% Inputs:
%  hfun:    [func handle] Evaluates h, returning the [scalar] function
%                         value and the [p x l] matrix whose columns are the
%                         l limiting gradients at the point given.
%  Ffun:    [func handle] Evaluates F, the black box simulation, returning
%                         a [1 x p] vector.
%  x0:      [1 x n dbl]   Starting point.
%  Low:     [1 x n dbl]   Lower bounds; set unbounded entries to ``-Inf``.
%  Upp:     [1 x n dbl]   Upper bounds; set unbounded entries to ``Inf``.
%  nf_max:  [int]         Maximum number of function evaluations.
%  TRSP_options: [struct] Options for solving the trust-region subproblem. Has
%            the same fields as the argument of the same name of goombah.m:
%            'solver' ('GAMS' or 'lhs'), and, for 'GAMS', 'file', 'solvers',
%            and (optionally) 'hfun_data'.
%  Options:  [struct, optional] May contain field 'spsolver', a handle to a
%            bound-constrained TRSP solver created by create_trsp_solver
%            (see pounders/m/create_trsp_solver.m). Defaults to minq5.
%
% Outputs:
%   X:      [k x n]   Points evaluated
%   F:      [k x p]   Their simulation values
%   hF:     [k x 1]   The values h(F(x))
%   xk_in:  [int]     Current trust region center

function [X, F, hF, xk_in] = goombah_wo_msp(hfun, Ffun, x0, Low, Upp, nf_max, TRSP_options, Options)

    [here_path, ~, ~] = fileparts(mfilename('fullpath'));
    repo_root = fullfile(here_path, '..', '..');
    addpath(fullfile(repo_root, 'pounders', 'm'));
    addpath(fullfile(repo_root, 'manifold_sampling', 'm'));
    addpath(fullfile(repo_root, 'goombah', 'm', 'subproblems'));

    if nargin < 8 || isempty(Options)
        Options = struct();
    end
    if ~isstruct(Options)
        error('Options must be a struct');
    end
    if ~isfield(Options, 'spsolver')
        Options.spsolver = create_trsp_solver(2); % Use minq5 by default
    end
    solve_trsp = Options.spsolver;

    % Deduce p from evaluating Ffun at x0
    try
        F0 = Ffun(x0);
    catch
        warning('Problem using Ffun. Exiting');
        X = [];
        F = [];
        hF = [];
        xk_in = [];
        flag = -1;
        return
    end

    [n, delta, printf, fq_pars, tol, X, F, hF, Hash, nf, ~, xk_in, Hres] = check_inputs_and_initialize(x0, F0, nf_max);
    assert(any(strcmp(TRSP_options.solver, {'GAMS', 'lhs'})), 'TRSP_options.solver must be ''GAMS'' or ''lhs''');
    if strcmp(TRSP_options.solver, 'lhs')
        warning("GOOMBAH:lhsTrspSolver", ...
                "The lhs TRSP solver should only be used for testing or debugging");
    end

    [hF(nf), ~, hashes_at_nf] = hfun(F(nf, :));

    I_n = eye(n);
    for i = 1:n
        [nf, X, F, hF, Hash] = call_user_scripts(nf, X, F, hF, Hash, Ffun, hfun, X(xk_in, :) + delta * I_n(i, :), tol, Low, Upp, 1);
    end

    while nf < nf_max
        [~, xk_in] = min(hF(1:nf));
        % ================================
        % Build p component models
        [Gres, Hres, X, F, hF, nf] = build_p_models(nf, nf_max, xk_in, delta, F, X, hF, Hres, fq_pars, tol, hfun, Ffun, Hash, Low, Upp);

        if isempty(Gres)
            disp(['Empty Gres. Delta = ' num2str(delta)]);
            X = X(1:nf, :);
            F = F(1:nf, :);
            hF = hF(1:nf, :);
            return
        end
        if nf >= nf_max
            return
        end
        % ================================

        % ================================
        % Construct Gen_k
        D_k = construct_Gen_k(hfun, xk_in, 2, nf, delta, F, X);
        % ================================

        lambda = project_zero_onto_convex_hull_2(Gres * D_k);
        [g_k, H_k, ng] = build_master_model(D_k, Gres, lambda, Hres, n);

        %     if printf
        %         if norm(g_k) < 1e-12
        %             counter = counter + 1;
        %             scatter(best(1),best(2),'ro','filled','SizeData',SD)
        %             saveas(gca,['contour_animation=' int2str(animation) '_count=' int2str(counter) '.eps'],'epsc')
        %         else
        %             if ~true
        %                 counter = counter + 1;
        %                 dir = -g_k./norm(g_k)/4;
        %                 quiver(best(1),best(2),dir(1),dir(2),'r','AutoScale','off','LineWidth',2)
        %                 saveas(gca,['contour_animation=' int2str(animation) '_count=' int2str(counter) '.eps'],'epsc')
        %             end
        %         end
        %     end
        %     % ================================

        % Convergence test: tiny master model gradient and tiny delta
        if ng <= tol.gtol && delta <= tol.mindelta
            disp('g is sufficiently small');
            X = X(1:nf, :);
            F = F(1:nf, :);
            hF = hF(1:nf, :);
            return
        end

        if delta < tol.eta2 * ng
            % ================================
            % Solve TRSP
            Lows = max(Low - X(xk_in, :), -delta);
            Upps = min(Upp - X(xk_in, :), delta);

            [s_k, ~, found_solution] = solve_trsp(H_k, g_k, Lows, Upps);
            if ~found_solution
                s_k = zeros(n, 1);
            end
            s_k = s_k';

            % if printf
            %     plot_again(X, xk_in, delta, s_k, [], nf, [], Low, Upp);
            % end

            if strcmp(TRSP_options.solver, 'lhs')
                [s_k_trial, pred_dec] = minimize_quadratics_lhs(Hres, Gres, F(xk_in, :), Lows, Upps, X(xk_in, :), X(xk_in, :) + s_k, hF(xk_in), hfun);
            else
                [s_k_trial, pred_dec] = save_quadratics_call_GAMS(Hres, Gres, F(xk_in, :), Lows, Upps, X(xk_in, :), X(xk_in, :) + s_k, hF(xk_in), TRSP_options, hfun);
            end
            if pred_dec == 0
                if delta <= tol.mindelta
                    X = X(1:nf, :);
                    F = F(1:nf, :);
                    hF = hF(1:nf, :);
                    return
                else
                    rho_k = -inf;
                end
            end
            % ================================

            % if printf
            %     trsp_fun = @(x) h_of_quad_models(x, X(xk_in, :), F(xk_in, :), Gres, Hres, hfun);
            %     plot_again(X, xk_in, delta, s_k_trial, [], nf, trsp_fun, Low, Upp);
            % end

            if pred_dec > 0
                % ================================
                % Evaluate F
                [nf, X, F, hF, Hash] = call_user_scripts(nf, X, F, hF, Hash, Ffun, hfun, X(xk_in, :) + s_k_trial, tol, Low, Upp, 1);
                rho_k = (hF(xk_in) - hF(nf)) / pred_dec;
            end
        else
            rho_k = -inf;
        end

        if rho_k > tol.eta1
            if norm(X(xk_in, :) - X(nf, :), 'inf') >= 0.9 * delta
                delta = min(delta * tol.gamma_inc, tol.maxdelta);
            end
            xk_in = nf;
        else
            delta = max(delta * tol.gamma_dec, tol.mindelta);
        end
        fprintf('nf: %8d; fval: %8e; ||g||: %8e; radius: %8e; \n', nf, hF(xk_in), ng, delta);
    end

end

function [g_k, H_k, ng] = build_master_model(D_k, Gres, lambda, Hres, n)

    w_k = D_k * lambda;
    g_k = Gres * w_k;
    ng = norm(g_k);
    H_k = zeros(n);
    for model = 1:size(D_k, 1)
        H_k = H_k + w_k(model) * Hres(:, :, model);
    end
end

function D_k = construct_Gen_k(hfun, xk_in, gentype, nf, delta, F, X)
    [~, grads_at_x] = hfun(F(xk_in, :));
    D_k = grads_at_x;
    if gentype == 2
        for i = 1:nf
            if norm(X(xk_in, :) - X(i, :)) <= delta
                [~, grads_at_pt_near_x] = hfun(F(i, :));
                D_k = unique([D_k, grads_at_pt_near_x]', 'rows')';
            end
        end
    end
end
