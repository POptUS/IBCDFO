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
%  nf_max:   [int]         Maximum number of function evaluations.
%  x0:      [1 x n dbl]   Starting point.
%  L:       [1 x n dbl]   Lower bounds.
%  U:       [1 x n dbl]   Upper bounds.
%  GAMS_options: [struct] TRSP options. Field 'trsp' is 'GAMS' (call GAMS via
%            save_quadratics_call_GAMS; also needs 'file' and 'solvers') or
%            'lhs' (Latin hypercube sampling and fmincon via
%            minimize_quadratics_lhs).
%  Options:  [struct, optional] May contain field 'spsolver', a handle to a
%            bound-constrained TRSP solver created by create_trsp_solver
%            (see pounders/m/create_trsp_solver.m). Defaults to minq5.
%
% Outputs:
%   X:      [nf_max x n]   Points evaluated
%   F:      [nf_max x p]   Their simulation values
%   h:      [nf_max x 1]   The values h(F(x))
%   xkin:   [int]         Current trust region center

function [X, F, h, xkin] = goombah_wo_msp(hfun, Ffun, nf_max, x0, L, U, GAMS_options, Options)

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
        h = [];
        xkin = [];
        flag = -1;
        return
    end

    [n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, ~, xkin, Hres] = check_inputs_and_initialize(x0, F0, nf_max);
    assert(any(strcmp(GAMS_options.trsp, {'GAMS', 'lhs'})), 'GAMS_options.trsp must be ''GAMS'' or ''lhs''');

    [h(nf), ~, hashes_at_nf] = hfun(F(nf, :));

    I_n = eye(n);
    for i = 1:n
        [nf, X, F, h, Hash] = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X(xkin, :) + delta * I_n(i, :), tol, L, U, 1);
    end

    while nf < nf_max
        [~, xkin] = min(h(1:nf));
        % ================================
        % Build p component models
        [Gres, Hres, X, F, h, nf] = build_p_models(nf, nf_max, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U);

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
        % ================================

        % ================================
        % Construct Gen_k
        col_vecs_for_Genk = construct_Gen_k(hfun, xkin, 2, nf, delta, F, X);
        % ================================

        lambda = project_zero_onto_convex_hull_2(Gres * col_vecs_for_Genk);
        [g_k, H_k, ng] = build_master_model(col_vecs_for_Genk, Gres, lambda, Hres, n);

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
            h = h(1:nf, :);
            return
        end

        if delta < tol.eta2 * ng
            % ================================
            % Solve TRSP
            Low = max(L - X(xkin, :), -delta);
            Upp = min(U - X(xkin, :), delta);

            [sk, ~, found_solution] = solve_trsp(H_k, g_k, Low, Upp);
            if ~found_solution
                sk = zeros(n, 1);
            end
            sk = sk';

            % if printf
            %     plot_again(X, xkin, delta, sk, [], nf, [], L, U);
            % end

            if strcmp(GAMS_options.trsp, 'lhs')
                [sk1, pred_dec] = minimize_quadratics_lhs(Hres, Gres, F(xkin, :), Low, Upp, X(xkin, :), X(xkin, :) + sk, h(xkin), hfun);
            else
                [sk1, pred_dec] = save_quadratics_call_GAMS(Hres, Gres, F(xkin, :), Low, Upp, X(xkin, :), X(xkin, :) + sk, h(xkin), GAMS_options, hfun);
            end
            if pred_dec == 0
                if delta <= tol.mindelta
                    X = X(1:nf, :);
                    F = F(1:nf, :);
                    h = h(1:nf, :);
                    return
                else
                    rho_k = -inf;
                end
            end
            % ================================

            % if printf
            %     trsp_fun = @(x) h_of_quad_models(x, X(xkin, :), F(xkin, :), Gres, Hres, hfun);
            %     plot_again(X, xkin, delta, sk1, [], nf, trsp_fun, L, U);
            % end

            if pred_dec > 0
                % ================================
                % Evaluate F
                [nf, X, F, h, Hash] = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X(xkin, :) + sk1, tol, L, U, 1);
                rho_k = (h(xkin) - h(nf)) / pred_dec;
            end
        else
            rho_k = -inf;
        end

        if rho_k > tol.eta1
            if norm(X(xkin, :) - X(nf, :), 'inf') >= 0.9 * delta
                delta = min(delta * tol.gamma_inc, tol.maxdelta);
            end
            xkin = nf;
        else
            delta = max(delta * tol.gamma_dec, tol.mindelta);
        end
        fprintf('nf: %8d; fval: %8e; ||g||: %8e; radius: %8e; \n', nf, h(xkin), ng, delta);
    end

end

function [g_k, H_k, ng] = build_master_model(col_vecs_for_Genk, Gres, lambda, Hres, n)

    w_k = col_vecs_for_Genk * lambda;
    g_k = Gres * w_k;
    ng = norm(g_k);
    H_k = zeros(n);
    for model = 1:size(col_vecs_for_Genk, 1)
        H_k = H_k + w_k(model) * Hres(:, :, model);
    end
end

function col_vecs_for_Genk = construct_Gen_k(hfun, xkin, gentype, nf, delta, F, X)
    [~, grads_at_x] = hfun(F(xkin, :));
    col_vecs_for_Genk = grads_at_x;
    if gentype == 2
        for i = 1:nf
            if norm(X(xkin, :) - X(i, :)) <= delta
                [~, grads_at_pt_near_x] = hfun(F(i, :));
                col_vecs_for_Genk = unique([col_vecs_for_Genk, grads_at_pt_near_x]', 'rows')';
            end
        end
    end
end
