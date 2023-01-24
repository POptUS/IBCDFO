function [s, tau, chi, lambda_star] = minimize_affine_envelope(f, f_bar, beta, G_k, H, delta, Low, Upp, H_k, subprob_switch)

    [G_k_smaller, cols] = uniquetol(G_k', 'ByRows', true);
    G_k_smaller = G_k_smaller';

    [n, p] = size(G_k_smaller);

    bk = -(f_bar' - f - beta);
    bk_smaller = bk(cols);
    H_k_smaller = H_k(cols, :, :);

    % check first if mu = 0 (i.e. TR is not active)
    A  = [-ones(p, 1) G_k_smaller'];
    ff = [1; zeros(n, 1)];
    HH = [0 zeros(1, n); zeros(n, 1) H];
    x0 = [max(-bk_smaller); zeros(n, 1)];

    if strcmp(subprob_switch, 'GAMS_QCP')
        [x, duals_g, duals_u, duals_l] = solve_matts_QCP(ff, A, bk_smaller, x0, delta, Low, Upp, H_k_smaller);
        duals_u = duals_u(2:n + 1);
        duals_l = duals_l(2:n + 1);
    elseif strcmp(subprob_switch, 'GAMS_LP')
        [x, duals_g, duals_u, duals_l] = solve_matts_LP(ff, A, bk_smaller, x0, Low, Upp);
        duals_u = duals_u(2:n + 1);
        duals_l = duals_l(2:n + 1);
    elseif strcmp(subprob_switch, 'linprog')
        try
            [x, ~, exitflag, ~, lambda] = linprog(ff, A, bk_smaller, [], [], [-Inf, Low]', [Inf, Upp]');
            if exitflag == 1 % successful termination
                duals_g = lambda.ineqlin;
                duals_u = lambda.upper(2:n + 1);
                duals_l = lambda.lower(2:n + 1);
            else
                duals_g = zeros(p, 1);
                duals_g(1) = 1.0;
                duals_l = zeros(n, 1);
                duals_u = zeros(n, 1);
                x = x0;
            end
        catch % sigh. for some reason linprog just hard fails rather than returning an exitflag sometimes.
            % it appears that this failure is related to the scaling of A. so we suggest:
            normA = norm(A(:, 2:end));
            rescaledA = zeros(size(A));
            rescaledA(:, 1) = -ones(p, 1);
            rescaledA(:, 2:end) = A(:, 2:end) / normA;
            [x, ~, exitflag, ~, lambda] = linprog(ff, rescaledA, bk_smaller, [], [], [-Inf, Low]', [Inf, Upp]');
            if exitflag == 1 % successful termination
                duals_g = lambda.ineqlin;
                duals_u = lambda.upper(2:n + 1) * normA;
                duals_l = lambda.lower(2:n + 1) * normA;
            else
                duals_g = zeros(p, 1);
                duals_g(1) = 1.0;
                duals_l = zeros(n, 1);
                duals_u = zeros(n, 1);
                x = x0;
            end
        end
    else
        error('Unrecognized subprob_switch');
    end

    lambda_star = sparse(size(G_k, 2), 1);
    lambda_star(cols) = duals_g;

    s = x(2:end);
    tau = max(-bk + G_k' * s) + 0.5 * s' * H * s;
    if tau > 0
        % something went wrong
        tau = 0;
        s = zeros(n, 1);
    end
    Low(duals_l <= 0) = 0;
    Upp(duals_u <= 0) = 0;
    chi = norm(G_k * lambda_star - duals_l + duals_u) + bk' * lambda_star - Low * duals_l + Upp * duals_u;
end
