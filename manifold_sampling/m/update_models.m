function [valid, Gres, Hres, X, F, h, nf, Hash] = update_models(hfun, Ffun, n, p, nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, geometry_pt_flag, Hash, tol, L, U)
    Cres = F(xkin, :);
    Res = zeros(size(F)); % Stores the residuals for model updates
    for i = 1:nf
        D = X(i, :) - X(xkin, :);
        for j = 1:length(Cres)
            Res(i, j) = (F(i, j) - Cres(j)) - .5 * D * Hres(:, :, j) * D';
        end
    end
    [Mdir, np, valid, Gres, Hresdel, ~] = formquad(X(1:nf, :), Res(1:nf, :), delta, xkin, fq_pars.npmax, fq_pars.Par, 0);
    % Evaluate geometry points
    if np < n && geometry_pt_flag % Must obtain and evaluate bounded geometry points
        [Mdir, np] = bmpts(X(xkin, :), Mdir(1:n - np, :), L, U, delta, fq_pars.Par(3));
        for i = 1:min(n - np, nfmax - nf)
            [nf, X, F, h, Hash] = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X(xkin, :) + Mdir(i, :), tol, L, U);
            D = Mdir(i, :);
            for j = 1:p
                Res(nf, j) = (F(nf, j) - Cres(j)) - .5 * D * Hres(:, :, j) * D';
            end
        end
        [~, ~, valid, Gres, Hresdel, ~] = formquad(X(1:nf, :), Res(1:nf, :), delta, xkin, fq_pars.npmax, fq_pars.Par, 0);
        if isempty(Gres)
            return
        end
    end

    if ~isempty(Gres) % We'll be doing evaluations; Hres will be updated after that
        Hres = Hres + Hresdel;
    end
end
