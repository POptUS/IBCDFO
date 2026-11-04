function [valid, Gres, Hres, X, F, h, nf, Hash] = update_models(hfun, Ffun, n, p, nf, nf_max, xk_in, delta, F, X, h, Hres, fq_pars, geometry_pt_flag, Hash, tol, Low, Upp)
    Cres = F(xk_in, :);
    Res = zeros(size(F)); % Stores the residuals for model updates

    assert(ndims(Hres) == 3, 'update_models: Hres must be a 3-D array, but has ndims(Hres) = %d.', ndims(Hres)); % GH Actions debug prints
    assert(size(Hres, 3) == length(Cres), 'update_models: size(Hres,3) = %d does not match size(F,2) = %d.', size(Hres, 3), length(Cres)); % GH Actions debug prints

    for i = 1:nf
        D = X(i, :) - X(xk_in, :);
        for j = 1:length(Cres)
            Res(i, j) = (F(i, j) - Cres(j)) - .5 * D * Hres(:, :, j) * D';
        end
    end
    [Mdir, np, valid, Gres, Hresdel, ~] = formquad(X(1:nf, :), Res(1:nf, :), delta, xk_in, fq_pars.npmax, fq_pars.Par, 0);
    % Evaluate geometry points
    if np < n && geometry_pt_flag % Must obtain and evaluate bounded geometry points
        [Mdir, np] = bmpts(X(xk_in, :), Mdir(1:n - np, :), Low, Upp, delta, fq_pars.Par(3));
        for i = 1:min(n - np, nf_max - nf)
            [nf, X, F, h, Hash] = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X(xk_in, :) + Mdir(i, :), tol, Low, Upp, 1);
            D = Mdir(i, :);
            for j = 1:p
                Res(nf, j) = (F(nf, j) - Cres(j)) - .5 * D * Hres(:, :, j) * D';
            end
        end
        [~, ~, valid, Gres, Hresdel, ~] = formquad(X(1:nf, :), Res(1:nf, :), delta, xk_in, fq_pars.npmax, fq_pars.Par, 0);
        if isempty(Gres)
            return
        end
    end

    if ~isempty(Gres) % We'll be doing evaluations; Hres will be updated after that
        Hres = Hres + Hresdel;
    end
end
