function [Gres, Hres, X, F, hF, nf, Hash] = build_p_models(nf, nf_max, xk_in, delta, F, X, hF, Hres, fq_pars, tol, hfun, Ffun, Hash, Low, Upp)
    n = size(X, 2);
    p = size(F, 2);

    [valid, Gres, Hres, X, F, hF, nf, Hash] = update_models(hfun, Ffun, n, p, nf, nf_max, xk_in, delta, F, X, hF, Hres, fq_pars, 1, Hash, tol, Low, Upp);

    % Evaluate model-improving points if necessary
    if ~valid && ~isempty(Gres)
        [Mdir, np, valid] = formquad(X(1:nf, :), F(1:nf, :), delta, xk_in, fq_pars.npmax, fq_pars.Par, 1);
        assert(~valid, "How did this switch to being valid?");
        [X, F, hF, nf, Hash] = evaluate_points_to_force_valid_model(n, nf, xk_in, delta, X, F, hF, Mdir, np, hfun, Ffun, Hash, fq_pars, tol, nf_max, Low, Upp);
        [~, Gres, Hres, X, F, hF, nf, Hash] = update_models(hfun, Ffun, n, p, nf, nf_max, xk_in, delta, F, X, hF, Hres, fq_pars, 1, Hash, tol, Low, Upp);
    end
end
