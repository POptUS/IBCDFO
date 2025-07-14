function [Gres, Hres, X, F, h, nf, Hash] = build_p_models(nf, nf_max, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U)
    n = size(X, 2);
    p = size(F, 2);

    [valid, Gres, Hres, X, F, h, nf, Hash] = update_models(hfun, Ffun, n, p, nf, nf_max, xkin, delta, F, X, h, Hres, fq_pars, 1, Hash, tol, L, U);

    % Evaluate model-improving points if necessary
    if ~valid && ~isempty(Gres)
        [Mdir, np, valid] = formquad(X(1:nf, :), F(1:nf, :), delta, xkin, fq_pars.npmax, fq_pars.Par, 1);
        assert(~valid, "How did this switch to being valid?");
        [X, F, h, nf, Hash] = evaluate_points_to_force_valid_model(n, nf, xkin, delta, X, F, h, tol.gentype, Mdir, np, hfun, Ffun, Hash, fq_pars, tol, nf_max, L, U);
        [~, Gres, Hres, X, F, h, nf, Hash] = update_models(hfun, Ffun, n, p, nf, nf_max, xkin, delta, F, X, h, Hres, fq_pars, 1, Hash, tol, L, U);
    end
end
