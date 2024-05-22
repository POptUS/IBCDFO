function Xsp = formquad_model_improvement(x_k, Cres, Gres, Hres, Mdir, np, Low, Upp, delta, Model, combinemodels)

    n = length(x_k); 

    % Update for modelimp; Cres unchanged b/c xk_in unchanged
    [G, H] = combinemodels(Cres, Gres, Hres);

    % Evaluate model-improving points to pick best one
    % ! May eventually want to normalize Mdir first for infty norm
    % Plus directions
    [Mdir1, np1] = bmpts(x_k, Mdir(1:n - np, :), Low, Upp, delta, Model.Par(3));
    for i = 1:n - np1
        D = Mdir1(i, :);
        Res(i, 1) = D * (G + .5 * H * D');
    end
    [a1, b] = min(Res(1:n - np1, 1));
    Xsp = Mdir1(b, :);
    % Minus directions
    [Mdir1, np2] = bmpts(x_k, -Mdir(1:n - np, :), Low, Upp, delta, Model.Par(3));
    for i = 1:n - np2
        D = Mdir1(i, :);
        Res(i, 1) = D * (G + .5 * H * D');
    end
    [a2, b] = min(Res(1:n - np2, 1));
    if a2 < a1
        Xsp = Mdir1(b, :);
    end

end

