load('formquad_inputs.mat')
n = 4;
m = 1;
nf = 490;

[Mdir, np, valid] = formquad(X(1:nf, :), F(1:nf, :), delta, xkin, npmax, Par, 1);
if ~(valid)  % ! One strategy for choosing model-improving point:
    % Update model (exists because delta & xkin unchanged)
    for i = 1:nf
        D = (X(i, :) - X(xkin, :));
        Res(i, :) = F(i, :) - Cres - .5 * D * reshape(D * reshape(Hres, n, m * n), n, m);
    end
    [~, ~, valid, Gres, Hresdel, Mind] = formquad(X(1:nf, :), Res(1:nf, :), delta, xkin, npmax, Par, 0);
    Hres = Hres + Hresdel;
end
