function [Mdir, np, valid, Gres, Hres, Mind] = gaussnewton(X, F, delta, xk_in, np_max, Pars, vf, aux)

    Gres = aux{xk_in}';
    [n, m] = size(Gres);
    Hres = zeros(n, n, m); 

    % silly things to make everything compatible in pounders
    valid = true; 
    np = n; 

    Mdir = []; % because valid is always true, pounders should never try to read this.
    Mind = []; % ditto above

end