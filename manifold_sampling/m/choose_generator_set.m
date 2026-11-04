function [D_k, Act_Z_k, f_k] = choose_generator_set(X, Hash, xk_in, nf, delta, F, hfun)
% Choose the generator set D_k for the manifold sampling subproblem at the current
% trust region center X(xk_in,:). The generators are the gradients, at the center
% F(xk_in,:), of a set of selection functions (manifolds) of h.
%
% The selection functions are those active at the center, plus those active at
% evaluated points near the center. Each is evaluated at the center F(xk_in,:),
% not at the point where it was found to be active. For an evaluated point X(i,:)
% other than the center, let h_i be the value at F(xk_in,:) of the first selection
% function active at X(i,:), and let hF_at_xk_in be the corresponding value for
% the center (that is, h at the center). All selection functions active at X(i,:)
% are added when
%   * ||X(i,:) - X(xk_in,:)||_inf <= delta   and h_i <= hF_at_xk_in, or
%   * ||X(i,:) - X(xk_in,:)||_inf <= delta^2 and h_i >  hF_at_xk_in,
% where each distance test has a relative tolerance of 1e-8. Selection functions
% whose gradients coincide are kept only once.
%
% Inputs:
%  X:       [nf_max x n]  Points evaluated (rows 1:nf are in use)
%  Hash:    [cell]        Hash(i, :) holds the hashes of the selection functions
%                         active at X(i,:); unused entries are empty
%  xk_in:   [int]         Index of the trust region center
%  nf:      [int]         Number of points evaluated so far
%  delta:   [dbl]         Trust region radius
%  F:       [nf_max x p]  Function values at the points in X
%  hfun:    [func handle] Evaluates h; given z and a set of hashes, returns the
%                         values and gradients of those selection functions
%
% Outputs:
%  D_k:     [p x l]       The generator set: columns are the gradients, at F(xk_in,:),
%                         of the l chosen selection functions
%  Act_Z_k: [1 x l cell]  Hashes of the chosen selection functions, in the order of
%                         the columns of D_k
%  f_k:     [1 x l]       Values of the chosen selection functions at F(xk_in,:)

Act_Z_k = Hash(xk_in, ~cellfun(@isempty, Hash(xk_in, :)));

hF_at_xk_in = hfun(F(xk_in, :), Act_Z_k);
for i = [1:xk_in - 1 xk_in + 1:nf]
    Act_tmp = Hash(i, ~cellfun(@isempty, Hash(i, :)));
    h_i = hfun(F(xk_in, :), Act_tmp);
    if norm(X(xk_in, :) - X(i, :), "inf") <= delta * (1 + 1e-8) && h_i(1) <= hF_at_xk_in(1)
        Act_Z_k = [Act_Z_k, Act_tmp];
    elseif norm(X(xk_in, :) - X(i, :), "inf") <= delta^2 * (1 + 1e-8) && h_i(1) > hF_at_xk_in(1)
        Act_Z_k = [Act_Z_k, Act_tmp];
    end
end
[f_k, D_k] = hfun(F(xk_in, :), Act_Z_k);
[D_k, inds] = unique(D_k', 'rows');
D_k = D_k';
Act_Z_k = Act_Z_k(inds);
f_k = f_k(inds);
end
