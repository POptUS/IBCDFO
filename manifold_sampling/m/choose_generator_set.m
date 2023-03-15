function [D_k, Act_Z_k, f_k] = choose_generator_set(X, Hash, gentype, xkin, nf, delta, F, hfun)
% Returns:

% D_k:     A set of gradients of h_j at points near X(xkin,:)
% Act_Z_k: A set of hashes for points in D_k (may contain duplicates in the
%          gentype == 2 case if D_k has multiple gradients from a given hash)

Act_Z_k = Hash(xkin, ~cellfun(@isempty, Hash(xkin, :)));
if gentype == 2
    for i = [1:xkin - 1 xkin + 1:nf]
        if norm(X(xkin, :) - X(i, :)) <= delta * (1 + 1e-8)
            Act_tmp = Hash(i, ~cellfun(@isempty, Hash(i, :)));

            Act_Z_k = [Act_Z_k, Act_tmp];
        end
    end
elseif gentype == 3
    hxkin = hfun(F(xkin, :), Act_Z_k);
    for i = [1:xkin - 1 xkin + 1:nf]
        Act_tmp = Hash(i, ~cellfun(@isempty, Hash(i, :)));
        h_i = hfun(F(xkin, :), Act_tmp);
        if norm(X(xkin, :) - X(i, :), "inf") <= delta * (1 + 1e-8) && h_i(1) <= hxkin(1)
            Act_Z_k = [Act_Z_k, Act_tmp];
        elseif norm(X(xkin, :) - X(i, :), "inf") <= delta^2 * (1 + 1e-8) && h_i(1) > hxkin(1)
            Act_Z_k = [Act_Z_k, Act_tmp];
        end
    end
end
[f_k, D_k] = hfun(F(xkin, :), Act_Z_k);
[D_k, inds] = unique(D_k', 'rows');
D_k = D_k';
Act_Z_k = Act_Z_k(inds);
f_k = f_k(inds);
end
