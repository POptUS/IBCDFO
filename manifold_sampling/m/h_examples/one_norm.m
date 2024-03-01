function [h, grads, Hash] = one_norm(z, H0)
    % Evaluates sum(abs(z_j))
    %
    % Inputs:
    %   z:    [1 x p] point where we are evaluating h
    %   H0:   (optional) [1 x l cell of strings] set of hashes where to evaluate z
    %
    % Outputs:
    %   h:    [dbl] function value
    %   grads: [p x l] gradients of each of the l manifolds active at z
    %   Hash:  [1 x l cell of strings] set of hashes for each of the l manifolds active at z

    if nargin < 2
        h = sum(abs(z));
        tol = 1e-8;
        p = length(z);
        grad_lists = cell(1, p);
        Hash_lists = cell(1, p);

        for i = 1:p
            if z(i) < -tol
                grad_lists{i} = -1;
                Hash_lists{i} = {'-'};
            elseif z(i) > tol
                grad_lists{i} = 1;
                Hash_lists{i} = {'+'};
            else
                grad_lists{i} = [-1, 1];
                Hash_lists{i} = {'-', '+'};
            end
        end

        grads = allcomb(grad_lists{:})';  % Can get this here: https://www.mathworks.com/matlabcentral/fileexchange/10064-allcomb-varargin-
        HashCombTmp = allcomb(Hash_lists{:});
        Hash = cell(1, size(HashCombTmp, 1));

        for i = 1:size(HashCombTmp, 1)
            Hash{i} = strjoin(HashCombTmp(i, :), '');
        end
    else
        J = length(H0);
        h = zeros(1, J);
        p = length(z);
        grads = ones(p, J);

        for j = 1:p
            for k = 1:J
                if strcmp(H0{k}(j), '-')
                    grads(j, k) = -1;
                end
                h(k) = dot(grads(:, k), z.');
            end
        end
    end
end
