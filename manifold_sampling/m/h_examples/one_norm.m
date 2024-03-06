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
                % Technically, we should return [1,-1] for the grad entry and
                % {'-','+'} for the Hash, but that causes issues for large dim(z)
                % because we get 2^dim(z) grads.... but they don't matter
                % really for the convex hull calculation
                grad_lists{i} = 0;
                Hash_lists{i} = {'0'};
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

        for k = 1:J
            ztemp = z;
            for j = 1:p
                if strcmp(H0{k}(j), '-')
                    grads(j, k) = -1;
                    ztemp(j) = -1 * ztemp(j);
                elseif strcmp(H0{k}(j), '0')
                    grads(j, k) = 0;
                    if ztemp(j) < 0
                        ztemp(j) = -1 * ztemp(j);
                    end
                end
            end
            h(k) = sum(ztemp);
        end
    end
end
