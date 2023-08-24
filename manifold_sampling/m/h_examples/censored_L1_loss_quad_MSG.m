function [h, grads, Hash] = censored_L1_loss_quad_MSG(z, H0)
% This a generalized version of Womersley's censored L1 loss function.

% Hashes are output (and must be input) in the following fashion:
%   Hash elements are strings of p integers.
%     1 in position j means [z(j) >= C(j)] and [D(j) - max(z(j),C(j) <= 0]
%     2 in position j means [z(j) <= C(j)] and [D(j) - max(z(j),C(j) <= 0]
%     3 in position j means [z(j) >= C(j)] and [D(j) - max(z(j),C(j) >= 0]
%     4 in position j means [z(j) <= C(j)] and [D(j) - max(z(j),C(j) >= 0]
%
%   Similarly, H0
%     1 in position j uses -(D(j) - z(j)) ...
%     2 in position j uses -(D(j) - C(j)) ...
%     3 in position j uses D(j) - z(j) ...
%     4 in position j uses D(j) - C(j) ...
%   ... in the calculation of h and grads.

% Get data from outside of this function
global C D n_h hvals_mat

eqtol = 1e-8;

% Ensure column vector and collect dimensions
zin = z(:);
z = z(:);
C = C(:);
D = D(:);
p = length(C);

coef = -0.08;
z = z + coef * z.^2;
if nargin == 1
    h = sum(abs(D - max(z, C)));
    g = cell(p, 1);
    H = cell(p, 1);

    lg = zeros(p, 1);
    lH = zeros(p, 1);
    for i = 1:p
        lg(i) = 0;
        lH(i) = 0;
        if z(i) <= C(i) || abs(z(i) - C(i)) < eqtol * (max(abs([z(i), C(i)]))) || abs(z(i) - C(i)) < eqtol
            lg(i) = lg(i) + 1;
            g{i}(lg(i)) = {0};
            if C(i) >= D(i)
                lH(i) = lH(i) + 1;
                H{i}(lH(i)) = {'2'};
            end
            if C(i) <= D(i)
                lH(i) = lH(i) + 1;
                H{i}(lH(i)) = {'4'};
            end
        end
        if z(i) >= C(i) || abs(z(i) - C(i)) < eqtol * (max(abs([z(i), C(i)]))) || abs(z(i) - C(i)) < eqtol
            lg(i) = lg(i) + 1;
            lH(i) = lH(i) + 1;
            if max(z(i), C(i)) == D(i) || abs(max(z(i), C(i)) - D(i)) < eqtol * (max(abs([max(z(i), C(i)), D(i)]))) || abs(max(z(i), C(i)) - D(i)) < eqtol
                g{i}(lg(i)) = {2 * coef * z(i) + 1};
                lg(i) = lg(i) + 1;
                g{i}(lg(i)) = {-2 * coef * z(i) - 1};

                H{i}(lH(i)) = {'1'};
                lH(i) = lH(i) + 1;
                H{i}(lH(i)) = {'3'};
            else
                g{i}(lg(i)) = {sign(z(i) - D(i)) * (2 * coef * zin(i) + 1)};
                if D(i) >= z(i)
                    H{i}(lH(i)) = {'3'};
                else
                    H{i}(lH(i)) = {'1'};
                end
            end
        end
    end

    if all(cellfun(@length, g) == 1)
        grads = cell2mat([g{:}])';
        hashes_as_mat = cell2mat([H{:}]);
    else
        grads = cell2mat(allcomb(g{:}))';  % Can get this here: https://www.mathworks.com/matlabcentral/fileexchange/10064-allcomb-varargin-
        hashes_as_mat = cell2mat(allcomb(H{:}));
    end

    b = size(hashes_as_mat, 1);
    Hash = cell(1, b);
    for i = 1:b
        Hash{i} = hashes_as_mat(i, :);
    end
    % % For debugging purposes:
    % % Verify that the hash and value are the same at this point
    % [a,b] = censored_L1_loss_quad_MSG(zin, Hash);
    % assert(a == h)
    % assert(all(b == grads));

elseif nargin == 2
    K = length(H0);

    h = zeros(1, K);
    grads = zeros(p, K);
    vals = zeros(p, K);

    for k = 1:K
        for j = 1:p
            switch H0{k}(j)
                case '1'
                    vals(j, k) = -(D(j) - z(j));
                    grads(j, k) = 2 * coef * zin(j) + 1;
                case '2'
                    vals(j, k) = -(D(j) - C(j));
                    grads(j, k) = 0;
                case '3'
                    vals(j, k) = D(j) - z(j);
                    grads(j, k) = -2 * coef * zin(j) - 1;
                case '4'
                    vals(j, k) = D(j) - C(j);
                    grads(j, k) = 0;
            end
        end
        h(k) = sum(vals(:, k));
    end
end

if ~isempty(n_h)
    n_h = n_h + 1;
    hvals_mat(n_h, :) = h;
end
