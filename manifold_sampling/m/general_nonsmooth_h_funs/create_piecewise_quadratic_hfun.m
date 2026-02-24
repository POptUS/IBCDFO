function [hfun] = create_piecewise_quadratic_hfun(Qs, zs, cs)
    % Please refer to the documentation for the Python version of this h function.

    % Have MATLAB automatically ensure that each actual C,D arguments passed to
    % this function are 1D finite, real column vectors of the same length.
    arguments
        Qs (:, :, :) {mustBeReal, mustBeFinite, mustBeNonempty}
        zs (:, :)    {mustBeReal, mustBeFinite, mustBeNonempty}
        cs (:, 1)    {mustBeReal, mustBeFinite, mustBeNonempty}
    end

    % Inputs:
    %  z:              [1 x p]   point where we are evaluating h
    %  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate

    % Outputs:
    %  h: [dbl]                       function value
    %  grads: [p x l]                 gradients of each of the l quadratics active at z
    %  Hash: [1 x l cell of strings]  set of hashes for each of the l quadratics active at z (in the same order as the elements of grads)

    % Hashes are output (and must be input) in the following fashion:
    %   Hash{i} = 'j' if quadratic j is active at z (or H0{i} = 'j' if the
    %   value/gradient of quadratic j at z is desired)

    % IMPORTANT: Aside from ensuring that cs is a column vector, don't alter
    % Qs, zs, or cs anywhere in this function.
    cs = cs(:);

    [n, J] = size(zs);
    if ~isequal(size(Qs), [n n J])
        error("POptUS:IncompatibleSizes", "zs & Qs sizes incompatible");
    elseif size(cs) ~= J
        error("POptUS:IncompatibleSizes", "zs & cs sizes incompatible");
    end

    hfun = @h_piecewise_quadratic;

    function [h, grads, Hash] = h_piecewise_quadratic(z, H0)
        global h_activity_tol

        if isempty(h_activity_tol)
            h_activity_tol = 0;
        end

        [n, ~] = size(zs);

        % Error check under the assumption that it is essentially only MSP,
        % which is under our control, calling this function.
        assert(isvector(z));
        if length(z) ~= n
            error("POptUS:IncompatibleSizes", "z size incompatible with zs");
        elseif any(~isreal(z))
            error("POptUS:NonrealValues", "z contains non-real values");
        elseif any(~isfinite(z))
            error("POptUS:NonfiniteValues", "z contains non-finite values");
        end

        % Ensure column vector
        z = z(:);

        if nargin == 1
            [n, J] = size(zs);
            manifolds = zeros(1, J);
            for j = 1:J
                manifolds(j) = (z - zs(:, j))' * Qs(:, :, j) * (z - zs(:, j)) + cs(j);
            end

            h = max(manifolds);

            atol = h_activity_tol;
            rtol = h_activity_tol;
            inds = find(abs(h - manifolds) <= atol + rtol * abs(manifolds));

            grads = zeros(n, length(inds));

            Hash = cell(1, length(inds));
            for j = 1:length(inds)
                Hash{j} = int2str(inds(j));
                grads(:, j) = 2 * Qs(:, :, inds(j)) * (z - zs(:, inds(j)));
            end

        elseif nargin == 2
            J = length(H0);
            h = zeros(1, J);
            grads = zeros(length(z), J);

            for k = 1:J
                j = str2num(H0{k});
                h(k) = (z - zs(:, j))' * Qs(:, :, j) * (z - zs(:, j)) + cs(j);
                grads(:, k) = 2 * Qs(:, :, j) * (z - zs(:, j));
            end
        end
    end

end
