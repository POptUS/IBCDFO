function [hfun, combinemodels] = create_squared_diff_from_mean_functions(alpha)
    % Please refer to the documentation for the Python version of this
    % h function.

    % Have MATLAB automatically ensure that each actual alpha argument passed to
    % this function is a finite, real scalar.
    arguments
        alpha {mustBeScalarOrEmpty, mustBeNonempty, mustBeReal, mustBeFinite}
    end

    function [h] = h_squared_diff_from_mean(F, alpha)
        h = sum((F - 1 / length(F) * sum(F)).^2) - alpha * (1 / length(F) * sum(F))^2;
    end

    function [G, H] = combine_squared_diff_from_mean(Cres, Gres, Hres, alpha)
        [n, ~, m] = size(Hres);

        m_sumF = mean(Cres);
        m_sumG = 1 / m * sum(Gres, 2);
        m_sumH = 1 / m * sum(Hres, 3);

        G = zeros(n, 1);
        for i = 1:m
            G = G + (Cres(i) - m_sumF) * (Gres(:, i) - m_sumG);
        end
        G = 2 * G - 2 * alpha * m_sumF * m_sumG;

        H = zeros(n, n);
        for i = 1:m
            H = H + (Cres(i) - m_sumF) * Hres(:, :, i) + (Gres(:, i) - m_sumG) * (Gres(:, i) - m_sumG)';
        end
        H = 2 * H;

        H = H - (2 * alpha) * (m_sumF * m_sumH + m_sumG * m_sumG');

        % [grad, Hess] = matlab_symbolic_grad(Cres,Gres,Hres);
    end

    hfun = @(F) h_squared_diff_from_mean(F, alpha);
    combinemodels = @(Cres, Gres, Hres) combine_squared_diff_from_mean(Cres, Gres, Hres, alpha);
end
