function [G, H] = combine_squared_diff_from_mean(Cres, Gres, Hres, alpha)
% Please refer to the documentation for the Python version of this
% function as well as the documentation for ``h_squared_diff_from_mean.m``.
[n, ~, m] = size(Hres);

m_sumF = mean(Cres);
m_sumG = 1 / m * sum(Gres, 2);
sumH = sum(Hres, 3);

G = zeros(n, 1);
for i = 1:m
    G = G + (Cres(i) - m_sumF) * (Gres(:, i) - m_sumG);
end
G = 2 * G - 2 * alpha * m_sumF * m_sumG;

H = zeros(n, n);
for i = 1:m
    H = H + (Cres(i) - m_sumF) * (Hres(:, :, i) + sumH) + (Gres(:, i) - m_sumG) * (Gres(:, i) - m_sumG)';
end
H = 2 * H;

H = H - (2 * alpha / m) * m_sumF * sumH - (2 * alpha) * m_sumG * m_sumG';

% [grad, Hess] = matlab_symbolic_grad(Cres,Gres,Hres);
