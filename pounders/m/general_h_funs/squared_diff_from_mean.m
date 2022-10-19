function [G, H] = squared_diff_from_mean(Cres, Gres, Hres)
% Combines models for the following h function
%    h = @(F)sum((F - 1/m*sum(F)).^2) - alpha*(1/m*sum(F))^2
% That is, the objective is to have the vector close to it's mean, and have
% a small mean (penalized by alpha)
alpha = 0;

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
