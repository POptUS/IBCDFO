function [G, H] = leastsquares(Cres, Gres, Hres)
[n, ~, m] = size(Hres);

G = 2 * Gres * Cres';
H = zeros(n);
for i = 1:m
    H = H + Cres(i) * Hres(:, :, i);
end
H = 2 * H + 2 * (Gres * Gres');
