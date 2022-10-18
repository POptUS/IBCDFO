function [G, H] = emittance(Cres, Gres, Hres)

[n, ~, m] = size(Hres);

% assert(m==3, "Emittance calculation requires exactly three quantities");

G = Cres(1) * Gres(:, 2) + Cres(2) * Gres(:, 1) - 2 * Cres(end) * Gres(:, end);
H = Cres(1) * Hres(:, :, 2) + Cres(2) * Hres(:, :, 1) + Gres(:, 2) * Gres(:, 1)' + Gres(:, 1) * Gres(:, 2)' - 2 * Cres(end) * Hres(:, :, end) - 2 * Gres(:, end) * Gres(:, end)';

% disp(H)
% disp(eigs(H))
% disp(eigs(Hres(:,:,1)))
% disp(eigs(Hres(:,:,2)))
% disp(eigs(Hres(:,:,3)))
