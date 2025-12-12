function [G, H] = neg_leastsquares(Cres, Gres, Hres)
[G, H] = leastsquares(Cres, Gres, Hres);
G = -G;
H = -H;
