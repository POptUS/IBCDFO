function [G, H] = combine_neg_leastsquares(Cres, Gres, Hres)
[G, H] = combine_leastsquares(Cres, Gres, Hres);
G = -G;
H = -H;
