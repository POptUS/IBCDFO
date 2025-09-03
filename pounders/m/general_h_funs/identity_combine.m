function [Gout, Hout] = identity_combine(~, Gres, Hres)
    Gout = squeeze(Gres);
    Hout = squeeze(Hres);
end
