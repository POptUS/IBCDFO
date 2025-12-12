function [Gout, Hout] = combine_identity(~, Gres, Hres)
    Gout = squeeze(Gres);
    Hout = squeeze(Hres);
end
