function [y] = calfun_wrapper_y(x, struct, probtype)
[y, fvec, ~] = calfun(x, struct, probtype);
end
