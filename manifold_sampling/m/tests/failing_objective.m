function [fvec] = failing_objective(x)
% This function will produce a NaN in one component on average in 1 out of 10
% queries, otherwise, it produces returns x.

fvec = x;

if rand < 0.1
    fvec(1) = NaN;
end
