function [h] = h_leastsquares(F)
    % Please refer to the documentation for the Python version of this
    % h function.
    h = sum(F.^2);
end
