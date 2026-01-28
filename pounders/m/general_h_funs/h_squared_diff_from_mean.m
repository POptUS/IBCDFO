function [h] = h_squared_diff_from_mean(F, alpha)
    % Please refer to the documentation for the Python version of this
    % h function.
    %
    % Users should construct |pounders|-compatible versions of this function
    % and its combinemethods function with code such as
    %
    % .. code:: matlab
    %
    %   ALPHA = X.Y;
    %   hfun = @(F) h_squared_diff_from_mean(F, ALPHA);
    %   combinemethods = @(Cres, Gres, Hres) combine_squared_diff_from_mean(Cres, Gres, Hres, ALPHA);
    %
    h = sum((F - 1 / length(F) * sum(F)).^2) - alpha * (1 / length(F) * sum(F))^2;
end
