function [fvec, J] = blackbox(X)

% Get data from outside of this function
global nprob n m n_fvec Fvecs_mat n_fvec_grads X_mat

% Call the fvec function
if nargout <= 1
    [~, fvec] = jacobian(m, n, X(:), nprob);
    if max(abs(fvec)) > 1e64 % This value should match jacobian.m
        fvec(abs(fvec) > 1e64) = sign(fvec(abs(fvec) > 1e64)) * 1e64;
    end
    if any(isnan(fvec))
        fvec(isnan(fvec)) = 1e64;
    end
    J = 0;
elseif nargout == 2
    [J, fvec] = jacobian(m, n, X(:), nprob);
    n_fvec_grads = n_fvec_grads + 1;
end
fvec = fvec(:)';
if ~isempty(n_fvec)
    n_fvec = n_fvec + 1;
    Fvecs_mat(n_fvec, :) = fvec;
    X_mat(n_fvec, :) = X;
end
