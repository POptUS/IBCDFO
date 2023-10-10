% bqmin.m, Version 0.1,    Modified 12/1/09  Stefan Wild
%
%  [X,f] = bqmin(A,B,L,U)
%
%  Minimizes the quadratic .5*X'*A*X + B subject to L<=X<=U using the
%  projected gradient method with a (semi) exact line search.
%  This will one day be replaced by a more efficient solver.
%  This approach is not recommended for n>100.
%
% --INPUTS-----------------------------------------------------------------
% A       [dbl] [n-by-n] (Symmetric) Hessian matrix
% B       [dbl] [n-by-1] Gradient vector
% L       [dbl] [1-by-n] Vector of lower bounds assumed to be nonpositive
% U       [dbl] [1-by-n] Vector of upper bounds, must have U(j)>=0>=L(j)
%
% --OUTPUTS----------------------------------------------------------------
% X       [dbl] [n-by-1] Approximate solution
% f       [dbl] Function value at X
%
function [X, f] = bqmin(A, B, L, U)
% --INTERMEDIATE-----------------------------------------------------------
% G       [dbl] [n-by-1]  Gradient at X
% it      [dbl] Iteration counter
% pap     [dbl] The A norm of the projected gradient
% Projg   [dbl] [n-by-1]  Projected gradient at X
% t       [dbl] Step length along projected gradient
%
% --INTERNAL PARAMETERS----------------------------------------------------
n = size(A, 2); % [int] Dimension (number of continuous variables)
maxit = 5000; % [int] maximum number of iterations
pgtol = 1e-13; % [dbl] tolerance on final projected gradient
% -------------------------------------------------------------------------

% Make everything a column vector:
B = B(:);
L = L(:);
U = U(:);

% Initial point (assumed feasible by L<=0<=U )
X = zeros(n, 1);
f =  X' * (.5 * A * X + B);
G = A * X + B;
Projg = X - max(min(X - G, U), L); % Projected gradient

it = 0; % Iteration counter
while it < maxit && norm(Projg) > pgtol
    it = it + 1;

    % Simple line search along the projected gradient
    t = 1; % By default take the full step
    pap = Projg' * A * Projg;
    if pap > 0
        t = min(1, (Projg' * G) / pap);
    end

    % Compute the next point and update everything
    X = X - t * Projg;
    f = X' * (.5 * A * X + B);
    G = A * X + B;
    Projg = X - max(min(X - G, U), L);
end
