% formquad.m, Version 0.1, Modified 3/2/10
% Stefan Wild and Jorge More', Argonne National Laboratory.
%
%  [Mdir,np,valid,G,H,Mind] = formquad(X,F,delta,xkin,npmax,Pars,vf)
%
% Computes the parameters for m quadratics
%       Q_i(x) = C(i) + G(:,i)'*x + 0.5*x'*H(:,:,i)*x,  i=1:m
% whose Hessians H are of least Frobenius norm subject to the interpolation
%       Q_i(X(Mind,:)) = F(Mind,i).
%
% The procedure works equally well with m=1 and m>1.
% The derivation is involved but may be found in "MNH: A Derivative-Free
% Optimization Algorithm Using Minimal Norm Hessians" by S Wild, 2008.
%
% --INPUTS-----------------------------------------------------------------
% X       [dbl] [nf-by-n] Locations of evaluated points
% F       [dbl] [nf-by-m] Function values of evaluated points
% delta   [dbl] Positive trust region radius
% xkin    [int] Index in (X and F) of the current center
% npmax   [int] Max # interpolation points (>=n+1) (.5*(n+1)*(n+2))
% Pars(1) [dbl] delta multiplier for checking validity
% Pars(2) [dbl] delta multiplier for all interpolation points
% Pars(3) [dbl] Pivot threshold for validity
% Pars(4) [dbl] Pivot threshold for additional points (.001)
% vf      [log] Flag indicating you just want to check model validity
%
% --OUTPUTS----------------------------------------------------------------
% Mdir    [dbl] [(n-np+1)-by-n]  Unit directions to improve model
% np      [int] Number of interpolation points (=length(Mind))
% valid   [log] Flag saying if model is valid within Pars(2)*delta
% G       [dbl] [n-by-m]  Matrix of model gradients at centered at X(xkin, :)
% H       [dbl] [n-by-n-by-m]  Array of model Hessians centered at X(xkin, :)
% Mind    [int] [npmax-by-1] Integer vector of model interpolation indices
%
function [Mdir, np, valid, G, H, Mind] = formquad(X, F, delta, xkin, npmax, Pars, vf)

% --DEPENDS ON-------------------------------------------------------------
% phi2eval : Evaluates the quadratic basis for vector inputs
% qrinsert, svd : Standard internal Matlab/Octave/LAPACK commands

% Internal parameters:
[nf, n] = size(X);
m = size(F, 2);

G = zeros(n, m);
H = zeros(n, n, m);

% Precompute the scaled displacements (could be expensive for larger nfmax)
D = zeros(nf, n); % Scaled displacements
Nd = zeros(nf, 1); % Norm of scaled displacements
for i = 1:nf
    D(i, :) = (X(i, :) - X(xkin, :)) / delta;
    Nd(i) = norm(D(i, :));
end

% Get n+1 sufficiently affinely independent points:
Q = eye(n);
R = []; % Initialize the QR factorization of interest
Mind = xkin; % Indices of model interpolation points
valid = false;
np = 0;  % Counter for number of interpolation points
for aff = 1:2
    for i = nf:-1:1
        if Nd(i) <= Pars(aff)
            proj = norm(D(i, :) * Q(:, np + 1:n), 2); % Project D onto null
            if proj >= Pars(aff + 2)  % add this index to Mind
                np = np + 1;
                Mind(np + 1, 1) = i;
                % MATLAB:
                % [Q,R] = qrinsert(Q,R,np,D(i,:)'); % Update QR
                % This bit is just for Octave
                if size(R, 1) == 0
                    [Q, R] = qr(D(i, :)');
                else
                    [Q, R] = qrinsert(Q, R, np, D(i, :)'); % Update QR
                end
                if np == n
                    break  % Breaks out of for loop
                end
            end
        end
    end

    if aff == 1 && np == n % Have enough points:
        Mdir = [];
        valid = true;
        break
    elseif aff == 2 && np < n % Need to evaluate more points, then recall
        Mdir = Q(:, np + 1:n)';  % Output Geometry directions
        G = [];
        H = [];
        return
    elseif aff == 1 % Output model-improving directions
        Mdir = Q(:, np + 1:n)';  % Will be empty if np=n
    end
    if vf % Only needed to do validity check
        return
    end
end

% Collect additional points
N = zeros(.5 * n * (n + 1), n + 1);
for np = 1:n + 1
    N(:, np) = phi2eval(D(Mind(np), :))';
end

M = [ones(n + 1, 1) D(Mind, :)]';
[Q, R] = qr(M');

% Now we add points until we have npmax starting with the most recent ones
i = nf;
while np < npmax || npmax == n + 1
    if Nd(i) <= Pars(2) && ~ismember(i, Mind)
        Ny = [N phi2eval(D(i, :))'];
        [Qy, Ry] = qrinsert(Q, R, np + 1, [1 D(i, :)], 'row'); % Update QR
        Ly = Ny * Qy(:, n + 2:np + 1);

        if min(svd(Ly)) > Pars(4)
            np = np + 1;
            Mind(np, 1) = i;
            N = Ny;
            Q = Qy;
            R = Ry;
            L = Ly;

            Z = Q(:, n + 2:np);
            M = [M [1; D(i, :)']]; % Note that M is growing
        end
    end

    i = i - 1;
    if i == 0 % Reached end of points
        if np == (n + 1) % Set outputs so that Hessian is zero
            L = 1;
            Z = zeros(n + 1, .5 * n * (n + 1));
            N = zeros(.5 * n * (n + 1), n + 1);
        end
        break
    end
end

F = F(Mind, :);
for k = 1:m
    % For L=N*Z, solve L'*L*Omega = Z'*f:
    Omega = L' \ (Z' * F(:, k));
    Omega = L \ Omega;
    Beta = L * Omega;
    Alpha = M' \ (F(:, k) - N' * Beta);

    G(:, k) = Alpha(2:n + 1);
    num = 0;
    for i = 1:n
        num = num + 1;
        H(i, i, k) = Beta(num);
        for j = i + 1:n
            num = num + 1;
            H(i, j, k) = Beta(num) / sqrt(2);
            H(j, i, k) = H(i, j, k);
        end
    end
end
H = H / delta^2;
G = G / delta;
