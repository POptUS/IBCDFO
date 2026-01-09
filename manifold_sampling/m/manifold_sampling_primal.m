function [X, F, h, xkin, flag] = manifold_sampling_primal(hfun, Ffun, x0, L, U, nf_max, subprob_switch)
% Run manifold sampling on the optimization problem specified by the given
% arguments.
%
% :param hfun: Handle to function that, given a :math:`\np \times 1` point
%     :math:`\zvec`, returns
%
%     * the value :math:`h(\zvec)`
%     * :math:`\np \times l` matrix that contains gradients for all
%       :math:`l` limiting gradients at :math:`\zvec`
%     * :math:`1 \times l` vector of hashes for each manifold active at :math:`\zvec`
%
%     Given point :math:`\zvec` and :math:`l` hashes :math:`H`, returns
%
%     * An :math:`1 \times l` vector that contains the values :math:`h_i(\zvec)`
%       for each hash in :math:`H`
%     * A :math:`p \times l` matrix that contains the gradients of
%       :math:`h_i(\zvec)` for each hash in :math:`H`
%
% :param Ffun: Function that returns the value :math:`\Ffun(\psp)` of the
%     blackbox simulation as an :math:`1 \times \nd` vector for given
%     :math:`\psp`
% :param x0: :math:`1 \times \np` vector that specifies the starting point
% :param L: :math:`1 \times \np` vector of lower bounds
% :param U: :math:`1 \times \np` vector of upper bounds
% :param nf_max: [int] Maximum number of function evaluations
% :param subprob_switch: **???**
%
% :return:
%      * **X** - :math:`\mathrm{nf\_max} \times \np` matrix containing the
%        points evaluated in the order in which they were evaluated
%      * **F** - :math:`\mathrm{nf\_max} \times \nd` matrix containing the
%        blackbox function values :math:`\Ffun(\psp)` for all points in ``X``
%        with matching ordering
%      * **h** - :math:`\mathrm{nf\_max} \times 1` matrix containing the values
%        :math:`\hfun(\Ffun(\psp))` for all points in ``X`` with matching
%        ordering.
%      * **xkin** - Current trust region center.  **ONE-BASED INDEX INTO
%        X?**
%      * **flag** - Termination criteria flag (See general documentation)

% Some other values
%  n:       [int]     Dimension of the domain of F (deduced from x0)
%  p:       [int]     Dimension of the domain of h (deduced from evaluating F(x0))
%  delta:   [dbl]     Positive starting trust region radius
% Intermediate Variables:
%   nf    [int]         Counter for the number of function evaluations
%   s_k   [dbl]         Step from current iterate to approx. TRSP solution
%   norm_g [dbl]        Stationary measure ||g||
%   Gres [n x p]        Model gradients for each of the p outputs from Ffun
%   Hres [n x n x p]    Model Hessians for each of the p outputs from Ffun
%   Hash [cell]         Contains the hashes active at each evaluated point in X
%   Act_Z_k [l cell]      Set of hashes for active selection functions in TR
%   G_k  [n x l]        Matrix of model gradients composed with gradients of elements in Act_Z_k
%   D_k  [p x l_2]      Matrix of gradients of selection functions at different points in p-space

% We use POUNDERS checkinputss function
[here_path, ~, ~] = fileparts(mfilename('fullpath'));
addpath(fullfile(here_path, '..', '..', 'pounders', 'm'));

% Deduce p from evaluating Ffun at x0
try
    F0 = Ffun(x0);
catch e
    fprintf(1, 'The identifier was:\n%s\n\n', e.identifier);
    fprintf(1, 'There was an error! The message was:\n%s\n', e.message);
    warning('Problem using Ffun. Exiting');
    X = [];
    F = [];
    h = [];
    xkin = [];
    flag = -1;
    return
end

[n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, successful, xkin, Hres] = check_inputs_and_initialize(x0, F0, nf_max);
[flag, x0, ~, F0, L, U] = checkinputss(hfun, x0, n, fq_pars.npmax, nf_max, tol.gtol, delta, 1, length(F0), F0, xkin, L, U);
if flag == -1 % Problem with the input
    X = x0;
    F = F0;
    h = [];
    return
end

% Evaluate user scripts at x_0
[h(nf), ~, hashes_at_nf] = hfun(F(nf, :));
Hash(nf, 1:length(hashes_at_nf)) = hashes_at_nf;

H_mm = zeros(n);

while nf < nf_max && delta > tol.mindelta
    bar_delta = delta;

    % Line 3: manifold sampling while loop
    while nf < nf_max

        % Line 4: build models
        [Gres, Hres, X, F, h, nf, Hash] = build_p_models(nf, nf_max, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U);
        if isempty(Gres)
            disp(['Model building failed. Empty Gres. Delta = ' num2str(delta)]);
            X = X(1:nf, :);
            F = F(1:nf, :);
            h = h(1:nf, :);
            flag = -1;
            return
        end
        if nf >= nf_max
            flag = 0; % Budget exceeded
            return
        end

        % Line 5: Build set of activities Act_Z_k, gradients D_k, G_k, and beta
        [D_k, Act_Z_k, f_bar] = choose_generator_set(X, Hash, xkin, nf, delta, F, hfun);
        G_k = Gres * D_k;
        beta = max(0, f_bar' - h(xkin));

        % Line 6: Choose Hessians
        H_k = zeros(size(G_k, 2), n + 1, n + 1);
        for i = 1:size(G_k, 2) % would like to vectorize this tensor product ...
            for j = 1:size(Hres, 3) % p
                H_k(i, 2:end, 2:end) = squeeze(H_k(i, 2:end, 2:end)) + D_k(j, i) * Hres(:, :, j);
            end
        end

        % Line 7: Find a candidate s_k by solving QP
        Low = max(L - X(xkin, :), -delta);
        Upp = min(U - X(xkin, :), delta);
        [s_k, tau_k, ~, lambda_k] = minimize_affine_envelope(h(xkin), f_bar, beta, G_k, H_mm, delta, Low, Upp, H_k, subprob_switch);

        % Line 8: Compute stationary measure chi_k
        Low = max(L - X(xkin, :), -1.0);
        Upp = min(U - X(xkin, :), 1.0);
        [~, ~, chi_k] = minimize_affine_envelope(h(xkin), f_bar, beta, G_k, zeros(n), delta, Low, Upp, zeros(size(G_k, 2), n + 1, n + 1), subprob_switch);

        % Lines 9-11: Convergence test: tiny master model gradient and tiny delta
        if chi_k <= tol.gtol && delta <= tol.mindelta
            disp('Convergence satisfied: small stationary measure and small delta');
            X = X(1:nf, :);
            F = F(1:nf, :);
            h = h(1:nf, :);
            flag = chi_k;
            return
        end

        if printf
            trsp_fun = @(x) max_affine(x, h(xkin), f_bar, beta, G_k, H_mm);
%             plot_again_j(X, xkin, delta, s_k, [], nf, trsp_fun, L, U);
        end

        % Line 12: Evaluate F
        [nf, X, F, h, Hash, hashes_at_nf] = ...
            call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X(xkin, :) + s_k', tol, L, U, 1);

        % Line 13: Compute rho_k
        ared = h(xkin) - h(nf);
        pred = -tau_k;
        rho_k = ared / pred;

        % Lines 14-16: Check for success
        if rho_k >= tol.eta1 && pred > 0
            successful = true; % iteration is successful
            break
        else % Line 17: Stay in the manifold sampling loop

            % Line 18: Check temporary activities after adding TRSP solution to X
            [~, tmp_Act_Z_k, ~] = choose_generator_set(X, Hash, xkin, nf, delta, F, hfun);

            % Lines 19: See if any new activities
            if all(ismember(tmp_Act_Z_k, Act_Z_k))

                % Line 20: See if intersection is nonempty
                if any(ismember(hashes_at_nf, Act_Z_k))
                    successful = false; % iteration is unsuccessful
                    break
                else
                    % Line 24: Shrink delta
                    delta = tol.gamma_dec * delta;
                end
            end
        end
    end

    if successful
        xkin = nf; % Line 15: Update TR center and radius
        if rho_k > tol.eta3 && norm(s_k, "inf") > 0.8 * bar_delta
            % Update delta if rho is sufficiently large
            delta = bar_delta * tol.gamma_inc;
            % h_activity_tol = min(1e-8, delta);
        end
    else
        % Line 21: iteration is unsuccessful; shrink Delta
        delta = max(bar_delta * tol.gamma_dec, tol.mindelta);
        % h_activity_tol = min(1e-8, delta);
    end

    fprintf('nf: %8d; fval: %8e; chi: %8e; radius: %8e; \n', nf, h(xkin), chi_k, delta);

    % Build primal master model Hessian for next iteration
    % H_mm = zeros(n);
    % for i = 1:size(G_k,2) % would like to vectorize this tensor product ...
    %    Hg = zeros(n);
    %    for j = 1:size(Hres,3) %p
    %        Hg = Hg + D_k(j,i)*Hres(:,:,j);
    %    end
    %    H_mm = H_mm + lambda_k(i)*Hg;
    % end
end

if nf >= nf_max
    flag = 0; % Budget exceeded
else
    X = X(1:nf, :);
    F = F(1:nf, :);
    h = h(1:nf, :);
    flag = chi_k;
end
end
