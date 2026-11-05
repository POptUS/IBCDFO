% Checks that evaluate_points_to_force_valid_model never requests a point that
% violates the bound constraints (so no clipping is needed in call_user_scripts).
function test_evaluate_points_bounds()
    [here_path, ~, ~] = fileparts(mfilename('fullpath'));
    ms_path = fullfile(here_path, '..');
    pounders_path = fullfile(here_path, '..', '..', '..', 'pounders', 'm');

    % Shadow call_user_scripts with a spy that records the *unclipped* request
    spy_dir = tempname;
    mkdir(spy_dir);
    fid = fopen(fullfile(spy_dir, 'call_user_scripts.m'), 'w');
    fprintf(fid, '%s\n', ...
        'function [nf, X, F, h, Hash, hashes_at_nf] = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, x_in, tol, L, U, allow_recalc)', ...
        '    global REQUESTED_POINTS', ...
        '    REQUESTED_POINTS(end + 1, :) = x_in(:)'';', ...
        '    nf = nf + 1; X(nf, :) = x_in; F(nf, :) = x_in; hashes_at_nf = {};', ...
        'end');
    fclose(fid);
    oldpath = addpath(spy_dir, ms_path, pounders_path);
    cleanup = onCleanup(@() restore_and_remove(oldpath, spy_dir));
    rehash;

    inf3 = inf(1, 3);
    % {incumbent, L, U, delta}
    cases = {
        {[0 0 0], zeros(1, 3), ones(1, 3), 0.1}              % at lower bound
        {[1 1 1], zeros(1, 3), ones(1, 3), 0.1}              % at upper bound ("always +" would fail)
        {[-1 1 0], -ones(1, 3), ones(1, 3), 0.25}            % mixed corner
        {[1 0 0.5], [0 0 0], [1 0.01 1], 0.5}                % narrow box -> coordinate fallback
        {[0 0 0], -inf3, inf3, 0.1}                          % unbounded
    };

    for c = 1:numel(cases)
        [xk, L, U, delta] = cases{c}{:};
        global REQUESTED_POINTS
        REQUESTED_POINTS = zeros(0, 3);
        n = 3;
        nf_max = 50;
        fq_pars.Par = [sqrt(n), max(10, sqrt(n)), 0.001, 0.001, false];
        fq_pars.npmax = 2 * n + 1;
        X = zeros(nf_max, n); X(1, :) = xk;
        F = X; h = zeros(nf_max, 1); Hash = {};
        tol.hfun_test_mode = 0;
        evaluate_points_to_force_valid_model(n, 1, 1, delta, X, F, h, eye(n), 0, [], [], Hash, fq_pars, tol, nf_max, L, U);

        assert(size(REQUESTED_POINTS, 1) > 0, 'case %d: no points requested', c);
        assert(all(all(REQUESTED_POINTS >= L - 1e-15)), 'case %d: lower bound violated', c);
        assert(all(all(REQUESTED_POINTS <= U + 1e-15)), 'case %d: upper bound violated', c);
    end
    disp('test_evaluate_points_bounds: all cases passed');
end

function restore_and_remove(oldpath, spy_dir)
    path(oldpath);
    rmdir(spy_dir, 's');
end
