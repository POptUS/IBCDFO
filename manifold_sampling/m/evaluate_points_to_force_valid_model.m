function [X, F, hF, nf, Hash] = evaluate_points_to_force_valid_model(n, nf, xk_in, delta, X, F, hF, Mdir, np, hfun, Ffun, Hash, fq_pars, tol, nf_max, Low, Upp)
    % global nprob Qs zs bs p x0 nf_max h_activity_tol row_in_dfo_dat s inst
    % A.n=n; A.nf=nf; A.xk_in=xk_in; A.delta=delta; A.X=X; A.F=F; A.hF=hF; A.Mdir=Mdir; A.np=np; A.hfun=hfun; A.Ffun=Ffun; A.Hash=Hash; A.fq_pars=fq_pars; A.tol=tol; A.nf_max=nf_max; A.Low=Low; A.Upp=Upp;

    % Evaluate model-improving points to pick best one
    % ! May eventually want to normalize Mdir first for infty norm
    % bmpts respects the bounds Low and Upp: for each direction it picks the sign (+/-)
    % with the larger feasible step and scales the step to stay inside the box (or
    % falls back to coordinate directions), so every X(xk_in, :) + Xsp below is
    % bound-feasible. The sign is chosen geometrically, not by predicted model decrease.
    % (Maybe a model-informed sign choice could be added here?)
    [Mdir1, np1] = bmpts(X(xk_in, :), Mdir(1:n - np, :), Low, Upp, delta, fq_pars.Par(3));
    for i = 1:n - np1
        Xsp = Mdir1(i, :);
        % Only do this evaluation if the point is new and nf < nf_max
        if ~ismember(X(xk_in, :) + Xsp, X(1:nf, :), 'rows') && nf < nf_max
            [nf, X, F, hF, Hash] = call_user_scripts(nf, X, F, hF, Hash, Ffun, hfun, X(xk_in, :) + Xsp, tol, Low, Upp, 1);
        end
    end
    [~, ~, valid] = formquad(X(1:nf, :), F(1:nf, :), delta, xk_in, fq_pars.npmax, fq_pars.Par, 1);
    if ~valid && nf < nf_max
        disp(nf);
        disp('Proceeding with nonvalid model! Report this to Stefan in Alg1');
        % uuid = char(java.util.UUID.randomUUID);
        % global mw_prob_num hfun
        % save(['first_failure_for_row_in_dfo_dat=' int2str(mw_prob_num) '_hfun=' func2str(hfun{1}) ], 'A');
        % error('a')
    end
end
