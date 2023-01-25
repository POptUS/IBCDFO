function best_lambda = project_zero_onto_convex_hull_2(G_k)

[G_k_smaller, cols] = uniquetol(G_k', 'ByRows', true);
G_k_smaller = G_k_smaller';

[n, p] = size(G_k_smaller);

best_norm = inf;
for i = 1:4
    if i == 1 || i == 3
        solveSubproblem_tol = 1e-8; % [dbl]
    else
        solveSubproblem_tol = 1e-16; % [dbl]
    end
    if i == 3
        G_k_smaller  = G_k_smaller / norm(G_k_smaller);
    end

    [lambda_smaller, ~, flag] = solveSubproblem(eye(n), G_k_smaller, zeros(p, 1), ones(p, 1) / p, solveSubproblem_tol);

    lambda = sparse(size(G_k, 2), 1);
    lambda(cols) = lambda_smaller;
    flags(i) = flag;

    nval = norm(G_k * lambda);
    if ~isnan(nval) && nval < best_norm && abs(sum(lambda_smaller) - 1) <= 1e-8 && all(lambda_smaller) >= 0
        best_lambda = lambda;
        best_norm = nval;
        best_flag = flag;
    end
end
assert(~isinf(best_norm), "This shouldn't happen. Never solved one of our four calls.");
if best_flag ~= 1
    if best_flag == 5
        disp('Some sign error in subproblem calculation; continuing');
    elseif best_flag == 2
        disp('Iteration limit reached when finding min-norm element of G_k, continuing with what was returned');
    elseif best_flag == 4
        disp('Error in the linear solve in the best_norm point. But other tests mean that this is okay to proceed with.');
    else
        error('Finding minimum norm element in conv(Gen^k) did not terminate happily');
    end
end
end
