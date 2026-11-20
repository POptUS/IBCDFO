function x = snap_to_bounds(x, Low, Upp, snap_tol, what)
    % Put ``x`` back inside ``[Low, Upp]``, complaining if it was not already there.
    %
    % Everything checked here comes from a subproblem solver or a user script,
    % so a large violation means something is wrong and silently snapping
    % would hide it (it has hidden real bugs before). How far outside decides
    % what happens:
    %
    %   * up to ~2e-13: snap, and say nothing
    %   * up to **snap_tol**: warn, then snap
    %   * beyond that: error
    %
    % The first threshold is fixed, so setting **snap_tol** below it buys
    % nothing: violations that small are always snapped in silence.
    %
    % This compares absolute violations and does not scale with ``x``. On a
    % problem whose variables are large, one unit in the last place of ``x``
    % already exceeds the default **snap_tol**, so a correct solver will trip
    % this check on rounding alone unless **snap_tol** is raised to suit.
    %
    % :param x:        [dbl] Point or step to check; a row per point.
    % :param Low:      [dbl] Lower bounds on ``x``.
    % :param Upp:      [dbl] Upper bounds on ``x``.
    % :param snap_tol: [dbl] Largest violation that is snapped rather than
    %     refused. The solvers default this to 1e-6, the feasibility
    %     tolerance that LP/QP solvers such as ``linprog`` promise; tighter
    %     values reject their output.
    % :param what:     [char] Describes ``x``, for the error/warning message.
    %
    % :return: **x** snapped into ``[Low, Upp]``

    assert(snap_tol >= 0, 'IBCDFO:badTolerances', 'snap_tol must be nonnegative');

    % Below this a snap is not worth mentioning: it is the slop that a solver
    % leaves on a step it considers feasible.
    noise_tol = 1e3 * eps;

    viol = max([Low - x, x - Upp], [], 'all');

    if viol > noise_tol
        if viol > snap_tol
            error('IBCDFO:boundViolation', ...
                  '%s violates its bounds by %g, which exceeds snap_tol = %g.', ...
                  what, viol, snap_tol);
        end
        warning('IBCDFO:boundViolation', ...
                '%s violates its bounds by %g; snapping it onto the bounds.', ...
                what, viol);
    end

    x = min(Upp, max(Low, x));
end
