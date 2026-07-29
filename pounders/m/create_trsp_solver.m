function [solver] = create_trsp_solver(spsolver)
    % Please refer to the documentation for the Python version of this
    % function.
    %
    % The returned solver satisfies the interface
    %
    % .. code:: matlab
    %
    %     [Xsp, mdec, found_solution] = solve_trsp(H, G, Low, Upp);
    %
    % where
    %
    % * ``H`` is an :math:`\np \times \np` matrix that provides the
    %   (symmetric) Hessian of the objective function,
    % * ``G`` is :math:`\np \times 1` vector that provides the
    %   gradient of the objective function,
    % * ``Low`` and ``Upp`` are :math:`1 \times \np` vectors that specify
    %   the bound constraints,
    % * ``Xsp`` :math:`\np \times 1` subproblem solution vector,
    % * ``mdec`` is the value of the subproblem objective function at
    %   the solution as a real scalar, and
    % * ``found_solution`` is True if a solution was found that should be
    %   acceptable for |pounders|'s purposes; False, otherwise.

    arguments
        spsolver {mustBeScalarOrEmpty, mustBeNonempty, mustBeInteger}
    end

    % ----- HARDCODED VALUES
    % Ensure that these match the analogous constants implemented for
    % POUNDERS/Python.  These same values might be used in the code that
    % directly tests this function.
    %
    % Both MATLAB and Python implementations should declare the union of all
    % solvers available even if they don't support one or more of the solvers.
    TRSP_SOLVER_SIMPLE = 1;
    TRSP_SOLVER_MINQ5 = 2;
    TRSP_SOLVER_MINQ8 = 3;

    % ----- DEFINE POUNDERS-COMPATIBLE INTERFACES ON SOLVERS
    % Stefan's crappy 10 line solver
    function [Xsp, mdec, found_solution] = bqmin_wrapper(H, G, Low, Upp)
        % Assume that solver error checks its arguments thoroughly and that
        % solver always finds valid solution.
        found_solution = true;
        [Xsp, mdec] = bqmin(H, G, Low, Upp);
    end

    % Arnold Neumaier's minq5
    function [Xsp, mdec, found_solution] = minq5_wrapper(H, G, Low, Upp)
        % Assume that solver error checks its arguments thoroughly.
        xx = zeros(size(H, 1), 1);
        [Xsp, mdec, minq_err] = minqsw(0, G, H, Low', Upp', 0, xx);
        % Continuous function restricted to (compact) k-cell.
        assert(minq_err ~= 1);
        % See comments in Python version of this function for info on handling
        % error code 99.
        % assert(minq_err ~= 99);
        found_solution = (minq_err >= 0);
    end

    % Arnold Neumaier's minq8
    function [Xsp, mdec, found_solution] = minq8_wrapper(H, G, Low, Upp)
        % Assume that solver error checks its arguments thoroughly and that
        % solver always finds valid solution.
        found_solution = true;

        n = size(H, 1);

        data.gam = 0;
        data.c = G;
        data.b = zeros(n, 1);
        [tmp1, tmp2] = ldl(H);
        data.D = diag(tmp2);
        data.A = tmp1';

        [Xsp, mdec] = minq8(data, Low', Upp', zeros(n, 1), 10 * n);
    end

    % ----- IDENTIFY DESIRED SOLVER
    if spsolver == TRSP_SOLVER_SIMPLE
        warning("POUNDERS:simpleTrspSolver", ...
                "The simple TRSP solver should only be used for testing or debugging");
        solver = @bqmin_wrapper;
    elseif spsolver == TRSP_SOLVER_MINQ5
        check_minq_installation(5);
        solver = @minq5_wrapper;
    elseif spsolver == TRSP_SOLVER_MINQ8
        check_minq_installation(8);
        solver = @minq8_wrapper;
    else
        error('POUNDERS:badValue', sprintf("Invalid TRSP solver %d", spsolver));
    end
end
