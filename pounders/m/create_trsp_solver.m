function [solver] = create_trsp_solver(spsolver)
    % Please refer to the documentation for the Python version of this
    % function.

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
    function [Xsp, mdec, trsp_err] = bqmin_wrapper(H, G, Low, Upp)
        % Assume that solver error checks its arguments thoroughly and that
        % solver always finds valid solution.
        trsp_err = 0;
        [Xsp, mdec] = bqmin(H, G, Low, Upp);
    end

    % Arnold Neumaier's minq5
    function [Xsp, mdec, trsp_err] = minq5_wrapper(H, G, Low, Upp)
        % Assume that solver error checks its arguments thoroughly.
        xx = zeros(size(H, 1), 1);
        [Xsp, mdec, trsp_err] = minqsw(0, G, H, Low', Upp', 0, xx);
        % Continuous function restricted to (compact) k-cell.
        assert(trsp_err ~= 1);
        % See comments in Python version of this function for info on handling
        % error code 99.
        % assert(trsp_err ~= 99);
    end

    % Arnold Neumaier's minq8
    function [Xsp, mdec, trsp_err] = minq8_wrapper(H, G, Low, Upp)
        % Assume that solver error checks its arguments thoroughly and that
        % solver always finds valid solution.
        trsp_err = 0;

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
