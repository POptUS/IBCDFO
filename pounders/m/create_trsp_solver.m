function [solver] = create_trsp_solver(spsolver)
    %
    % :param spsolver:

    % ----- HARDCODED VALUES
    TRSP_BQMIN = 1;
    TRSP_MINQ5 = 2;
    TRSP_MINQ8 = 3;

    % ----- DEFINE POUNDERS-COMPATIBLE INTERFACES ON SOLVERS
    % Stefan's crappy 10 line solver
    function [Xsp, mdec, trsp_err] = bqmin_wrapper(G, H, Low, Upp)
        trsp_err = 0;
        [Xsp, mdec] = bqmin(H, G, Low, Upp);
    end

    % Arnold Neumaier's minq5
    function [Xsp, mdec, trsp_err] = minq5_wrapper(G, H, Low, Upp)
        xx = zeros(size(H, 1), 1);
        [Xsp, mdec, minq_err] = minqsw(0, G, H, Low, Upp, 0, xx);
        if minq_err < 0
            trsp_err = -4;
        else
            trsp_err = 0;
        end
    end

    % Arnold Neumaier's minq8
    function [Xsp, mdec, trsp_err] = minq8_wrapper(G, H, Low, Upp)
        trsp_err = 0;

        n = size(H, 1);

        data.gam = 0;
        data.c = G;
        data.b = zeros(n, 1);
        [tmp1, tmp2] = ldl(H);
        data.D = diag(tmp2);
        data.A = tmp1';

        [Xsp, mdec] = minq8(data, Low, Upp, zeros(n, 1), 10 * n);
    end

    % ----- IDENTIFY DESIRED SOLVER
    if spsolver == TRSP_BQMIN
        solver = @bqmin_wrapper;
    elseif spsolver == TRSP_MINQ5
        check_minq_installation(5);
        solver = @minq5_wrapper;
    elseif spsolver == TRSP_MINQ8
        check_minq_installation(8);
        solver = @minq8_wrapper;
    else
        error(sprintf("Invalid TRSP solver %d", spsolver));
    end
end
