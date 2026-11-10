% This function is called by manifold_sampling_primal and goombah to trim X, F,
% and hF to the nf points that were evaluated, to set the exit flag, and to
% display the reason for exiting.
%
% :param X, F, hF: Storage allocated for nf_max points
% :param nf:       [int] Number of points evaluated
% :param exit_flag: [dbl] Termination criteria flag:
%
%     * 0 - normal termination: stationary measure and radius met their tolerances
%     * > 0 - the budget of function evaluations was exhausted; the value is the
%       final stationary measure
%     * -1 - model construction failed (an empty or invalid local model)
%     * -2 - trust-region subproblem failed
%     * -6 - the trust-region radius reached delta_min without satisfying g_tol
%
% :param printf:   [int] No message is displayed if 0

function [X, F, hF, exit_flag] = prepare_ms_outputs_before_return(X, F, hF, nf, exit_flag, printf)

    X = X(1:nf, :);
    F = F(1:nf, :);
    hF = hF(1:nf, :);

    if printf
        if exit_flag == 0
            disp('Convergence satisfied: small stationary measure and small delta');
        elseif exit_flag > 0
            disp('Number of function evals exceeded');
        elseif exit_flag == -1
            disp('Model building failed. Empty Gres. Exiting.');
        elseif exit_flag == -2
            disp('Trust-region subproblem failed. Exiting.');
        elseif exit_flag == -6
            disp('Terminating because delta_min reached without satisfying g_tol.');
        end
    end
end
