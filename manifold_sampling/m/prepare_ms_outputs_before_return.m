% This function is called by manifold_sampling_primal and goombah to trim X, F,
% and hF to the nf points that were evaluated, to set the exit flag, and to
% display the reason for exiting.
%
% :param X, F, hF: Storage allocated for nf_max points
% :param nf:       [int] Number of points evaluated
% :param exit_flag: [dbl] Termination criteria flag:
%
%     * 0 - normal termination: stationary measure and radius met their tolerances
%     * 1 - the budget of function evaluations was exhausted
%     * -1 - model construction failed (an empty or invalid local model)
%     * -2 - trust-region subproblem failed
%     * -6 - the trust-region radius reached delta_min without satisfying g_tol.
%       This is how many successful manifold sampling runs end, so the final
%       stationary measure chi_k is displayed to help judge the result.
%
% :param printf:   [int] No message is displayed if 0
% :param chi_k:    [dbl] (optional) Final stationary measure, displayed with
%     exit_flag -6

function [X, F, hF, exit_flag] = prepare_ms_outputs_before_return(X, F, hF, nf, exit_flag, printf, chi_k)

    X = X(1:nf, :);
    F = F(1:nf, :);
    hF = hF(1:nf, :);

    if printf
        if exit_flag == 0
            disp('Convergence satisfied: small stationary measure and small delta');
        elseif exit_flag == 1
            disp('Number of function evals exceeded');
        elseif exit_flag == -1
            disp('Model building failed. Empty Gres. Exiting.');
        elseif exit_flag == -2
            disp('Trust-region subproblem failed. Exiting.');
        elseif exit_flag == -6
            if nargin >= 7
                disp(['Terminating because delta_min reached without satisfying g_tol. Final chi_k = ' num2str(chi_k, '%.3e')]);
            else
                disp('Terminating because delta_min reached without satisfying g_tol.');
            end
        end
    end
end
