% This function is called to cleanup X and F, set the exit value, and display
% reason for exiting

% flag    [dbl] Termination criteria flag:
%               = 0 normal termination because of grad,
%               > 0 exceeded nfmax evals,   flag = norm of grad at final X
%               = -1 if input was fatally incorrect (error message shown)
%               = -2 model failure
%               = -3 error from TRSP Solver or if a NaN was encountered
%               = -4 error in TRSP Solver

function [X, F, exit_flag] = clean_up_before_return(X, F, nf, exit_flag)

    X = X(1:nf, :);
    F = F(1:nf, :);

    if exit_flag == -4
        disp("An occured in minq. Exiting.");
    elseif exit_flag == -3
        disp("A NaN was encountered in an objective evaluation. Exiting.");
    elseif exit_flag == -2
        disp('Terminating because mdec == 0 with a valid model and no improvement from TRSP solution');
    elseif exit_flag == 0
        disp('g is sufficiently small');
    end
