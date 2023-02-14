% This function is called to cleanup X and F, set the exit value, and display
% reason for exiting.

function [X, F, exit_flag] = clean_up_before_return(X, F, nf, exit_flag)

    X = X(1:nf, :);
    F = F(1:nf, :);

    if exit_flag == -4
        disp("A minq input error occurred. Exiting.");
    elseif exit_flag == -3
        disp("A NaN was encountered in an objective evaluation. Exiting.");
    elseif exit_flag == -2
        disp('Terminating because mdec == 0 with a valid model and no improvement from TRSP solution');
    elseif exit_flag == 0
        disp('g is sufficiently small');
    end
