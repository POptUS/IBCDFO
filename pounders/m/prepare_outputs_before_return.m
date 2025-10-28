% This function is called to cleanup X and F, set the exit value, and display
% reason for exiting.

function [X, F, hF, exit_flag] = prepare_outputs_before_return(X, F, hF, nf, exit_flag)

    X = X(1:nf, :);
    F = F(1:nf, :);
    hF = hF(1:nf);

    if exit_flag == 0
        disp('g is sufficiently small');
    elseif exit_flag == -1
        disp('Number of residuals in output of fun does not match supplied m. Exiting.');
    elseif exit_flag == -2
        disp('Terminating because no improvement from TRSP solution (mdec == 0) with a valid model and small delta.');
    elseif exit_flag == -3
        disp("A NaN was encountered in an objective evaluation. Exiting.");
    elseif exit_flag == -4
        disp("A minq input error occurred. Exiting.");
    elseif exit_flag == -5
        disp('Unable to improve model with current Pars; try dividing Par(3:4) by 10.');
    elseif exit_flag == -6
        disp('Terminating because delta_min reached with a valid model.');
    end
