% This function writes a GDX file of the C and D data that defined the
% censored_L1_loss objective

%%%%%%%%%%%
% Inputs:
%%%%%%%%%%%
% C:       [p-by-1] array of p censors
% D:       [p-by-1] array of p data

function [] = save_censored_L1_loss_data(C, D)

    % Put all of the problem information into the correct data structures

    cardI = length(C);

    % C and D have p elements; the GAMS model (.gms file) indexes them by its set 'P'.
    Is.name = 'I';
    Is.ts   = 'set of quadratic functions';
    Is.type = 'set';
    Is.uels = {1:cardI};

    Cs.name = 'c';
    Cs.ts   = 'censor terms';
    Cs.type = 'parameter';
    Cs.val = C;
    Cs.form = 'full';
    Cs.dim = 1;
    Cs.uels = Is.uels;

    Ds.name = 'd';
    Ds.ts   = 'data terms';
    Ds.type = 'parameter';
    Ds.val = D;
    Ds.form = 'full';
    Ds.dim = 1;
    Ds.uels = Is.uels;

    wgdx ('censored_L1_loss_data', Cs, Ds);
    fprintf('Data written to GDX file censored_L1_loss_data.gdx\n');

end
