% This function writes a GDX file of the Q, z, and c data that defined the
% piecewise quadratic objective

%%%%%%%%%%%
% Inputs:
%%%%%%%%%%%
% Q:       [P-by-P-by-L] array of objective quadratic terms
% z:       [P-by-L]      array of objective linear terms
% c:       [L-by-1]      array of objective constant terms

function [] = save_piecewise_quadratic_data(Q, z, c)

    % Put all of the problem information into the correct data structures

    [cardP, cardL] = size(z);

    Ps.name = 'P';
    Ps.ts   = 'domain of quadratic functions';
    Ps.type = 'set';
    Ps.uels = {1:cardP};

    Ls.name = 'L';
    Ls.ts   = 'set of quadratic functions';
    Ls.type = 'set';
    Ls.uels = {1:cardL};

    Qs.name = 'Q';
    Qs.ts   = 'Objective quadratic terms';
    Qs.type = 'parameter';
    Qs.val = Q;
    Qs.form = 'full';
    Qs.dim = 3;
    Qs.uels{1} = Ps.uels;
    Qs.uels{2} = Ps.uels;
    Qs.uels{3} = Ls.uels;

    zs.name = 'z';
    zs.ts   = 'Objective linear terms';
    zs.type = 'parameter';
    zs.val = z;
    zs.form = 'full';
    zs.dim = 2;
    zs.uels{1} = Ps.uels;
    zs.uels{2} = Ls.uels;

    cs.name = 'c';
    cs.ts   = 'Objective constant terms';
    cs.type = 'parameter';
    cs.val = c;
    cs.form = 'full';
    cs.dim = 1;
    cs.uels = Ls.uels;

    wgdx ('piecewise_quadratic_data', Ls, Qs, zs, cs);
    fprintf('Data written to GDX file piecewise_quadratic_data.gdx\n');

end
