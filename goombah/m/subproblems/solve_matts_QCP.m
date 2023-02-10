% This function calls GAMS to solve a trust region subproblem
%
%       minimize_x  c'x
%         s.t.      Ax <= b
%                   lb <= x <= ub

% This function writes a GDX file via a GDXMRW's wgdx() call. This generates
% data that is loaded into a GAMS model. The solution from the GAMS model is
% then written to file, and read back into this function.

%%%%%%%%%%%
% Inputs:
%%%%%%%%%%%
% c:       [n-by-1] array of objective parameters
% A:       [m-by-n] array of constraints
% b:       [m-by-1] array of right-hand sides
% x0:      [n-by-1] initial point
% lb:      [n-by-1] lower bounds on components of x
% ub:      [n-by-1] upper bounds on components of x

%%%%%%%%%%%
% Outputs:
%%%%%%%%%%%
% x        [n-by-1] array containing the solution

function [x, duals_g, duals_u, duals_l] = solve_matts_QCP(c, A, b, x0, delta, Low, Upp, H)

% First put all of the problem information into the correct data structures

[m, n] = size(A);

assert(size(H, 1) == m, 'Wrong size');
assert(size(H, 2) == n, 'Wrong size');
assert(size(H, 3) == n, 'Wrong size');

Ns.name = 'N';
Ns.ts   = 'number of elements in x';
Ns.type = 'set';
Ns.uels = {1:n};

Ms.name = 'M';
Ms.ts   = 'number of constraints in A';
Ms.type = 'set';
Ms.uels = {1:m};

cs.name = 'c';
cs.ts   = 'objective coefficients';
cs.type = 'parameter';
cs.val = c;
cs.form = 'full';
cs.dim = 1;
cs.uels = Ns.uels;

As.name = 'A';
As.ts   = 'constraint matrix';
As.type = 'parameter';
As.val = A;
As.form = 'full';
As.dim = 2;
As.uels{1} = Ms.uels;
As.uels{2} = Ns.uels;

%Hs.name = 'H';
%Hs.ts   = 'quadratic terms';
%Hs.type = 'parameter';
%Hs.val = H;
%Hs.form = 'full';
%Hs.dim = 3;
%Hs.uels{1} = Ms.uels;
%Hs.uels{2} = Ns.uels;
%Hs.uels{3} = Ns.uels;

bs.name = 'b';
bs.ts   = 'constraints right-hand sides';
bs.type = 'parameter';
bs.val = b;
bs.form = 'full';
bs.dim = 1;
bs.uels = Ms.uels;

x0s.name = 'x0';
x0s.ts   = 'initial point';
x0s.type = 'parameter';
x0s.val = x0;
x0s.form = 'full';
x0s.dim = 1;
x0s.uels = Ns.uels;

solver.name = 'solver';
solver.ts   = 'Solver flag';
solver.type = 'parameter';

deltas.name = 'delta';
deltas.ts   = 'trust region radius';
deltas.type = 'parameter';
deltas.val = delta;
deltas.form = 'full';

Lows.name = 'Low';
Lows.ts = 'lower bounds';
Lows.type = 'parameter';
Lows.val = [-Inf, Low];
Lows.form = 'full';
Lows.dim = 1;
Lows.uels = Ns.uels;

Upps.name = 'Upp';
Upps.ts = 'upper bounds';
Upps.type = 'parameter';
Upps.val = [Inf, Upp];
Upps.form = 'full';
Upps.dim = 1;
Upps.uels = Ns.uels;

% Then loop over solvers
for i = 1:1
    solver.val = i;

    % Put problem data to a gdx file
    dat_file_name = 'matts_QCP_dat';
    wgdx(dat_file_name, Ns, Ms, cs, As, bs, x0s, solver, deltas, Lows, Upps);
    fprintf(['Matlab data written to GDX file ' dat_file_name '.gdx\n']);

    % Remove old solutions file for safety
    solGDX = 'matts_QCP_sol.gdx';
    if exist(solGDX, 'file')
        delete(solGDX);
    end

    % Copy the template gams file
    copyfile('../../subproblems/solve_matts_QCP.gms', ['./matts_QCP' int2str(i) '.gms']);

    % Perform the gams run
    flag = system(['gams matts_QCP' int2str(i) '.gms lo=2']);

    assert(flag == 0, 'gams run failed: rc = %d', flag);
    assert(exist(solGDX, 'file') == 2, ['Results file ', solGDX, ' does not exist after gams run']);

    fprintf('Model matts_QCP.gms finished\n');

    % now get the outputs from the GDX file produced by the GAMS run
    rs = struct('name', 'modelStat', 'form', 'full');
    r = rgdx (solGDX, rs);
    modelStat(i) = r.val;

    rs.name = 'solveStat';
    r = rgdx (solGDX, rs);
    solveStat(i) = r.val;

    rs.name = 'constraints';
    rs.uels = Ms.uels;
    rs.field = 'm';
    r = rgdx (solGDX, rs);

    % Giving back the negative of the duals because of some disagreement
    % between the problem formulations in Matlab's optimization toolbox and
    % GAMS.
    duals_g = -r.val;

    rs.name = 'bounds_LB';
    rs.uels = Ns.uels;
    rs.field = 'm';
    r = rgdx (solGDX, rs);
    duals_l = r.val; % should be negative?

    rs.name = 'bounds_UB';
    rs.uels = Ns.uels;
    rs.field = 'm';
    r = rgdx (solGDX, rs);
    duals_u = r.val; % should be negative?

    assert(length(duals_g) == m);
    assert(length(duals_l) == n);
    assert(length(duals_u) == n);

    assert(solveStat <= 4, "Didn't solve the problem");

    rs.name = 'x';
    rs.uels = Ns.uels;
    rs.field = 'l';
    r = rgdx(solGDX, rs);
    x = r.val;
end

end
