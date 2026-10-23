% A general MATLAB-to-GAMS interface.
%
% This function
% - Writes out information defining `P` quadratics out a gdx file in the current directory
% - Copies the gams_file_path file to the current directory
% - Calls GAMS on that copied file (needs to be written to read-in & write-out its solution) for many solver flags
% - Asserts that the GAMS call produced a file and reads in the solution.
% - Checks that the GAMS solver status indicates success for at least one of the solvers
% - Computes h_fun on all solver solutions that were successful
% - Returns the min of those solutions
%
% This function writes a GDX file via a GDXMRW's wgdx() call.

%%%%%%%%%%%
% Inputs:
%%%%%%%%%%%
% H:          [n-by-n-by-P] array of P model quadratic terms
% g:          [n-by-P] array of P model linear terms
% b:          [P-by-1] array of P model constant terms
% delta:      [scalar] trust region radius
% x0:         [n-by-1] trust region center
% x1:         [n-by-1] a candidate starting point for use by GAMS
% val_at_x0:  [scalar] the objective value at x0

%%%%%%%%%%%
% Outputs:
%%%%%%%%%%%
% x:          [1-by-n] array containing the solution
% pred_dec:   [scalar] predicted decrease from x0 to x

function [s_k, pred_dec] = save_quadratics_call_GAMS(H, g, b, Low, Upp, x0, x1, val_at_x0, GAMS_options, h_fun)

% First put all of the problem information into the correct data structures

[n, P] = size(g);

% Resolve the GAMS template path before we change directories below.
gams_file_full = GAMS_options.file;
if ~(gams_file_full(1) == filesep || (ispc && length(gams_file_full) > 1 && gams_file_full(2) == ':'))
    gams_file_full = fullfile(pwd, gams_file_full);
end

% Run everything inside a fresh temporary directory so this call does not
% write GDX/GAMS files into (and potentially clobber) the current directory.
orig_dir = pwd;
tmp_dir = tempname;
mkdir(tmp_dir);
cleanupObj = onCleanup(@() cleanup_temp_dir(orig_dir, tmp_dir)); %#ok<NASGU>
cd(tmp_dir);

% Optionally write the h-specific data (Q/z/c or C/D) into this temp dir,
% so callers no longer have to do this as a side effect before calling us.
if isfield(GAMS_options, 'hfun_data') && ~isempty(GAMS_options.hfun_data)
    hfun_data = GAMS_options.hfun_data;
    if isfield(hfun_data, 'Q')
        save_piecewise_quadratic_data(hfun_data.Q, hfun_data.z, hfun_data.c);
    elseif isfield(hfun_data, 'C')
        save_censored_L1_loss_data(hfun_data.C, hfun_data.D);
    else
        error('GAMS_options.hfun_data must have fields Q/z/c or C/D');
    end
end

Ns.name = 'N';
Ns.ts   = 'elements of x';
Ns.type = 'set';
Ns.uels = {1:n};

Ps.name = 'P';
Ps.ts   = 'elements of z';
Ps.type = 'set';
Ps.uels = {1:P};

Hs_mod.name = 'H';
Hs_mod.ts   = 'quadratic terms';
Hs_mod.type = 'parameter';
Hs_mod.val = H;
Hs_mod.form = 'full';
Hs_mod.dim = 3;
Hs_mod.uels{1} = Ns.uels;
Hs_mod.uels{2} = Ns.uels;
Hs_mod.uels{3} = Ps.uels;

gs_mod.name = 'g';
gs_mod.ts   = 'linear terms';
gs_mod.type = 'parameter';
gs_mod.val = g;
gs_mod.form = 'full';
gs_mod.dim = 2;
gs_mod.uels{1} = Ns.uels;
gs_mod.uels{2} = Ps.uels;

bs_mod.name = 'b';
bs_mod.ts   = 'constant terms';
bs_mod.type = 'parameter';
bs_mod.val = b;
bs_mod.form = 'full';
bs_mod.dim = 1;
bs_mod.uels = Ps.uels;

x0s.name = 'x0';
x0s.ts   = 'Trust region center';
x0s.type = 'parameter';
x0s.val = x0;
x0s.form = 'full';
x0s.dim = 1;
x0s.uels = Ns.uels;

x1s.name = 'x1';
x1s.ts   = 'Starting point for optimization';
x1s.type = 'parameter';
x1s.val = x1;
x1s.form = 'full';
x1s.dim = 1;
x1s.uels = Ns.uels;

solver.name = 'solver';
solver.ts   = 'Solver flag';
solver.type = 'parameter';

Lows.name = 'Low';
Lows.ts   = 'Lower bounds on step';
Lows.type = 'parameter';
Lows.val = Low;
Lows.form = 'full';
Lows.dim = 1;
Lows.uels = Ns.uels;

Upps.name = 'Upp';
Upps.ts   = 'Upper bounds on step';
Upps.type = 'parameter';
Upps.val = Upp;
Upps.form = 'full';
Upps.dim = 1;
Upps.uels = Ns.uels;

x1s.name = 'x1';
x1s.ts   = 'Starting point for optimization';
x1s.type = 'parameter';
x1s.val = x1;
x1s.form = 'full';
x1s.dim = 1;
x1s.uels = Ns.uels;

num_solvers = numel(GAMS_options.solvers);
allx = inf * ones(num_solvers, n);
solveStat = inf * ones(num_solvers, 1);
modelStat = inf * ones(num_solvers, 1);
obj_vals_GAMS = inf * ones(num_solvers, 1);

% Then loop over solvers
for j = 1:num_solvers
    i = GAMS_options.solvers(j);
    solver.val = i;

    % Put problem data to a gdx file
    wgdx('quad_model_data', Ns, Ps, Hs_mod, gs_mod, bs_mod, x0s, x1s, solver, Lows, Upps);
    fprintf('Matlab data written to GDX file quads_model_data.gdx\n');

    % Remove old solutions file for safety
    solGDX = 'solution.gdx';
    if exist(solGDX, 'file')
        delete(solGDX);
    end

    % Copy the template gams file
    copyfile(gams_file_full, ['./TRSP_' int2str(i) '.gms']);

    % Perform the gams run
    flag = system(['gams TRSP_' int2str(i) '.gms lo=2']);

    assert(flag == 0, 'gams run failed: rc = %d', flag);
    assert(exist(solGDX, 'file') == 2, ['Results file ', solGDX, ' does not exist after gams run']);

    fprintf(['TRSP_' int2str(i) '.gms finished\n']);

    % now get the outputs from the GDX file produced by the GAMS run
    rs = struct('name', 'modelStat', 'form', 'full');
    r = rgdx (solGDX, rs);
    modelStat(j) = r.val;

    rs.name = 'solveStat';
    r = rgdx (solGDX, rs);
    solveStat(j) = r.val;

    rs.name = 'tau';
    rs.field = 'l';
    r = rgdx (solGDX, rs);
    obj_vals_GAMS(j) = r.val;

    rs.name = 'x';
    rs.uels = Ns.uels;
    r = rgdx (solGDX, rs);
    x = r.val;
    allx(j, :) = x';
end

if ~any(solveStat <= 4)
    error(['None of the methods can solve this problem']);
end

solved = find(solveStat <= 4);

obj_vals_GAMS = obj_vals_GAMS(solved);
allx = allx(solved, :);

for j = 1:size(allx, 1)
    allx(j, :) =  min(Upp + x0, max(Low + x0, allx(j, :)));
end

z = zeros(1, P);
obj_vals_MATLAB = inf * ones(size(allx, 1), 1);
for j = 1:size(allx, 1)
    x = allx(j, :);
    for i = 1:P
        z(i) = 0.5 * (x - x0) * H(:, :, i) * (x - x0)' + (x - x0) * g(:, i) + b(i);
    end
    obj_vals_MATLAB(j) = h_fun(z);
end

atol = 1e-4;
rtol = 1e-4;
if any(abs(obj_vals_GAMS(:) - obj_vals_MATLAB(:)) > atol + rtol * abs(obj_vals_MATLAB(:)))
    disp("MATLAB and GAMS calculated different objective values for some reason");
    pred_dec = 0;
    s_k = zeros(1, n);
    return
end
% assert(all(abs(obj_vals_GAMS(:) - obj_vals_MATLAB(:)) <= atol + rtol * abs(obj_vals_MATLAB(:))), "MATLAB and GAMS calculated different objective values for some reason");

[val_at_new, ind] = min(obj_vals_MATLAB);
s_k = allx(ind, :) - x0;
pred_dec = val_at_x0 - val_at_new;

% assert(pred_dec >= 0)

if pred_dec < 0
    pred_dec = 0;
end

end

function cleanup_temp_dir(orig_dir, tmp_dir)
cd(orig_dir);
rmdir(tmp_dir, 's');
end
