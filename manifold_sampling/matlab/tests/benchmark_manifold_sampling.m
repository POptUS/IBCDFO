% This wrapper tests various algorithms against the Benchmark functions from the
% More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
function [] = benchmark_manifold_sampling()

global BenDFO
BenDFO.probtype = 'smooth';

addpath('../')
addpath('../../../../BenDFO/m/')
addpath('../../../../BenDFO/data/')

% addpath('../../manifold_sampling/matlab/')
% addpath('../../manifold_sampling/matlab/subproblem_scripts/') % project_zero_onto_convex_hull_2, solveSubproblem
% addpath('../../manifold_sampling/matlab/subproblem_scripts/gqt/') % mgqt_2
addpath('../h_examples/')
% addpath('./test_problems/')
addpath('../../../pounders/matlab') % formquad, bmpts, boxline, phi2eval


mkdir('benchmark_results')

nfmax = 500;
factor = 10;

subprob_switch = 'linprog';

load dfo.dat

filename = ['./benchmark_results/manifold_samplingM_nfmax=' num2str(nfmax) '.mat'];
Results = cell(1,53);

% for row = find(cellfun(@length,Results)==0)
for row = [1, 2, 7, 8, 43, 44, 45]
    row
    nprob = dfo(row,1);
    n = dfo(row,2);
    m = dfo(row,3);
    factor_power = dfo(row,4);

    BenDFO.nprob = nprob;
    BenDFO.n = n;
    BenDFO.m = m;

    LB = -Inf*ones(1,n);
    UB = Inf*ones(1,n);

    xs = dfoxs(n,nprob,factor^factor_power);

    SolverNumber = 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Manifold sampling
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for jj = 1
        if jj == 1
            hfun = @pw_maximum_squared;
        end
        if jj == 2
            hfun = @pw_maximum_abs;
        end
    end
    Ffun = @calfun;
    x0 = xs';

    [X,F,h,xkin, flag] = manifold_sampling_primal(hfun,Ffun,x0,LB,UB,nfmax,subprob_switch);

    SolverNumber = SolverNumber + 1;
    Results{SolverNumber,row}.alg = 'Manifold sampling';
    Results{SolverNumber,row}.problem = ['problem ' num2str(row) ' from More/Wild with hfun='  ];
    Results{SolverNumber,row}.Fvec = F;
    Results{SolverNumber,row}.H = h;
    Results{SolverNumber,row}.X = X;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save(filename, 'Results');
    % save('-mat7-binary', filename, 'Results') % Octave save
end
save(filename, 'Results');
