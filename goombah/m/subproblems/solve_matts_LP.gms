* Solve the problem:
*
*       minimize_x  c'x
*         s.t.      Ax <= b
*                   lb <= x <= ub
* where M is a defined by P quadratics mapping from R^n (0.5*(x-x0)^T * H_p * (x-x0) + g_p * (x-x0) + c_p)

SETS
  N   'elements of x'
  M   'constraints in A'
;

PARAMETERS
c(N)    'objective coefficients'
A(M,N)  'constraint matrix'
b(M)    'right-hand sides'
Low(N)  'lower bounds'
Upp(N)  'upper bounds'
x0(N)   'Starting point'
solver  'solver int'
;

$if NOT exist matts_LP_dat.gdx  $abort File matts_LP_dat.gdx does not exist: run the calling script from Matlab to create it.
$gdxin matts_LP_dat.gdx
$load N M c A b Low Upp x0 solver
$gdxin

FREE VARIABLES
  tau     Objective value
  x(N)    decision variable

option decimals=8;

* Declare model equations
EQUATIONS
obj
constraints
bounds_LB
bounds_UB
;

* Define model equations
obj..               tau =e= sum(N, c(N)*x(N));
constraints(M)..    sum(N, A(M,N)*x(N)) =l= b(M);
bounds_LB(N)..      Low(N) =l= x(N);
bounds_UB(N)..      Upp(N) =g= x(N);

model matts_LP / ALL /;

x.lo(N) = Low(N);
x.up(N) = Upp(N);
x.l(N) = x0(N);
* tau.l = sum(N, c(N)*x0(N));

option limrow = 1000;
option limcol = 1000;
option Optca = 0;
option Optcr = 0;
option reslim = 30;

* display x0;
* display scale;
* display delta;
* display Q;
* display z;
* display b;
* display H;
* display g;
* display c;
* display 'lastly, ', tau.l;
* display 'lastly, ', m_F.l;
* display 'lastly, ', y.l;

matts_LP.optfile = 1;
matts_LP.trylinear = 1;

$onecho > minos.opt
Feasibility tolerance  0
Optimality  tolerance  0
$offecho

* option DNLP=LINDOGLOBAL;
* option DNLP=CONOPT;
* option DNLP=couenne;
* option DNLP=mosek;
* option DNLP=minos;
* option DNLP=examiner;
* option DNLP=snopt;

if (solver = 1,
option LP=cplex;
);
option QCP=antigone;

*SOLVE matts_LP MINIMIZING tau USING LP;
SOLVE matts_LP MINIMIZING tau USING QCP;

scalars modelStat, solveStat;
modelStat = matts_LP.modelstat;
solveStat = matts_LP.solvestat;

execute_unload 'matts_LP_sol', modelStat, solveStat, x, constraints, bounds_LB, bounds_UB;
