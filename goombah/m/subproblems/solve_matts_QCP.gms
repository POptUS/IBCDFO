SETS
  N   'elements of x'
  M   'constraints in A'
;

PARAMETERS
c(N)    'objective coefficients'
A(M,N)  'constraint matrix'
b(M)    'right-hand sides'
x0(N)   'Starting point'
solver  'solver int'
delta   'trust-region radius'
Low(N)	'lower bounds'
Upp(N)	'upper bounds'
;

$if NOT exist matts_QCP_dat.gdx  $abort File matts_QCP_dat.gdx does not exist: run the calling script from Matlab to create it.
$gdxin matts_QCP_dat.gdx
$load N M c A b x0 solver delta Low Upp
$gdxin

FREE VARIABLES
  tau     Objective value
  x(N)    decision variable

option decimals=8;

* Declare model equations
EQUATIONS
obj
constraints
tr_const
bounds_LB
bounds_UB
;

* Define model equations
obj..               tau =e= sum(N, c(N)*x(N));
constraints(M)..    sum(N, A(M,N)*x(N)) =l= b(M);
tr_const..          sum(N$(ord(N)>=2), x(N)*x(N)) =l= power(delta,2);
bounds_LB(N)..      Low(N) =l= x(N);
bounds_UB(N)..      Upp(N) =g= x(N);

model matts_QCP / ALL /;

x.l(N) = x0(N);

option limrow = 1000;
option limcol = 1000;
option Optca = 0;
option Optcr = 0;
option reslim = 30;

* display x0;
* display scale;
* display delta;
* display Low;
* display Upp;
* display Q;
* display z;
* display b;
* display H;
* display g;
* display c;
* display 'lastly, ', tau.l;
* display 'lastly, ', m_F.l;
* display 'lastly, ', y.l;

matts_QCP.optfile = 1;
matts_QCP.trylinear = 1;

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

SOLVE matts_QCP MINIMIZING tau USING QCP;

scalars modelStat, solveStat;
modelStat = matts_QCP.modelstat;
solveStat = matts_QCP.solvestat;

execute_unload 'matts_QCP_sol', modelStat, solveStat, x, constraints, bounds_LB, bounds_UB;
