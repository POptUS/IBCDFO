* Solve the problem:
*
*       minimize min_p M(x)
*         s.t.   Low <= (x - x0) <= Upp
*
* where M is a defined by P quadratics mapping from R^n (0.5*(x - x0)^T * H_p * (x - x0) + g_p * (x - x0) + c_p)

SETS
  N   'elements of x'
  P   'set of quadratic models'
;

Alias(N, M)

PARAMETERS
H(N, M, P)  'Model quadratic terms'
g(N, P)     'Model linear terms'
b(P)        'Model constant terms'
solver      'Flag for solver'
x0(N)       'Trust region center'
x1(N)       'Candidate starting point for TRSP'
Low(N)      'Lower bound on step'
Upp(N)      'Upper bound on step'
;

$if NOT exist quad_model_data.gdx $abort File quad_model_data.gdx does not exist: run the calling script from Matlab to create it.
$gdxin quad_model_data.gdx
$load N P H g b x0 x1 solver Low Upp
$gdxin

VARIABLES
  tau     Objective value
  x(N)     decision
  m_F(P) value of each quadratic

option decimals=8;

* Declare model equations
EQUATIONS
obj                     Objective
each_model
bounds_LB
bounds_UB
;

* Define model equations
obj..               tau =e= smin(P, m_F(P));

each_model(P)..     m_F(P) =e= 0.5*sum((N, M), (x(N) - x0(N))*H(N, M, P)*(x(M) - x0(M))) + sum(N, g(N, P)*(x(N) - x0(N))) + b(P);

bounds_LB(N)..      x(N) - x0(N) =g= Low(N);
bounds_UB(N)..      x(N) - x0(N) =l= Upp(N);

model TRSP / ALL /;

x.l(N) = x0(N);
m_F.l(P) = b(P);
tau.l = smin(P, m_F.l(P));

option limrow = 1000;
option limcol = 1000;
option Optca = 0;
option Optcr = 0;
option reslim = 30;

* display x0;
* display scale;
* display Q;
* display z;
* display b;
* display H;
* display g;
* display c;
* display 'lastly, ', tau.l;
* display 'lastly, ', m_F.l;

TRSP.optfile = 1;
TRSP.trylinear = 1;

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
  option DNLP=conopt;
);
if (solver = 2,
$onecho > minos.opt
$offecho
  option DNLP=minos;
);
if (solver = 3,
$onecho > snopt.opt
$offecho
  option DNLP=snopt;
);

SOLVE TRSP MINIMIZING tau USING DNLP;

scalars modelStat, solveStat;
modelStat = TRSP.modelstat;
solveStat = TRSP.solvestat;

execute_unload 'solution', modelStat, solveStat, x, tau, m_F;
