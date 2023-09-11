* Given a set of quadratics Q_l, linear terms z_l, and constant terms b_l, for
* l=1, ..., L, solve the problem:
*
*       minimize max_l || M(x) - z_l ||_{Q_l}^2 + c_l
*         s.t.   Low <= (x - x0) <= Upp
*
* for a given a set of quadratic models m_F_p = 0.5*(x - x0)^T * H_p * (x - x0) + g_p * (x - x0) + b_p

SETS
  N   'elements of x'
  P   'set of quadratic models'
  L   'set of quadratic mappings'
;

Alias(N, M)
Alias(P, P1)

PARAMETERS
H(N, M, P)  'Model quadratic terms'
g(N, P)     'Model linear terms'
b(P)        'Model constant terms'
solver      'Flag for solver'
Q(P, P1, L) 'Outer quadratic terms'
z(P, L)     'Outer linear terms'
c(L)        'Outer constant terms'
x0(N)       'Trust region center'
x1(N)       'Candidate starting point for TRSP'
Low(N)      'Lower bound on step'
Upp(N)      'Upper bound on step'
dsc(N)      'automatic scaling factor for d := x - x0'
gamma(N)    'manual scaling factor for d := x - x0'
;

$if NOT exist quad_model_data.gdx $abort File quad_model_data.gdx does not exist: run the calling script from Matlab to create it.
$gdxin quad_model_data.gdx
$load N P H g b x0 x1 solver Low Upp
$gdxin

$if NOT exist piecewise_quadratic_data.gdx  $abort File piecewise_quadratic_data.gdx does not exist.
$gdxin piecewise_quadratic_data.gdx
$load L Q z c
$gdxin

$ontext
For variable scaling:
  Let v_u be the variable seen by the user (i.e. in the original model)
  Let v_a be the variable passed on to the algorithm
  Let c be the scale factor
    v_a = v_u / c

For equation scaling:
  Let e_u be the equation seen by the user (i.e. in the original model)
  Let e_a be the equation passed on to the algorithm
  Let d be the scale factor
    e_a = e_u / d
$offtext

VARIABLES
  tau     Objective value
  x(N)    decision
  d(N)    'd := (x - x0) / dsc, i.e. d * dsc := (x - x0)'
  m_F(P)  value of each quadratic
  y(L)    auxiliary variable

option decimals=8;

* Declare model equations
EQUATIONS
obj                     Objective
each_model
* bounds_LB
* bounds_UB

inner_term
;

* Define model equations
obj(L)..            tau =g= y(L);

each_model(P)..     m_F(P) =e= 0.5*sum{(N, M), (d(N)*gamma(N))*H(N, M, P)*(d(M)*gamma(N))} + sum{N, g(N, P)*d(N)*gamma(N)} + b(P);

* bounds_LB(N)..      x(N) - x0(N) =g= Low(N);
* bounds_UB(N)..      x(N) - x0(N) =l= Upp(N);

inner_term(L)..     y(L) =e= 1.0*(sum((P, P1), (m_F(P) - z(P, L))*Q(P, P1, L)*(m_F(P1) - z(P1, L))) + c(L));

* model TRSP / ALL /;
model TRSP / obj, each_model, inner_term /;

dsc(N) = 1;
gamma(N) = max[abs(low(N)),abs(upp(N))];

$ifthen set AUTOSCALE
  dsc(N) = gamma(N);
  gamma(N) = 1;
  TRSP.scaleopt = 1;
$endif

d.lo(N) = Low(N) / gamma(N);
d.up(N) = Upp(N) / gamma(N);

x.l(N) = x0(N);
d.l(N) = 0;
d.scale(N) = dsc(N);
m_F.l(P) = b(P);
y.l(L) = 1.0*(sum((P, P1), (b(P) - z(P, L))*Q(P, P1, L)*(b(P1) - z(P1, L))) + c(L));
tau.l = smax(L, y.l(L));

option limrow = 1000;
option limcol = 1000;
option Optca = 0;
option Optcr = 0;
option reslim = 30;

* display x0;
* display Q;
* display z;
* display b;
* display H;
* display g;
* display c;
* display Low;
* display Upp;
* display 'lastly, ', tau.l;
* display 'lastly, ', m_F.l;
* display 'lastly, ', y.l;

TRSP.optfile = 1;
TRSP.trylinear = 1;

$onecho > minos.opt
Feasibility tolerance  0
Optimality  tolerance  0
$offecho

* option NLP=LINDOGLOBAL;
* option NLP=CONOPT;
* option NLP=couenne;
* option NLP=mosek;
* option NLP=minos;
* option NLP=examiner;

if (solver = 1,
$onecho > knitro.opt
opttol 0.0
opttolabs 0.0
xtol 0.0
ftol 0.0
ftol_iters 20
$offecho
option NLP=knitro;
);
if (solver = 2,
$onecho > snopt.opt
$offecho
  option NLP=snopt;
);
if (solver = 3,
$onecho > lindo.opt
$offecho
  option NLP=LINDOGLOBAL;
);
if (solver = 4,
$onecho > minos.opt
$offecho
  option NLP=minos;
);
SOLVE TRSP MINIMIZING tau USING NLP;

scalars modelStat, solveStat;
modelStat = TRSP.modelstat;
solveStat = TRSP.solvestat;

parameters
  Nerr(N)
  xd(N)    'x - x0'
  ;
xd(N) = d.L(N)*gamma(N);
Nerr(N) = max(0, Low(N) - xd(N));
* abort$[sum{N, Nerr(N)}] 'x-x0 < Low', Nerr;
Nerr(N) = max(0, xd(N)-Upp(N));
* abort$[sum{N, Nerr(N)}] 'x-x0 > Upp', Nerr;

m_F.L(P) = 0.5*sum{(N, M), xd(N)*H(N, M, P)*xd(M)} + sum{N, g(N, P)*xd(N)} + b(P);
x.L(N) = x0(N) + xd(N);
execute_unload 'solution', modelStat, solveStat, x, tau, m_F;
