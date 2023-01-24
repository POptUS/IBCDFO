* Solve the problem:
*
*       minimize sum_p | D_p - max{m_F_p(x - x0), C_p} |
*         s.t.   Low <= (x - x0) <= Upp
*
* for a given a set of quadratic models m_F_p = 0.5*(x - x0)^T * H_p * (x - x0) + g_p * (x - x0) + b_p
*
* Todd Munson helped develop this more GAMS-friendly formulation of this sum/abs/max objective

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
D(P)        'Data'
C(P)        'Censor value'
x0(N)       'Trust region center'
x1(N)       'Candidate starting point for TRSP'
Low(N)      'Lower bound on step'
Upp(N)      'Upper bound on step'
;

$if NOT exist quad_model_data.gdx $abort File quad_model_data.gdx does not exist: run the calling script from Matlab to create it.
$gdxin quad_model_data.gdx
$load N P H g b x0 x1 solver Low Upp
$gdxin

$if NOT exist censored_L1_loss_data.gdx  $abort File censored_L1_loss_data.gdx does not exist.
$gdxin censored_L1_loss_data.gdx
$load C D
$gdxin

VARIABLES
  tau     Objective value
  x(N)     decision
  m_F(P) value of each quadratic
  w(P)
  lambda(P)
  gamma(P)
  mu(P)
  s(P)
  r(P)

option decimals=8;

* Declare model equations
EQUATIONS
obj                     Objective
each_model
bounds_LB
bounds_UB

eq1
eq2
eq2_1
eq3
eq4
eq5
eq6
eq7
eq8
eq9
eq10
eq11
;

* Define model equations
obj..               tau =e= sum(P, w(P));

each_model(P)..     m_F(P) =e= 0.5*sum((N, M), (x(N) - x0(N))*H(N, M, P)*(x(M) - x0(M))) + sum(N, g(N, P)*(x(N) - x0(N))) + b(P);

bounds_LB(N)..      x(N) - x0(N) =g= Low(N);
bounds_UB(N)..      x(N) - x0(N) =l= Upp(N);

eq1(P)..   1 - lambda(P) - gamma(P) =e= 0;
eq2_1..                     mu('1') =e= m_F('1');
eq2(P)$(ord(P)>=2)..   mu(P) - m_F(P) =e= s(P);
eq3(P)$(ord(P)>=2)..   mu(P) - C(P)     =e= r(P);
eq4(P)..   lambda(P)*s(P)   =l= 0;
eq5(P)..   gamma(P)*r(P)    =l= 0;
eq6(P)..   -w(P) =l= D(P) - mu(P);
eq7(P)..             D(P) - mu(P) =l= w(P);
eq8(P)..   s(P) =g= 0;
eq9(P)..   r(P) =g= 0;
eq10(P)..  gamma(P) =g= 0;
eq11(P)..  lambda(P) =g= 0;

model TRSP / ALL /;

* x.l(N) = x1(N);
* m_F.l(P) = 0.5*sum((N, M), (x1(N) - x0(N))*H(N, M, P)*(x1(M) - x0(N))) + sum(N, g(N, P)*(x1(N) - x0(N))) + b(P);
m_F.l(P) = b(P);
mu.l(P) = max(C(P), m_F.l(P));
w.l(P) = abs(D(P) - mu.l(P));
tau.l = sum(P, w.l(P));
x.l(N) = x0(N);
s.l(P)$(mu.l(P) = C(P)) = mu.l(P) - m_F.l(P);
r.l(P)$(mu.l(P) = m_F.l(P)) = mu.l(P) - C(P);

lambda.l(P)$(mu.l(P) <> C(P)) = 1;
gamma.l(P)$(mu.l(P) <> m_F.l(P)) = 1;


option limrow = 1000;
option limcol = 1000;
option Optca = 0;
option Optcr = 0;
option reslim = 30;

* display x0;
* display x1;
* display tau.l;
* display m_F.l;
* display D;
* display C;
* display g;
* display H;
* display 'lastly, ', solver;

TRSP.optfile = 1;
TRSP.trylinear = 1;

$onecho > minos.opt
Feasibility tolerance  0
Optimality  tolerance  0
$offecho

if (solver = 1,
  option NLP=conopt;
);
if (solver = 2,
$onecho > minos.opt
$offecho
  option NLP=minos;
);
if (solver = 3,
$onecho > snopt.opt
$offecho
  option NLP=snopt;
);
if (solver = 4,
$onecho > baron.opt
threads 100
$offecho
  option NLP=baron;
);

* option NLP=LINDOGLOBAL;
* option NLP=couenne;
SOLVE TRSP MINIMIZING tau USING NLP;

scalars modelStat, solveStat;
modelStat = TRSP.modelstat;
solveStat = TRSP.solvestat;

execute_unload 'solution', modelStat, solveStat, x, tau, m_F;
