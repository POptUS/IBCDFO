$include sets.inc
SET null /0/;

ALIAS(N,M)

VARIABLES
obj_val, s(N)

PARAMETERS G(N,K) "Generators G_k "      
/
$ondelim
$include G.csv
$offdelim
/
;
PARAMETERS f_bar(K) "f_bar" 
/
$ondelim
$include f_bar.csv
$offdelim
/
;
PARAMETERS f_x_k(null) "f_x_k" 
/
$ondelim
$include f_x_k.csv
$offdelim
/
;
PARAMETERS beta(K) "beta" 
/
$ondelim
$include beta.csv
$offdelim
/
;
PARAMETERS H(K,N,N) "Hessian terms" 
/
$ondelim
$include H.csv
$offdelim
/

PARAMETERS U(N) "Upper bounds" 
/
$ondelim
$include U.csv
$offdelim
/

PARAMETERS L(N) "Lower bounds" 
/
$ondelim
$include L.csv
$offdelim
/
;

EQUATIONS
objective
lower
upper
;

objective(K)..  obj_val =g= (f_bar(K) + sum(N, G(N,K)*s(N)) - beta(K)) + 0.5 * sum((N,M), s(N)*H(K,N,M)*s(M)) + f_x_k('0');
lower(N).. L(N) =g= s(N);
upper(N).. s(N) =l= U(N);


model wanping / ALL /;
option limrow = 10;
option limcol = 10;
option Optca = 0;
option Optcr = 0;
option NLP=BARON;

SOLVE wanping MINIMIZING obj_val USING NLP;

file results /'results.txt'/;
results.nd = 9;

put results
  / 'obj_val: ' obj_val.l:0:8 /
  loop(N, put 's'N.tl s.l(N):0:16/)
;


file sout /'s.out'/;
sout.nd = 9;

put sout
  loop(N, put s.l(N):0:16/)
;

