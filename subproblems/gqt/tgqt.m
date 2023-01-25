n = 100 ;
ncase = 1 ;
[a,b,delta] = hard(n,ncase) ;

rtol = 1.0e-5 ;
itmax = 20 ;

t = cputime;
par = 0.0 ;
[x,qmin,parf] = mgqt(a,b,delta,rtol,itmax,par) ;
qmin
parf
itmax
disp (['Time for mgqt  ',num2str(cputime-t)])

t = cputime;
par = 0.0 ;
[x,qmin,parf,iter] = gqt(a,b,delta,rtol,itmax,par) ;
qmin
parf
iter
disp (['Time for gqt  ',num2str(cputime-t)])
