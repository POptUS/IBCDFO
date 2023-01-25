% set path for the test generator

% p = path ;
% path(p,'/home/more/matlab/qps') ;

rtol = 1.0e-1;
itmax = 20;

disp(sprintf('  ncase   n      av_iters    max_iters    max_err   nprobs'));
% disp(sprintf('     qmin           par      iters      time'));

nprobs = 5;
xerrs = zeros(nprobs,1);
itmaxs = zeros(nprobs,1);

for ncase = [1:7]
  rand('state',0);
  for n = [10,20,40,60,80,100]
    np = 0;
    for np = 1: nprobs
      t=cputime;
      [a,b,delta] = hard(n,ncase) ;
      par = norm(b)/delta;
      [x,qmin,parf,iters] = gqt(a,b,delta,rtol,itmax,par);
  %    disp(sprintf('%12.6e %12.3e %6i %12.2e',qmin, parf, iters, cputime-t))
      xnorm = norm(x);
      if (parf == 0.0e0 & xnorm <= (1+rtol)* delta)
        xerr = 0.0e0;
      else
        xerr = (xnorm-delta)/delta;
      end
      if (abs(xerr) >= 0.2)
	xnorm;
	parf;
      end
      xerrs(np) = xerr;
      itmaxs(np) = iters;
    end
    av_iters = sum(itmaxs(1:nprobs))/nprobs;
    max_iters = max(itmaxs(1:nprobs));
    max_err = max(abs(xerrs(1:nprobs)));
    disp(sprintf('%5i %5i %12.2f %12.2f %12.2e %5i',ncase,n,av_iters,max_iters,max_err,nprobs))
  end
end

return;

% Javier Pena's example.

  n = 3 ;

  q = [ 1 1 1 ; 2 -1 -1 ; 0 -3 3 ] ;
  a = inv(q)*diag([0 1 2])*q ;
  b = zeros(n,1) ;
  delta = 1.0 ;
  rtol = 0.1 ;
  itmax = 20 ;
  par = 0.0 ;
  [x,qmin,parf,iters] = gqt(a,b,delta,rtol,itmax,par);
  disp(sprintf('%12.6e %12.3e %6i %12.2e',qmin, parf, iters, cputime-t))