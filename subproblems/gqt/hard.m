function [a,b,delta] = hard(n,ncase)
%
%   function [a,b,delta] = hard(n,ncase)
%
%   This function generates random hard problems for gqt.
%

   e = ones(n,1) ;
   evals = e - 2*rand(n,1) ;
   b0 = e - 2*rand(n,1) ;
   [evmin,index] = min(evals) ;

%  Hard case

   if ncase == 2
      b0(index) = 0.0 ;
   end

%  Saddle point case

   if ncase == 3
      b0 = zeros(n,1) ;
   end

%  Positive definite case

   if ncase == 4
      evals = abs(evals) ;
   end

%  Positive semi-definite case

   if ncase == 5
      evals = abs(evals) ;
      [evmin,index] = min(evals) ;
      evals(index) = 0.0 ;
   end

%  Positive semi-definite and hard case

   if ncase == 6
      evals = abs(evals) ;
      [evmin,index] = min(evals) ;
      evals(index) = 0.0 ;
      b0(index) = 0.0 ;
   end

%  Positive semi-definite and saddle point case

   if ncase == 7
      evals = abs(evals) ;
      [evmin,index] = min(evals) ;
      evals(index) = 0.0 ;
      b0 = zeros(n,1) ;
   end

   a = diag(evals) ;
   b = b0 ;
   evmin = min(evals) ;

%  Generate a random matrix Q and form Q'*A*Q and Q'*b

   for k = 1: 3
       w = rand(n,1) ;
       beta = 2/norm(w)^2 ;
       a = a - beta*w*(a'*w)' ;
       b = b - beta*w*(b'*w) ;
       a = a - beta*(a*w)*w' ;
   end

%  Compute delta

   delta = 100*rand ;
