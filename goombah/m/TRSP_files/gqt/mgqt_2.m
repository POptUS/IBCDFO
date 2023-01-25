function [x,qmin,par,iter] = mgqt_2(a,b,delta,rtol,itmax,par)
% This represents a modified version of Jorge More's gqt.m
% Updated many times by Stefan Wild (last mod 9/24/08)
% Should be revisited eventually, especially since some of the backslashes
% reveal that r is nearly singular (eg.- line 79)
%
%    function [x,qmin,par,iter] = mgqt(a,b,delta,rtol,itmax,par)
%
%     Given an n by n symmetric matrix A, an n-vector b, and a
%     positive number delta, this function determines a vector
%     x which minimizes the quadratic function
%           f(x) = (x,Ax)/2 + (b,x)
%     subject to the constraint
%           norm(x) .le. delta.

printer=0; % To suppress output from mgqt

n = size(a,1) ;
bnorm = norm(b) ;
diaga = diag(a) ;
rowsum = sum(abs(a))' ;
growsum = rowsum - abs(diaga) ;

pars = max(-diaga) ;
parl = max(diaga+growsum) ;
paru = max(-diaga+growsum) ;

anorm = max(rowsum);
pars = max(pars,-anorm);
parl = max(parl,-anorm);
paru = max(paru,-anorm);

parl = max([0.0,bnorm/delta-parl,pars]) ;
paru = max(0.0,bnorm/delta+paru) ;

%  If the input par lies outside of the interval (parl,paru),
%  set par to the closer endpoint

par = max(par,parl) ;
par = min(par,paru) ;

%  Special case: parl = paru

paru = max(paru,(1+rtol)*parl) ;

%  Special case: paru = 0.0

%    if paru == 0.0
%       x = zeros(n,1) ;
%       iter = 0 ;
%       qmin = 0.0 ;
%       return
%    end

for iter = 1: itmax

    %     Safeguard par

    if (par <= pars) && paru > 0.0
        par = max([0.001,sqrt(parl/paru)])*paru ;
    end

    % disp(' ')
    % disp (sprintf('parl = %6.3e, paru = %6.3e',parl,paru))
    % disp (sprintf('par  = %6.3e, pars = %6.3e',par ,pars))

    c = a + par*eye(n) ;

    %     Attempt the Cholesky decomposition

    [r,indef] = chol(c) ;

    %     Case 1: A + par*I is positive definite

    if indef == 0

        %        Compute an approximate solution x

        x = - r\((r')\b) ;
        xnorm = norm(x) ;

        %        Test for convergence

        if (abs(xnorm-delta) <= rtol*delta || (par == 0.0 && ...
                xnorm <= (1+rtol)*delta))
            % disp (sprintf('Exit with par = %6.3e', par))
            % disp (sprintf('parl = %6.3e, paru = %6.3e',parl,paru))
            qmin = 0.5*x'*(a*x) + x'*b ;
            return
        end

        %        Compute a direction of negative curvature

        [v,d] = eig(c) ;
        [evalmin,i] = min(diag(d)) ;
        z =  v(:,i);

        %        Compute a negative curvature solution of the form
        %        x + alpha*z where norm(x + alpha*z) = delta.

        if xnorm < delta
            bf = (z'*x)/delta ;
            cf = ((delta-xnorm)/delta)*((delta+xnorm)/delta) ;
            alpha = delta*(abs(bf)+sqrt(bf^2+cf)) ;
            if bf > 0.0
                alpha = -alpha ;
            end
            z = alpha*z ;

            %           Test for convergence

            if norm(r*z)^2 <= rtol*(2-rtol)*(norm(r*x)^2 + par*delta^2)
                x = x + z ;
                if printer==1
                    disp('Exit with negative curvature')
                end
                % disp (sprintf('parl = %6.3e, paru = %6.3e',parl,paru))
                qmin = 0.5*x'*(a*x) + x'*b ;
                return
            end

            %           Test for convergence on exceptional cases

            if xnorm == 0.0
                qmin = 0.5*x'*(a*x) + x'*b ;
                return
            end
        end

        %        Compute the Newton correction parc to par.

        if xnorm == 0.0
            parc = -par ;
        else
            q = x/xnorm ;
            q = (r')\q ;
            qnorm = norm(q) ;
            parc = (((xnorm - delta)/delta)/qnorm)/qnorm ;
        end

        %        Update parl or paru.

        if xnorm > delta
            parl = max(parl,par) ;
        else
            paru = min(paru,par) ;
        end

        %     Case 2: A + par*I is not positive definite

    else

        %        In this version of the code parc = 0
        %        and choose the new par by simple extrapolation

        parc = 0.0 ;
        pars = max(pars,par+parc) ;

        par = min(10*par,paru) ;

        %        If necessary, increase paru slightly

        paru = max(paru,(1+rtol)*pars) ;

    end

    %     Use pars to update parl

    parl = max(parl,pars) ;

    %     Compute an improved estimate for par

    par = max(parl,par+parc) ;

end
if printer==1
    disp('WARNING: Exit after itmax iterations')
end
qmin = 0.5*x'*(a*x) + x'*b ; % ADDED BY SMW 9/23 !!!!!
% disp (sprintf('parl = %6.3e, paru = %6.3e',parl,paru))
% disp (sprintf('pars = %6.3e, par = %6.3e',pars,par))
