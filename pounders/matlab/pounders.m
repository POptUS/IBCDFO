% POUNDerS Version 0.1,    Modified 04/9/2010. Copyright 2010
% Stefan Wild and Jorge More', Argonne National Laboratory.
%
%   [X,F,flag,xkin] = ...
%        pounders(fun,X0,n,npmax,nfmax,gtol,delta,nfs,m,F0,xkin,L,U,printf)
%
% This code minimizes a blackbox function, solving
% min { f(X)=sum_(i=1:m) F_i(x)^2, such that L_j <= X_j <= U_j, j=1,...,n }
% where the user-provided F is specified in the handle fun. Evaluation of
% this F must result in the return of a 1-by-m row vector. Bounds must be
% specified in U and L but can be set to L=-Inf(1,n) and U=Inf(1,n) if the
% unconstrained solution is desired. The algorithm will not evaluate F
% outside of these bounds, but it is possible to take advantage of function
% values at infeasible X if these are passed initially through (X0,F0).
% In each iteration, the algorithm forms an interpolating quadratic model
% of the function and minimizes it in an infinity-norm trust region.
%
% This is an older MATLAB/OCTAVE implementation of POUNDerS (Practical
% Optimization Using No Derivatives for sums of Squares).
% It comes with no warranty, is not bug-free, and is not for industrial use
% or public distribution. Direct requests and bugs to wild@mcs.anl.gov.
% A technical report/manual is forthcoming, a brief description is in
% Nuclear Energy Density Optimization. Phys. Rev. C, 82:024313, 2010.
%
% --INPUTS-----------------------------------------------------------------
% fun     [f h] Function handle so that fun(x) evaluates F (@calfun)
% X0      [dbl] [max(nfs,1)-by-n] Set of initial points  (zeros(1,n))
% n       [int] Dimension (number of continuous variables)
% npmax   [int] Maximum number of interpolation points (>n+1) (2*n+1)
% nfmax   [int] Maximum number of function evaluations (>n+1) (100)
% gtol    [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
% delta   [dbl] Positive trust region radius (.1)
% nfs     [int] Number of function values (at X0) known in advance (0)
% m       [int] Number of residual components
% F0      [dbl] [nfs-by-m] Set of known function values  ([])
% xkin    [int] Index of point in X0 at which to start from (1)
% L       [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
% U       [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
% printf  [log] 1 Indicates you want output to screen (1)
% --OUTPUTS----------------------------------------------------------------
% X       [dbl] [nfmax+nfs-by-n] Locations of evaluated points
% F       [dbl] [nfmax+nfs-by-m] Function values of evaluated points
% flag    [dbl] Termination criteria flag:
%               = 0 normal termination because of grad,
%               > 0 exceeded nfmax evals,   flag = norm of grad at final X
%               = -1 if input was fatally incorrect (error message shown)
%               = -2 model failure 
% xkin    [int] Index of point in X representing approximate minimizer
%
% --DEPENDS ON-------------------------------------------------------------
% bqmin or minqsw : Subproblem solver (could be updated!)
% checkinputss : Checks the inputs are feasible
% formquad, phi2eval  :  Forms interpolation set and fits quadratic models
% bmpts, boxline : Generates feasible model-improving points
function [X,F,flag,xkin] = ...
    pounders(fun,X0,n,npmax,nfmax,gtol,delta,nfs,m,F0,xkin,L,U,printf)

% Choose your solver:
global spsolver
%spsolver=1; % Stefan's crappy solver
if spsolver == 2
    addpath('../../minq/minq5/matlab'); % Arnold Neumaier's minq5
elseif spsolver == 3
    addpath('../../minq/minq8/matlab'); % Arnold Neumaier's minq8
end

% 0. Check inputs
[flag,X0,npmax,F0,L,U] = ...
    checkinputss(fun,X0,n,npmax,nfmax,gtol,delta,nfs,m,F0,xkin,L,U);
if (flag == -1), X=[]; F=[]; return; end; % Problem with the input

% --INTERNAL PARAMETERS [won't be changed elsewhere, defaults in ( ) ]-----
maxdelta = min(.5*min(U-L),1e3*delta); % [dbl] Maximum tr radius
mindelta = min(delta*1e-13,gtol/10); % [dbl] Min tr radius (technically 0)
gam0 = .5;      % [dbl] Parameter in (0,1) for shrinking delta  (.5)
gam1 = 2;       % [dbl] Parameter >1 for enlarging delta   (2)
eta1 = .05;     % [dbl] Parameter 2 for accepting point, 0<eta1<1 (.2)
Par(1) = sqrt(n); % [dbl] delta multiplier for checking validity
Par(2) = max(10,sqrt(n)); % [dbl] delta multiplier for all interp. points
Par(3) = 1e-3;  % [dbl] Pivot threshold for validity (1e-5)
Par(4) = .001;  % [dbl] Pivot threshold for additional points (.001)
if printf
    disp('  nf   delta    fl  np       f0           g0       ierror');
    progstr = '%4i %9.2e %2i %3i  %11.5e %12.4e %11.3e\n'; % Line-by-line
end
% -------------------------------------------------------------------------

% --INTERMEDIATE VARIABLES-------------------------------------------------
% D       [dbl] [1-by-n] Generic displacement vector
% G       [dbl] [n-by-1] Model gradient at X(xkin,:)
% H       [dbl] [n-by-n] Model Hessian at X(xkin,:)
% Hdel    [dbl] [n-by-n] Change to model Hessian at X(xkin,:)
% Lows    [dbl] [1-by-n] Vector of subproblem lower bounds
% Upps    [dbl] [1-by-n] Vector of subproblem upper bounds
% Mdir    [dbl] [n-by-n] Unit row directions to improve model/geometry
% Mind    [int] [npmax-by-1] Integer vector of model interpolation indices
% Xsp     [dbl] [1-by-n] Subproblem solution
% c       [dbl] Model value at X(xkin,:)
% mdec    [dbl] Change predicted by the model, m(nf)-m(xkin)
% nf      [int] Counter for the number of function evaluations
% ng      [dbl] Norm of (projection of) G
% np      [int] Number of model interpolation points
% rho     [dbl] Ratio of actual decrease to model decrease
% valid   [log] Flag saying if model is fully linear within Par(1)*delta
% -------------------------------------------------------------------------

if nfs==0 % Need to do the first evaluation
    X = [X0; zeros(nfmax-1,n)]; % Stores the point locations
    F = zeros(nfmax,m); % Stores the function values
    nf = 1;
    F(nf,:) = fun(X(nf,:));
    if printf
        fprintf('%4i    Initial point  %11.5e\n',nf,sum(F(nf,:).^2));
    end
else % Have other function values around
    X = [X0(1:max(1,nfs),:); zeros(nfmax,n)]; % Stores the point locations
    F = [F0(1:nfs,:); zeros(nfmax,m)]; % Stores the function values
    nf = nfs;
    nfmax = nfmax+nfs;
end
Fs = zeros(nfmax+nfs,1); % Stores the sum of squares of evaluated points
for i=1:nf
    Fs(i) = sum(F(i,:).^2);
end

Res = zeros(size(F)); % Stores the residuals for model updates
Cres = F(xkin,:); Hres = zeros(n,n,m);
%H = zeros(n); G = zeros(n,1); c = Fs(xkin);

%! NOTE: Currently do not move to a geometry point (including in
% the first iteration!) if it has a lower f value

while nf<nfmax
    % 1a. Compute the interpolation set.
    for i=1:nf
        D = X(i,:)-X(xkin,:);
        for j=1:m
            Res(i,j) = (F(i,j)-Cres(j))-.5*D*Hres(:,:,j)*D';
        end
    end
    [Mdir,np,valid,Gres,Hresdel,Mind] = ...
        formquad(X(1:nf,:),Res(1:nf,:),delta,xkin,npmax,Par,0);
    if np<n  % Must obtain and evaluate bounded geometry points
        [Mdir,np] = bmpts(X(xkin,:),Mdir(1:n-np,:),L,U,delta,Par(3));
        for i=1:min(n-np,nfmax-nf)
            nf = nf+1;
            X(nf,:) = min(U,max(L,X(xkin,:)+Mdir(i,:))); % Temp safeguard
            F(nf,:) = fun(X(nf,:)); Fs(nf) = sum(F(nf,:).^2);
            if printf
                fprintf('%4i   Geometry point  %11.5e\n',nf,Fs(nf));
            end
            D = Mdir(i,:);
            for j=1:m
                Res(nf,j) = (F(nf,j)-Cres(j))-.5*D*Hres(:,:,j)*D';
            end
        end
        if nf>=nfmax; break; end
        [~,np,valid,Gres,Hresdel,Mind] = ...
            formquad(X(1:nf,:),Res(1:nf,:),delta,xkin,npmax,Par,0);
    end
    
    % 1b. Update the quadratic model
    Cres = F(xkin,:); Hres = Hres+Hresdel;
    c = Fs(xkin);  
    G = 2*Gres*F(xkin,1:m)';
    H = zeros(n);
    for i=1:m
        H = H + F(xkin,i)*Hres(:,:,i);
    end
    H = 2*H + 2*(Gres*Gres');
    ng = norm(G.*( and(X(xkin,:)>L,G'>0) + and(X(xkin,:)<U,G'<0) )');
    
    if printf   % Output stuff: ---------(can be removed later)------------
        IERR = zeros(1,size(Mind,1));
        for i=1:size(Mind,1)
            D = (X(Mind(i),:)-X(xkin,:));
            IERR(i) = (c-Fs(Mind(i)))+D*(G+.5*H*D');
        end
        for i=1:size(Mind,1)
            D = (X(Mind(i),:)-X(xkin,:));
            for j=1:m
                jerr(i,j) = (Cres(j)-F(Mind(i),j))+D*(Gres(:,j)+.5*Hres(:,:,j)*D');
            end
        end
        ierror=norm(IERR./max(abs(Fs(Mind,:)'),0),inf); % Interp. error
        fprintf(progstr, nf, delta, valid, np, Fs(xkin), ng, ierror);
    end %------------------------------------------------------------------
    
    
    % 2. Criticality test invoked if the projected model gradient is small
    if ng<gtol
        % Check to see if the model is valid within a region of size gtol
        delta = max(gtol, max(abs(X(xkin,:)))*eps); % Safety for tiny gtols
        [Mdir,~,valid] = ...
            formquad(X(1:nf,:),F(1:nf,:),delta,xkin,npmax,Par,1);
        if ~valid % Make model valid in this small region
            [Mdir,np] = bmpts(X(xkin,:),Mdir,L,U,delta,Par(3));
            for i = 1:min(n-np,nfmax-nf)
                nf = nf + 1;
                X(nf,:) = min(U,max(L,X(xkin,:)+Mdir(i,:))); % Temp safeg.
                F(nf,:) = fun(X(nf,:)); Fs(nf) = sum(F(nf,:).^2);
                if printf
                    fprintf('%4i   Critical point  %11.5e\n',nf,Fs(nf));
                end
            end
            if nf>=nfmax; break; end
            % Recalculate gradient based on a MFN model
            [~,~,valid,Gres,Hres,Mind] = ...
                formquad(X(1:nf,:),F(1:nf,:),delta,xkin,npmax,Par,0);
            G = 2*Gres*F(xkin,1:m)';
            H = zeros(n);
            for i=1:m
                H = H + F(xkin,i)*Hres(:,:,i);
            end
            H = 2*H + 2*(Gres*Gres');
            ng = norm(G.*(and(X(xkin,:)>L,G'>0)+and(X(xkin,:)<U,G'<0))');
        end
        if ng<gtol % We trust the small gradient norm and return
            if printf, disp('g is sufficiently small'), end
            X = X(1:nf,:);  F = F(1:nf,:);  flag = 0;  return;
        end
    end
   
    
    % 3. Solve the subproblem min{G'*s+.5*s'*H*s : Lows <= s <= Upps }
    Lows = max((L-X(xkin,:)),-delta);
    Upps = min((U-X(xkin,:)),delta);
    if spsolver==1 % Stefan's crappy 10line solver
        [Xsp,mdec] = bqmin(H,G,Lows,Upps);
    elseif spsolver==2 % Arnold Neumaier's minq5
        [Xsp,mdec] = minqsw(0,G,H,Lows',Upps',0,zeros(n,1));
                     
    elseif spsolver==3 % Arnold Neumaier's minq8
        
        data.gam = 0;
        data.c = G;
        data.b = zeros(n,1);
        [tmp1,tmp2] = ldl(H);
        data.D = diag(tmp2);
        data.A = tmp1';
        
        [Xsp,mdec] = minq8(data,Lows',Upps',zeros(n,1),10*n);
    end
    Xsp = Xsp'; % Solvers currently work with column vectors
    step_norm = norm(Xsp,inf);

    % 4. Evaluate the function at the new point (provided mdec isn't zero with an invalid model)
    if (step_norm >= 0.01*delta || valid) && ~(mdec==0 && ~valid)

        Xsp = min(U,max(L,X(xkin,:)+Xsp));  % Temp safeguard; note Xsp is not a step anymore
        
        % Project if we're within machine precision
        for i=1:n %! This will need to be cleaned up eventually
            if U(i)-Xsp(i)<eps*abs(U(i)) && U(i)>Xsp(i) && G(i)>=0
                Xsp(i) = U(i); disp('eps project!')
            elseif Xsp(i)-L(i)<eps*abs(L(i)) && L(i)<Xsp(i) && G(i)>=0
                Xsp(i) = L(i);  disp('eps project!')
            end
        end

        if mdec == 0 && valid && all(Xsp == X(xkin,:))
            disp('Terminating because mdec == 0 with a valid model and no change in Xsp')
            X = X(1:nf,:);  F = F(1:nf,:);  flag = -2;  return;
        end

        nf = nf + 1;
        X(nf,:) = Xsp;
        F(nf,:) = fun(X(nf,:)); Fs(nf) = sum(F(nf,:).^2);

        if mdec ~= 0
            rho = (Fs(nf)-Fs(xkin))/mdec;
        else % Note: this conditional only occurs when model is valid
            if Fs(nf)==Fs(xkin)
                disp('Terminating because mdec == 0 with a valid model and Fs(nf) == Fs(xkin)')
                X = X(1:nf,:);  F = F(1:nf,:);  flag = -2;  return;
            else
                rho = np.inf*sign(Fs(nf)-Fs(xkin));
            end
        end

        % 4a. Update the center
        if (rho >= eta1)  || ((rho>0) && (valid))
            %  Update model to reflect new center
            Cres = F(xkin,:);
            xkin = nf; % Change current center
        end
        
        % 4b. Update the trust-region radius:
        if (rho>=eta1)  &&  (step_norm>.75*delta)
            delta = min(delta*gam1,maxdelta);
        elseif (valid)
            delta = max(delta*gam0,mindelta);
        end
    else % Don't evaluate f at Xsp
        rho = -1; % Force yourself to do a model-improving point
        if printf, disp('Warning: skipping sp soln!---------'); end
    end
    
    % 5. Evaluate a model-improving point if necessary
    if ~(valid) && (nf<nfmax) && (rho<eta1) % Implies xkin,delta unchanged
        % Need to check because model may be valid after Xsp evaluation
        [Mdir,np,valid] = ...
            formquad(X(1:nf,:),F(1:nf,:),delta,xkin,npmax,Par,1);
        if ~(valid)  %! One strategy for choosing model-improving point:
            % Update model (exists because delta & xkin unchanged)
            for i=1:nf
                D = (X(i,:)-X(xkin,:));
                for j=1:m
                    Res(i,j) = (F(i,j)-Cres(j))-.5*D*Hres(:,:,j)*D';
                end
            end
            [~,~,valid,Gres,Hresdel,Mind] = ...
                formquad(X(1:nf,:),Res(1:nf,:),delta,xkin,npmax,Par,0);
            Hres = Hres+Hresdel;
            % Update for modelimp; Cres unchanged b/c xkin unchanged
            G = 2*Gres*F(xkin,1:m)';
            H = zeros(n);
            for i=1:m
                H = H + F(xkin,i)*Hres(:,:,i);
            end
            H = 2*H + 2*(Gres*Gres');
            
            % Evaluate model-improving points to pick best one
            %! May eventually want to normalize Mdir first for infty norm
            % Plus directions
            [Mdir1,np1] = bmpts(X(xkin,:),Mdir(1:n-np,:),L,U,delta,Par(3));
            for i = 1:n-np1
                D = Mdir1(i,:);
                Res(i,1) = D*(G+.5*H*D');
            end
            [a1,b] = min(Res(1:n-np1,1));
            Xsp = Mdir1(b,:);
            % Minus directions
            [Mdir1,np2] = bmpts(X(xkin,:),-Mdir(1:n-np,:),L,U,delta,Par(3));
            for i = 1:n-np2
                D = Mdir1(i,:);
                Res(i,1) = D*(G+.5*H*D');
            end
            [a2,b] = min(Res(1:n-np2,1));
            if a2<a1
                Xsp = Mdir1(b,:);
            end
            
            nf = nf + 1;
            X(nf,:) = min(U,max(L,X(xkin,:)+Xsp)); % Temp safeguard
            F(nf,:) = fun(X(nf,:)); Fs(nf) = sum(F(nf,:).^2);
            if printf
                fprintf('%4i   Model point     %11.5e\n',nf,Fs(nf));
            end
            if Fs(nf,:)<Fs(xkin,:)  %! Eventually check suff decrease here!
                if printf, disp('**improvement from model point****');  end
                %  Update model to reflect new base point
                D = (X(nf,:)-X(xkin,:));
                xkin = nf; % Change current center
                Cres = F(xkin,:);     
                % Don't actually use:
                for j=1:m, Gres(:,j)=Gres(:,j)+Hres(:,:,j)*D'; end
            end
        end
    end
end
if printf, disp('Number of function evals exceeded'); end
flag = ng;
