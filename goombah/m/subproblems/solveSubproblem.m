function [x,A,f,c] = solveSubproblem(W,P,a,A,t)

% Code developed by Frank Curtis and Xiaocun Que as a part of the project described in
% https://doi.org/10.1080/10556788.2012.714781
%
% Description : Active-set algorithm for the problem
%                 min  a'*x + (1/2)*x'*P'*W*P*x
%                 s.t. e'*x = 1, x >= 0.
%               Implements the "Quadratic Programming for Direction Finding
%               (QPDF)" method proposed by Kiwiel [IMA Journal of Numerical
%               Analysis, 6 (1986), pp. 137--152].  The only difference is
%               that in this implementation W need not be the identity.
% Input       : W ~ quadratic term data
%               P ~ quadratic term data
%               a ~ linear term data
%               A ~ initial guess of positive elements of x
%                   A(j) = 0 => x(j) = 0; A(j) = 1 => x(j) > 0
%               t ~ optimality tolerance
% Output      : x ~ final solution
%               A ~ final set of positive elements of x
%               f ~ termination flag
%                     0 ~ no flag set (shouldn't happen!)
%                     1 ~ solved successfully
%                     2 ~ iteration limit reached
%                     3 ~ error in input
%                     4 ~ error in linear system solve
%                     5 ~ error in sign calculation (some value should have been negative!)
%               c ~ iteration counter

% Set size of P
[d,n] = size(P);
%W = eye(d);
% Initialize solution
x = sparse(n,1);

% Initialize number of iterations
c = 1;

% Check for error in input size
if size(W,1) ~= d | size(W,2) ~= d | size(a,1) ~= n | size(a,2) ~= 1 | length(A) > n, f = 3; return; end;

% Initialize termination flag
f = 0;

% Convert binary vector to vector of indices
A = find(A==1);

% Check for empty set
if isempty(A)

  % Compute ind = i with val = min_i {p(i)'*W*p(i)+a(i)}
  ind = -1; val = inf;
  for i = 1:n
    temp = P(:,i)'*W*P(:,i) + a(i);
    if temp < val
      ind = i; val = temp;
    end
  end

  % Check for sign error
  [x,A,f] = check_sign_error(x,A,f,ind,n,[]); if f > 0, return; end;

  % Initialize active set
  A = ind;

end

% Initialize objective data for nonzero elements
P_ = P(:,A); a_ = a(A);

% Initialize number of active columns
k = length(A);

% Initialize Cholesky factor
[R,err] = chol(ones(k,k)+P_'*W*P_);

% Check initial factorization
if err > 0

  % Simplify initial solution
  A = A(1); P_ = P(:,A); a_ = a(A); k = 1; R = sqrt(1+P_'*W*P_);

end

% Initialize intermediate vectors in subproblem solution
ye = (ones(k,1)'/R)'; ya = ((-a_)'/R)';

% Initialize multiplier in subproblem solution
v = (ye'*ye + ye'*ya - 1)/(ye'*ye);

% Solve subproblem
y = R\(ya + (1-v)*ye); y_ = y;

% Check initial factorization
if sum(y_ < 0) > 0

  % Simplify initial solution
  A = A(1); P_ = P(:,A); a_ = a(A); k = 1; R = sqrt(1+P_'*W*P_); ye = (ones(k,1)'/R)'; ya = ((-a_)'/R)'; v = (ye'*ye + ye'*ya - 1)/(ye'*ye); y = R\(ya + (1-v)*ye); y_ = y;

end

% Check for solve error
[x,A,f] = check_solve_error(x,A,f,{y,v},n,y_); if f > 0, return; end;

% Set maximum iterations
% c_max = min(1000,2^(max(n,d)));
c_max = 1000;

% Iteration loop
while c < c_max

  % Compute KKT vector
  kkt = v + P'*W*P_*y_ + a;

  % Zero-out KKT vector elements in A
  kkt(A) = 0;

  % Find minimum of KKT vector (m < 0 => A augmented w/ j)
  [m,j] = min(kkt);

  % Optimality check
  if m >= -t, f = 1; break; end;

  % Increment iteration counter
  c = c + 1;

  % Solve for intermediate vector in least squares system
  r = ((ones(k,1)+P_'*W*P(:,j))'/R)';

  % Check for solve error
  [x,A,f] = check_solve_error(x,A,f,{r},n,y_); if f > 0, return; end;

  % Compute new diagonal for R (squared)
  rho2 = max(0,1 + P(:,j)'*W*P(:,j) - r'*r);

  % Initialize linear independence check boolean
  li = 0;

  % Check for sufficiently large new diagonal for R (squared)
  if rho2 > 1e3*eps*(1 + P(:,j)'*W*P(:,j))

    % Compute new diagonal for R
    rho = sqrt(rho2);

    % Linear independence check satisfied
    li = 1;

  else

    % Solve least squares system
    yt = R\r;

    % Check for solve error
    [x,A,f] = check_solve_error(x,A,f,{yt},n,y_); if f > 0, return; end;

    % Compute residuals in least squares system
    delta = sum(yt) - 1; Delta = P_*yt - P(:,j);

    % Compute new diagonal for R
    rho = sqrt(max(rho2,delta^2+Delta'*W*Delta));

    % Check for sufficiently negative delta
    if delta < -1e3*eps, li = 1;
    else

      % Compute ind = i with val = min_i {y(i)/yt(i) : yt(i) > 0}
      ind = -1; val = inf;
      for i = 1:k
        if yt(i) > 0
          temp = y(i)/yt(i);
          if temp < val
            ind = i; val = temp;
          end
        end
      end

      % Check for sign error
      [x,A,f] = check_sign_error(x,A,f,ind,n,y_); if f > 0, return; end;

      % Compute objective change
      Deltaw = (1/2)*val^2*norm(W*(Delta-delta*P(:,j)))^2 + val*(1+delta)*(v + P(:,j)'*W*P_*y + a(j));

      % Check objective change
      if Deltaw >= 1e-2*val*(v + P(:,j)'*W*P_*y + a(j)), li = 1; end

    end

  end

  % Choose between column augmentation and column exchange
  if li == 1

    % Set augmentation
    [A,P_,a_,y_,k] = set_augment(A,j,P_,P(:,j),a_,a(j),y_,0,k);

    % Augment Cholesky factor
    R = [R r; zeros(1,k-1) rho];

    % Augment intermediate vector in subproblem solution
    ye = [ye; (1-r'*ye)/rho]; ya = [ya; (-a(j)-r'*ya)/rho];

    % Update multiplier in subproblem solution
    v = (ye'*ye + ye'*ya - 1)/(ye'*ye);

    % Check for solve error
    [x,A,f] = check_solve_error(x,A,f,{ye,v},n,y_); if f > 0, return; end;

  else

    % Compute ind = i with val = min_i {y_(i)/yt(i) : yt(i) > 0}
    ind = -1; val = inf;
    for i = 1:k
      if yt(i) > 0
        temp = y_(i)/yt(i);
        if temp < val
          ind = i; val = temp;
        end
      end
    end

    % Check for sign error
    [x,A,f] = check_sign_error(x,A,f,ind,n,y_); if f > 0, return; end;

    % Update solution
    y_ = y_ - val*yt;

    % Set reduction
    [A,P_,a_,y_,k] = set_reduce(A,P_,a_,y_,k,ind);

    % Remove column from R
    R(:,ind) = [];

    % Apply Givens rotations
    [R,Y] = givens(R,ind,[ye ya r]);

    % Split Y
    ye = Y(:,1); ya = Y(:,2); r = Y(:,3);

    % Remove entries in intermediate vectors
    ye(end) = []; ya(end) = []; r(end) = [];

    % Set augmentation
    [A,P_,a_,y_,k] = set_augment(A,j,P_,P(:,j),a_,a(j),y_,val,k);

    % Modify entry if yt computed
    if rho2 <= 1e3*eps*(1 + P(:,j)'*W*P(:,j)), y_(end) = y_(end)*sum(yt); end;

    % Augment Cholesky factor
    R = [R r; zeros(1,k-1) rho];

    % Augment intermediate vector in subproblem solution
    if length(r) > 0, ye = [ye; (1-r'*ye)/rho]; ya = [ya; (-a(j)-r'*ya)/rho]; else ye = 1/R; ya = -a(j)/R; end;

    % Initialize multiplier in subproblem solution
    v = (ye'*ye + ye'*ya - 1)/(ye'*ye);

    % Check for solve error
    [x,A,f] = check_solve_error(x,A,f,{ye,v},n,y_); if f > 0, return; end;

  end

  % Subproblem solution loop
  while(1)

    % Solve subproblem
    y = R\(ya + (1-v)*ye);

    % Check for solve error
    [x,A,f] = check_solve_error(x,A,f,{y},n,y); if f > 0, return; end;

    % Check for feasibility
    if sum(y <= 0) == 0, y_ = y; break;
    else

      % Compute ind = i with val = min_i {y_(i)/(y_(i)-y(i)) : y(i) < 0}
      ind = -1; val = inf;
      for i = 1:k
        if y(i) < 0
          temp = y_(i)/(y_(i)-y(i));
          if temp < val
            ind = i; val = temp;
          end
        end
      end

      % Check for sign error
      [x,A,f] = check_sign_error(x,A,f,ind,n,y); if f > 0, return; end;

      % Update value
      val = min(1,val);

      % Update dual solution
      y_ = val*y + (1-val)*y_;

      % Set reduction
      [A,P_,a_,y_,k] = set_reduce(A,P_,a_,y_,k,ind);

      % Remove column from R
      R(:,ind) = [];

      % Apply Givens rotations
      [R,Y] = givens(R,ind,[ye ya]);

      % Split Y
      ye = Y(:,1); ya = Y(:,2);

      % Remove entry in intermediate vector in subproblem solution
      ye(end) = []; ya(end) = [];

      % Initialize multiplier in subproblem solution
      v = (ye'*ye + ye'*ya - 1)/(ye'*ye);

      % Check for solve error
      [x,A,f] = check_solve_error(x,A,f,{ye,v},n,y); if f > 0, return; end;

    end

  end

end

% Check for maximum iterations
if c >= c_max, f = 2; end;

% Set solution
[x,A] = construct_solution(n,A,y_);

end

% Linear system solve error checker
function [x,A,f] = check_solve_error(x,A,f,Y,n,y)

% Loop through elements of Y
for i = 1:length(Y)

  % Update error indicator
  if sum(isnan(Y{i})) > 0 | sum(isinf(Y{i})) > 0, f = 4; end;

end

% If error, then set solution
if f > 0, [x,A] = construct_solution(n,A,y); end;

end

% Sign error checker
function [x,A,f] = check_sign_error(x,A,f,v,n,y)

% Set error indicator
if v == -1, f = 5; end;

% If error, then set solution
if f > 0, [x,A] = construct_solution(n,A,y); end;

end

% Solution setter
function [x,A] = construct_solution(n,A,y)

% Construct final solution
x = sparse(n,1); x(A) = y;

% Construct active set
A_final = zeros(1,n); A_final(A) = 1; A = A_final;

end

% Givens rotator
function [R,Y] = givens(R,k,Y)

% Loop over indices
for i = k:(size(R,1)-1)

  % Set values
  p = max(R(i,i),R(i+1,i));
  q = min(R(i,i),R(i+1,i));
  r = abs(p)*sqrt(1+(q/p)^2);
  c =  R(i,i)  /r;
  s = -R(i+1,i)/r;

  % Compute Givens rotation
  Q = [c -s; s c];

  % Loop to transform
  for j = i:size(R,2)

    % Transform
    R(i:i+1,j) = Q*R(i:i+1,j);

  end

  % Transform y
  for j = 1:size(Y,2)

    % Transform
    Y(i:i+1,j) = Q*Y(i:i+1,j);

  end

end

% Set R
R = R(1:end-1,:);

end

% Set augmenter
function [A,P_,a_,y_,k] = set_augment(A,j,P_,p,a_,a,y_,y,k)

% Add j to A
A = [A j];

% Add column to P_
P_ = [P_ p];

% Add element to a_
a_ = [a_; a];

% Add element to y_
y_ = [y_; y];

% Increment k
k = k + 1;

end

% Set reducer
function [A,P_,a_,y_,k] = set_reduce(A,P_,a_,y_,k,i)

% Remove ith entry from A
A(i) = [];

% Remove ith column from P_
P_(:,i) = [];

% Remove ith element from a_
a_(i) = [];

% Remove ith element from y_
y_(i) = [];

% Decrement k
k = k - 1;

end
