% bmpts.m    Modified 04/9/2010. Copyright 2010
% Stefan Wild and Jorge More', Argonne National Laboratory.
function [Modeld, np] = bmpts(X, Modeld, Low, Upp, delta, theta)
[num, n] = size(Modeld);

% For each ray, find largest t to remain feasible
T = zeros(2, num);
for j = 1:num
    T(1, j) = boxline(delta * Modeld(j, :), X, Low, Upp);
    T(2, j) = boxline(-delta * Modeld(j, :), X, Low, Upp);
end

if min(max(T)) >= theta % Safe to use our directions:
    np = n - num;
    for  j = 1:num
        if  T(1, j) >= T(2, j)
            Modeld(j, :) = delta * Modeld(j, :) * T(1, j);
        else
            Modeld(j, :) = -delta * Modeld(j, :) * T(2, j);
        end
    end
else
    % May want to turn this display off
    disp('Note: Geometry points need to be coordinate directions!');
    np = 0;
    Modeld = zeros(n);
    for j = 1:n
        t1 = min(X(j) - Low(j), delta);
        t2 = min(Upp(j) - X(j), delta);
        if t1 >= t2
            Modeld(j, j) = -t1;
        else
            Modeld(j, j) =  t2;
        end
    end
end
