function [nf, X, F, h, Hash, hashes_at_nf] = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, x_in, tol, Low, Upp, allow_recalc)
    % Call user scripts Ffun and hfun; saves their output to the appropriate arrays
    if nargin < 12
        allow_recalc = 0;
    end

    xnew = min(Upp, max(Low, x_in));

    for i = 1:length(xnew) % ! This will need to be cleaned up eventually
        if Upp(i) - xnew(i) < eps * abs(Upp(i)) && Upp(i) > xnew(i)
            xnew(i) = Upp(i);
            disp('eps project!');
        elseif xnew(i) - Low(i) < eps * abs(Low(i)) && Low(i) < xnew(i)
            xnew(i) = Low(i);
            disp('eps project!');
        end
    end

    if ~allow_recalc && ismember(xnew, X(1:nf, :), 'rows')
        error('Your method requested a point that has already been requested. FAIL!');
    end

    nf = nf + 1;
    X(nf, :) = xnew;
    F(nf, :) = Ffun(X(nf, :));
    [h(nf), ~, hashes_at_nf] = hfun(F(nf, :));
    Hash(nf, 1:length(hashes_at_nf)) = hashes_at_nf;

    assert(~any(isnan(F(nf, :))), 'Got a NaN. FAIL!');

    % It must be the case hfun values and gradients are the same when called with
    % and without hashes. Because we assume h is cheap to evaluate, we can
    % possibly detect errors in the implementation of hfun by checking each time
    % that the values and gradients agree:
    if tol.hfun_test_mode
        assert(iscell(hashes_at_nf), "hfun must return a cell of hashes");
        assert(all(cellfun(@ischar, hashes_at_nf)), "Hashes must be character arrays");

        [h_dummy1, grad_dummy1, hash_dummy] = hfun(F(nf, :));
        [h_dummy2, grad_dummy2] = hfun(F(nf, :), hash_dummy);
        assert(any(h_dummy1 == h_dummy2), ['hfun values do not agree when ' func2str(hfun) ' is re-called with the same inputs']);
        assert(all(all(grad_dummy1 == grad_dummy2)), ['hfun gradients do not agree when '  func2str(hfun) ' is re-called with the same inputs']);
    end
end
