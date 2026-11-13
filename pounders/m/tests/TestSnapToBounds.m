% From MATLAB and the directory containing this file, execute
%     >> runtests
%
% If you would like to run this from a different folder that includes these
% tests as a subfolder, then from that folder execute
%     >> runtests("IncludeSubfolders", true)
%

classdef TestSnapToBounds < matlab.unittest.TestCase
    properties (Constant)
        ID = 'IBCDFO:boundViolation'
        LOW = [-1.0, -1.0, -1.0]
        UPP = [1.0, 1.0, 1.0]
        SNAP_TOL = 1e-6
    end

    properties
        oldpath
    end

    methods (TestMethodSetup)

        function setUp(testCase)
            warning("on");
            [here_path, ~, ~] = fileparts(mfilename('fullpath'));
            testCase.oldpath = addpath(fullfile(here_path, '..'));
        end

    end

    methods (TestMethodTeardown)

        function tearDown(testCase)
            path(testCase.oldpath);
        end

    end

    methods

        function f = snapper(testCase, x, Low, Upp, snap_tol)
            % Handle to snap_to_bounds with this test class's defaults.
            if nargin < 3
                Low = testCase.LOW;
                Upp = testCase.UPP;
            end
            if nargin < 5
                snap_tol = testCase.SNAP_TOL;
            end
            f = @() snap_to_bounds(x, Low, Upp, snap_tol, 'x');
        end

    end

    methods (Test)

        function testFeasiblePointIsUnchanged(testCase)
            x = [-1.0, 0.25, 1.0];
            snapped = testCase.verifyWarningFree(testCase.snapper(x));
            testCase.verifyEqual(snapped, x);
        end

        function testViolationBelowTheNoiseFloorIsSilent(testCase)
            x = [-1.0 - 1e-15, 0.0, 1.0 + 1e-15];
            snapped = testCase.verifyWarningFree(testCase.snapper(x));
            testCase.verifyEqual(snapped, [-1.0, 0.0, 1.0]);
        end

        function testViolationWithinSnapTolWarnsAndSnaps(testCase)
            x = [0.0, 0.0, 1.0 + 1e-10];
            snapped = testCase.verifyWarning(testCase.snapper(x), testCase.ID);
            testCase.verifyEqual(snapped, [0.0, 0.0, 1.0]);
        end

        function testViolationBeyondSnapTolErrors(testCase)
            bad = {[0.0, 0.0, 1.1], [-1.1, 0.0, 0.0]};
            for i = 1:numel(bad)
                testCase.verifyError(testCase.snapper(bad{i}), testCase.ID);
            end
        end

        function testEachPointOfAMatrixIsChecked(testCase)
            X = [0.0, 0.0, 0.0; 0.0, 0.0, 1.1];
            testCase.verifyError(testCase.snapper(X), testCase.ID);
        end

        function testInfiniteBoundsAreNeverViolated(testCase)
            x = [-1.0e300, 0.0, 1.0e300];
            snapped = testCase.verifyWarningFree(testCase.snapper(x, -inf(1, 3), inf(1, 3)));
            testCase.verifyEqual(snapped, x);
        end

        function testLargeXTripsOnItsOwnRounding(testCase)
            % The check is absolute, so it does not scale with x. One unit in
            % the last place of 1e10 is ~1.9e-6, past the default snap_tol,
            % and a solver that lands exactly on the bound still trips the
            % check. This pins that documented limitation; raising snap_tol
            % is the remedy.
            Upp = 1.0e10 * ones(1, 3);
            x = [0.0, 0.0, 1.0e10 + eps(1.0e10)];
            testCase.assertGreaterThan(x(3) - Upp(3), testCase.SNAP_TOL);
            testCase.verifyError(testCase.snapper(x, -Upp, Upp), testCase.ID);
            testCase.verifyWarning(testCase.snapper(x, -Upp, Upp, 1e-5), testCase.ID);
        end

    end
end
