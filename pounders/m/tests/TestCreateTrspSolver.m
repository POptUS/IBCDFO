% To run this test, you must first install a MINQ clone and add
%    /path/to/MINQ/m/minq5
%    /path/to/MINQ/m/minq8
% to the MATLAB path.
%
% From MATLAB and the directory containing this file, execute
%     >> runtests
%
% If you would like to run this from a different folder that includes these
% tests as a subfolder, then from that folder execute
%     >> runtests("IncludeSubfolders", true)
%
% To execute the test suite with coverage enabled and to generate an HTML-format
% coverage report, execute from /path/to/IBCDFO/pounders/m
%     >> runtests("IncludeSubfolders", true, "ReportCoverageFor", pwd)
%

classdef TestCreateTrspSolver < matlab.unittest.TestCase
    properties (Constant)
        WARNING_SIMPLE = 'POUNDERS:simpleTrspSolver'
        ERROR_BAD_SOLVER = 'POUNDERS:badValue'
        % These should match the values used in create_trsp_solver().
        SOLVER_SIMPLE = 1
        SOLVER_MINQ5 = 2
        SOLVER_MINQ8 = 3
    end

    properties
        oldpath
        solvers
        officialSolvers
        emitWarnings
    end

    methods (TestMethodSetup)

        function setUp(testCase)
            testCase.solvers = [testCase.SOLVER_SIMPLE, ...
                                testCase.SOLVER_MINQ5, ...
                                testCase.SOLVER_MINQ8];
            testCase.officialSolvers = [testCase.SOLVER_MINQ5, ...
                                        testCase.SOLVER_MINQ8];

            testCase.emitWarnings = struct([]);
            testCase.emitWarnings(1).solver = testCase.SOLVER_SIMPLE;
            testCase.emitWarnings(1).warnID = testCase.WARNING_SIMPLE;

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

    methods (Test)

        function testErrors(testCase)
            badSolvers = [min(testCase.solvers) - 1, ...
                          max(testCase.solvers) + 1];
            for i = 1:length(badSolvers)
                bad = badSolvers(i);
                testCase.assertError(@() create_trsp_solver(bad), ...
                                     testCase.ERROR_BAD_SOLVER);
            end
        end

        function testWarnings(testCase)
            for i = 1:length(testCase.emitWarnings)
                idx = testCase.emitWarnings(i).solver;
                warnID = testCase.emitWarnings(i).warnID;
                testCase.assertWarning(@() create_trsp_solver(idx), warnID);
            end
        end

        function test1D(testCase)
            % Specify problem
            N = 1;
            G = [-1.1];
            H = [[2.2]];
            Low = [-1.9];
            Upp = [0.9];
            testCase.assertTrue(H(1, 1) > 0.0);

            % Known solutions
            s_expected = 0.5;
            f_expected = -0.275;

            too_small = [0.25];
            s_small = too_small(1);
            f_small = -0.20625;

            too_large = [0.8];
            s_large = too_large(1);
            f_large = -0.176;

            % Expected emission of specific warnings tested in testWarnings.
            % Ignore only those to silence expected warnings without
            % inadvertently silencing unintended warnings.
            warning("off", testCase.WARNING_SIMPLE);
            for i = 1:length(testCase.solvers)
                idx = testCase.solvers(i);

                solve_trsp = create_trsp_solver(idx);
                testCase.assertTrue(isa(solve_trsp, 'function_handle'));

                % Unconstrained solution in bounds
                [s_0, f_0, found_solution] = solve_trsp(H, G, Low, Upp);
                testCase.assertTrue(found_solution);
                testCase.assertEqual(ndims(s_0), 2);
                testCase.assertEqual(size(s_0), [N 1]);
                testCase.assertEqual(ndims(f_0), 2);
                testCase.assertEqual(size(f_0), [1 1]);
                rel_err = abs(1.0 - s_0 / s_expected);
                testCase.assertTrue(rel_err <= 110.0 * eps);
                rel_err = abs(1.0 - f_0 / f_expected);
                testCase.assertTrue(rel_err <= 110.0 * eps);

                % Unconstrained solution outside bounds
                [s_0, f_0, found_solution] = solve_trsp(H, G, Low, too_small);
                testCase.assertTrue(found_solution);
                testCase.assertEqual(ndims(s_0), 2);
                testCase.assertEqual(size(s_0), [N 1]);
                testCase.assertEqual(ndims(f_0), 2);
                testCase.assertEqual(size(f_0), [1 1]);
                testCase.assertEqual(s_0, s_small);
                rel_err = abs(1.0 - f_0 / f_small);
                testCase.assertTrue(rel_err <= 35.0 * eps);

                if idx ~= testCase.SOLVER_SIMPLE
                    % The simple sampler requires that Low <= 0 <= Upp
                    [s_0, f_0, found_solution] = solve_trsp(H, G, too_large, Upp);
                    testCase.assertTrue(found_solution);
                    testCase.assertEqual(ndims(s_0), 2);
                    testCase.assertEqual(size(s_0), [N 1]);
                    testCase.assertEqual(ndims(f_0), 2);
                    testCase.assertEqual(size(f_0), [1 1]);
                    testCase.assertEqual(s_0, s_large);
                    rel_err = abs(1.0 - f_0 / f_large);
                    testCase.assertTrue(rel_err <= 500.0 * eps);
                end
            end
            warning("on", testCase.WARNING_SIMPLE);
        end

        function test2D(testCase)
            % Specify problem
            N = 2;
            G = [1.2; -2.3];
            H = [[1.1 -1.2]
                 [-1.2 4.5]];
            [lambdas] = eig(H, "vector");
            testCase.assertTrue(isequal(H', H));
            testCase.assertTrue(all(lambdas > 0.5));
            Low = [-7.0 -4.0];
            Upp = [ 5.0  4.0];

            % Known solution
            s_expected = [-0.75213675; 0.31054131];
            f_expected = -0.8084045584045583;

            % Expected emission of specific warnings tested in testWarnings.
            % Ignore only those to silence expected warnings without
            % inadvertently silencing unintended warnings.
            warning("off", testCase.WARNING_SIMPLE);
            for i = 1:length(testCase.solvers)
                idx = testCase.solvers(i);

                solve_trsp = create_trsp_solver(idx);
                testCase.assertTrue(isa(solve_trsp, 'function_handle'));

                % Unconstrained solution in bounds
                [s_0, f_0, found_solution] = solve_trsp(H, G, Low, Upp);
                testCase.assertTrue(found_solution);
                testCase.assertEqual(ndims(s_0), 2);
                testCase.assertEqual(size(s_0), [N 1]);
                testCase.assertEqual(ndims(f_0), 2);
                testCase.assertEqual(size(f_0), [1 1]);
                max_rel_err = max(abs(1.0 - s_0 ./ s_expected));
                testCase.assertTrue(max_rel_err <= 5.0e-9);
                rel_err = abs(1.0 - f_0 / f_expected);
                testCase.assertTrue(rel_err <= 75.0 * eps);
            end
            warning("on", testCase.WARNING_SIMPLE);
        end

        function test5D(testCase)
            % Specify problem
            N = 5;
            G = [1.2; -2.3; 0.7; -0.4; 3.4];
            H = [[30.25 38.5 115.5 -19.25 8.25]
                 [38.5 50.0 131.0 -14.5 5.5]
                 [115.5 131.0 818.0 -211.5 67.5]
                 [-19.25 -14.5 -211.5 388.5 -5.5]
                 [8.25 5.5 67.5 -5.5 64.5]];
            [lambdas] = eig(H, "vector");
            testCase.assertTrue(isequal(H', H));
            testCase.assertTrue(all(lambdas > 0.01));
            Low = [-150.0 -10.0 -1.0 -2.0 -1.0];
            Upp = [  10.0 100.0  3.0  0.5  7.0];

            % Known solution
            s_expected = [-128.82782698
                            90.82291477
                             2.8264531
                            -1.37446078
                             5.60555695];
            f_expected = -170.94945062515922;

            % Setting maxit=600,000 in bqmin yielded a solution that was of
            % similar quality to MINQ5's solution.  Since, that's far more that
            % the real budget, we skip it.  We consider a passing 2D test as
            % sufficient evidence of correct functionality for that unofficial
            % solver.
            for i = 1:length(testCase.officialSolvers)
                idx = testCase.officialSolvers(i);

                solve_trsp = create_trsp_solver(idx);
                testCase.assertTrue(isa(solve_trsp, 'function_handle'));

                % Unconstrained solution in bounds
                [s_0, f_0, found_solution] = solve_trsp(H, G, Low, Upp);
                testCase.assertTrue(found_solution);
                testCase.assertEqual(ndims(s_0), 2);
                testCase.assertEqual(size(s_0), [N 1]);
                testCase.assertEqual(ndims(f_0), 2);
                testCase.assertEqual(size(f_0), [1 1]);
                max_rel_err = max(abs(1.0 - s_0 ./ s_expected));
                testCase.assertTrue(max_rel_err <= 2.5e-9);
                rel_err = abs(1.0 - f_0 / f_expected);
                testCase.assertTrue(rel_err <= 7.5e-11);
            end
        end

    end
end
