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
        emitWarnings
    end

    methods (TestMethodSetup)
        function setUp(testCase)
            testCase.solvers = [testCase.SOLVER_SIMPLE, ...
                                testCase.SOLVER_MINQ5, ...
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

        function testSuccessful(testCase)
            % Expected emission of specific warnings tested in testWarnings.
            % Ignore only those to silence expected warnings without
            % inadvertently silencing unintended warnings.
            warning("off", testCase.WARNING_SIMPLE);
            for i = 1:length(testCase.solvers)
                idx = testCase.solvers(i);
                solve_trsp = create_trsp_solver(idx);
                testCase.assertTrue(isa(solve_trsp, 'function_handle'));
            end
            warning("on", testCase.WARNING_SIMPLE);
        end
    end

end
