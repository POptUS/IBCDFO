% To run this test, you must first install a BenDFO clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
% and a MINQ clone and add
%    /path/to/MINQ/m/minq5
% to the MATLAB path.  The tests solve the GOOMBAH TRSP with Latin hypercube
% sampling and fmincon, so GAMS is not needed.
%
% From MATLAB and the directory containing this file, execute
%     >> runtests
%
% If you would like to run this from a different folder that includes these
% tests as a subfolder, then from that folder execute
%     >> runtests("IncludeSubfolders", true)
%
% To execute the test suite with coverage enabled and to generate an HTML-format
% coverage report, execute from /path/to/IBCDFO/goombah/m
%     >> runtests("IncludeSubfolders", true, "ReportCoverageFor", pwd)
%

classdef Testgoombah < matlab.unittest.TestCase
    methods (TestMethodSetup)

        function disableLhsWarning(testCase)
            % The tests use the LHS TRSP solver, which warns that it is only
            % for testing and debugging.
            oldwarn = warning('off', 'GOOMBAH:lhsTrspSolver');
            testCase.addTeardown(@warning, oldwarn);
        end

    end

    methods (Test)

        function shortTests(testCase)
            test_minimize_quadratics_lhs;
            test_goombah_failures;
            test_goombah_converge;
            test_goombah_budget_sweep;
        end

        function longTest(testCase)
            test_goombah_vs_msp;
            test_goombah_vs_pounders;
        end

    end
end
