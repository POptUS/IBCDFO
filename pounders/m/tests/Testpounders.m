% To run this test, you must first install a BenDFO clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
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

classdef Testpounders < matlab.unittest.TestCase
    methods (Test)

        function shortTests(testCase)
            test_failing_objective;
            %test_bounds_and_sp1;
            %test_bmpts;
        end

        %function longTest(testCase)
        %    benchmark_pounders;
        %end

    end
end
