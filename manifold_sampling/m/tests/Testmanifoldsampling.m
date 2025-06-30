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
% coverage report, execute from /path/to/IBCDFO/manifold_sampling/m
%     >> runtests("IncludeSubfolders", true, "ReportCoverageFor", pwd)
%

classdef Testmanifoldsampling < matlab.unittest.TestCase
    methods (Test)

        function shortTests(testCase)
            test_failing_objective;
            test_manifold_sampling_simple;
        end

        function longTest(testCase)
            benchmark_manifold_sampling;
        end

    end
end
