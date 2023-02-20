classdef Testpounders < matlab.unittest.TestCase
    methods (Test)

        function shortTests(testCase)
            test_failing_objective;
            test_bounds_and_sp1;
            test_bmpts;
        end

        function longTest(testCase)
            benchmark_pounders;
        end

    end
end
