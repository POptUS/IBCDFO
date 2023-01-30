classdef testPounders < matlab.unittest.TestCase
    methods (Test)

        function realSolution(testCase)
            test_failing_objective;
        end

        function all(testCase)
            benchmark_pounders;
        end

        function more(testCase)
            test_bounds_and_sp1;
        end

    end
end
