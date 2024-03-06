classdef Testmanifoldsampling < matlab.unittest.TestCase
    methods (Test)

        function shortTests(testCase)
            test_failing_objective;
            test_one_norm;
        end

        function longTest(testCase)
            benchmark_manifold_sampling;
        end

    end
end
