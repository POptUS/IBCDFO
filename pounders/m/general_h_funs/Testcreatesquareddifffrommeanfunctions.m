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

classdef Testcreatesquareddifffrommeanfunctions < matlab.unittest.TestCase
    properties
        n;
        m;
        F;
        Cres;
        Gres;
        Hres;
    end

    methods (TestMethodSetup)

        function setup(testCase)
            % Test problem 1 setup
            %   - Here and in tests *_bar refers to the average of *
            testCase.n = 2;
            testCase.m = 3;
            testCase.F = [3.3; -1.1; 4.4];
            % F_bar = 2.2;
            % F_bar_sqr = 4.84;
            % Delta_F = F - F_bar = [1.1; -3.3; 2.2];
            % F_sum = 16.94;

            testCase.Cres = testCase.F;

            testCase.Gres = [[1.1 0.3 2.2]
                             [2.2 0.3 1.4]];
            % G_bar = [1.2 1.3];

            testCase.Hres = zeros([testCase.n testCase.n testCase.m]);
            testCase.Hres(:, :, 1) = [[1.2 -2.3]
                                      [-2.3 3.4]];
            testCase.Hres(:, :, 2) = [[-1.4 3.2]
                                      [3.2 -2.1]];
            testCase.Hres(:, :, 3) = [[3.5 1.8]
                                      [1.8 -2.5]];
        end

    end

    methods (Test)

        function badArguments(testCase)
            testCase.verifyError( ...
                @()create_squared_diff_from_mean_functions([]), ...
                'MATLAB:validators:mustBeNonempty' ...
            );

            testCase.verifyError( ...
                @()create_squared_diff_from_mean_functions([1.1 2.2]), ...
                'MATLAB:validators:mustBeScalarOrEmpty' ...
            );

            testCase.verifyError( ...
                @()create_squared_diff_from_mean_functions("hello"), ...
                'MATLAB:validators:mustBeReal' ...
            );

            for bad = [inf -inf nan]
                testCase.verifyError( ...
                    @()create_squared_diff_from_mean_functions(bad), ...
                    'MATLAB:validators:mustBeFinite' ...
                );
                testCase.verifyError( ...
                    @()create_squared_diff_from_mean_functions([bad]), ...
                    'MATLAB:validators:mustBeFinite' ...
                );
            end
        end

        function confirmImmutability(testCase)
            % Construct using variable declared in this scope & collect results
            alpha = 1.2;
            [hfun, combinemodels] = ...
                    create_squared_diff_from_mean_functions(alpha);
            hF = hfun(testCase.F);
            [G, H] = combinemodels(testCase.Cres, ...
                                   testCase.Gres, ...
                                   testCase.Hres);

            % Alter same construction variable & confirm that it yields
            % different results
            alpha = -2.3 * alpha;
            [hfun_2, combinemodels_2] = ...
                    create_squared_diff_from_mean_functions(alpha);
            hF_2 = hfun_2(testCase.F);
            [G_2, H_2] = combinemodels_2(testCase.Cres, ...
                                         testCase.Gres, ...
                                         testCase.Hres);
            testCase.assertNotEqual(hF, hF_2);
            testCase.assertFalse(isequal(G, G_2));
            testCase.assertFalse(isequal(H, H_2));

            % Confirm that changes to actual alpha argument used to construct
            % hfun and combinemodels do not alter those functions.  This check
            % is motivated by technical subtleties seen with Python.
            hF_new = hfun(testCase.F);
            [G_new, H_new] = combinemodels(testCase.Cres, ...
                                           testCase.Gres, ...
                                           testCase.Hres);
            testCase.assertEqual(hF, hF_new);
            testCase.assertTrue(isequal(G, G_new));
            testCase.assertTrue(isequal(H, H_new));
        end

        function testFunctions(testCase)
            % Handworked intermediate results for test problem 1
            H_bar = [[1.1 0.9]
                     [0.9 -0.4]];
            H_JmGJmG = [[3.64 1.82]
                        [1.82 3.64]];
            H_GG = [[2.88 3.12]
                    [3.12 3.38]];
            H_FH = [[27.28 -18.26]
                    [-18.26 10.34]];

            % Include
            % - Negative and positive
            % - What look like integer and double literals
            % - The special case of alpha = 0.0
            for alpha = [-1.1 -5 0.0 2.3 4]
                [hfun, combinemodels] = ...
                    create_squared_diff_from_mean_functions(alpha);

                % Check F = F_bar = 0.0 super-duper special case
                F = zeros([10 1]);
                [G, H] = combinemodels(F, ...
                                       testCase.Gres, ...
                                       testCase.Hres);
                testCase.assertEqual(size(G), [testCase.n 1]);
                testCase.assertEqual(size(H), [testCase.n testCase.n]);
                testCase.assertTrue(isequal(H, H'));

                testCase.assertEqual(0.0, hfun(F));

                testCase.assertEqual(G, zeros(size(G)));

                H_f = H_JmGJmG - alpha * H_GG;
                max_abs_err = max(abs(H - H_f), [], "all");
                testCase.assertTrue(max_abs_err <= 35.0 * eps);

                % Check F - F_bar = 0 with F_bar != 0 special cases
                for F_bar = [-10.1 3.3]
                    F = F_bar * ones([6 1]);
                    hF = hfun(F);
                    [G, H] = combinemodels(F, ...
                                           testCase.Gres, ...
                                           testCase.Hres);
                    testCase.assertEqual(size(G), [testCase.n 1]);
                    testCase.assertEqual(size(H), [testCase.n testCase.n]);
                    testCase.assertTrue(isequal(H, H'));

                    abs_err = abs(hF + alpha * F_bar^2);
                    testCase.assertEqual(0.0, abs_err);

                    grad_f = -2.0 * alpha * F_bar * [1.2; 1.3];
                    abs_err = max(abs(G - grad_f));
                    testCase.assertTrue(abs_err <= 130.0 * eps);

                    H_f = H_JmGJmG - alpha * (H_GG + 2.0 * F_bar * H_bar);
                    max_abs_err = max(abs(H - H_f), [], "all");
                    testCase.assertTrue(max_abs_err <= 195.0 * eps);
                end

                % Check generic test problem
                hF = hfun(testCase.F);
                [G, H] = combinemodels(testCase.Cres, ...
                                       testCase.Gres, ...
                                       testCase.Hres);
                testCase.assertEqual(size(G), [testCase.n 1]);
                testCase.assertEqual(size(H), [testCase.n testCase.n]);
                testCase.assertTrue(isequal(H, H'));

                hF_expected = 16.94 - alpha * 4.84;
                abs_err = abs(hF - hF_expected);
                testCase.assertTrue(abs_err <= 20.0 * eps);

                grad_f = [10.12; 9.02] - alpha * [5.28; 5.72];
                abs_err = max(abs(G - grad_f));
                testCase.assertTrue(abs_err <= 35.0 * eps);

                H_f = H_JmGJmG + H_FH - alpha * (H_GG + 2.0 * 2.2 * H_bar);
                max_abs_err = max(abs(H - H_f), [], "all");
                testCase.assertTrue(max_abs_err <= 70.0 * eps);
            end
        end

    end
end
