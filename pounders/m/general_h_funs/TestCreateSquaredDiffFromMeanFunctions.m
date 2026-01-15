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

% TODO: If this is a good idea, then thest the combinemodel function in the
% same way.
classdef TestCreateSquaredDiffFromMeanFunctions < matlab.unittest.TestCase
    methods (Test)

        function badArguments(testCase)
            testCase.verifyError(...
                @()create_squared_diff_from_mean_functions([]), ...
                'MATLAB:validators:mustBeNonempty' ...
            )

            testCase.verifyError(...
                @()create_squared_diff_from_mean_functions([1.1 2.2]), ...
                'MATLAB:validators:mustBeScalarOrEmpty' ...
            )

            testCase.verifyError(...
                @()create_squared_diff_from_mean_functions("hello"), ...
                'MATLAB:validators:mustBeReal' ...
            )

            for bad = [inf -inf nan]
                testCase.verifyError(...
                    @()create_squared_diff_from_mean_functions(bad), ...
                    'MATLAB:validators:mustBeFinite' ...
                )
                testCase.verifyError(...
                    @()create_squared_diff_from_mean_functions([bad]), ...
                    'MATLAB:validators:mustBeFinite' ...
                )
            end
        end

        function actualArguments(testCase)
            F = [1.1 2.2 3.3 4.4];
            expected_0 = 6.05;
            expected_2 = -9.075;

            alpha = 0.0;
            [hfun_0, cm_0] = create_squared_diff_from_mean_functions(alpha);
            h_0 = hfun_0(F);
            testCase.assertTrue(abs(h_0 - expected_0) <= 5 * eps);
            testCase.assertEqual(h_0, hfun_0(F));

            % Confirm that we get same result as for actual scalar.
            [hfun_tmp, cm_tmp] = create_squared_diff_from_mean_functions([alpha]);
            testCase.assertEqual(h_0, hfun_tmp(F));

            % Intentionally alter variable used to create *_0 functions
            alpha = 2.0;
            [hfun_2, cm_2] = create_squared_diff_from_mean_functions(alpha);
            h_2 = hfun_2(F);
            testCase.assertEqual(h_2, expected_2);
            testCase.assertEqual(h_2, hfun_2(F));
            testCase.assertNotEqual(h_0, h_2);
            % Confirm that changes to actual alpha argument used to construct
            % hfun_0 and cm_0 do not alter those functions.  This check is
            % motivated by technical subtleties seen with Python.
            testCase.assertEqual(h_0, hfun_0(F));
        end

    end
end
