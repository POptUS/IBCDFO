% From MATLAB and the directory containing this file, execute
%     >> runtests
%
% If you would like to run this from a different folder that includes these
% tests as a subfolder, then from that folder execute
%     >> runtests("IncludeSubfolders", true)
% Other tests run during execution might require adding other folders to the
% path.
%
% To execute the test suite with coverage enabled and to generate an HTML-format
% coverage report, execute from /path/to/IBCDFO/pounders/m
%     >> runtests("IncludeSubfolders", true, "ReportCoverageFor", pwd)
%

classdef TestCreateCensoredL1LossHfun < matlab.unittest.TestCase
    properties
        C
        D
        hfun
        Z
        hF
        grads
        Hash
        C_short
        D_short
        path_orig
    end

    methods (TestMethodSetup)

        function setup(testCase)
            % tests need product_of_cells
            [here_path, ~, ~] = fileparts(mfilename('fullpath'));
            testCase.path_orig = addpath(fullfile(here_path, '..'));

            % Confirm that we have good arguments for use in testing
            M = 10;

            testCase.C = 1:M;
            testCase.D = -1.1 * (1:M);
            [testCase.hfun] = create_censored_L1_loss_hfun(testCase.C, ...
                                                           testCase.D);

            % The test values were not determined by hand and are, therefore,
            % not known to be correct.  Rather, they were gathered from test
            % output at some time and are used here to catch regressions and to
            % confirm that the Python and MATLAB code are returning the same
            % values for the same problems.
            testCase.Z = 2.1 * (0:length(testCase.C) - 1);
            [testCase.hF, testCase.grads, testCase.Hash] = ...
                testCase.hfun(testCase.Z);
            testCase.assertEqual(156.0, testCase.hF);
            expected = ones([M 1]);
            expected(1) = 0;
            testCase.assertEqual(testCase.grads, expected);
            testCase.assertTrue(iscell(testCase.Hash));
            [tmp, n_hash] = size(testCase.Hash);
            testCase.assertEqual(tmp, 1);
            testCase.assertEqual(n_hash, 1);
            testCase.assertEqual(size(testCase.grads), [M, 1]);
            expected = {'2111111111'};
            testCase.assertEqual(testCase.Hash, expected);

            % Sanity check hash result/H0
            [hF_H0, grads_H0] = testCase.hfun(testCase.Z, testCase.Hash);
            testCase.assertEqual(hF_H0, testCase.hF);
            testCase.assertEqual(grads_H0, testCase.grads);

            % Good but different size for testing size incompatibilities
            testCase.C_short = [1.1 2.2 -3.3];
            testCase.D_short = [3.3 -1.1 -2.2];
            M_short = length(testCase.C_short);
            testCase.assertNotEqual(length(testCase.C_short), length(testCase.C));
            testCase.assertNotEqual(length(testCase.D_short), length(testCase.D));

            [hfun] = create_censored_L1_loss_hfun(testCase.C_short, ...
                                                  testCase.D_short);

            % As above in terms of correctness and use of expected results
            % here.
            Z_short = 2.1 * (1:length(testCase.C_short));
            [hF, grads, Hash] = hfun(Z_short);
            testCase.assertEqual(15.0, hF);
            expected = ones([M_short 1]);
            expected(1) = -1;
            testCase.assertEqual(grads, expected);
            testCase.assertTrue(iscell(Hash));
            [tmp, n_hash] = size(Hash);
            testCase.assertEqual(tmp, 1);
            testCase.assertEqual(n_hash, 1);
            testCase.assertEqual(size(grads), [M_short, 1]);
            expected = {'311'};
            testCase.assertEqual(Hash, expected);

            % Sanity check hash result/H0
            [hF_H0, grads_H0] = hfun(Z_short, Hash);
            testCase.assertEqual(hF_H0, hF);
            testCase.assertEqual(grads_H0, grads);
        end

    end

    methods (TestMethodTeardown)

        function restorePath(testCase)
            path(testCase.path_orig);
        end

    end

    methods (Test)

        function badArguments(testCase)
            SHAPE_2D = [5 2];

            % Accepts row and column vectors ...
            testCase.verifyTrue(isrow(testCase.C));
            testCase.verifyTrue(isrow(testCase.D));

            C = testCase.C(:);
            testCase.verifyTrue(iscolumn(C));
            [hfun_c] = create_censored_L1_loss_hfun(C, testCase.D);
            [hF_c, grads_c, Hash_c] = hfun_c(testCase.Z);
            testCase.assertEqual(hF_c, testCase.hF);
            testCase.assertEqual(grads_c, testCase.grads);
            testCase.assertEqual(Hash_c, testCase.Hash);

            D = testCase.D(:);
            testCase.verifyTrue(iscolumn(D));
            [hfun_c] = create_censored_L1_loss_hfun(C, D);
            [hF_c, grads_c, Hash_c] = hfun_c(testCase.Z);
            testCase.assertEqual(hF_c, testCase.hF);
            testCase.assertEqual(grads_c, testCase.grads);
            testCase.assertEqual(Hash_c, testCase.Hash);

            % but not matrices
            C_2D = reshape(testCase.C, SHAPE_2D);
            D_2D = reshape(testCase.D, SHAPE_2D);
            testCase.assertEqual(numel(C_2D), length(testCase.C));
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun(C_2D, testCase.D), ...
                'MATLAB:validation:IncompatibleSize' ...
            );
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun(testCase.C, D_2D), ...
                'MATLAB:validation:IncompatibleSize' ...
            );
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun(C_2D, D_2D), ...
                'MATLAB:validation:IncompatibleSize' ...
            );

            % or an empty array
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun([], testCase.D), ...
                'MATLAB:validators:mustBeNonempty' ...
            );
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun(testCase.C, []), ...
                'MATLAB:validators:mustBeNonempty' ...
            );
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun([], []), ...
                'MATLAB:validators:mustBeNonempty' ...
            );

            % C & D must have at least two elements
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun([1.1], testCase.D), ...
                'POptUS:IncompatibleSizes' ...
            );
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun(testCase.C, [2.2]), ...
                'POptUS:IncompatibleSizes' ...
            );
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun([1.1], [2.2]), ...
                'POptUS:ArrayTooShort' ...
            );

            % Finite real values only please
            for bad = ["bad", 1j, 1.0 - 2.0 * 1j]
                testCase.verifyError( ...
                    @()create_censored_L1_loss_hfun(bad, testCase.D), ...
                    'MATLAB:validators:mustBeReal' ...
                );
                testCase.verifyError( ...
                    @()create_censored_L1_loss_hfun(testCase.C, bad), ...
                    'MATLAB:validators:mustBeReal' ...
                );
            end

            for bad = [inf -inf nan]
                testCase.verifyError( ...
                    @()create_censored_L1_loss_hfun(bad, testCase.D), ...
                    'MATLAB:validators:mustBeFinite' ...
                );
                testCase.verifyError( ...
                    @()create_censored_L1_loss_hfun(testCase.C, bad), ...
                    'MATLAB:validators:mustBeFinite' ...
                );
            end

            % C & D must be the same length ...
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun(testCase.C_short, testCase.D), ...
                'POptUS:IncompatibleSizes' ...
            );
            testCase.verifyError( ...
                @()create_censored_L1_loss_hfun(testCase.C, testCase.D_short), ...
                'POptUS:IncompatibleSizes' ...
            );
        end

        function badZArgument(testCase)
            M = length(testCase.Z);

            % Must be 1D
            for bad = ["bad", 1.1, 1j, []]
                testCase.verifyError( ...
                    @()testCase.hfun(bad), 'POptUS:IncompatibleSizes' ...
                );
            end

            Z_2D = zeros([M, 2]);
            testCase.verifyError( ...
                @()testCase.hfun(Z_2D), 'MATLAB:assertion:failed' ...
            );

            % z shape must match C & D
            Z_short = zeros([M - 1, 1]);
            Z_long = zeros([M + 1, 1]);
            testCase.verifyError( ...
                @()testCase.hfun(Z_short), 'POptUS:IncompatibleSizes' ...
            );
            testCase.verifyError( ...
                @()testCase.hfun(Z_long), 'POptUS:IncompatibleSizes' ...
            );

            % z should only by finite real vector
            for bad = [1j, 1.0 - 2.0 * 1j]
                for i = 1:length(testCase.Z)
                    z_bad = testCase.Z;
                    z_bad(i) = bad;
                    testCase.verifyError( ...
                        @()testCase.hfun(z_bad), 'POptUS:NonrealValues' ...
                    );
                end
            end

            for bad = [inf -inf nan]
                for i = 1:length(testCase.Z)
                    z_bad = testCase.Z;
                    z_bad(i) = bad;
                    testCase.verifyError( ...
                        @()testCase.hfun(z_bad), 'POptUS:NonfiniteValues' ...
                    );
                end
            end
        end

        function confirmImmutable(testCase)
            % Construct using variable declared in this scope & collect results
            C = testCase.C;
            D = testCase.D;
            [hfun] = create_censored_L1_loss_hfun(C, D);
            [hF, grads, Hash] = hfun(testCase.Z);

            % Alter same construction variable & confirm that it yields
            % different results
            C = -2.3 * C;
            [hfun_2] = create_censored_L1_loss_hfun(C, D);
            [hF_2, grads_2, Hash_2] = hfun_2(testCase.Z);
            testCase.assertNotEqual(hF_2, hF);
            testCase.assertNotEqual(grads_2, grads);
            testCase.assertNotEqual(Hash_2, Hash);

            D = -4.1 * D;
            [hfun_3] = create_censored_L1_loss_hfun(C, D);
            [hF_3, grads_3, Hash_3] = hfun_3(testCase.Z);
            testCase.assertNotEqual(hF_3, hF);
            testCase.assertNotEqual(hF_3, hF_2);
            testCase.assertNotEqual(grads_3, grads);
            testCase.assertNotEqual(grads_3, grads_2);
            testCase.assertNotEqual(Hash_3, Hash);
            testCase.assertNotEqual(Hash_3, Hash_2);

            % Confirm that changes to actual C, D arguments used to construct
            % hfun do not alter that function.  This check is motivated by
            % technical subtleties seen with Python.
            [hF_new, grads_new, Hash_new] = hfun(testCase.Z);
            testCase.assertEqual(hF_new, hF);
            testCase.assertEqual(grads_new, grads);
            testCase.assertEqual(Hash_new, Hash);
        end

    end
end
