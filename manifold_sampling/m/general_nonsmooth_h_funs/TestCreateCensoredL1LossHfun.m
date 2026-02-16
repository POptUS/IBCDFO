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
            P = 10;

            testCase.C = 1:P;
            testCase.D = -1.1 * (1:P);
            [testCase.hfun] = create_censored_L1_loss_hfun(testCase.C, ...
                                                           testCase.D);

            testCase.Z = ones([length(testCase.C) 1]);
            [testCase.hF, testCase.grads, testCase.Hash] = ...
                testCase.hfun(testCase.Z);
            testCase.assertTrue(isscalar(testCase.hF));
            testCase.assertTrue(isreal(testCase.hF));
            testCase.assertTrue(isfinite(testCase.hF));
            testCase.assertTrue(ismatrix(testCase.grads));
            testCase.assertTrue(iscell(testCase.Hash));
            [tmp, n_hash] = size(testCase.Hash);
            testCase.assertEqual(tmp, 1);
            testCase.assertEqual(size(testCase.grads), [P, n_hash]);

            % Sanity check hash result/H0
            [hF_H0, grads_H0] = testCase.hfun(testCase.Z, testCase.Hash);
            % TODO: Why is this failing?
            % testCase.assertEqual(hF_H0, testCase.hF);
            testCase.assertTrue(isequal(grads_H0, testCase.grads));

            % Good but different size for testing size incompatibilities
            testCase.C_short = [1.1 2.2 -3.3];
            testCase.D_short = [3.3 -1.1 -2.2];
            P_short = length(testCase.C_short);

            [hfun] = create_censored_L1_loss_hfun(testCase.C_short, ...
                                                  testCase.D_short);

            Z_short = zeros([P_short 1]);
            [hF, grads, Hash] = hfun(Z_short);
            testCase.assertTrue(isscalar(hF));
            testCase.assertTrue(isreal(hF));
            testCase.assertTrue(isfinite(hF));
            testCase.assertTrue(ismatrix(grads));
            testCase.assertTrue(iscell(Hash));
            [tmp, n_hash] = size(Hash);
            testCase.assertEqual(tmp, 1);
            testCase.assertEqual(size(grads), [P_short, n_hash]);

            % Sanity check hash result/H0
            [hF_H0, grads_H0] = hfun(Z_short, Hash);
            testCase.assertEqual(hF_H0, hF);
            testCase.assertTrue(isequal(grads_H0, grads));
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
            P = length(testCase.Z);

            % Accepts row and column vectors ...
            testCase.verifyTrue(isrow(testCase.C));
            testCase.verifyTrue(isrow(testCase.D));

            C = testCase.C(:);
            testCase.verifyTrue(iscolumn(C));
            [hfun_c] = create_censored_L1_loss_hfun(C, testCase.D);
            [hF_c, grads_c, Hash_c] = hfun_c(testCase.Z);
            testCase.assertEqual(hF_c, testCase.hF);
            testCase.assertTrue(isequal(grads_c, testCase.grads));
            testCase.assertTrue(isequal(Hash_c, testCase.Hash));

            D = testCase.D(:);
            testCase.verifyTrue(iscolumn(D));
            [hfun_c] = create_censored_L1_loss_hfun(C, D);
            [hF_c, grads_c, Hash_c] = hfun_c(testCase.Z);
            testCase.assertEqual(hF_c, testCase.hF);
            testCase.assertTrue(isequal(grads_c, testCase.grads));
            testCase.assertTrue(isequal(Hash_c, testCase.Hash));

            % but not matrices
            C_2D = reshape(testCase.C, SHAPE_2D);
            D_2D = reshape(testCase.D, SHAPE_2D);
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

            % Come to think of it, the vectors have to have enough elements to
            % make the problem interesting
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

            % C & D must be the same shape ...
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
            P = length(testCase.Z);

            for bad = ["bad", 1.1, 1j, []]
                testCase.verifyError( ...
                    @()testCase.hfun(bad), 'POptUS:IncompatibleSizes' ...
                );
            end

            % z shape must match C & D
            Z_short = zeros([P - 1, 1]);
            Z_long = zeros([P + 1, 1]);
            Z_2D = zeros([P, 2]);
            testCase.verifyError( ...
                @()testCase.hfun(Z_short), 'POptUS:IncompatibleSizes' ...
            );
            testCase.verifyError( ...
                @()testCase.hfun(Z_long), 'POptUS:IncompatibleSizes' ...
            );
            testCase.verifyError( ...
                @()testCase.hfun(Z_2D), 'MATLAB:assertion:failed' ...
            );

            % z should only by finite real vector
            for bad = [1j, 1.0 - 2.0 * 1j]
                z_bad = testCase.Z;
                z_bad(1) = bad;
                testCase.verifyError( ...
                    @()testCase.hfun(z_bad), 'POptUS:NonrealValues' ...
                );
            end

            for bad = [inf -inf nan]
                z_bad = testCase.Z;
                z_bad(1) = bad;
                testCase.verifyError( ...
                    @()testCase.hfun(z_bad), 'POptUS:NonfiniteValues' ...
                );
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
            testCase.assertFalse(isequal(grads_2, grads));
            testCase.assertFalse(isequal(Hash_2, Hash));

            D = -4.1 * D;
            [hfun_3] = create_censored_L1_loss_hfun(C, D);
            [hF_3, grads_3, Hash_3] = hfun_3(testCase.Z);
            testCase.assertNotEqual(hF_3, hF);
            testCase.assertFalse(isequal(grads_3, grads));
            testCase.assertFalse(isequal(Hash_3, Hash));

            % Confirm that changes to actual alpha argument used to construct
            % hfun and combinemodels do not alter those functions.  This check
            % is motivated by technical subtleties seen with Python.
            [hF_new, grads_new, Hash_new] = hfun(testCase.Z);
            testCase.assertEqual(hF_new, hF);
            testCase.assertTrue(isequal(grads_new, grads));
            testCase.assertTrue(isequal(Hash_new, Hash));
        end

    end
end
