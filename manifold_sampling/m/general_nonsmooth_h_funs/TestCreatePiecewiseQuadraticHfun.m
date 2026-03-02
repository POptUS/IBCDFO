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

% Claude Sonnet 4.5 via Argo was used to generate a starting point for this
% test.  It was provided with the current working versions of
% 1. create_piecewise_quadratic_hfun.m, and
% 2. create_censored_L1_loss_hfun.m,
% 3. TestCreateCensoredL1LossHfun.m
% at that time and asked to build up this test of (1) so that it tests (1) in a
% way analogous to how (3) tests (2).
%
% That code was then reviewed, cleaned, and evolved with no further
% contributions or alterations by a generative AI tool.  Developers are, as
% expected, taking responsibility for the correctness of this content and its
% ability to test the related code.

classdef TestCreatePiecewiseQuadraticHfun < matlab.unittest.TestCase
    properties
        Qs
        zs
        cs
        hfun
        Z
        hF
        grads
        Hash
        Qs_long
        zs_long
        cs_long
        path_orig
    end

    methods (TestMethodSetup)

        function setup(testCase)
            % tests need product_of_cells
            [here_path, ~, ~] = fileparts(mfilename('fullpath'));
            testCase.path_orig = addpath(fullfile(here_path, '..'));

            % Confirm that we have good arguments for use in testing
            M = 3;
            L = 4;

            % Create test data
            testCase.Qs = reshape(1:(M * M * L), [M, M, L]);
            testCase.zs = reshape(1:(M * L), [M, L]);
            testCase.cs = 1:L;
            testCase.cs = testCase.cs(:);

            [testCase.hfun] = create_piecewise_quadratic_hfun(testCase.Qs, ...
                                                              testCase.zs, ...
                                                              testCase.cs);

            % The test values were not determined by hand and are, therefore,
            % not known to be correct.  Rather, they were gathered from test
            % output at some time and are used here to catch regressions and to
            % confirm that the Python and MATLAB code are returning the same
            % values for the same problems.
            testCase.Z = 1.1 * (1:M);
            [testCase.hF, testCase.grads, testCase.Hash] = ...
                testCase.hfun(testCase.Z);
            expected = 22285.6;
            rel_diff = abs(1.0 - testCase.hF / expected);
            testCase.assertTrue(rel_diff <= 5 * eps);
            expected = [-1635.6; -1688.4; -1741.2];
            rel_diff = max(abs(1.0 - expected ./ testCase.grads));
            testCase.assertTrue(rel_diff <= 5 * eps);
            testCase.assertTrue(iscell(testCase.Hash));
            [tmp, n_hash] = size(testCase.Hash);
            testCase.assertEqual(tmp, 1);
            testCase.assertEqual(n_hash, 1);
            testCase.assertEqual(size(testCase.grads), [M, 1]);
            % In the Python version of this test, the hash is "3".  This is due
            % to the fact that for this problem the hash result is the index to
            % the active quadratic piece and indices in Python are 0-based
            % instead of 1-based.
            testCase.assertEqual(testCase.Hash, {'4'});

            % Sanity check hash result/H0
            [hF_H0, grads_H0] = testCase.hfun(testCase.Z, testCase.Hash);
            testCase.assertEqual(hF_H0, testCase.hF);
            testCase.assertEqual(grads_H0, testCase.grads);

            % Good but different size for testing size incompatibilities
            M_long = M + 1;
            L_long = L - 1;
            testCase.Qs_long = reshape(1:(M_long * M_long * L_long), ...
                                       [M_long, M_long, L_long]);
            testCase.zs_long = reshape(1:(M_long * L_long), ...
                                       [M_long, L_long]);
            testCase.cs_long = 1:(L_long);
            testCase.cs_long = testCase.cs_long(:);

            [hfun] = create_piecewise_quadratic_hfun(testCase.Qs_long, ...
                                                     testCase.zs_long, ...
                                                     testCase.cs_long);

            % As above in terms of correctness and use of expected results
            % here.
            Z_long = 2.1 * (1:length(testCase.zs_long));
            [hF, grads, Hash] = hfun(Z_long);
            testCase.assertEqual(hF, 17286.0);
            expected = [-1594; -1636; -1678; -1720];
            rel_diff = max(abs(1.0 - expected ./ grads));
            testCase.assertTrue(rel_diff <= 5 * eps);
            testCase.assertTrue(iscell(Hash));
            [tmp, n_hash] = size(Hash);
            testCase.assertEqual(tmp, 1);
            testCase.assertEqual(n_hash, 1);
            testCase.assertEqual(size(grads), [M_long, 1]);
            % Similar to above comment, the hash here is one more than the
            % result in the Python version of this test.
            testCase.assertEqual(Hash, {'3'});

            % Sanity check hash result/H0
            [hF_H0, grads_H0] = hfun(Z_long, Hash);
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
            M = size(testCase.zs, 1);
            L = size(testCase.zs, 2);

            % Qs must be a 3D array
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun([], testCase.zs, testCase.cs), ...
                'MATLAB:validators:mustBeNonempty' ...
            );
            Qs_2D = ones([M, M]);
            testCase.verifyEqual(ndims(Qs_2D), 2);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(Qs_2D, testCase.zs, testCase.cs), ...
                'POptUS:IncompatibleSizes' ...
            );
            Qs_4D = ones([M, M, L, 2]);
            testCase.verifyEqual(ndims(Qs_4D), 4);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(Qs_4D, testCase.zs, testCase.cs), ...
                'MATLAB:validation:IncompatibleSize' ...
            );

            % zs must be a matrix (not 3D)
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, [], testCase.cs), ...
                'MATLAB:validators:mustBeNonempty' ...
            );
            zs_1D = ones([L 1]);
            testCase.verifyEqual(ndims(zs_1D), 2);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, zs_1D, testCase.cs), ...
                'POptUS:IncompatibleSizes' ...
            );
            zs_1D = ones([1 L]);
            testCase.verifyEqual(ndims(zs_1D), 2);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, zs_1D, testCase.cs), ...
                'POptUS:IncompatibleSizes' ...
            );
            zs_3D = ones([M M L]);
            testCase.verifyEqual(ndims(zs_3D), 3);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, zs_3D, testCase.cs), ...
                'MATLAB:validation:IncompatibleSize' ...
            );

            % cs must be a (2D) vector
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, testCase.zs, []), ...
                'MATLAB:validators:mustBeNonempty' ...
            );
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, testCase.zs, 1.1), ...
                'POptUS:IncompatibleSizes' ...
            );
            cs_2D = ones([M, L]);
            testCase.verifyEqual(ndims(cs_2D), 2);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, testCase.zs, cs_2D), ...
                'MATLAB:validation:IncompatibleSize' ...
            );

            % cs can be row or column vector
            cs_row = testCase.cs(:)';
            testCase.verifyTrue(iscolumn(testCase.cs));
            testCase.verifyTrue(isrow(cs_row));
            [hfun_c] = create_piecewise_quadratic_hfun(testCase.Qs, ...
                                                       testCase.zs, ...
                                                       cs_row);
            [hF_c, grads_c, Hash_c] = hfun_c(testCase.Z);
            testCase.assertEqual(hF_c, testCase.hF);
            testCase.assertEqual(grads_c, testCase.grads);
            testCase.assertEqual(Hash_c, testCase.Hash);

            % Finite real values only please
            for bad = [1j, 1.0 - 2.0 * 1j]
                for k = 1:L
                    cs_bad = testCase.cs;
                    cs_bad(k) = bad;
                    testCase.assertError( ...
                        @()create_piecewise_quadratic_hfun( ...
                            testCase.Qs, testCase.zs, cs_bad ...
                        ), ...
                        'MATLAB:validators:mustBeReal' ...
                    );

                    for i = 1:M
                        zs_bad = testCase.zs;
                        zs_bad(i, k) = bad;
                        testCase.assertError( ...
                            @()create_piecewise_quadratic_hfun( ...
                                testCase.Qs, zs_bad, testCase.cs ...
                            ), ...
                            'MATLAB:validators:mustBeReal' ...
                        );

                        for j = 1:M
                            Qs_bad = testCase.Qs;
                            Qs_bad(i, j, k) = bad;
                            testCase.assertError( ...
                                @()create_piecewise_quadratic_hfun( ...
                                    Qs_bad, testCase.zs, testCase.cs ...
                                ), ...
                                'MATLAB:validators:mustBeReal' ...
                            );
                        end
                    end
                end
            end

            for bad = [inf -inf nan]
                for k = 1:L
                    cs_bad = testCase.cs;
                    cs_bad(k) = bad;
                    testCase.assertError( ...
                        @()create_piecewise_quadratic_hfun( ...
                            testCase.Qs, testCase.zs, cs_bad ...
                        ), ...
                        'MATLAB:validators:mustBeFinite' ...
                    );

                    for i = 1:M
                        zs_bad = testCase.zs;
                        zs_bad(i, k) = bad;
                        testCase.assertError( ...
                            @()create_piecewise_quadratic_hfun( ...
                                testCase.Qs, zs_bad, testCase.cs ...
                            ), ...
                            'MATLAB:validators:mustBeFinite' ...
                        );

                        for j = 1:M
                            Qs_bad = testCase.Qs;
                            Qs_bad(i, j, k) = bad;
                            testCase.assertError( ...
                                @()create_piecewise_quadratic_hfun( ...
                                    Qs_bad, testCase.zs, testCase.cs ...
                                ), ...
                                'MATLAB:validators:mustBeFinite' ...
                            );
                        end
                    end
                end
            end

            % Qs, zs, and cs must have compatible sizes
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs_long, testCase.zs, testCase.cs), ...
                'POptUS:IncompatibleSizes' ...
            );
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, testCase.zs_long, testCase.cs), ...
                'POptUS:IncompatibleSizes' ...
            );
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, testCase.zs, testCase.cs_long), ...
                'POptUS:IncompatibleSizes' ...
            );

            % Test mismatched dimensions more explicitly
            Qs_wrong = zeros(M + 1, M, L);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(Qs_wrong, testCase.zs, testCase.cs), ...
                'POptUS:IncompatibleSizes' ...
            );

            Qs_wrong = zeros(M, M + 1, L);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(Qs_wrong, testCase.zs, testCase.cs), ...
                'POptUS:IncompatibleSizes' ...
            );

            Qs_wrong = zeros(M, M, L + 1);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(Qs_wrong, testCase.zs, testCase.cs), ...
                'POptUS:IncompatibleSizes' ...
            );

            zs_wrong = zeros(M + 1, L);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, zs_wrong, testCase.cs), ...
                'POptUS:IncompatibleSizes' ...
            );

            zs_wrong = zeros(M, L + 1);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, zs_wrong, testCase.cs), ...
                'POptUS:IncompatibleSizes' ...
            );

            cs_wrong = zeros(L + 1, 1);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, testCase.zs, cs_wrong), ...
                'POptUS:IncompatibleSizes' ...
            );

            cs_wrong = zeros(1, L + 1);
            testCase.verifyError( ...
                @()create_piecewise_quadratic_hfun(testCase.Qs, testCase.zs, cs_wrong), ...
                'POptUS:IncompatibleSizes' ...
            );
        end

        function badZArgument(testCase)
            M = size(testCase.zs, 1);

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

            % z shape must match zs
            Z_short = zeros(M - 1, 1);
            Z_long = zeros(M + 1, 1);
            testCase.verifyError( ...
                @()testCase.hfun(Z_short), 'POptUS:IncompatibleSizes' ...
            );
            testCase.verifyError( ...
                @()testCase.hfun(Z_long), 'POptUS:IncompatibleSizes' ...
            );

            % z should only be finite real vector
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
            % This test case is motivated by technical subtleties seen with
            % Python.

            % Construct using variable declared in this scope & collect results
            Qs = testCase.Qs;
            zs = testCase.zs;
            cs = testCase.cs;
            [hfun] = create_piecewise_quadratic_hfun(Qs, zs, cs);
            [hF, grads, Hash] = hfun(testCase.Z);

            % Alter same Qs variable and confirm different result
            Qs = -2.3 * Qs;
            [hfun_2] = create_piecewise_quadratic_hfun(Qs, zs, cs);
            [hF_2, grads_2, Hash_2] = hfun_2(testCase.Z);
            testCase.assertNotEqual(hF_2, hF);
            testCase.assertNotEqual(grads_2, grads);
            testCase.assertNotEqual(Hash_2, Hash);

            % Confirm that changing Qs didn't alter the original function
            [hF_new, grads_new, Hash_new] = hfun(testCase.Z);
            testCase.assertEqual(hF_new, hF);
            testCase.assertEqual(grads_new, grads);
            testCase.assertEqual(Hash_new, Hash);

            % Alter same zs variable and confirm different result
            zs(:, 1) = -1.1 * zs(:, 1);
            [hfun_3] = create_piecewise_quadratic_hfun(Qs, zs, cs);
            [hF_3, grads_3, Hash_3] = hfun_3(testCase.Z);
            testCase.assertNotEqual(hF_3, hF);
            testCase.assertNotEqual(hF_3, hF_2);
            testCase.assertNotEqual(grads_3, grads);
            testCase.assertNotEqual(grads_3, grads_2);
            testCase.assertNotEqual(Hash_3, Hash);
            testCase.assertNotEqual(Hash_3, Hash_2);

            % Confirm that changing zs didn't alter the original function
            [hF_new, grads_new, Hash_new] = hfun(testCase.Z);
            testCase.assertEqual(hF_new, hF);
            testCase.assertEqual(grads_new, grads);
            testCase.assertEqual(Hash_new, Hash);

            % Alter same cs variable and confirm different result
            cs(1) = 500.6 + cs(1);
            [hfun_4] = create_piecewise_quadratic_hfun(Qs, zs, cs);
            [hF_4, grads_4, Hash_4] = hfun_4(testCase.Z);
            testCase.assertNotEqual(hF_4, hF);
            testCase.assertNotEqual(hF_4, hF_3);
            testCase.assertNotEqual(grads_4, grads);
            testCase.assertNotEqual(grads_4, grads_3);
            testCase.assertNotEqual(Hash_4, Hash);
            testCase.assertNotEqual(Hash_4, Hash_3);

            % Confirm that changing cs didn't alter the original function
            [hF_new, grads_new, Hash_new] = hfun(testCase.Z);
            testCase.assertEqual(hF_new, hF);
            testCase.assertEqual(grads_new, grads);
            testCase.assertEqual(Hash_new, Hash);
        end

    end
end
