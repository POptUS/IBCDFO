function check_minq_installation(minq_version)
    % Confirm that the folder containing the code for the given MINQ version has
    % been added to the MATLAB path and that the git clone that contains it is
    % set to the required commit.
    %
    % An error is thrown if the installation cannot be found or is deemed
    % invalid.
    %
    % Inputs:
    %   minq_version - 5 to check for a valid MINQ5 installation; 8, for a
    %                  valid MINQ8 installation.

    % ----- HARDCODED VALUES
    % We need the valid MINQ commit to be stored so that it is available
    % repository-wide since this information should be valid, in fact, for all
    % methods in the package and regardless of implementation language.
    %
    % We also need, for instance, for actions to be able to load the commit's
    % SHA so that they can always checkout the correct MINQ commit even when
    % developers are in the process of moving to a new version of MINQ - we
    % change the SHA in one place and all aspects of our infrastructure adapt
    % as needed automatically.
    %
    % Therefore, we load the MINQ commit SHA here dynamically from that file.
    [HERE_PATH, ~, ~] = fileparts(mfilename('fullpath'));
    MINQ_VERSION_FILE = fullfile(HERE_PATH, '..', '..', 'REQUIRED_MINQ_COMMIT');
    COMMIT_INFO = readlines(MINQ_VERSION_FILE);
    assert(length(COMMIT_INFO) >= 1);
    for i = 2:length(COMMIT_INFO)
        assert(COMMIT_INFO(i) == "");
    end
    VALID_MINQ_SHA = COMMIT_INFO(1);
    assert(strlength(VALID_MINQ_SHA) == 40);

    % ----- FIND MINQ CLONE INSTALLATION
    % Users are required to add to the MATLAB path the specific folder
    % containing their target version of MINQ.
    if minq_version == 5
        function_path = which("minqsw");
        if function_path == ""
            error("Please add /path/to/MINQ/m/minq5 to MATLAB path");
        end
    elseif minq_version == 8
        function_path = which("minq8");
        if function_path == ""
            error("Please add /path/to/MINQ/m/minq8 to MATLAB path");
        end
    else
        error(sprintf("Invalid MINQ version %d", minq_version));
    end
    [minq_path, ~, ~] = fileparts(function_path);
    % This lovely functionality was only introduced in 2023b, so we don't use
    % it for now.
    % minq_repo = gitrepo(minq_path);
    % git_sha = minq_repo.LastCommit.ID;

    cwd_original = cd(minq_path);
    [status, git_sha] = system("git rev-parse HEAD");
    cd(cwd_original);

    git_sha = strip(git_sha);
    assert(status == 0);
    assert(strlength(git_sha) == 40);

    % ----- CONFIRM DESIRED VERSION
    if git_sha ~= VALID_MINQ_SHA
        msg = sprintf(['Please set MINQ clone to commit %s\n' ...
                       'See User Guide (https://ibcdfo.readthedocs.io) ' ...
                       'for more information and instructions.'], ...
                       VALID_MINQ_SHA);
        error(msg);
    end
end
