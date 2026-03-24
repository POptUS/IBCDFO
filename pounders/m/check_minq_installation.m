function [is_valid] = check_minq_installation(minq_version)
    % Confirm that the folder containing the given MINQ version has been added
    % to the MATLAB path and that the git clone that contains it is set to the
    % required commit.
    %
    % An error is thrown if the installation cannot be found or is deemed
    % invalid.
    %
    % Inputs:
    %   minq_version - 5 to check for a valid the MINQ5 installation; 8, for a
    %                  valid MINQ8 installation.

    % ----- HARDCODED VALUES
    % This must be the full git commit hash
    VALID_MINQ_HASH = "7749b83645ea21e303a94e1200542f7028499bb8";

    % ----- FIND MINQ CLONE INSTALLATION
    % Users are required to add the specific folder containing their target
    % version of MINQ.
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
    %minq_repo = gitrepo(minq_path);
    %git_hash = minq_repo.LastCommit.ID;

    cwd_original = cd(minq_path);
    [status, git_hash] = system("git rev-parse HEAD");
    cd(cwd_original);

    git_hash = strip(git_hash);
    assert(status == 0);

    % ----- CONFIRM DESIRED VERSION
    if git_hash ~= VALID_MINQ_HASH
        error(sprintf("Please set MINQ clone to commit %s", VALID_MINQ_HASH));
    end
end
