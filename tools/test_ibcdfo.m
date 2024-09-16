% ----- HARDCODED VALUES
[HERE_PATH, ~, ~] = fileparts(mfilename('fullpath'));
CLONE_ROOT = fullfile(HERE_PATH, '..');

BENDFO_ENV_VAR = 'BENDFO_PATH';

% ----- SEARCH FOR BenDFO DEPENDENCE
bendfo_default = fullfile(CLONE_ROOT, '..', 'BenDFO');
bendfo_value = getenv(BENDFO_ENV_VAR);
if exist(bendfo_default, 'dir')
    bendfo_path = bendfo_default;
elseif ~isempty(bendfo_value)
    bendfo_path = bendfo_value;
else
    msg = "IBCDFO tests use BenDFO: https://github.com/POptUS/BenDFO.  ";
    msg = msg + "Either clone that repo next to your IBCDFO repo, or set the ";
    msg = msg + BENDFO_ENV_VAR + " env var to the location of your clone.";
    error(msg);
end
disp(" ");
disp(['Using BenDFO at', bendfo_path]);
disp(" ");

bendfo_m_path    = fullfile(bendfo_path, 'm');
bendfo_data_path = fullfile(bendfo_path, 'data');
if ~exist(bendfo_m_path, 'dir') || ~exist(bendfo_data_path, 'dir')
    error('The BenDFO clone is missing the m or data folders.');
end

% ----- ADD DEPENDENCIES TO PATH
old_path = addpath(bendfo_m_path);
addpath(bendfo_data_path);

% ----- RUN TESTS FOR ALL MATLAB SUBPACKAGES IN IBCDFO
old_cwd = cd(CLONE_ROOT);

% Unfortunately we cannot simply run from the root of the repository and have
% runtests automatically discover all subpackage tests because it will
% "discover" this script.  In such cases, this test script will run itself.
SUBPACKAGES = [string(fullfile(CLONE_ROOT, 'minq', 'm')),
               string(fullfile(CLONE_ROOT, 'pounders', 'm')),
               string(fullfile(CLONE_ROOT, 'manifold_sampling', 'm'))];
runtests(SUBPACKAGES, "IncludeSubfolders", true, "ReportCoverageFor", pwd)

cd(old_cwd);

% ----- RESTORE TO INCOMING MATLAB SETUP
path(old_path);
