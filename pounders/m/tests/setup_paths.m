addpath('../');
addpath('../general_h_funs');
bendfo_location = '../../../../BenDFO';

if ~exist(bendfo_location, 'dir')
    error("These tests depend on the BenDFO repo: https://github.com/POptUS/BenDFO. Make sure BenDFO is on your path in MATLAB");
end

addpath([bendfo_location, '/m']);
addpath([bendfo_location, '/data']);

minq_location = '../../../minq/m';
addpath([minq_location, '/minq5']);
addpath([minq_location, '/minq8']);
