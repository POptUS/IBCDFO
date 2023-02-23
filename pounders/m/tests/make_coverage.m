clear all;
setup_paths;
suite = testsuite("Testpounders.m");
import matlab.unittest.plugins.CodeCoveragePlugin
import matlab.unittest.plugins.codecoverage.CoverageReport
runner = testrunner("textoutput");
sourceCodeFolder = "../";
% sourceCodeFolder = "../../../../minq/m/";
reportFolder = "coverageReport";
reportFormat = CoverageReport(reportFolder);
p = CodeCoveragePlugin.forFolder(sourceCodeFolder, "Producing", reportFormat, "IncludingSubfolders", true);
runner.addPlugin(p);
results = runner.run(suite);
