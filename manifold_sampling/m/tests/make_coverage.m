clear all;
setup_paths;
suite = testsuite("Testmanifoldsampling.m");
import matlab.unittest.plugins.CodeCoveragePlugin
import matlab.unittest.plugins.codecoverage.CoverageReport
runner = testrunner("textoutput");
sourceCodeFolder = "../";
% sourceCodeFolder = "../../../../minq/m/";
reportFolder = "coverageReport";
reportFormat = CoverageReport(reportFolder);
% p = CodeCoveragePlugin.forFolder(sourceCodeFolder, "Producing", reportFormat, "IncludingSubfolders", true);
p = CodeCoveragePlugin.forFolder(["../", "../h_examples/"], "Producing", reportFormat);
runner.addPlugin(p);
results = runner.run(suite);
