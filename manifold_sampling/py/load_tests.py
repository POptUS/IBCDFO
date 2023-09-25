from pathlib import Path


def load_tests(loader, suite, _):
    """
    This function implements the load_tests protocol of the python unittest
    package.  In particular, it adds to the given suite all tests needed to test
    the pounders sub-package.

    Developers and users can run tests using this indirectly via
                         python -m unittest ibcdfo
    to run as part of the package's full suite or via
                         python -m unittest ibcdfo.manifold_sampling
    to run alone.

    Parameters:
        loader - the unittest.TestLoader instance doing the loading
        suite - test suite being built at the time of call to this function
    """
    start_dir = Path(__file__).resolve().parent.joinpath("tests")

    pkg_tests = loader.discover(
        start_dir=str(start_dir),
        top_level_dir=str(start_dir),
        pattern="[Tt]est*.py",
    )
    suite.addTests(pkg_tests)

    return suite
