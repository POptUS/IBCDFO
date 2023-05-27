"""
TODO: Jeff to insert high-level description for here.  After installing with
pip, this should be visible with `python -m pydoc ibcdfo`.
"""

import pkg_resources

# Follow typical version-access interface used by other packages
# (e.g., numpy, scipy, pandas, matplotlib)
__version__ = pkg_resources.get_distribution("ibcdfo").version
