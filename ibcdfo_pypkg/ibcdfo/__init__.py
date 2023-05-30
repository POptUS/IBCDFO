"""
TODO: Jeff to insert high-level description for here.  After installing with
pip, this should be visible with `python -m pydoc ibcdfo`.
"""

from importlib.metadata import version

# Follow typical version-access interface used by other packages
# (e.g., numpy, scipy, pandas, matplotlib)
__version__ = version("ibcdfo")

from . import pounders
