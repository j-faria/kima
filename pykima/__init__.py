from .version import kima_version
__version__ = kima_version
from .showresults import showresults
from .results import KimaResults
load = KimaResults.load

__all__ = [__version__, showresults, load]
