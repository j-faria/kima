# flake8: noqa

from .version import kima_version as __version__

from .showresults import showresults

from .results import KimaResults
load = KimaResults.load

from numpy import array
from .pykepler import keplerian as keplerian_list
keplerian = lambda *args, **kwargs: array(keplerian_list(*args, **kwargs))


__all__ = [__version__, showresults, load, keplerian]

