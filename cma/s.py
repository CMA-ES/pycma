"""versatile shortcuts for quick typing in an (i)python shell or even
``from cma.s import *`` in interactive sessions.

Provides various aliases from within the `cma` package, to be reached like
``cma.s....``

Don't use for stable code.

This is not actively maintained.
"""
import warnings as _warnings
try: from matplotlib import pyplot as _pyplot  # like this it doesn't show up in the interface
except ImportError:
    _pyplot = None
    _warnings.warn('Could not import matplotlib.pyplot, therefore'
                   ' ``cma.plot()`` etc. is not available')
# from . import fitness_functions as ff
from . import evolution_strategy as es
from . import fitness_transformations as ft
from . import transformations as tf
from . import constraints_handler as ch
from .utilities import utils
from .evolution_strategy import CMAEvolutionStrategy as CMAES
from .utilities.utils import pprint
from .utilities.math import Mh
# from .fitness_functions import elli as felli

if _pyplot:
    def figshow():
        """`pyplot.show` to make a plotted figure show up"""
        # is_interactive = matplotlib.is_interactive()

        _pyplot.ion()
        _pyplot.show()
        # if we call now matplotlib.interactive(True), the console is
        # blocked
    figsave = _pyplot.savefig

if 11 < 3:
    try:
        from matplotlib.pyplot import savefig as figsave, close as figclose, ion as figion
    except:
        figsave, figclose, figshow = 3 * ['not available']
        _warnings.warn('Could not import matplotlib.pyplot, therefore ``cma.plot()``'
                       ' etc. is not available')

def _cdict(obj, exclude='_'):
    """return public attributes of `obj` as a `dict`.

    Pass '__' as second argument to see "private" attributes.
    """
    return {d[0]:d[1] for d in obj.__dict__.items() if not d[0].startswith(exclude)}
def clean(obj, exclude='_'):
    """return "public" elements of a `list` or `dict` or of ``object.__dict__``.

    Ignore entries starting with `exclude`. Return a 'list` when `obj`
    is a `list` or a `dict`, return a `dict` otherwise. This is versatile
    and could change in future.

    Typical usage: ``clean(dir())`` or ``clean(obj)`` or ``clean(dir(), '__')``.
    """
    if isinstance(obj, (list, dict)):
        return [d for d in obj if not d.startswith(exclude)]
    return _cdict(obj, exclude=exclude)
