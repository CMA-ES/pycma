"""versatile shortcuts for quick typing in an (i)python shell,

 or even using ``from cma.s import *`` in interactive sessions.

Provides various aliases from within the `cma` package, to be reached like
``cma.s....``

Do not use this module for stable code developments!

This may not be actively maintained.
"""
import warnings as _warnings
import math as ma
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
from .evolution_strategy import CMAEvolutionStrategy as _CMAES
from .utilities.utils import pprint
from .utilities.math import Mh, testchisquare, testranksum
from .utilities.utils import figure
# from .fitness_functions import elli as felli

log_tick_labels_candidates = [
        [1],
        [1, 3],  # leads to up to 2 * (log_tick_labels_number - 1) + 1 = 9 labels
        [1, 2, 5],  # factors 2, 2.5
        [1, 2, 4, 6],  # factors 2, 1.5
        # [1, 2, 3, 5, 7],  # factors 2, 1.7, 1.4
        [1, 1.5, 2, 3, 5, 7],  # 1.5 adds a useful grid line, factors 1.5, 1.3, 1.7, 1.4
        # [1, 1.5, 2, 3, 4, 6, 8],  # doesn't look too good
        [1, 1.5, 2, 3, 4, 5, 7],
        [1, 1.5, 2, 3, 4, 5, 6, 8],
        [1, 1.5, 2, 3, 4, 5, 6, 7, 8],
        [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
        ]
'''candidate tick label positions [times 10^i]'''
log_tick_labels_number = 5
'''minimal number of tick labels'''

class CMAES(_CMAES):
    def __init__(self, *args, **kwargs):
        _warnings.warn("Deprecated (renamed). Use `cma.CMA` instead of `cma.s.CMAES`",
                       FutureWarning)
        super(CMAES, self).__init__(*args, **kwargs)

if _pyplot:
    def figshow():
        """`pyplot.show` to make a plotted figure show up"""
        # is_interactive = matplotlib.is_interactive()

        _pyplot.ion()
        _pyplot.show()
        # if we call now matplotlib.interactive(True), the console is
        # blocked
    figsave = _pyplot.savefig
    def grid(visible=True, which='both', **kwargs):
        """arguments are passed to `matplotlib.pyplot.grid`"""
        _pyplot.grid(visible, which, **kwargs)
    def figpolish(**kwargs):
        """set better `matplotlib` figure default values

        for scrutinizing data, in particular `grid` and `clip_on`.

        Set the axis limits, such that regression or grid lines plotted _after_
        calling `figpolish` conveniently do not change the displayed area.

        Accepts the following kw-arguments:

        ``axis: str='tight-ish'`` where ``'image', 'scaled' or 'square'`` are
        other notable values which all lead to equal scaling, ``True, 'auto',
        'tight'`` are further alternatives, see ``help(plt.axis)``.

        ``clip_on: bool=False`` do not clip lines outside of the frame, lines
        after calling `figpolish` are still clipped

        ``frame: bool=True`` show frame (default in `matplotlib`)

        ``grid: dict={'visible': True, 'which': 'both'}``

        ``legend: dict={}`` works for setting a legend title

        ``patch: float=3`` is the relative width between data and frame in
        percent. This is not a ``matplotlib`` parameter and ignored when
        ``'axis'`` is given.
        """
        frame = kwargs.get('frame', True)
        clip_on = kwargs.get('clip_on', False)
        patch = kwargs.get('patch', 3)  # inner border patch in percentage of frame size
        # axis = kwargs.get('axis', 'tight')
        grid = kwargs.get('grid', {})
        grid.setdefault('visible', True)
        grid.setdefault('which', 'both')

        p = _pyplot

        if 'axis' not in kwargs or kwargs.get('axis') == 'tight-ish':
            p.axis('tight')
            p.xlim(better_limits(*p.xlim(), p.gca().get_xscale(), patch))
            p.ylim(better_limits(*p.ylim(), p.gca().get_yscale(), patch))
        else:
            p.axis(kwargs.get('axis'))

        # set tick labels on log axes
        # Caveat: p.?lim() may reflect the data rather than the current limits
        #         which may lead to surprising results
        if p.gca().get_xscale() == 'log':
            p.gca().set_xticks(*log_tick_labels(*p.xlim()))
        if p.gca().get_yscale() == 'log':
            p.gca().set_yticks(*log_tick_labels(*p.ylim()))

        p.grid(**grid)
        if any(p.gca().get_legend_handles_labels()):
            p.legend(**kwargs.get('legend', {}))
        for spine in p.gca().spines.values():
            spine.set_visible(frame)
        for line in p.gca().get_lines():
            line.set_clip_on(clip_on)
        # p.tight_layout()

try:
    from matplotlib.pyplot import savefig as figsave, ion as figion  # noqa: I001
except ImportError:
    figsave, figion = 2 * ['not available']

def better_limits(l, u, log='linear', patch=3):
    """``l, u`` are the ``plt.axis('tight')`` limits, return new limits.

    This is a quick hack to compute tight-ish plot limits that do not cover up
    any extreme lines or symbols.
    """
    def oom(n):  # floor order of magnitude
        return 10**ma.floor(ma.log10(n))
    def next_log_int(n, round):
        l = oom(n)
        return round(n / l) * l
    if log == 'log':
        ll, lu = ma.log10(l), ma.log10(u)
        ld = (lu - ll) * patch / 100  # 3%
        xl = 10**(ll - ld)
        xu = 10**(lu + ld)
        # let the limit be the "next integer" when this has little effect
        xln = next_log_int(xl, ma.floor)
        if xl / xln < (xu / xl)**(patch/100):
            xl = xln
        xun = next_log_int(xu, ma.ceil)
        if xun / xu < (xu / xl)**(patch/100):
            xu = xun
        # treat special case of limit between 1 and 2 more generously
        if xu / xl >= 20:  # otherwise we usually annotate also 1.5
            if 1 < xl / oom(xl) < 1.4:
                xl = 10**ma.floor(ll)
            if 1.42 < xu / oom(xu) < 2:
                xu = 2 * 10**ma.floor(lu)
        # if xu - xl > 10:
        #     if xl > 10:
        #         xl = ma.floor(xl)  # 0.1 / 20 = 0.5%, xl is always positive
        #     if xu > 20:  # xl may be very small and not rounded
        #         xu = ma.floor(xu + 0.9)
    else:
        d = (u - l) * patch / 100
        xl = l - d
        xu = u + d
        if d > 0.5:
            xl = ma.floor(xl + 0.1)
            xu = ma.ceil(xu - 0.1)
    return xl, xu

def formated_number(n):
    """return $10^{{-3}}$, $0.01$,...,$1000$, $10^{{4}}$..."""
    def as_int(val):
        return int(val) if val == val // 1 else ma.floor(val * 1e11 + 0.5) / 1e11
    fln = int(ma.floor(ma.log10(n)))  # floor returns a float in Python 2
    if 1e-2 <= n < 1e4:
        # only since Python 3.1, format can handle empty (implicit) references {}
        return "{0}".format(as_int(n))
    if ma.isclose(n / 10**fln, 1):
        return "$10^{{{0}}}$".format(fln)
    return r"${0}\times10^{{{1}}}$".format(as_int(n / 10**fln), fln)

def log_tick_labels(l, u):
    """return ``positions, labels`` for ``plt.gca().set_?ticks`` with log-log plots.

    `l` and `u` are the axis limits, ``*plt.?lim()``, tick labels are chosen
    based on ``log_tick_labels_candidates`` and ``log_tick_labels_number``.

    Example::

        if plt.gca().get_xscale() == 'log':
            plt.gca().set_xticks(*log_tick_labels(*plt.xlim()))

    Details: ``(1,2,4,6)`` leaves three empty lines 7-9, hence ``(1,2,5,7)``
    seems preferable where also the smallest factors ``7/5`` and ``10/7`` are
    close to ``sqrt(2)`` and ``>=1.4`` while ``6/4=1.33`` is smaller.
    """
    ll = int(ma.log10(l)) - 2  # caveat: int(-1.9) == -1
    lu = int(ma.log10(u)) + 3
    for facs in log_tick_labels_candidates:
        positions = [fac * 10**i  # are sorted
                        for i in range(ll, lu)
                            for fac in facs
                     if l <= fac * 10**i <= u]
        if len(positions) >= log_tick_labels_number:
            break            
    if u / l > 10000:
        return positions, [formated_number(p) for p in positions]

    # add empty labels to overwrite possible matplotlib labels at 4 and 6
    all_positions = {fac * 10**i
                        for i in range(ll, lu)
                            for fac in range(1, 10)
                                if l <= fac * 10**i <= u}
    all_positions.update(positions)  # add 1.5, 15,...
    all_positions = sorted(all_positions)  # list(.) should be fine too
    return all_positions, [formated_number(p) if p in positions else ''
                           for p in all_positions]

def _cdict(obj, exclude='_'):
    """return public attributes of `obj` as a `dict`.

    Pass '__' as second argument to see "private" attributes.
    """
    return {d[0]:d[1] for d in obj.__dict__.items() if not d[0].startswith(exclude)}
def clean(*args, **kwargs):
    """Deprecated, renamed to `ddir`"""
    _warnings.warn("Deprecated (renamed). Use `ddir` instead of `clean`", FutureWarning)
    return ddir(*args, **kwargs)
def ddir(obj, exclude='__', truncate_value=99):
    """return "public" elements of an object.

    Ignore entries starting with `exclude` which may be a `tuple`.

    Return a 'list` when `obj`
    is a `list` or a `dict`, return a `dict` otherwise. This is versatile
    and could change in future.

    Typical usage: ``s.ddir(obj)`` or ``s.ddir(dir(), '_')``.
    """
    names = dir(obj)
    res = {}
    for i, name in enumerate(names):
        if name.startswith(exclude):
            continue
        val = getattr(obj, name, None)  # catch not implemented error on staticmethod!
        if len(str(val)) > truncate_value:
            val = str(val)[:truncate_value]
        res[name] = '<method>' if callable(val) else val
    return res
def vvars(obj, exclude='__'):
    """return "public" elements of a `list` or `dict` or of ``object.__dict__``.

    Ignore entries starting with `exclude`.

    `vars` may do this job good enough most of the time.

    Return a 'list` when `obj` is a `list` or a `dict`, return a `dict`
    otherwise. This is versatile and could change in future.

    Typical usage: ``cma.s.vvars(obj)``.
    """
    if isinstance(obj, (list, dict)):
        return [d for d in obj if not d.startswith(exclude)]
    return _cdict(obj, exclude=exclude)
