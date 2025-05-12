# -*- coding: utf-8 -*-
"""A collection of variable boundaries (AKA box constraints) handling classes.
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
import warnings as _warnings
import numpy as np
from .utilities import utils as _utils
from .utilities.utils import rglen
from .transformations import BoxConstraintsLinQuadTransformation
del absolute_import, division, print_function  #, unicode_literals

def normalize_bounds(bounds, copy=True):
    """return ``[lower_bounds, upper_bounds]`` such that each of them

    is either a nonempty `list` or `None`.

    `copy` only makes a shallow copy.

    On input, `bounds` must be a `list` of length 2.
    """
    for i in [0, 1]:
        try:
            if len(bounds[i]) == 0:  # bails when bounds[i] is a scalar
                if copy:
                    bounds = list(bounds)
                    copy = False
                bounds[i] = None  # let's use None instead of empty list
        except TypeError:
            if copy:
                bounds = list(bounds)
                copy = False
            bounds[i] = [bounds[i]]
        if not _utils.is_(bounds[i]) or all(
                bounds[i][j] is None or not np.isfinite(bounds[i][j])
                    for j in rglen(bounds[i])):
            if copy:
                bounds = list(bounds)
                copy = False
            bounds[i] = None
        if bounds[i] is not None and any(bounds[i][j] == (-1)**i * np.inf
                                            for j in rglen(bounds[i])):
            raise ValueError('lower/upper is +inf/-inf and ' +
                                'therefore no finite feasible solution is available')
    return bounds

def none_to_inf(bounds, copy=True):
    """replace any None by inf or -inf.

    This code was never tested and is not in use.
    """
    copy_i = [copy, copy]
    for ib, val in [[0, -np.inf], [1, np.inf]]:
        if bounds[ib] is None:
            if copy:
                bounds = list(bounds)
                copy = False
            bounds[ib] = [val]
            continue
        for i in rglen(bounds[ib]):
            if bounds[ib][i] is None:
                if copy_i[ib]:
                    bounds[ib] = list(bounds[ib])
                    copy_i[ib] = False
                bounds[ib][i] = val
    return bounds

class BoundDomainTransform(object):
    """create a `callable` with unbounded domain from a function with bounded domain,

    for example an objective or constraints function. The new "unbounded"
    `callable` maps ``x`` to ``function(self.transform(x))``, first
    "projecting" the input (e.g., candidate solutions) into the bounded
    domain with `transform` before to evaluate the result on the original
    function. The "projection" is smooth and differentiable, namely
    coordinate-wise piecewise linear or quadratic. The "projection" is not
    idempotent.

    Bounds are passed as ``boundaries = [lower_bounds, upper_bounds]``
    where each ``*_bounds`` can be `None`, or a scalar, or an iterable (a
    vector) whose values can be `None` too. If the iterable has fewer
    values than the solutions, the last value is recycled, if it has more
    values, trailing values are ignored.

    For example, ``boundaries = [None, [0, 0, None]]`` means no lower
    bounds and an upper bound of zero for the first two variables.

    Example:

    >>> import cma
    >>> fun = cma.boundary_handler.BoundDomainTransform(
    ...             cma.ff.sphere,  # is composed with fun.transform into fun
    ...             [[0.02, 0.01], None])  # boundaries for the "original" problem
    >>> x, es = cma.fmin2(fun, 3 * [0.5], 0.5, {'verbose':-9})
    >>> assert all(x - 1e-4 < [-0.03, -0.04, -0.04])  # x is in the unbounded domain
    >>> print("solution in the original (bounded) domain = {}"
    ...       .format(fun.transform(es.result.xfavorite)))
    solution in the original (bounded) domain = [0.02 0.01 0.01]

    The original function can be accessed and called via the attribute
    ``function`` like ``fun.function(...)``. For code simplicity,
    attributes of ``function`` are "inherited" to ``fun``.

    Details: the resulting function has a repetitive landscape with a
    period slightly larger than twice the boundary interval. The
    ``BoundDomainTransform(function,...)`` instance emulates the original function
    on attribute access by calling ``__getattr__``. To access shadowed
    attributes or for debugging, replace ``.attrname`` with
    ``.function.attrname``.
    """
    def __init__(self, function, boundaries):
        """return a callable that evaluates `function` only within `boundaries`"""
        self.function = function
        self.boundary_handler = BoundTransform(boundaries)
        self.transform = self.boundary_handler.transform  # alias for convenience
    def __call__(self, x, *args, **kwargs):
        return self.function(self.transform(x), *args, **kwargs)
    def __getattr__(self, name):
        """return ``getattr(self.function, name)`` when ``not hasattr(self, name)``.

        This emulates the `function` interface, kinda like blind inheritance.
        """
        return getattr(self.function, name)

class _BoundDomainPenalty(object):
    """[WIP early] function wrapper for penalty boundary handler, looks kinda complex"""
    def __init__(self, function, boundaries):
        """return a callable that evaluates `function` only within `boundaries`"""
        self.function = function
        self.boundary_handler = BoundPenalty(boundaries)
        self.transform = self.boundary_handler.repair  # alias for convenience
    def __call__(self, x, *args, **kwargs):
        return self.function(self.transform(x), *args, **kwargs)
    def callback(self):
        """needs the mean and the fitness values?"""
        raise NotImplementedError
    def __getattr__(self, name):
        """return ``getattr(self.function, name)`` when ``not hasattr(self, name)``.

        This emulates the `function` interface, kinda like blind inheritance.
        """
        return getattr(self.function, name)

class BoundaryHandlerBase(object):
    """quick hack versatile base class.

    To guaranty that modifications of the attribute `bounds` after this
    instance was already used are effective, the class attribute
    ``_bounds_dict`` must be reset like ``_bounds_dict = {}``.
    """
    use_cached_values = False
    '''default behavior as to whether to use cached values'''
    def __init__(self, bounds):
        """`bounds` can be ``None`` or ``[lb, ub]``

        where ``lb`` and ``ub`` are either ``None`` or a vector (which can have
        ``None`` entries).

        On return, the ``bounds`` attribute of ``self`` are the bounds in a
        normalized form.

        To compute bounds for any dimension, the last entry of bounds is
        then recycled for variables with ``indices >= len(bounds[i]) for i
        in (0,1)``.

        """
        self.use_cached_values = BoundaryHandlerBase.use_cached_values
        if bounds in [None, (), []]:
            self.bounds = None
        else:
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(
                    "bounds must be None, empty, or a list of length 2"
                    " where each element may be a scalar, list, array,"
                    " or None; type(bounds) was: %s" % str(type(bounds)))
            self.bounds = normalize_bounds(list(bounds), copy=False)
        self._bounds_dict = {}
        '''saved return values of get_bounds(i, dim). Changing `self.bounds` may
           only be effective when this is reset.'''

    def amend_bounds_for_integer_variables(self, integer_indices, at=0.5, offset=1e-9):
        """set bounds away from ``at=0.5`` such that

        a repaired solution is always rounded into the feasible domain.
        """
        if not integer_indices or not self.has_bounds():
            return
        def set_bounds(which, idx, integer_indices):
            """idx is a `bool` index array of ``bound % 1 == 0.5``"""
            assert which in (0, 1), which
            if not np.any(idx):
                return
            for i in np.nonzero(idx)[0]:
                if i not in integer_indices:
                    idx[i] = False
            if not np.any(idx):
                return
            dimension = max((len(self.bounds[which]),
                             np.max(np.nonzero(idx)[0]) + 1))
            bounds = self.get_bounds(which, dimension)
            if len(bounds) < len(idx):
                idx = idx[:len(bounds)]  # last nonzero entry determined len
            elif len(bounds) > len(idx):
                idx = np.hstack([idx, np.zeros(len(bounds) - len(idx), dtype=bool)])
            bounds[idx] += (1 - 2 * which) * offset * np.maximum(1, np.abs(bounds[idx]))
            self.bounds[which] = bounds
            self._bounds_dict = {}

        dimension = max(integer_indices) + 1
        for which in (0, 1):
            if not self.has_bounds('upper' if which else 'lower'):
                continue
            bounds = self.get_bounds(which, dimension)
            set_bounds(which, np.mod(bounds, 1) == at, integer_indices)

    def __call__(self, solutions, *args, **kwargs):
        """return penalty or list of penalties, by default zero(s).

        This interface seems too specifically tailored to the derived
        BoundPenalty class, it should maybe change.

        """
        return 0.0 if np.isscalar(solutions[0]) else len(solutions) * [0.0]

    def update(self, *args, **kwargs):
        """end-iteration callback of boundary handler (abstract/empty)"""
        return self

    def repair(self, x, copy_if_changed=True):
        """projects infeasible values on the domain bound, might be
        overwritten by derived class """
        copy = copy_if_changed
        if self.bounds is None:
            return x
        for ib in [0, 1]:
            if self.bounds[ib] is None:
                continue
            for i in rglen(x):
                idx = min([i, len(self.bounds[ib]) - 1])
                if self.bounds[ib][idx] is not None and \
                        (-1)**ib * x[i] < (-1)**ib * self.bounds[ib][idx]:
                    if copy:
                        x = np.array(x, copy=True)
                        copy = False
                    x[i] = self.bounds[ib][idx]
        return x

    def inverse(self, y, copy_if_changed=True):
        """inverse of repair if it exists, at least it should hold
         ``repair == repair o inverse o repair``"""
        return y

    def get_bounds(self, which, dimension):
        """``get_bounds('lower', 8)`` returns the lower bounds in 8-D"""
        if which in ['lower', 0, '0']:
            return self._get_bounds(0, dimension)
        elif which in ['upper', 1, '1']:
            return self._get_bounds(1, dimension)
        else:
            raise ValueError("argument which must be 'lower' or 'upper'")

    def _get_bounds(self, ib, dimension):
        """ib == 0/1 means lower/upper bound, return a vector of length
        `dimension` """
        if not self.use_cached_values:
            self._bounds_dict = {}  # recover original behavior
        if (ib, dimension) in self._bounds_dict:
            return self._bounds_dict[(ib, dimension)]
        sign_ = 2 * ib - 1
        assert sign_**2 == 1
        if self.bounds is None or self.bounds[ib] is None:
            self._bounds_dict[(ib, dimension)] = sign_ * np.inf + np.zeros(dimension)
            return self._bounds_dict[(ib, dimension)]
        res = []
        for i in range(dimension):
            res.append(self.bounds[ib][min([i, len(self.bounds[ib]) - 1])])
            if res[-1] is None:
                res[-1] = sign_ * np.inf
        self._bounds_dict[(ib, dimension)] = np.asarray(res)
        return self._bounds_dict[(ib, dimension)]

    def has_bounds(self, which='both'):
        """return `True` if any variable is bounded"""
        valid_whichs = (None, 'both', 'lower', 'upper')
        if which not in valid_whichs:
            raise ValueError("`which` parameter must be in {0} but was ={1}"
                                .format(valid_whichs, which))
        bounds = self.bounds
        if bounds is None or bounds in (False, [], ()) or (
                bounds[0] is None and bounds[1] is None):
            return False
        for ib, bound in enumerate(bounds):
            if bound is None or (
                ib == 0 and which == 'upper') or (
                ib == 1 and which == 'lower'):
                continue
            sign_ = 2 * ib - 1
            for bound_i in bound:
                if bound_i is not None and sign_ * bound_i < np.inf:
                    return True
        return False

    def is_in_bounds(self, x):
        """return `True` if `x` is in bounds.

        >>> import numpy as np
        >>> import cma

        >>> b = cma.boundary_handler.BoundaryHandlerBase(bounds=[[-0.5], 0.2])
        >>> for n in [2, 4, 9]:
        ...     x = np.random.randn(n)
        ...     assert (all(-0.5 <= x) and all(0.2 >= x)) is b.is_in_bounds(x)

        """
        if self.bounds is None:
            return True
        for ib in [0, 1]:
            if self.bounds[ib] is None:
                continue
            for i in rglen(x):
                if i == len(self.bounds[ib]) - 1 and (
                        self.bounds[ib][i] is None or
                        (-1)**(1 - ib) * self.bounds[ib][i] == np.inf):
                    break
                idx = min([i, len(self.bounds[ib]) - 1])
                if self.bounds[ib][idx] is None:
                    continue
                if ((ib == 0 and x[i] < self.bounds[ib][idx]) or (
                     ib == 1 and x[i] > self.bounds[ib][idx])):
                    return False
        return True

    def into_bounds(self, x, copy=True):
        """set out-of-bound values on bounds and return `x`.

        Make a copy when `x` is changed and ``copy is True``,
        otherwise change in place.

        >>> import numpy as np
        >>> import cma

        >>> b = cma.boundary_handler.BoundaryHandlerBase(bounds=[[0.1], 0.2])
        >>> assert all(0.1 <= b.into_bounds(np.random.randn(22)))
        >>> assert all(0.2 >= b.into_bounds(np.random.randn(11)))
        >>> b = cma.boundary_handler.BoundaryHandlerBase(bounds=[[-0.1], np.inf])
        >>> assert all(-0.1 <= b.into_bounds(np.random.randn(22)))
        >>> b = cma.boundary_handler.BoundaryHandlerBase(bounds=[[-0.1], [None]])
        >>> assert all(-0.1 <= b.into_bounds(np.random.randn(22)))
        >>> b = cma.boundary_handler.BoundaryHandlerBase(bounds=[-np.inf, 0.1])
        >>> assert all(0.1 >= b.into_bounds(np.random.randn(22)))

        """
        if self.bounds is None:
            return x
        for ib in [0, 1]:
            if self.bounds[ib] is None:
                continue
            for i in rglen(x):
                if i == len(self.bounds[ib]) - 1 and (
                          self.bounds[ib][i] is None or
                          (-1)**(1 - ib) * self.bounds[ib][i] == np.inf):
                    break
                idx = min([i, len(self.bounds[ib]) - 1])
                if self.bounds[ib][idx] is None:
                    continue
                if ((ib == 0 and x[i] >= self.bounds[ib][idx]) or (
                     ib == 1 and x[i] <= self.bounds[ib][idx])):
                    continue
                if copy:
                    x = np.array(x, copy=True)
                    copy = False  # copy only once
                x[i] = self.bounds[ib][idx]
        return x

    def idx_out_of_bounds(self, x):
        """return index list of out-of-bound values in `x`.

        ``if bounds.idx_out_of_bounds`` evaluates to `True` if and only if
        `x` is out of bounds.
        """
        if self.bounds is None:
            return []
        idxs = []
        for ib in [0, 1]:
            if self.bounds[ib] is None:
                continue
            for i in rglen(x):
                idx = min([i, len(self.bounds[ib]) - 1])
                if self.bounds[ib][idx] is None:
                    continue
                if ((ib == 0 and x[i] < self.bounds[ib][idx]) or (
                     ib == 1 and x[i] > self.bounds[ib][idx])):
                   idxs += [i]
        return sorted(idxs)

    def get_bound(self, index):
        """return lower and upper bound of variable with index `index`"""
        if self.bounds is None:
            return [-np.inf, np.inf]
        res = []
        for ib in [0, 1]:
            if self.bounds[ib] is None or len(self.bounds[ib]) == 0:
                b = None
            else:
                b = self.bounds[ib][min((index, len(self.bounds[ib]) - 1))]
            res.append([-np.inf, np.inf][ib] if b is None else b)
        return res

    def to_dim_times_two(self, bounds=None):
        """return boundaries in format ``[[lb0, ub0], [lb1, ub1], ...]``,
        as used by ``BoxConstraints...`` class.

        Use by default ``bounds = self.bounds``.
        """
        if bounds is None:
            try:
                bounds = self.bounds
            except AttributeError:
                pass
        if not bounds:
            return [[None, None]]
        bounds = normalize_bounds(list(bounds), copy=False)
        for i in [0, 1]:
            if bounds[i] is None:  # replace None with [None] which is easier below
                bounds[i] = [None]  # [-np.inf if i == 0 else np.inf]
        l = [len(bounds[i]) for i in [0, 1]]
        if l[0] != l[1] and 1 not in l and None not in (
                bounds[0][-1], bounds[1][-1]):  # warn on different lengths
            _warnings.warn(
                "lower and upper bounds do not have the same length or length\n"
                " one or `None` as last element (the last"
                    " element is always recycled).\n"
                "Lengths are {0} = [len(b) for b in bounds={1}])".format(l, bounds))
        b = []  # bounds in different format
        try:
            for i in range(max(l)):
                b.append([bounds[0][min((i, l[0] - 1))],
                            bounds[1][min((i, l[1] - 1))]])
        except (TypeError, IndexError):
            print("boundaries must be provided in the form " +
                    "[scalar_of_vector, scalar_or_vector]")
            raise
        return b

class BoundNone(BoundaryHandlerBase):
    """no boundaries"""
    def __init__(self, bounds=None):
        if bounds is not None:
            raise ValueError()
        # BoundaryHandlerBase.__init__(self, None)
        super(BoundNone, self).__init__(None)
    def is_in_bounds(self, x):
        return True

class BoundTransform(BoundaryHandlerBase):
    """Handle boundaries by a smooth, piecewise linear and quadratic
    transformation into the feasible domain.

    >>> import numpy as np
    >>> import cma
    >>> from cma.boundary_handler import BoundTransform
    >>> from cma import fitness_transformations as ft
    >>> veq = cma.utilities.math.Mh.vequals_approximately
    >>> b = BoundTransform([0, None])
    >>> assert b.bounds == [[0], None]
    >>> assert veq(b.repair([-0.1, 0, 1, 1.2]), np.array([0.0125, 0.0125, 1, 1.2])), b.repair([-0.1, 0, 1, 1.2])
    >>> assert b.is_in_bounds([0, 0.5, 1])
    >>> assert veq(b.transform([-1, 0, 1, 2]), [0.9, 0.0125,  1,  2  ]), b.transform([-1, 0, 1, 2])
    >>> bounded_sphere = ft.ComposedFunction([
    ...         cma.ff.sphere,
    ...         BoundTransform([[], 5 * [-1] + [np.inf]]).transform
    ...     ])
    >>> o1 = cma.fmin(bounded_sphere, 6 * [-2], 0.5)  # doctest: +ELLIPSIS
    (4_w,9)-aCMA-ES (mu_w=2.8,w_1=49%) in dimension 6 (seed=...
    >>> o2 = cma.fmin(cma.ff.sphere, 6 * [-2], 0.5, options={
    ...    'BoundaryHandler': cma.boundary_handler.BoundTransform,
    ...    'bounds': [[], 5 * [-1] + [np.inf]] })  # doctest: +ELLIPSIS
    (4_w,9)-aCMA-ES (mu_w=2.8,w_1=49%) in dimension 6 (seed=...
    >>> assert o1[1] < 5 + 1e-8 and o2[1] < 5 + 1e-8
    >>> b = BoundTransform([-np.random.rand(120), np.random.rand(120)])
    >>> for i in range(0, 100, 9):
    ...     x = (-i-1) * np.random.rand(120) + i * np.random.randn(120)
    ...     x_to_b = b.repair(x)
    ...     x2 = b.inverse(x_to_b)
    ...     x2_to_b = b.repair(x2)
    ...     x3 = b.inverse(x2_to_b)
    ...     x3_to_b = b.repair(x3)
    ...     assert veq(x_to_b, x2_to_b)
    ...     assert veq(x2, x3)
    ...     assert veq(x2_to_b, x3_to_b)
    >>> for _ in range(5):
    ...     lb = np.random.randn(4)
    ...     ub = lb + 1e-7 + np.random.rand(4)
    ...     b = BoundTransform([lb, ub])
    ...     for x in [np.random.randn(4) / np.sqrt(np.random.rand(4)) for _ in range(22)]:
    ...         assert all(lb <= b.transform(x)), (lb, ub, b.__dict__)
    ...         assert all(b.transform(x) <= ub), (lb, ub, b.__dict__)

    Details: this class uses ``class BoxConstraintsLinQuadTransformation``

    """
    def __init__(self, bounds=None):
        """Argument bounds can be `None` or ``bounds[0]`` and ``bounds[1]``
        are lower and upper domain boundaries, each is either `None` or
        a scalar or a list or array of appropriate size.

        """
        # BoundaryHandlerBase.__init__(self, bounds)
        super(BoundTransform, self).__init__(bounds)
        self.bounds_tf = BoxConstraintsLinQuadTransformation(self.to_dim_times_two(bounds))

    def repair(self, x, copy_if_changed=True):
        """transforms ``x`` into the bounded domain.
        """
        copy = copy_if_changed
        if self.bounds is None or (self.bounds[0] is None and
                                   self.bounds[1] is None):
            return x
        return np.asarray(self.bounds_tf(x, copy))

    def transform(self, x):
        return self.repair(x)

    def inverse(self, x, copy_if_changed=True):
        """inverse transform of ``x`` from the bounded domain.

        """
        if self.bounds is None or (self.bounds[0] is None and
                                   self.bounds[1] is None):
            return x
        return np.asarray(self.bounds_tf.inverse(x, copy_if_changed))  # this doesn't exist

class BoundPenalty(BoundaryHandlerBase):
    """Compute a bound penalty and update coordinate-wise penalty weights.

    An instance must be updated each iteration using the `update` method.

    Details:

    - The penalty computes like ``sum(w[i] * (x[i]-xfeas[i])**2)``,
      where ``xfeas`` is the closest feasible (in-bounds) solution from
      ``x``. The weight ``w[i]`` should be updated during each iteration
      using the update method.

    Example how this boundary handler is used with `cma.fmin` via the
    options (`CMAOptions`) of the class `cma.CMAEvolutionStrategy`:

    >>> import cma
    >>> res = cma.fmin(cma.ff.elli, 6 * [0.9], 0.1,
    ...     {'BoundaryHandler': cma.BoundPenalty,
    ...      'bounds': [-1, 1],
    ...      'tolflatfitness': 10,
    ...      'fixed_variables': {0: 0.012, 2:0.234}
    ...     })  # doctest: +ELLIPSIS
    (4_w,8)-aCMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=...
    >>> if res[1] >= 13.76: print(res)  # should never happen

    Reference: Hansen et al 2009, A Method for Handling Uncertainty...
    IEEE TEC, with addendum, see
    https://ieeexplore.ieee.org/abstract/document/4634579
    https://hal.inria.fr/inria-00276216/file/TEC2008.pdf

    **todo**: implement a more generic interface, where this becomes a
    fitness wrapper which adds the desired penalty and the `update`
    method is used as callback argument for `fmin` like::

        f = cma.BoundPenalty(cma.ff.elli, bounds=[-1, 1])
        res = cma.fmin(f, 6 * [1], callback=f.update)

    where callback functions should receive the same arguments as
    `tell`, namely an `CMAEvolutionStrategy` instance, an array of the
    current solutions and their respective f-values. Such change is
    relatively involved. Consider also that bounds are related with the
    geno- to phenotype transformation.
    """
    def __init__(self, bounds=None):
        """Argument bounds can be `None` or ``bounds[0]`` and ``bounds[1]``
        are lower  and upper domain boundaries, each is either `None` or
        a scalar or a `list` or `np.array` of appropriate size.
        """
        # #
        # bounds attribute reminds the domain boundary values
        # BoundaryHandlerBase.__init__(self, bounds)
        super(BoundPenalty, self).__init__(bounds)

        self.gamma = 1  # a very crude assumption
        self.weights_initialized = False  # gamma becomes a vector after initialization
        self.hist = []  # delta-f history

    def repair(self, x, copy_if_changed=True):
        """sets out-of-bounds components of ``x`` on the bounds.

        TODO: unify with BoundaryHandlerBase.repair
        """
        # TODO (old data): CPU(N,lam,iter=20,200,100): 3.3s of 8s for two bounds, 1.8s of 6.5s for one bound
        # remark: np.max([bounds[0], x]) is about 40 times slower than max((bounds[0], x))
        copy = copy_if_changed
        if self.has_bounds():
            bounds = self.bounds
            if copy:
                x = np.array(x, copy=True)
            if bounds[0] is not None:
                if np.isscalar(bounds[0]):
                    for i in rglen(x):
                        x[i] = max((bounds[0], x[i]))
                else:
                    for i in rglen(x):
                        j = min([i, len(bounds[0]) - 1])
                        if bounds[0][j] is not None:
                            x[i] = max((bounds[0][j], x[i]))
            if bounds[1] is not None:
                if np.isscalar(bounds[1]):
                    for i in rglen(x):
                        x[i] = min((bounds[1], x[i]))
                else:
                    for i in rglen(x):
                        j = min((i, len(bounds[1]) - 1))
                        if bounds[1][j] is not None:
                            x[i] = min((bounds[1][j], x[i]))
        return x

    # ____________________________________________________________
    #
    def __call__(self, x, archive, gp):
        """returns the boundary violation penalty for `x`,
        where `x` is a single solution or a list or np.array of solutions.

        """
        # if x in (None, (), []):  # breaks when x is a nparray
        #     return x
        if not self.has_bounds():
            return 0.0 if np.isscalar(x[0]) else [0.0] * len(x)  # no penalty

        x_is_single_vector = np.isscalar(x[0])
        if x_is_single_vector:
            x = [x]

        # add fixed variables to self.gamma
        try:
            gamma = list(self.gamma)  # fails if self.gamma is a scalar
            for i in sorted(gp.fixed_values):  # fails if fixed_values is None
                gamma.insert(i, 0.0)
            gamma = np.asarray(gamma)
        except TypeError:
            gamma = self.gamma
        pen = []
        for xi in x:
            # CAVE: this does not work with already repaired values!!
            # CPU(N,lam,iter=20,200,100)?: 3s of 10s, np.array(xi): 1s
            # remark: one deep copy can be prevented by xold = xi first
            xpheno = xi if gp.isidentity else gp.pheno(archive[xi]['geno'])
            # necessary, because xi was repaired to be in bounds
            xinbounds = self.repair(xpheno)
            # could be omitted (with unpredictable effect in case of external repair)
            fac = 1  # exp(0.1 * (log(self.scal) - np.mean(self.scal)))
            pen.append(sum(gamma * ((xinbounds - xpheno) / fac)**2) / len(xi))
        return pen[0] if x_is_single_vector else pen

    # ____________________________________________________________
    #
    def feasible_ratio(self, solutions):
        """counts for each coordinate the number of feasible values in
        ``solutions`` and returns an `np.array` of length
        ``len(solutions[0])`` with the ratios.
        """
        raise NotImplementedError

    # ____________________________________________________________
    #
    def update(self, function_values, es):
        """updates the weights for computing a boundary penalty.

        Arguments
        =========
        ``function_values``:
            all function values of recent population of solutions
        ``es``:
            `CMAEvolutionStrategy` object instance, in particular
            mean and variances and the methods from the attribute
            `gp` of type `GenoPheno` are used.

        """
        if self.bounds is None or (self.bounds[0] is None and
                                   self.bounds[1] is None):
            return self

        N = es.N
        # ## prepare
        # compute varis = sigma**2 * C_ii
        if 11 < 3:  # old
            varis = es.sigma**2 * np.array(N * [es.C] if np.isscalar(es.C) else (# scalar case
                                    es.C if np.isscalar(es.C[0]) else  # diagonal matrix case
                                    [es.C[i][i] for i in range(N)]))  # full matrix case
        else:
            varis = es.sigma**2 * es.sm.variances

        # relative violation in geno-space
        dmean = (es.mean - es.gp.geno(self.repair(es.gp.pheno(es.mean)))) / varis**0.5

        # ## Store/update a history of delta fitness value
        fvals = sorted(function_values)
        _l = 1 + len(fvals)
        val = fvals[3 * _l // 4] - fvals[_l // 4]  # exact interquartile range apart interpolation
        val = val / np.mean(varis)  # new: val is normalized with sigma of the same iteration
        # insert val in history
        if np.isfinite(val) and val > 0:
            self.hist.insert(0, val)
        elif val == np.inf and len(self.hist) > 1:
            self.hist.insert(0, max(self.hist))
        else:
            pass  # ignore 0 or nan values
        if len(self.hist) > 20 + (3 * N) / es.popsize:
            self.hist.pop()

        # ## prepare
        dfit = np.median(self.hist)  # median interquartile range
        damp = min(1, es.sp.weights.mueff / 10. / N)

        # ## set/update weights
        # Throw initialization error
        if len(self.hist) == 0:
            raise ValueError('wrongful initialization, no feasible solution sampled. ' +
                'Reasons can be mistakenly set bounds (lower bound not smaller than upper bound) or a too large initial sigma0 or... ' +
                'See description of argument func in help(cma.fmin) or an example handling infeasible solutions in help(cma.CMAEvolutionStrategy). ')
        # initialize weights
        if dmean.any() and (not self.weights_initialized or es.countiter == 2):  # TODO
            self.gamma = np.array(N * [2 * dfit])  ## BUGBUGzzzz: N should be phenotypic (bounds are in phenotype), but is genotypic
            self.weights_initialized = True
        # update weights gamma
        if self.weights_initialized:
            edist = np.array(abs(dmean) - 3 * max(1, N**0.5 / es.sp.weights.mueff))
            if 1 < 3:  # this is better, around a factor of two
                # increase single weights possibly with a faster rate than they can decrease
                #     value unit of edst is std dev, 3==random walk of 9 steps
                self.gamma *= np.exp((edist > 0) * np.tanh(edist / 3) / 2.)**damp
                # decrease all weights up to the same level to avoid single extremely small weights
                #    use a constant factor for pseudo-keeping invariance
                self.gamma[self.gamma > 5 * dfit] *= np.exp(-1. / 3)**damp
                #     self.gamma[idx] *= exp(5*dfit/self.gamma[idx] - 1)**(damp/3)
            elif 1 < 3 and (edist > 0).any():  # previous method
                # CAVE: min was max in TEC 2009
                self.gamma[edist > 0] *= 1.1**min(1, es.sp.weights.mueff / 10. / N)
                # max fails on cigtab(N=12,bounds=[0.1,None]):
                # self.gamma[edist>0] *= 1.1**max(1, es.sp.weights.mueff/10./N) # this was a bug!?
                # self.gamma *= exp((edist>0) * np.tanh(edist))**min(1, es.sp.weights.mueff/10./N)
            else:  # alternative version, but not better
                solutions = es.pop  # this has not been checked
                r = self.feasible_ratio(solutions)  # has to be the averaged over N iterations
                self.gamma *= np.exp(np.max([N * [0], 0.3 - r], axis=0))**min(1, es.sp.weights.mueff / 10 / N)
        es.more_to_write += list(self.gamma) if self.weights_initialized else N * [1.0]
        # ## return penalty
        # es.more_to_write = self.gamma if not np.isscalar(self.gamma) else N*[1]
        return self  # bound penalty values

