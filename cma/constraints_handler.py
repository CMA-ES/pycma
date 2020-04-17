# -*- coding: utf-8 -*-
"""A collection of boundary and (in future) constraints handling classes.
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
# __package__ = 'cma'
import warnings as _warnings
import numpy as np
from numpy import logical_and as _and, logical_or as _or, logical_not as _not
from .utilities.utils import rglen
from .utilities.math import Mh as _Mh
from . import logger as _logger
from .transformations import BoxConstraintsLinQuadTransformation
from .utilities.python3for2 import range
del absolute_import, division, print_function  #, unicode_literals

class BoundaryHandlerBase(object):
    """quick hack versatile base class"""
    def __init__(self, bounds):
        """bounds are not copied, but possibly modified and
        put into a normalized form: ``bounds`` can be ``None``
        or ``[lb, ub]`` where ``lb`` and ``ub`` are
        either None or a vector (which can have ``None`` entries).

        Generally, the last entry is recycled to compute bounds
        for any dimension.

        """
        if bounds in [None, (), []]:
            self.bounds = None
        else:
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(
                    "bounds must be None, empty, or a list of length 2"
                    " where each element may be a scalar, list, array,"
                    " or None; type(bounds) was: %s" % str(type(bounds)))
            l = [None, None]  # figure out lengths
            for i in [0, 1]:
                try:
                    l[i] = len(bounds[i])
                except TypeError:
                    bounds[i] = [bounds[i]]
                    l[i] = 1
                if all([bounds[i][j] is None or not np.isfinite(bounds[i][j])
                        for j in rglen(bounds[i])]):
                    bounds[i] = None
                if bounds[i] is not None and any([bounds[i][j] == (-1)**i * np.inf
                                                  for j in rglen(bounds[i])]):
                    raise ValueError('lower/upper is +inf/-inf and ' +
                                     'therefore no finite feasible solution is available')
            self.bounds = bounds

    def __call__(self, solutions, *args, **kwargs):
        """return penalty or list of penalties, by default zero(s).

        This interface seems too specifically tailored to the derived
        BoundPenalty class, it should maybe change.

        """
        if np.isscalar(solutions[0]):
            return 0.0
        else:
            return len(solutions) * [0.0]

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
        sign_ = 2 * ib - 1
        assert sign_**2 == 1
        if self.bounds is None or self.bounds[ib] is None:
            return np.array(dimension * [sign_ * np.Inf])
        res = []
        for i in range(dimension):
            res.append(self.bounds[ib][min([i, len(self.bounds[ib]) - 1])])
            if res[-1] is None:
                res[-1] = sign_ * np.Inf
        return np.array(res)

    def has_bounds(self):
        """return `True` if any variable is bounded"""
        bounds = self.bounds
        if bounds is None or all(b is None for b in bounds):
            return False
        for ib, bound in enumerate(bounds):
            if bound is not None:
                sign_ = 2 * ib - 1
                for bound_i in bound:
                    if bound_i is not None and sign_ * bound_i < np.inf:
                        return True
        return False

    def is_in_bounds(self, x):
        """not yet tested"""
        if self.bounds is None:
            return True
        for ib in [0, 1]:
            if self.bounds[ib] is None:
                continue
            for i in rglen(x):
                idx = min([i, len(self.bounds[ib]) - 1])
                if self.bounds[ib][idx] is not None and \
                        (-1)**ib * x[i] < (-1)**ib * self.bounds[ib][idx]:
                    return False
        return True

    def to_dim_times_two(self, bounds):
        """return boundaries in format ``[[lb0, ub0], [lb1, ub1], ...]``,
        as used by ``BoxConstraints...`` class.

        """
        if not bounds:
            b = [[None, None]]
        else:
            l = [None, None]  # figure out lenths
            for i in [0, 1]:
                try:
                    l[i] = len(bounds[i])
                except TypeError:
                    bounds[i] = [bounds[i]]
                    l[i] = 1
            if l[0] != l[1] and 1 not in l and None not in (
                    bounds[0][-1], bounds[1][-1]):  # disallow different lengths
                raise ValueError(
                    "lower and upper bounds must have the same length\n"
                    "or length one or `None` as last element (the last"
                    " element is always recycled).\n"
                    "Lengths were %s"
                    % str(l))
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
    >>> from cma.constraints_handler import BoundTransform
    >>> from cma import fitness_transformations as ft
    >>> veq = cma.utilities.math.Mh.vequals_approximately
    >>> b = BoundTransform([None, 1])
    >>> assert b.bounds == [[None], [1]]
    >>> assert veq(b.repair([0, 1, 1.2]), np.array([ 0., 0.975, 0.975]))
    >>> assert b.is_in_bounds([0, 0.5, 1])
    >>> assert veq(b.transform([0, 1, 2]), [ 0.   ,  0.975,  0.2  ])
    >>> bounded_sphere = ft.ComposedFunction([
    ...         cma.ff.sphere,
    ...         BoundTransform([[], 5 * [-1] + [np.inf]]).transform
    ...     ])
    >>> o1 = cma.fmin(bounded_sphere, 6 * [-2], 0.5)  # doctest: +ELLIPSIS
    (4_w,9)-aCMA-ES (mu_w=2.8,w_1=49%) in dimension 6 (seed=...
    >>> o2 = cma.fmin(cma.ff.sphere, 6 * [-2], 0.5, options={
    ...    'BoundaryHandler': cma.s.ch.BoundTransform,
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
    >>> res = cma.fmin(cma.ff.elli, 6 * [1], 1,
    ...     {'BoundaryHandler': cma.BoundPenalty,
    ...      'bounds': [-1, 1],
    ...      'tolflatfitness': 10,
    ...      'fixed_variables': {0: 0.012, 2:0.234}
    ...     })  # doctest: +ELLIPSIS
    (4_w,8)-aCMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=...
    >>> if res[1] >= 13.76: print(res)  # should never happen

    Reference: Hansen et al 2009, A Method for Handling Uncertainty...
    IEEE TEC, with addendum, see
    http://www.lri.fr/~hansen/TEC2009online.pdf

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

        """
        # TODO (old data): CPU(N,lam,iter=20,200,100): 3.3s of 8s for two bounds, 1.8s of 6.5s for one bound
        # remark: np.max([bounds[0], x]) is about 40 times slower than max((bounds[0], x))
        copy = copy_if_changed
        bounds = self.bounds
        if bounds not in (None, [None, None], (None, None)):  # solely for effiency
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
        if x in (None, (), []):
            return x
        if self.bounds in (None, [None, None], (None, None)):
            return 0.0 if np.isscalar(x[0]) else [0.0] * len(x)  # no penalty

        x_is_single_vector = np.isscalar(x[0])
        if x_is_single_vector:
            x = [x]

        # add fixed variables to self.gamma
        try:
            gamma = list(self.gamma)  # fails if self.gamma is a scalar
            for i in sorted(gp.fixed_values):  # fails if fixed_values is None
                gamma.insert(i, 0.0)
            gamma = np.array(gamma, copy=False)
        except TypeError:
            gamma = self.gamma
        pen = []
        for xi in x:
            # CAVE: this does not work with already repaired values!!
            # CPU(N,lam,iter=20,200,100)?: 3s of 10s, np.array(xi): 1s
            # remark: one deep copy can be prevented by xold = xi first
            xpheno = gp.pheno(archive[xi]['geno'])
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
        l = 1 + len(fvals)
        val = fvals[3 * l // 4] - fvals[l // 4]  # exact interquartile range apart interpolation
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

class PopulationEvaluator(object):
    """evaluate and store f- and g-values of a population in attributes F and G.

    If the g-function (`constraints`) has an `insert` method, `x`-values
    are "inserted" first.

    If the `constraints` function has a `true_g` attribute (assigned during
    the call) and ``offset_from_true_g is True``, population constraints
    values are corrected by an offset to guaranty that ``g >= 0`` if
    ``true_g > 0``. Named solutions are not offset.

    """
    def __init__(self, objective, constraints, insert=True, offset_from_true_g=False):
        self.objective = objective
        self.constraints = constraints
        self.offset_from_true_g = offset_from_true_g
        self.insert = insert

    def __call__(self, X, **kwargs):
        """`kwargs` are named solutions resulting in

    ::

            self.name['x'] = kwargs[name]
            self.name['f'] = self.objective(kwargs[name])
            self.name['g'] = self.constraints(kwargs[name])

        Store result in attributes `F` and `G`.
        """
        self.X = X
        self.F = [self.objective(x) for x in X]
        if self.insert:
            try:  # use constraints.insert method if available
                for x in X:
                    self.constraints.insert(x)
                for name, x in kwargs.items():
                    self.constraints.insert(x)
            except (AttributeError, TypeError):
                pass
        try:  # use constraints.true_g attribute if available
            if not hasattr(self.constraints, 'true_g'):
                raise AttributeError  # avoid to repeat one evaluation in next line
            self.G_all = [(self.constraints(x), self.constraints.true_g) for x in X]
        except AttributeError:  # "regular" execution path
            self.G = [self.constraints(x) for x in X]
        else:  # process constraints.true_g attribute values
            self.G = [G[0] for G in self.G_all]
            self.G_true = [G[1] for G in self.G_all]
            # offset g values for which g < 0 < true_g, TODO: avoid the double loop?
            if self.offset_from_true_g:  # process true g-values and offset g-values if necessary
                for j in range(len(self.G[0])):  # for each constraint
                    offset = 0
                    for i in range(len(self.G)):  # compute offset from all candidates
                        if self.G_true[i][j] > 0 and self.G[i][j] + offset < 0:
                            offset = -self.G[i][j]  # use smallest negative value of infeasible solution
                            assert offset >= 0
                    for i in range(len(self.G)):  # add offset on each infeasible candidate
                        if self.G_true[i][j] > 0:
                            self.G[i][j] += offset
                assert np.all([np.all(np.asarray(self.G[i])[np.asarray(self.G_true[i]) > 0] >= 0)
                            for i in range(len(self.G))])
        for name, x in kwargs.items():
            setattr(self, name, {'x': x,
                                 'f': self.objective(x),
                                 'g': self.constraints(x)})
        return self

    @property
    def feasibility_ratios(self):
        """or bias for equality constraints"""
        return np.mean(np.asarray(self.G) <= 0, axis=0)
        # return [np.mean(g <= 0) for g in np.asarray(self.G).T]

class AugmentedLagrangian(object):
    """Augmented Lagrangian with adaptation of the coefficients

    for minimization, implemented after Atamna et al FOGA 2017,
    Algorithm 1 and Sect 8.2, https://hal.inria.fr/hal-01455379/document.

    Input `dimension` is the search space dimension, boolean `equality`
    may be an iterable of length of number of constraints indicating the
    type for each constraint.

    Below, the objective function value is denoted as ``f = f(x)``, the
    constraints values as ``g = g(x) <= 0``, the penalties compute to
    ``penalties(x) = self(g(x)) = lam g + mu g^2 / 2`` (if g > -lam / mu) for
    each element of g, as returned by calling the instance with g as argument.

    The penalized "fitness" value ``f + sum(self(g))`` shall be minimized.
    lam and mu are the Lagrange multipliers or coefficients.

    An additional method, `set_coefficients` allows to initialize the
    Lagrange multipliers from data.

    A short example (and doctest):

    >>> import cma
    >>> from cma.constraints_handler import AugmentedLagrangian, PopulationEvaluator
    >>> m = 2  # number of constraints
    >>> def objective(x):
    ...     return sum(x[m:]**2) + sum((x[:m] - 1)**2) - m
    >>> def constraints(x):
    ...     return x[:m]
    >>> es = cma.CMAEvolutionStrategy(3 * [1], 1, {
    ...          'termination_callback': lambda es: sum(es.mean**2) < 1e-8})  #doctest: +ELLIPSIS
    (3_w,7)-aCMA-ES...
    >>> al = AugmentedLagrangian(es.N)  # lam and mu still need to be set
    >>> # al.chi_domega = 1.15  # is the new default, which seems to give better results than the original value
    >>> # al.lam, al.mu = ...  # we could set the initial Lagrange coefficients here
    >>> while not es.stop():
    ...     eva = PopulationEvaluator(objective, constraints)(es.ask(), m=es.mean)
    ...     al.set_coefficients(eva.F, eva.G)  # set lam and mu, not part of the original algorithm
    ...     al.update(eva.m['f'], eva.m['g'])
    ...     es.tell(eva.X, [f + sum(al(g)) for f, g in zip(eva.F, eva.G)])
    >>> if es.result.evaluations > 3100:
    ...     print("evaluations %d !< 3100. 1500 is normal, 2700 happens rarely" % es.result.evaluations)
    >>> assert 'callback' in es.stop()
    >>> assert len(eva.feasibility_ratios) == m
    >>> assert sum(eva.feasibility_ratios < 0) == sum(eva.feasibility_ratios > 1) == 0

    Details: the input `dimension` is needed to compute the default change
    rate `chi_domega` (if ``chi_domega is None``), to compute initial
    coefficients and to compare between h and g to update mu. The default
    dependency of `chi_domega` on the dimension seems to be however
    suboptimal. Setting ``self.chi_domega = 1.15`` as is the current
    default seems to give better results than the original setting.

    """
    def __init__(self, dimension, equality=False, chi_domega=2**0.2):
        """if ``chi_domega is None``, set to the original (worse) setting ``2**(0.2 / dimension)``"""
        self.dimension = dimension  # maybe not desperately needed
        self.lam, self.mu = None, None  # will become np arrays
        self._initialized = np.array(False)  # only used for setting, not for update
        self._equality = np.array(equality, dtype=bool)
        self.k1 = 3
        self.k2 = 5
        self.dgamma = 5  # damping for lambda change
        if chi_domega:
            self.chi_domega = chi_domega  # 2**0.2 seems to work better than the default
        else:
            self.chi_domega = 2**(1. / 5 / dimension)  # factor for mu change, 5 == domega
        self.f, self.g = 2 * [None]  # store previous values
        self.count = 0  # number of actual updates after any mu > 0 was set

        self.lam_opt = None  # only for display in logger
        self.logging = True
        self._init_()

    def _init_(self):
        """allow to reset the logger with a single call"""
        self.logger = _logger.Logger(self,
            callables=[lambda s: s.lam if s.lam_opt is None else np.log10(np.abs(s.lam - s.lam_opt) + 1e-9),
                       lambda s: np.log10(s.mu + 1e-9)],
            labels=['lambda' if self.lam_opt is None else 'lg(lambda-lam_opt)',
                    'lg(mu)'],
            name='outauglag',
            )
        self.logger_mu_conditions = _logger.Logger("mu_conditions", labels=[
                        r'$\mu$ increases',
                        r'$\mu g^2 < %.0f |\Delta h| / n$' % self.k1,
                        r'$|\Delta g| < |g| / %.0f$' % self.k2])

    @property
    def m(self):
        """number of constraints, raise `TypeError` if not set yet"""
        return len(self.lam)

    @property
    def feasibility_ratios(self):
        """or bias for equality constraints, versatile interface may change"""
        try: return [np.mean(np.asarray(g) <= 0) for g in np.asarray(self.G).T]
        except AttributeError: return None

    def set_m(self, m):
        """initialize attributes which depend on the number of constraints.

        This requires the `lam` attribute to be `None` and deletes all
        previously set or adapted `mu` coefficients.
        """
        assert self.lam is None  # better safe than sorry
        self.lam = np.zeros(m)
        self.mu = np.zeros(m)
        self._initialized = np.zeros(m, dtype=bool)

    def _check_dtypes(self):
        """in case the user set the attributes"""
        for name in ['mu', 'lam']:
            if getattr(self, name).dtype != 'float':
                setattr(self, name, np.array(getattr(self, name), dtype='float'))
        assert self._initialized.dtype == 'bool'

    def set_coefficients(self, F, G):
        """compute initial coefficients based on some f- and g-values.

        The formulas to set the coefficients::

            lam = iqr(f) / (sqrt(n) * iqr(g))
            mu = 2 * iqr(f) / (5 * n * (iqr(g) + iqr(g**2)))

        are taken out of thin air and not thoroughly tested. They are
        additionally protected against division by zero.

        Each row of `G` represents the constraints of one sample measure.

        Set lam and mu until a population contains more than 10% infeasible
        and more than 10% feasible at the same time. Afterwards, this at least...?...
        """
        self.F, self.G = F, G  # versatile storage
        if self.mu is not None and all(self._initialized * (self.mu > 0)):
            return  # we're all set
        G = np.asarray(G).T  # now row G[i] contains all values of constraint i
        sign_average = np.mean(np.sign(G), axis=1)  # caveat: equality vs inequality
        if self.lam is None:  # destroys all coefficients
            self.set_m(len(G))
        elif not self._initialized.shape:
            self.lam_old, self.mu_old = self.lam[:], self.mu[:]  # in case the user didn't mean to
            self._initialized = np.array(self.m * [self._initialized])
        self._check_dtypes()
        idx = _and(_or(_not(self._initialized), self.mu == 0),
                   _or(sign_average > -0.8, self.isequality))
        # print(len(F), len(self.G))
        if np.any(idx):
            df = _Mh.iqr(F)
            dG = np.asarray([_Mh.iqr(g) for g in G])      # only needed for G[idx], but like
            dG2 = np.asarray([_Mh.iqr(g**2) for g in G])  # this simpler in later indexing
            if np.any(df == 0) or np.any(dG == 0) or np.any(dG2 == 0):
                _warnings.warn("iqr(f), iqr(G), iqr(G**2)) == %s, %s, %s" % (str(df), str(dG), str(dG2)))
            assert np.all(df >= 0) and np.all(dG >= 0) and np.all(dG2 >= 0)
            # 1 * dG2 leads to much too small values for Himmelblau
            mu_new = 2. / 5 * df / self.dimension / (dG + 1e-6 * dG2 + 1e-11 * (df + 1))  # TODO: totally out of thin air
            idx_inequ = _and(idx, _not(self.isequality))
            if np.any(idx_inequ):
                self.lam[idx_inequ] = df / (self.dimension * dG[idx_inequ] + 1e-11 * (df + 1))

            # take min or max with existing value depending on the sign average
            # we don't know whether this is necessary
            isclose = np.abs(sign_average) <= 0.2
            idx1 = _and(_and(_or(isclose,
                                 _and(_not(self.isequality), sign_average <= 0.2)),
                             _or(self.mu == 0, self.mu > mu_new)),
                             idx)  # only decrease
            idx2 = _and(_and(_and(_not(isclose),
                                  _or(self.isequality, sign_average > 0.2)),
                                  self.mu < mu_new),
                                  idx)  # only increase
            idx3 = _and(idx, _not(_or(idx1, idx2)))  # others
            assert np.sum(_and(idx1, idx2)) == 0
            if np.any(idx1):  # decrease
                iidx1 = self.mu[idx1] > 0
                if np.any(iidx1):
                    self.lam[idx1][iidx1] *= mu_new[idx1][iidx1] / self.mu[idx1][iidx1]
                self.mu[idx1] = mu_new[idx1]
            if np.any(idx2):  # increase
                self.mu[idx2] = mu_new[idx2]
            if np.any(idx3):
                self.mu[idx3] = mu_new[idx3]
            if 11 < 3:  # simple version, this may be good enough
                self.mu[idx] = mu_new[idx]
            self._initialized[_and(idx, _or(self.count > 2 + self.dimension,  # in case sign average remains 1
                                            np.abs(sign_average) < 0.8))] = True
        elif all(self._initialized) and all(self.mu > 0):
            _warnings.warn(
                "Coefficients are already fully initialized. This can (only?) happen if\n"
                "the coefficients are set before the `_initialized` array.")

    @property
    def isequality(self):
        """bool array, `True` if `i`-th constraint is an equality constraint"""
        try:
            len(self._equality)
        except TypeError:
            self._equality = np.asarray(self.m * [True if self._equality else False])
        return self._equality

    def __call__(self, g):
        """return `list` of AL penalties for constraints values in `g`.

        Penalties are zero in the optimum and can be negative down to
        ``-lam**2 / mu / 2``.
        """
        if self.lam is None:
            return [0.] * len(g)
        assert len(self.lam) == len(self.mu)
        if len(g) != len(self.lam):
            raise ValueError("len(g) = %d != %d = # of Lagrange coefficients"
                             % (len(g), len(self.lam)))
        assert self._equality.dtype == 'bool'
        idx = _and(_not(self._equality), self.mu * g < -1 * self.lam)
        if any(idx):
            g = np.array(g, copy=True)
            g[idx] = -self.lam[idx] / self.mu[idx]
        return [self.lam[i] * g[i] + 0.5 * self.mu[i] * g[i]**2
                for i in range(len(g))]

    def update(self, f, g):
        """f is a scalar, g is a vector.

        Update Lagrange multipliers based on Atamna et al 2017. f and g are
        supposed to have been computed from the distribution mean.

        Details: do nothing if Lagrange coefficients `lam` were not yet set.
        """
        if self.lam is None:
            try: self._count_noupdate += 1
            except AttributeError: self._count_noupdate = 1
            if self._count_noupdate % (1 + self._count_noupdate**0.5) < 1:
                _warnings.warn("no update for %d calls (`lam` and `mu` need "
                               "to be initialized first)" % self._count_noupdate)
            return
        if self.m == 0:
            return
        self._check_dtypes()
        if self.g is not None and np.any(self.mu > 0):  # mu==0 makes a zero update anyway
            assert len(self.lam) == len(self.mu) == len(g)
            if not self.count and self.chi_domega < 1.05:
                _warnings.warn("chi_omega=%f as by default, however values <<1.1 may not work well"
                                % self.chi_domega)
            dg = np.asarray(g) - self.g
            dh = f + sum(self(g)) - self.f - sum(self(self.g))  # using the same coefficients
            for i in range(len(self.lam)):
                if self.logging:
                    condk1 = bool(self.mu[i] * g[i]**2 < self.k1 * np.abs(dh) / self.dimension)
                    condk2 = bool(self.k2 * np.abs(dg[i]) < np.abs(self.g[i]))
                    self.logger_mu_conditions.add(i - 0.1 + 0.25 * np.asarray(
                                [max((condk1, condk2)), 1.25 + condk1, 2.5 + condk2]))
                if self.mu[i] == 0:
                    continue  # for mu==0 all updates are zero anyway
                # address when g[i] remains inactive ("large" negative)
                if not self.isequality[i] and g[i] * self.mu[i] < -self.lam[i]:
                    continue  # not in the original algorithm
                    # prevent that lam gets negative and mu diverges temporarily (as observed)
                # lambda update
                self.lam[i] += self.mu[i] * g[i] / self.dgamma
                if not self.isequality[i] and self.lam[i] < 0:  # clamp to zero
                    # if we stay in the feasible domain, lam would diverge to -inf (as observed)
                    self.lam[i] = 0
                    # and mu would diverge to +inf, because abs(g) >> dg -> 0 (as observed)
                    continue  # don't update mu (it may diverge)
                # mu update
                if self.mu[i] * g[i]**2 < self.k1 * np.abs(dh) / self.dimension or (
                    self.k2 * np.abs(dg[i]) < np.abs(self.g[i])):  # this condition is always true if constraint i is not active
                    self.mu[i] *= self.chi_domega**0.25  # 4 / 1 is the stationary odds ratio of increment / decrement
                else:
                    self.mu[i] /= self.chi_domega
            self.count += 1
        assert np.all((self.lam >= 0) + self.isequality)
        self.f, self.g = f, g  # self(g) == 0 if mu=lam=0
        if self.logging:
            self.logger.push()
            self.logger_mu_conditions.push()

