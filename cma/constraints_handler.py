"""A collection of boundary and (in future) constraints handling classes.
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
# __package__ = 'cma'
import numpy as np
from .utilities.utils import rglen
# from .utilities.math import Mh
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
    ...      'fixed_variables': {0: 0.012, 2:0.234}
    ...     })  # doctest: +ELLIPSIS
    (4_w,8)-aCMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=...
    >>> assert res[1] < 13.76

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
