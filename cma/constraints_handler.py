# -*- coding: utf-8 -*-
"""A collection of (nonlinear) constraints handling classes and helper functions.
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
import warnings as _warnings
import collections as _collections
import functools as _functools
import numpy as np
from numpy import logical_and as _and, logical_or as _or, logical_not as _not
from .utilities import utils
from .utilities.math import Mh as _Mh, moving_average
from .logger import Logger as _Logger  # we can assign _Logger = cma.logger.LoggerDummy to turn off logging
from .optimization_tools import BestSolution2
del absolute_import, division, print_function  #, unicode_literals

_warnings.filterwarnings('once', message="``import moarchiving`` failed.*")

def _g_pos_max(gvals):
    return max(gi if gi > 0 else 0 for gi in gvals)
def _g_pos_sum(gvals):
    return sum(gi for gi in gvals if gi > 0)
def _g_pos_squared_sum(gvals):
    return sum(gi**2 for gi in gvals if gi > 0)
class ConstrainedSolutionsArchive(object):
    """Biobjective Pareto archive to store some Pareto optimal solutions
    for constrained optimization.

    The user can define the aggregator for the constraints values which
    is by default the sum of the positive parts.

    The Pareto archive is maintained in the `archive` attribute and the
    Pareto optimal solutions can be recovered in `archive.infos`.
"""
    def __init__(self, aggregator=_g_pos_sum):
        self.aggregator = aggregator
        self.archive = None
        self.count = 0
        self.maxlen = 10
        try:
            import moarchiving
        except ImportError:
            m = ("``import moarchiving`` failed, hence convergence tracking "
                 "is disabled. \n  'pip install moarchiving' should fix this.")
            _warnings.warn(m)
            return
        self.archive = moarchiving.BiobjectiveNondominatedSortedList()
    def update(self, f, g, info=None):
        self.count += 1
        if self.archive is not None:
            gagg = self.aggregator(g)
            try:
                self.archive.add([f, gagg], info=info)
            except TypeError:
                self.archive.add([f, gagg])  # previous interface
            while len(self.archive) > self.maxlen:
                if self.archive[1][1] > 0:  # keep at least one infeasible solution
                    self.archive.remove(self.archive[0])
                else:
                    self.archive.remove(self.archive[-1])
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

class DequeCDF(_collections.deque):
    """a queue of vectors with element-wise cdf computation.

    The `deque` is here used like a `list` with maximum length functionality,
    the (inherited) constructor takes `maxlen` as keyword argument (since Python 2.6).

    >>> import cma
    >>> d = cma.constraints_handler.DequeCDF(maxlen=22)
    >>> for i in range(5):
    ...     d.append([i])
    >>> ' '.join(['{0:.2}'.format(x) for x in
    ...                   [d.cdf(0, 0), d.cdf(0, 2), d.cdf(0, 2.1), d.cdf(0, 22.1), d.cdf(0, 4, 2)]])
    '0.1 0.5 0.6 1.0 0.75'

    """
    def cdf(self, i, val=0, len_=None):
        """return ecdf(`val`) from the `i`-th element of the last `len_` values in self,

        in other words, the ratio of values in ``[v[i] for v in
        self[-len_:]]`` that are smaller than `val` where equality counts
        1/2.
        """
        j0 = int(min((len_ or len(self), len(self))))
        data = np.asarray([self[j][i] for j in range(-j0, 0)])
        return np.mean((data < val) + 0.5 * (data == val))
    def cdf1(self, val=0, len_=None):
        """return ECDF at `val` in case this is a `deque` of scalars"""
        j0 = int(min((len_ or len(self), len(self))))
        data = np.asarray([self[j] for j in range(-j0, 0)])
        return np.mean((data < val) + 0.5 * (data == val))

class LoggerList(list):
    """list of loggers with plot method"""
    def plot(self, moving_window_width=7):
        """versatile plot method, argument list positions may change"""
        import matplotlib
        from matplotlib import pyplot as plt
        # _, axes = plt.gcf().subplots(2, 2)  # gcf().subplots return array of subplots, plt.subplots returns array of axes
        # axes = list(axes[0]) + list(axes[1])
        for i, logger in enumerate(self):
            # plt.sca(axes[i])
            with _warnings.catch_warnings():  # catch misfiring add_subplot warning
                _warnings.filterwarnings('ignore', message='Adding an axes using the same arguments')
                plt.subplot(2, 2, i + 1)
            logger.plot()
            if len(logger.data.shape) > 1 and logger.data.shape[1] > 1:
                for j, d in enumerate(logger.data.T):
                    if j < len(logger.data.T) - 1:  # variable number annotation
                        plt.text(len(d), d[-1], str(j))
                    if i == 1:  # plot rolling average
                        plt.plot(moving_average(d, min((moving_window_width, len(d)))),
                                 color='r', linewidth=0.15)
                if i == 0:
                    v = np.abs(logger.data)
                    min_val = max((1e-9, np.min(v[v>0])))
                    if utils.version_diff(matplotlib.__version__, '3.3') >= 0:
                        # a terrible interface change that swallows the new/old parameter and breaks code
                        plt.yscale('symlog', linthresh=min_val)  # see matplotlib.scale.SymmetricalLogScale
                    else:  # old kwarg
                        plt.yscale('symlog', linthreshy=min_val)

def _log_g(s):
    return s.g + [0]
def _log_feas_events(s):
    return [i + 0.5 * (gi > 0) - 0.25 + 0.2 * np.tanh(gi) for i, gi in enumerate(s.g)
            ] + [len(s.g) + np.any(np.asarray(s.g) > 0)]
def _log_lam(s):
    """for active constraints, lam is generally positive because Dg and Df are opposed"""
    v = np.log10(np.maximum(1e-9, np.abs(s.lam - (0 if s.lam_opt is None else s.lam_opt))))
    return np.hstack([v, 0])  # add column to get same colors as _log_feas_events
def _log_mu(s):
    v = np.log10(np.maximum(s.mu, 1e-9))
    return np.hstack([v, 0])  # add column to get same colors as _log_feas_events

class CountLastSameChanges(object):
    """An array/list of successive same-sign counts.

    ``.same_changes[i]`` counts how often the same sign was successively
    observed during ``update(..., i, change)``. When the sign flips,
    ``.same_changes[i]`` is reset to 1 or -1.

    `init` needs to be called before the class instance can be used.
    A valid use case is ``lsc = CountLastSameChanges().init(dimension)``.

    TODO: parameters may need to depend on population size?
    """
    def __init__(self, parent=None):
        self.same_changes = None
        self.parent = parent  # get easy access for hacks, not in use
        "  count of the number of changes with the same sign, where"
        "  the count sign signifies the sign"
        self.chi_exponent_threshold = None  # waiting and ramp up time
        self.chi_exponent_factor = None  # steepness of ramp up
        "  increase chi exponent up to given threshold, can make up for dimension dependent chi"
    def init(self, dimension, force=False):
        """do not overwrite user-set values unless `force`"""
        if force or self.same_changes is None:
            self.same_changes = dimension * [0]
        if force or self.chi_exponent_threshold is None:  # threshold for number of consecutive changes
            # self.chi_exponent_threshold = 5 + self.dimension**0.5  # previous value, probably too small in >- 20-D
            # self.chi_exponent_threshold = 5 + self.dimension**0.75  # sweeps in aug-lag4-mu-update.ipynv
            self.chi_exponent_threshold = 2 + self.dimension
        if force or self.chi_exponent_factor is None:  # factor used in shape_exponent
            # self.chi_exponent_factor = 3 / dimension**0.5 # previous value
            # self.chi_exponent_factor = 1 * dimension**0.25 / self.chi_exponent_threshold  # steepness of ramp up
            self.chi_exponent_factor = 4 / self.chi_exponent_threshold  # steepness of ramp up
        return self
    @property
    def dimension(self):
        return len(self.same_changes)
    def update(self, i, change):
        """update ``same_changes[i]`` count based on the sign of `change`"""
        if change * self.same_changes[i] < 0:
            self.same_changes[i] = 0
        self.same_changes[i] += change
        return self
    def shape_exponent(self, i):
        """return value to be added to exponent from change count.

        return a positive/negative value (or zero) when the last change was
        respectively postive/negative.

        The value is zero for the first `chi_exponent_threshold` same changes
        and increases linearly for the next `chi_exponent_threshold` same changes
        and then stays at the threshold value as long as the change does not
        flip sign.
        """
        d = self.same_changes[i]
        s = np.sign(d)
        d -= s * self.chi_exponent_threshold  # wait until increase
        d *= d * s > 0  # sign must not have changed
        return self.chi_exponent_factor * s * min((self.chi_exponent_threshold, s * d))  # bound change

def constraints_info_dict(count, x, f, g, g_al):
    """return dictionary with arg values, mainly there to unify names"""
    return {'x': x,  # caveat: x is not a copy
            'f': f, 'g': g, 'f_al': f + sum(g_al), 'g_al': g_al,
            'count': count}

class AugmentedLagrangian(object):
    """Augmented Lagrangian with adaptation of the coefficients

    for minimization, implemented after Atamna et al FOGA 2017,
    Algorithm 1 and Sect 8.2, https://hal.inria.fr/hal-01455379/document.

    `cma.ConstrainedFitnessAL` provides a (more) user-friendly interface to
    the `AugmentedLagrangian` class.

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

    More testing based on the simpler `ConstrainedFitnessAL` interface:

    >>> import cma
    >>> for algorithm, evals in zip((0, 1, 2, 3), (2000, 2200, 1500, 1800)):
    ...     alf = cma.ConstrainedFitnessAL(cma.ff.sphere, lambda x: [x[0] + 1], 3,
    ...                                    find_feasible_first=True)
    ...     _ = alf.al.set_algorithm(algorithm)
    ...     alf.al.logging = False
    ...     x, es = cma.fmin2(alf, 3 * [1], 0.1, {'verbose':-9, 'seed':algorithm},  # verbosity+seed for doctest only
    ...                       callback=alf.update)
    ...     assert sum((es.mean - [-1, 0, 0])**2) < 1e-9, (algorithm, es.mean)
    ...     assert es.countevals < evals, (algorithm, es.countevals)
    ...     assert alf.best_feas.f < 10, (algorithm, str(alf.best_feas))
    ...     # print(algorithm, es.countevals, ) #alf.best_feas.__dict__)

"""
    def __init__(self, dimension, equality=False):
        """use `set_algorithm()` and ``set_...()`` methods to change defaults"""
        self.algorithm = 0  # 1 = original, < 1 is now default
        self.dimension = dimension  # maybe not desperately needed
        self.lam, self.mu = None, None  # will become np arrays
        self._initialized = np.array(False)  # only used for setting, not for update
        self._equality = np.array(equality, dtype=bool)
        self.k2 = 5
        self.dgamma = 5  # damping for lambda change
        self.mucdf3_horizon = int(5 + self.dimension**1)
        "  number of data to compute cdf for mu adaptation"
        self.f, self.g = 2 * [None]  # store previous values
        self.count = 0  # number of actual updates after any mu > 0 was set
        self.count_g_in_penalized_domain = 0  # will be an array
        "  number of times g induced a penality in __call__ since last update"
        self.count_mu_last_same_changes = CountLastSameChanges()
        "  number of same changes of mu, -3 means it decreased in all last three iterations"
        self.counts = _collections.defaultdict(int)
        self.g_history = DequeCDF(maxlen=self.dimension + 20)
        "  g-values from calling update"
        self.g_all = _collections.defaultdict(_functools.partial(DequeCDF, maxlen=2 * self.dimension + 20))
        "  all g-values from calling the class, recorded but not in use"

        self.count_calls = 0
        self.lam_opt = None  # only for display in logger
        self.logging = 1
        self._set_parameters()
        self._init_()

    def set_atamna2017(self):
        """Atamna et al 2017 parameter settings"""
        self.k1 = 3
        self.chi_domega = 2**(1. / 5 / self.dimension)  # factor for mu change, 5 == domega
        return self
    def set_dufosse2020(self):
        """Dufosse & Hansen 2020 parameter settings"""
        self.k1 = 10
        self.chi_domega = 2**(1. / self.dimension**0.5)
        return self
    def _set_parameters(self):
        """set parameters based on the value of `self.algorithm`"""
        if self.algorithm <= 0:  # default is 3
            self.set_atamna2017()
        elif self.algorithm in (1, 3, 4):
            self.set_atamna2017()
        elif self.algorithm == 2:
            self.set_dufosse2020()
        else:
            raise ValueError("Algorithm id {0} is not recognized".format(self.algorithm))
    def set_algorithm(self, algorithm=None):
        """if algorithm not `None`, set it and return self,

        otherwise return current algorithm value which should be an integer.

        Values < 1 are the (new) default, 1 == Atamna et al 2017, 2 == modified 1.
        """
        if algorithm is None:  # like get_algorithm
            return self.algorithm
        elif algorithm != self.algorithm:
            # modify parameters, if necessary
            self.algorithm = algorithm
            self._set_parameters()
        return self

    def _init_(self):
        """allow to reset the logger with a single call"""
        self.loggers = LoggerList()
        if self.logging > 0:
            self.loggers.append(_Logger(self, callables=[_log_g],
                            labels=['constraint values'],
                            name='outauglagg'))
            self.loggers.append(_Logger(self, callables=[_log_feas_events],
                            labels=['sign(gi) / 2 + i',  'overall feasibility'],
                            name='outauglagfeas'))
            self.loggers.append(_Logger(self, callables=[_log_lam],
                labels=['lg(abs(lambda))' if self.lam_opt is None
                        else 'lg(abs(lambda-lam_opt))'],
                name='outauglaglam'))
            self.loggers.append(_Logger(self, callables=[_log_mu],
                            labels=['lg(mu)'], name='outauglagmu'))
            self.logger_mu_conditions = None
            if self.algorithm in (1, 2):
                self.logger_mu_conditions = _Logger("mu_conditions", labels=[
                            r'$\mu$ increases',
                            r'$\mu g^2 < %.0f |\Delta h| / n$' % self.k1,
                            r'$|\Delta g| < |g| / %.0f$' % self.k2])

    @property
    def m(self):
        """number of constraints, raise `TypeError` if not set yet"""
        return len(self.lam)
    @property
    def is_initialized(self):
        try:
            return all(self._initialized)
        except TypeError:
            return bool(self._initialized)
    @property
    def count_initialized(self):
        """number of constraints with initialized coefficients"""
        return 0 if self.mu is None else sum(self._initialized * (self.mu > 0))
    @property
    def feasibility_ratios(self):
        """or bias for equality constraints, versatile interface may change"""
        try:
            return [np.mean(np.asarray(g) <= 0) for g in np.asarray(self.G).T]
        except AttributeError:
            return None

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

            lam = iqr(f) / (n * iqr(g))
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
        sign_average = -1 + 2 * np.mean(G > 0, axis=1)  # caveat: equality vs inequality
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
        try:
            self.count_g_in_penalized_domain += self.mu * g > -1 * self.lam
        except TypeError:
            pass
        for i in range(len(g)):
            self.g_all[i].append(g[i])
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
        self.count_calls += 1
        return [self.lam[i] * g[i] + 0.5 * self.mu[i] * g[i]**2
                for i in range(len(g))]

    def muplus2(self, g, i, threshold=0.4):
        """provisorial function to correct mu plus condition, original is `True`"""
        # cplus = g[i] > 0 or self.g[i] > 0   # is too strong -> too small mu as learning rate
        n = int(1 + self.dimension**0.5)  # number of data to compute cdf
        cplus = not threshold < self.g_history.cdf(i, len_=n + n % 2) < 1 - threshold
        return cplus and g[i] * self.g[i] > 0
    def muminus2(self, g, i, threshold=0.05):
        """provisorial function to correct mu minus condition, original is `True`"""
        res = threshold < self.g_history.cdf(i) < 1 - threshold
        #if not res:
        #    print('muminus1 %f threshold triggered var %d: %f' % (threshold, i, self.g_history.cdf(i)))
        return res
    def muplus3(self, g, i, threshold=0.95):
        """return `True` if mu should be increased, else `False`.

        TODO: because the stationary condition is g == 0, it seems
        unreasonable to always increase mu when g > 0?
        """
        if g[i] > 0:
            return True
        if g[i] * self.mu[i] > -self.lam[i]:  # in penalized but feasible domain
            p_feas = self.g_history.cdf(i, len_=self.mucdf3_horizon)
            # p_feas = self.g_all[i].cdf1(len_=n)  # TODO: try to understand why this leads to increase of mu on a linear fct with n linear constraints
                                                   #       p_feas seem to be smaller than 0.5, but why?
            if p_feas > threshold:
                """too many feasible solutions were penalized, the constraint may be inactive,
                hence we move the penalization boundary closer to the constraint boundary by
                increasing mu"""
                return True
        return False
    def muminus3(self, g, i, threshold=0.9):
        """return `True` if mu should be reduced, else `False`.

        If mu is large, the augmented function is steeper on the infeasible
        side. Consequently, typically g < 0 and, if the constraint is
        active, some solution will still be infeasible. If this is the
        case, we decrease mu.

        TODO: if more than 50% are infeasible, we may not want to decrease
        mu further? This usually does not happen, because we increase mu
        when g > 0. However, if the constraint is a nonsmooth 1/2^n sector,
        the feasiblity ratio goes inevitably to 1/2^n.
        """
        p_feas = self.g_history.cdf(i, len_=self.mucdf3_horizon)
        # p_feas = self.g_all[i].cdf1(len_=n)  # TODO: try to understand why this leads to increase of mu on a linear fct with n linear constraints
        if p_feas < threshold and 0 > g[i]:  # * self.mu[i] > -1 * self.lam[i]:
            """g is feasible but penalized and >10% are infeasible"""
            return True
        return False
        # return g[i] < 0  # asymmetric condition is less desirable

    def muplus4(self, g, i, threshold=0.95):
        """return `True` if mu should be increased, else `False`"""
        if self.g_all[i].cdf1(len_=2*self.dimension) == 0:  # all samples were infeasible
            return True
        if g[i] * self.mu[i] > -self.lam[i]:  # in penalized but feasible domain
            p_feas = self.g_history.cdf(i, len_=self.mucdf3_horizon)
            if p_feas > threshold:
                """too many feasible solutions were penalized, the constraint may be inactive,
                hence we move the penalization boundary closer to the constraint boundary by
                increasing mu"""
                return True
        return False
    def muminus4(self, g, i, threshold=0.9):
        """return `True` if mu should be reduced, else `False`"""
        if g[i] * self.mu[i] < -1 * self.lam[i] and (
                self.g_all[i].cdf1(len_=2*self.dimension) < 1):
            """g is feasible but penalized and >10% are infeasible"""
            return True
        return False
        # return g[i] < 0  # asymmetric condition is less desirable

    def update(self, f, g):
        """f is a scalar, g is a vector.

        Update Lagrange multipliers based on Atamna et al 2017. f and g are
        supposed to have been computed from the distribution mean.

        Details: do nothing if Lagrange coefficients `lam` were not yet set.
        """
        self.g_history.append(g)  # is a deque with maxlen
        if self.lam is None:
            try:
                self._count_noupdate += 1
            except AttributeError:
                self._count_noupdate = 1
            if self._count_noupdate % (1 + self._count_noupdate**0.5) < 1:
                _warnings.warn("no update for %d calls (`lam` and `mu` need "
                               "to be initialized first)" % self._count_noupdate)
            return
        if self.m == 0:
            return
        self._check_dtypes()
        if self.g is not None and np.any(self.mu > 0):  # mu==0 makes a zero update anyway
            assert len(self.lam) == len(self.mu) == len(g)
            if 11 < 3 and not self.count and self.chi_domega < 1.05:
                _warnings.warn("chi_omega=%f as by default, however values <<1.1 may not work well"
                                % self.chi_domega)
            dg = np.asarray(g) - self.g
            dh = f + sum(self(g)) - self.f - sum(self(self.g))  # using the same coefficients
            self.count_mu_last_same_changes.init(len(self.mu))
            for i in range(len(self.lam)):
                if self.logging > 0 and not (self.count + 1) % self.logging and self.logger_mu_conditions:
                    condk1 = bool(self.mu[i] * g[i]**2 < self.k1 * np.abs(dh) / self.dimension)
                    condk2 = bool(self.k2 * np.abs(dg[i]) < np.abs(self.g[i]))
                    self.logger_mu_conditions.add(i - 0.1 + 0.25 * np.asarray(
                                [max((condk1, condk2)), 1.25 + condk1, 2.5 + condk2]))
                if self.mu[i] == 0:
                    continue  # for mu==0 all updates are zero anyway
                # lambda update unless constraint is entirely inactive
                if self.isequality[i] or g[i] * self.mu[i] > -self.lam[i] or (
                    not utils.is_(self.count_g_in_penalized_domain) or self.count_g_in_penalized_domain[i] > 0):
                    # in the original algorithm this update is unconditional
                    self.lam[i] += self.mu[i] * g[i] / self.dgamma
                if not self.isequality[i] and self.lam[i] < 0:  # clamp to zero
                    # if we stay in the feasible domain, lam would diverge to -inf (as observed)
                    self.lam[i] = 0
                    # and mu would diverge to +inf, because abs(g) >> dg -> 0 (as observed)
                    continue  # don't update mu (it may diverge)
                # address when g[i] remains inactive ("large" negative)
                if not self.isequality[i] and g[i] * self.mu[i] < -self.lam[i]:
                    continue  # not in the original algorithm
                    # prevent that mu diverges temporarily (as observed)
                # mu update
                if self.algorithm == 1:
                    if self.mu[i] * g[i]**2 < self.k1 * np.abs(dh) / self.dimension or (
                        self.k2 * np.abs(dg[i]) < np.abs(self.g[i])):  # this condition is always true if constraint i is not active
                        self.mu[i] *= self.chi_domega**0.25  # 4 / 1 is the stationary odds ratio of increment / decrement
                    else:
                        self.mu[i] /= self.chi_domega
                elif self.algorithm == 2:
                    if self.mu[i] * g[i]**2 < self.k1 * np.abs(dh) / self.dimension or (
                        self.k2 * np.abs(dg[i]) < np.abs(self.g[i])):  # this condition is always true if constraint i is not active
                        self.counts['%d:muup0' % i] += 1
                        if self.muplus2(g, i):
                            self.mu[i] *= self.chi_domega**0.25  # 4 / 1 is the stationary odds ratio of increment / decrement
                            self.counts['%d:muup1' % i] += 1
                    else:
                        self.counts['%d:mudown0' % i] += 1  # only for the record
                        if self.muminus2(g, i):
                            self.counts['%d:mudown1' % i] += 1
                            self.mu[i] /= self.chi_domega
                elif self.algorithm in (0, 3):
                    if self.muplus3(g, i):
                        self.mu[i] *= self.chi_domega**(0.25 * (1 +
                            self.count_mu_last_same_changes.update(i, 1).shape_exponent(i)))
                    elif self.muminus3(g, i):
                        self.mu[i] *= self.chi_domega**(-0.25 * (1 -
                            self.count_mu_last_same_changes.update(i, -1).shape_exponent(i)))
                elif self.algorithm == 4:
                    if self.muplus4(g, i):
                        self.mu[i] *= self.chi_domega**(0.25 * (1 +
                            self.count_mu_last_same_changes.update(i, 1).shape_exponent(i)))
                    elif self.muminus4(g, i):
                        self.mu[i] *= self.chi_domega**(-0.25 * (1 -
                            self.count_mu_last_same_changes.update(i, -1).shape_exponent(i)))
                else:
                    raise NotImplementedError("algorithm number {0} is not known"
                                              .format(self.algorithm))
            self.count += 1
        assert np.all((self.lam >= 0) + self.isequality)
        self.f, self.g = f, g  # self(g) == 0 if mu=lam=0
        self.count_g_in_penalized_domain *= 0
        if self.logging > 0 and not self.count % self.logging:
            for logger in self.loggers:
                logger.push()
            self.logger_mu_conditions and self.logger_mu_conditions.push()

def _get_favorite_solution(es):
    "avoid unpicklable lambda construct"
    return es.ask(1, sigma_fac=0)[0]
class ConstrainedFitnessAL(object):
    """Construct an unconstrained objective function from constraints.

    This class constructs an unconstrained "fitness" function (to be
    minimized) from an objective function and an inequality constraints
    function (which returns a list of constraint values). An
    equality constraint ``h(x) == 0`` must be expressed as two inequality
    constraints like ``[h(x) - eps, -h(x) - eps]`` with ``eps >= 0``.
    Non-positive values <= 0 are considered feasible.

    The `update` method of the class instance needs to be called after each
    iteration. Depending on the setting of `which`, `update` may call
    ``get_solution(es)`` which shall return the solution to be used for the
    constraints handling update, by default ``_get_favorite_solution ==
    lambda es: es.ask(1, sigma_fac=0)[0]``. The additional evaluation of
    objective and constraints is avoided by the default ``which='best'``,
    using the best solution in the current iteration.

    `find_feasible_first` optimizes to get a feasible solution first, which
    only works if no equality constraints are implemented. For this reason
    the default is `False`.

    Minimal example (verbosity set for doctesting):

    >>> import cma
    >>> def constraints(x):  # define the constraint
    ...     return [x[0] + 1, x[1]]  # shall be <= 0
    >>> cfun = cma.ConstrainedFitnessAL(cma.ff.sphere, constraints,
    ...                                 find_feasible_first=True)
    >>> es = cma.CMAEvolutionStrategy(3 * [1.1], 0.1,
    ...                   {'tolstagnation': 0, 'verbose':-9})  # verbosity for doctest only
    >>> es = es.optimize(cfun, callback=cfun.update)
    >>> x = es.result.xfavorite

    The best `x` return value of `cma.fmin2` may not be useful, because
    the underlying function changes over time. Therefore, we use
    `es.result.xfavorite`, which is still not guarantied to be a feasible
    solution. Alternatively, `cfun.best_feas.x` contains the best evaluated
    feasible solution. However, this is not necessarily expected to be a
    good solution, see below.

    >>> assert sum((x - [-1, 0, 0])**2) < 1e-9, x
    >>> assert es.countevals < 2200, es.countevals
    >>> assert cfun.best_feas.f < 10, str(cfun.best_feas)
    >>> # print(es.countevals, cfun.best_feas.__dict__)

    To find a final feasible solution (close to `es.result.xfavorite`) we
    can use the current `CMAEvolutionStrategy` instance `es`:

    >>> x = cfun.find_feasible(es)  # uses es.optimize to find (another) feasible solution
    >>> assert constraints(x)[0] <= 0, (x, cfun.best_feas.x)
    >>> assert cfun.best_feas.f < 1 + 2e-6, str(cfun.best_feas)
    >>> assert len(cfun.archives) == 3
    >>> assert cma.ConstrainedFitnessAL(cma.ff.sphere, constraints, archives=False).archives == []

    Details: The fitness, to be minimized, is changing over time such that
    the overall minimal value does not indicate the best solution.

    The construction is based on the `AugmentedLagrangian` class. If, as by
    default, ``self.finding_feasible is False``, the fitness equals ``f(x)
    + sum_i (lam_i x g_i + mu_i x g_i / 2)`` where ``g_i = max(g_i(x),
    -lam_i / mu_i)`` and lam_i and mu_i are generally positive and
    dynamically adapted coefficients. Only lam_i can change the position of
    the optimum in the feasible domain (and hence must converge to the
    right value).

    When ``self.finding_feasible is True``, the fitness equals to ``sum_i
    (g_i > 0) x g_i^2`` and omits ``f + sum_i lam_i g_i`` altogether.
    Whenever a feasible solution is found, the `finding_feasible` flag is
    reset to `False`.

    `find_feasible(es)` sets ``finding_feasible = True`` and uses
    `es.optimize` to optimize `self.__call__`. This works well with
    `CMAEvolutionStrategy` but may easily fail with solvers that do not
    consistently pass over the optimum in search space but approach the
    optimum from one side only. This is not advisable if the feasible
    domain has zero volume, e.g. when `g` models an equality like
    ``g = lambda x: [h(x), -h(x)]``.

    An equality constraint, h(x) = 0, cannot be handled like h**2 <= 0,
    because the Augmented Lagrangian requires the derivative at h == 0 to
    be non-zero. Using abs(h) <= 0 leads to divergence of coefficient mu
    and the condition number. The best way is apparently using the two
    inequality constraints [h <= 0, -h <= 0], which seems to work perfectly
    well. The underlying `AugmentedLagrangian` class also accepts equality
    constraints.

"""
    archive_aggregators = (_g_pos_max, _g_pos_sum, _g_pos_squared_sum)
    def __init__(self, fun, constraints,
                 dimension=None,
                 which='best',
                 find_feasible_first=False,
                 get_solution=_get_favorite_solution,
                 logging=None,
                 archives=archive_aggregators,
                 ):
        """constructor with lazy initialization.

        If ``which in ['mean', 'solution']``, `get_solution` is called
        (with the argument passed to the `update` method) to determine the
        solution used to update the AL coefficients.

        If `find_feasible_first`, only the constraints are optimized until
        the first (fully) feasible solution is found.

        `logging` is the iteration gap for logging constraints related
        data, in `AugmentedLagrangian`. 0 means no logging and negative
        values have unspecified behavior.

        `archives` are the aggregator functions for constraints for
        non-dominated biobjective archives. By default, the second
        objective is ``max(g_+)``, ``sum(g_+)`` or ``sum(g_+^2)``,
        respectively. ``archives=True`` invokes the same behavior.
        ``archives=False`` or an empty `tupel` prevents maintaining
        archives.

        """
        self.fun = fun
        self.constraints = constraints
        self.dimension = dimension
        self.get_solution = get_solution
        self.which = which
        self.finding_feasible = find_feasible_first
        self.find_feasible_aggregator = _g_pos_squared_sum  # sum(max(0, g)^2) = 2 x (g_al - lam x g) / mu if g > 0
        self.logging = logging
        self.omit_f_calls_when_possible = True
        self._reset()  # assign dynamic variables
        try:  # treat archives as list of constraints aggregation functions
            self.archives = [ConstrainedSolutionsArchive(fun) for fun in archives]
        except TypeError:  # treat archives as flag
            if archives:
                self.archives = [ConstrainedSolutionsArchive(fun) for fun in
                                 ConstrainedFitnessAL.archive_aggregators]
            else:
                self.archives = []
    def _reset(self):
        self._al = None
        self.F = []
        self.G = []
        self.F_plus_sum_al_G = []
        self.foffset = 0  # not in use yet
        self.best_aug  = BestSolution2()
        self.best_feas = BestSolution2()
        self.best_f_plus_gpos = BestSolution2()
        self.count_calls = 0
        self.count_updates = 0
        self._set_coefficient_counts = []
    def reset(self):
        """reset dynamic components"""
        self._reset()
        self._reset_archives()
    def _reset_archives(self):
        for i, a in enumerate(self.archives):
            self.archives[i] = ConstrainedSolutionsArchive(a.aggregator)
    def _reset_arrays(self):
        self.F, self.G, self.F_plus_sum_al_G = [], [], []

    def _is_feasible(self, gvals=None):
        """return True if last evaluated solution (or `gvals`) was feasible"""
        if gvals is None:
            gvals = self.G[-1]
        return all(gi <= 0 for gi in gvals)  # same as _g_pos_sum(gvals) == 0

    @property
    def al(self):
        """`AugmentedLagrangian` class instance"""
        if not self._al and self.dimension:
            self._al = AugmentedLagrangian(self.dimension)
            if self.logging is not None:
                self._al.logging = self.logging
        return self._al

    def set_mu_lam(self, mu, lam):
        """set AL coefficients"""
        self.al.set_mu_lam(mu, lam)

    def initialize(self, dimension):
        """set search space dimension explicitely"""
        self.dimension = dimension

    def __call__(self, x):
        """return AL fitness, append f and g values to `self.F` and `self.G`.

        If `self.finding_feasible`, `fun(x)` is not called and ``f = np.nan``.
        """
        if not self.dimension:
            self.initialize(len(x))
        self.count_calls += 1
        self.G += [self.constraints(x)]
        if self._is_feasible(self.G[-1]):
            self.finding_feasible = False  # found
        elif self.finding_feasible:
            # the boundary of sum(g) can still be a sharp ridge, even when one side is a plateau
            self.F += [np.nan]  # the aggregated F can be easily re-computed from G
            self.F_plus_sum_al_G += [np.nan]
            return self.find_feasible_aggregator(self.G[-1])
        self.F += [self.fun(x)]
        g_al = self.al(self.G[-1])
        self.F_plus_sum_al_G += [self.F[-1] + sum(g_al)]
        self._update_best(x, self.F[-1], self.G[-1], g_al)
        return self.F_plus_sum_al_G[-1] - self.foffset

    def find_feasible(self, es, termination=('maxiter', 'maxfevals'), aggregator=None):  # es: OOOptimizer, find_feasible -> solution
        """find feasible solution by calling ``es.optimize(self)``.

        Return best ever feasible solution `self.best_feas.x`.
        See also `self.best_feas.info`.

        `aggregator`, defaulting to `self.find_feasible_aggregator`, is the
        constraints aggregation function used as objective function to be
        minimized. `aggregator` takes as input all constraint values and
        returns a value <= 0 if and only if the solution is feasible.

        Terminate when either (another) feasible solution was found or any
        of the `termination` keys is matched in `es.stop()`.
    """
        # we could compare self.best_feas.count with self.count_call
        # but can't really know whether count_call was done in the last iteration
        self.finding_feasible = True
        self(es.result.xfavorite)  # register solution, check feasibility and update best
        if self.finding_feasible:  # was set back to False when xfavorite was feasible
            if aggregator:  # set objective
                self.find_feasible_aggregator, aggregator_ = aggregator, self.find_feasible_aggregator
            while self.finding_feasible and not any(any(d == m for m in termination)
                                                    for d in es.stop()):
                X = es.ask()
                es.tell(X, [self(x) for x in X])
                self.update(es)  # calls fun if finding_feasible finished and method is not 'best'
                es.logger.add()  # relies on calling log_in_es during update
                if self.finding_feasible:
                    self._reset_arrays()
            if aggregator:  # reset `find_feasible_aggregator` to original value
                self.find_feasible_aggregator = aggregator_
        # warn when no feasible solution was found
        if self.best_feas.x is None or self.finding_feasible:
            _warnings.warn("ConstrainedFitnessAL.find_feasible: "
                           " No {0}feasible solution found, stop() == {1}".format(
                               "new " if self.best_feas.x is not None else "", es.stop()))
        assert self.best_feas.x is not None or self.finding_feasible, (self.best_feas, self.finding_feasible)
        return self.best_feas.x

    @property
    def _best_fg(self):
        """return current best f, g, where best is determined by the Augmented Lagrangian"""
        i = np.nanargmin(self.F_plus_sum_al_G)  # raises ValueError when all values are nan
        return self.F[i], self.G[i]

    def _fg_values(self, es):
        """f, g values used to update the Augmented Lagrangian coefficients"""
        if self.which == 'mean' or self.which == 'solution':
            self(self.get_solution(es))  # also update best and in case reset finding_feasible
            return self.F[-1], self.G[-1]
        else:
            return self._best_fg

    def _update_best(self, x, f, g, g_al=None):
        """keep track of best solution and best feasible solution"""
        if g_al is None:
            g_al = self.al(g)
        d = constraints_info_dict(self.count_calls, x, f, g, g_al)
        self.best_aug.update(d['f_al'], x, d)
        self.best_f_plus_gpos.update(f + sum([gi for gi in g if gi > 0]), x, d)
        if self._is_feasible(g):
            self.best_feas.update(f, x, d)
        if np.isfinite(f):
            for a in self.archives:
                a.update(f, g, d)

    def update(self, es):
        """update AL coefficients, may be used as callback to `OOOptimizer.optimize`.

        TODO: decide what happens when `__call__` was never called:
              ignore (as for now) or update based on xfavorite by calling self(xfavorite),
              assuming that update was called on purpose?
              When method is not best, it should work without even call self(xfavorite).
        """
        if not len(self.F) == len(self.G) == len(self.F_plus_sum_al_G):
            _warnings.warn("len(F, G, F_plus_sum_al_G) = ({0}, {1}, {2}) differ."
                           "This is probably a bug!".format(
                           len(self.F), len(self.G), len(self.F_plus_sum_al_G)))
        if len(self.G) == 0:  # TODO: we could first self(es.result.xfavorite) and should be fine
            # Caveat: here we rely on the fact that log_in_es aggregates [np.nan] smoothly
            self.log_in_es(es, np.nan, [np.nan])  # TODO: we could use self.best_feas
            return
        elif self.finding_feasible:
            # TODO: using the last values is a hack, using
            #       the argmin(finding_feasible_aggregator(g) for g in self.G) may be better
            #       or we could use self.best_feas
            self.log_in_es(es, self.F[-1], self.G[-1])
            # CAVEAT: we have not reset the list-arrays
            return
        self.count_updates += 1
        if not self.dimension:
            self.initialize(len(self.solution(es)))
        if not self.al.is_initialized and np.isfinite(self.F).all():
            s = self.al.count_initialized
            self.al.set_coefficients(self.F, self.G)
            if self.al.count_initialized > s:
                self._set_coefficient_counts += [self.count_updates]
        f, g = self._fg_values(es)
        self._reset_arrays()  # arrays are used only in _fg_values
        if np.isfinite(f):
            self.al.update(f, g)
        self.log_in_es(es, f, g)
        if 11 < 3 and self.best_feas and np.isfinite(self.best_feas.f):
            # Using the best feasible f-value as offset leads to consistently
            # negative and increasing values that converge to zero.
            # Using the worst infeasible that is better than the best feasible
            # f-value leads to positive values but the graph stepps up and down
            # looks too difficult to interpret for my taste but it does converge.
            new_offset = self.best_feas.f
            if self.archives:
                # find the largest f-value that is smaller than the best feasible
                fvals = []
                for a in self.archives:
                    for i in range(len(a.archive) - 1, -1, -1):
                        if a.archive[i][1] <= 0:  # solution is feasible
                            continue  # this is expected to be the best seen feasible solution
                        fval = a.archive[i][0]  # fval is decreasing in the loop
                        if fval < self.best_feas.f:  # should always be the case
                            fvals += [fval]
                            break
                new_offset = max(fvals) if fvals else self.foffset
            if new_offset != self.foffset:
                print(self.count_updates, self.foffset)
            self.foffset = new_offset

    def log_in_es(self, es, f, g):
        """a hack to have something in the cma-logger divers plot.

        Append the sum of positive g-values and the number of infeasible
        constraints, displayed like 10**(number/10) (mapping [0, 10] to [1,
        10]) if number < 10, to `es.more_to_write`.
        """
        g_pos = _g_pos_sum(g)
        n_infeas = sum(gi > 0 for gi in g)
        # al_pen = sum(self.al(g))  # Lagrange penalization, equals zero initially, may also be negative
        try:
            es.more_to_write += [g_pos if g_pos else np.nan,  # nan avoids zero in log plot
                                 10**(n_infeas / 10) if n_infeas < 10 else n_infeas,  # symlog-like
                                 ]
        except Exception:
            pass
