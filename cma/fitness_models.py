# -*- coding: utf-8 -*-
"""Fitness surrogate model classes and handler for incremental evaluations.
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals, with_statement
del absolute_import, division, print_function  #, unicode_literals, with_statement
___author__ = "Nikolaus Hansen"
__license__ = "BSD 3-clause"

import os
import warnings
from collections import defaultdict  # since Python 2.5
import numpy as np
try:  # TODO: remove dependency
    from scipy.stats import kendalltau as _kendalltau
except ImportError:
    def _kendalltau(x, y):
        return _kendall_tau(x, y), None
from .utilities import utils
from .logger import LoggerDummy as Logger
from .utilities.utils import DefaultSettings as DefaultSettings


def _kendall_tau(x, y):
    """return Kendall tau rank correlation coefficient.

    Implemented only to potentially remove dependency on `scipy.stats`.

    This

    >>> import numpy as np
    >>> from cma.fitness_models import _kendall_tau
    >>> kendalltau = lambda x, y: (_kendall_tau(x, y), 0)
    >>> # from scipy.stats import kendalltau  # incomment if not available
    >>> for dim in np.random.randint(3, 22, 5):
    ...     x, y = np.random.randn(dim), np.random.randn(dim)
    ...     t1, t2 = _kendall_tau(x, y), kendalltau(x, y)[0]
    ...     # print(t1, t2)
    ...     assert np.isclose(t1, t2)

    """
    equal_value_contribution = 1 / 2  # value used in case of agreeing equality

    assert len(x) == len(y)
    x, y = np.asarray(x), np.asarray(y)
    s = 0  # sum of products of two signs (mostly in -1/1)
    for i in range(len(x)):
        if 1 < 3:  # 20 times faster with len(x)=200
            dx = np.sign(x[i] - x[:i])
            dy = np.sign(y[i] - y[:i])
            s += sum(dx * dy)
            if equal_value_contribution:
                s += equal_value_contribution * sum((dx == 0) * (dy == 0))
        else:
            for j in range(i):
                s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
                if equal_value_contribution:
                    s += equal_value_contribution * (x[i] == x[j]) * (y[i] == y[j])
    tau = s * 2. / (len(x) * (len(x) - 1))
    if 11 < 3:  # TODO: testing should be commented out some time
        from scipy.stats import kendalltau
        t = kendalltau(x, y)[0]
        if np.isfinite(t):  # kendalltau([3,3,3], [3,3,3]) is nan
            if t < tau - 1 / len(x) or t > tau + 1 / len(x):
                warnings.warn('tau=%f not close to stats.tau=%f' % (tau, t))
    return tau

def kendall_tau(x, y):
    """return rank correlation coefficient between data `x` and `y`
    """
    if 11 < 3:  # TODO: make default
        tau = _kendall_tau(x, y)
    else:
        try:
            tau = _kendalltau(x, y)[0]
        except TypeError:  # kendalltau([3,3,3], [3,3,3]) == 1
            tau = 0
        if not np.isfinite(tau):  # kendalltau([3,3,3], [3,3,3]) is nan
            tau = 0
    return tau

class SurrogatePopulationSettings(DefaultSettings):
    minimum_model_size = 3  # absolute minimum number of true evaluations before to build the model
    n_for_tau = lambda popsi, nevaluated: int(max((15, min((1.2 * nevaluated, 0.75 * popsi)))))
    model_max_size_factor = 1  # times popsize, was: 3
    tau_truth_threshold = 0.85  # tau between model and ground truth
    tau_truth_threshold_correction = 0  # loosen threshold for increasing evaluations
    min_evals_percent = 2  # eval int(1 + min_evals_percent / 100) unconditionally
    model_sort_globally = False
    return_true_fitnesses = True  # return true fitness if all solutions are evaluated
    # change_threshold = -1.0     # not in use tau between previous and new model; was: 0.8
    # crazy_sloppy = 0  # number of loops only done on the model, should depend on tau.tau?

class SurrogatePopulation(object):
    """surrogate f-values for a population.

    See also `__call__` method.

    What is new:

    - the model is built with >=3 evaluations (compared to LS-CMA and lmm-CMA)
    - the model is linear at first, then diagonal, then full
    - the model uses the fitness ranks as weights for weighted regression.

    >>> import cma
    >>> import cma.fitness_models as fm
    >>> from cma.fitness_transformations import Function as FFun  # adds evaluations attribute
    >>> # fm.Logger, Logger = fm.LoggerDummy, fm.Logger
    >>> surrogate = fm.SurrogatePopulation(cma.ff.elli)

    Example using the ask-and-tell interface:

    >>> for fitfun in [FFun(cma.ff.elli), FFun(cma.ff.sectorsphere)]:
    ...     es = cma.CMAEvolutionStrategy(5 * [1], 2.2,
    ...                    {'CMA_injections_threshold_keep_len': 1,
    ...                     'ftarget':1e-9, 'verbose': -9, 'seed':3})
    ...     surrogate = fm.SurrogatePopulation(fitfun)
    ...     while not es.stop():
    ...         X = es.ask()
    ...         es.tell(X, surrogate(X))  # surrogate evaluation
    ...         es.inject([surrogate.model.xopt])
    ...         # es.disp(); es.logger.add()  # ineffective with verbose=-9
    ...     print(fitfun.evaluations)  # was: (sig=2.2) 12 161, 18 131, 18 150, 18 82, 15 59, 15 87, 15 132, 18 83, 18 55
    ...     assert 'ftarget' in es.stop()
    18
    55

    Example using the ``parallel_objective`` interface to `cma.fmin`:

    >>> for fitfun, evals in [[FFun(cma.ff.elli), 22], [FFun(cma.ff.ellirot), 40]]:
    ...     surrogate = fm.SurrogatePopulation(fitfun)
    ...     inject_xopt = fm.ModelInjectionCallback(surrogate.model)  # must use the same model
    ...     xopt, es = cma.fmin2(None, 5 * [1], 2.2,
    ...                      {'CMA_injections_threshold_keep_len': 1,
    ...                       'ftarget':1e-12, 'verbose': -9},
    ...                      parallel_objective=surrogate,
    ...                      callback=inject_xopt)
    ...     # print(fitfun.evaluations)
    ...     assert fitfun.evaluations == es.result.evaluations
    ...     assert es.result[1] < 1e-12
    ...     assert es.result[2] < evals
    >>> # fm.Logger = Logger

    """
    def __init__(self,
                 fitness,
                 model=None,
                 model_max_size_factor=None,
                 # model_min_size_factor=None,
                 tau_truth_threshold=None,
                 ):
        """

        If ``model is None``, a default `LQModel` instance is used. By
        setting `self.model` to `None`, only the `fitness` for each
        population member is evaluated in each call.

        """
        self.fitness = fitness
        self.model = model if model else LQModel()
        # set 2 parameters of settings from locals() which are not attributes of self
        self.settings = SurrogatePopulationSettings(locals(), 2, self)
        self.count = 0
        self.evaluations = 0
        self.logger = Logger(self, labels=['tau0', 'tau1', 'evaluated ratio'])
        self.logger_eigenvalues = Logger(self.model, ['eigenvalues'])

    class EvaluationManager:
        """Manage incremental evaluation of a population of solutions.

        Evaluate solutions, add them to the model and keep track of which
        solutions were evaluated.

        Uses `model.add_data_row` and `model.eval`.

        Details: for simplicity and avoiding the copy construction, we do not
        inherit from `list`. Hence we use ``self.X`` instead of ``self``.

        """
        def __init__(self, X):
            """all is based on the population (list of solutions) `X`"""
            self.X = X
            self.evaluated = len(X) * [False]
            self.fvalues = len(X) * [np.nan]
        def add_eval(self, i, fval):
            """add fitness(self.X[i]), not in use"""
            self.fvalues[i] = fval
            self.evaluated[i] = True
        def eval(self, i, fitness, model_add):
            """add fitness(self.X[i]) to model data, mainly for internal use"""
            if self.evaluated[i]:  # need to decide what to do in this case
                raise ValueError("i=%d, evals=%d, len=%d" % (i, self.evaluations, len(self)))
            self.fvalues[i] = fitness(self.X[i])
            self.evaluated[i] = True
            model_add(self.X[i], self.fvalues[i])
        def eval_sequence(self, number, fitness, model_add, idx=None):
            """evaluate unevaluated entries of X[idx] until `number` entries are
            evaluated *overall*.

            Assumes that ``sorted(idx) == list(range(len(self.X)))``.

            ``idx`` defines the evaluation sequence.

            The post condition is ``self.evaluations == min(number, len(self.X))``.
            """
            if idx is None:
                idx = range(len(self.X))
            assert len(self.evaluated) == len(self.X)
            if not self.evaluations < number:
                warnings.warn("Expected evaluations=%d < number=%d, popsize=%d"
                              % (self.evaluations, number, len(self.X)))
            self.last_evaluations = number - self.evaluations  # used in surrogate loop
            for i in idx:
                if self.evaluations >= number:
                    break
                if not self.evaluated[i]:
                    self.eval(i, fitness, model_add)
            else:
                if self.evaluations < number and self.evaluations < len(self.X):
                    warnings.warn("After eval: evaluations=%d < number=%d, popsize=%d"
                                  % (self.evaluations, number, len(self.X)))
                return
            assert self.evaluations == number or self.evaluations == len(self.X) < number
        def surrogate_values(self, model_eval, true_values_if_all_available=True,
                             f_offset=None):
            """return surrogate values of `model_eval` with smart offset.
            """
            if true_values_if_all_available and self.evaluations == len(self.X):
                return self.fvalues
            F_model = [model_eval(x) for x in self.X]
            if f_offset is None:
                f_offset = np.nanmin(self.fvalues)  # must be added last to prevent numerical erasion
            if np.isfinite(f_offset):
                m_offset = np.nanmin(F_model)  # must be subtracted first to get close to zero
                return [f - m_offset + f_offset for f in F_model]
            else:
                return F_model
        def __len__(self):
            """should replace ``len(self.X)`` etc, not fully in use yet"""
            return len(self.fvalues)
        @property
        def evaluation_fraction(self):
            return self.evaluations / len(self.fvalues)
        @property
        def evaluations(self):
            return sum(self.evaluated)
        @property
        def remaining(self):
            """number of not yet evaluated solutions"""
            return len(self.X) - sum(self.evaluated)

    def __call__(self, X):
        """return population f-values.

        Also update the underlying model. Evaluate at least one solution on the
        true fitness. The smallest returned value is never smaller than the
        smallest truly evaluated value.

        Uses (this may not be a complete list):
        `model.settings.max_absolute_size`, `model.settings.truncation_ratio`,
        `model.size`, `model.sort`, `model.eval`, `model.reset`, `model.add_data_row`,
        `model.kendall`, `model.adapt_max_relative_size`, relies on default value
        zero for ``max_absolute_size``.

        """
        self.count += 1
        model = self.model  # convenience shortcut
        # a trick to see whether the population size has increased (from a restart)
        # model.max_absolute_size is by default initialized with zero
        # TODO: remove dependency on the initial value of model.max_absolute_size
        if self.settings.model_max_size_factor * len(X) > model.settings.max_absolute_size:
            if model.settings.max_absolute_size:  # do not reset in first call, in case model was initialized meaningfully
                model.reset()  # reset, because the population size changed
            model.settings.max_absolute_size = self.settings.model_max_size_factor * len(X)
        evals = SurrogatePopulation.EvaluationManager(X)
        self.evals = evals  # only for the record

        if 11 < 3:
            # make minimum_model_size unconditional evals in the first call and quit
            if model.size < self.settings.minimum_model_size:
                evals.eval_sequence(self.settings.minimum_model_size - model.size,
                                    self.fitness, model.add_data_row)
                self.evaluations += evals.evaluations
                model.sort(self.settings.model_sort_globally or evals.evaluations)
                return evals.surrogate_values(model.eval, self.settings.return_true_fitnesses)

        if 11 < 3 and self.count % (1 + self.settings.crazy_sloppy):
            return evals.surrogate_values(model.eval, f_offset=model.F[0])

        number_evaluated = int(1 + max((len(X) * self.settings.min_evals_percent / 100,
                                        3 / model.settings.truncation_ratio - model.size)))
        while evals.remaining:
            idx = np.argsort([model.eval(x) for x in X]) if model.size > 1 else None
            evals.eval_sequence(number_evaluated, self.fitness,
                                model.add_data_row, idx)
            model.sort(number_evaluated)  # makes only a difference if elements of X are pushed out on later adds in evals
            tau = model.kendall(self.settings.n_for_tau(len(X), evals.evaluations))
            if evals.last_evaluations == number_evaluated:  # first call to evals.eval_sequence
                self.logger.add(tau)  # log first tau
            if tau >= self.settings.tau_truth_threshold - self.settings.tau_truth_threshold_correction * evals.evaluation_fraction:
                break
            number_evaluated += int(np.ceil(number_evaluated / 2))
            """multiply with 1.5 and take ceil
                [1, 2, 3, 5, 8, 12, 18, 27, 41, 62, 93, 140, 210, 315, 473]
                +[1, 1, 2, 3, 4,  6,  9, 14, 21, 31, 47,  70, 105, 158]
            """

        model.sort(self.settings.model_sort_globally or evals.evaluations)
        model.adapt_max_relative_size(tau)
        self.logger.add(tau)  # log last tau
        self.evaluations += evals.evaluations
        if self.evaluations == 0:  # can currently not happen
            # a hack to have some grasp on zero evaluations from outside
            self.evaluations = 1e-2  # hundred zero=iterations sum to one evaluation
        self.logger.add(evals.evaluations / len(X))
        self.logger.push()
        return evals.surrogate_values(model.eval, self.settings.return_true_fitnesses)

class ModelInjectionCallbackSettings(DefaultSettings):
    sigma_distance_lower_threshold = 0  # 0 == never decrease sigma
    sigma_factor = 1 / 1.1

class ModelInjectionCallback(object):
    """inject `model.xopt` and decrease `sigma` if `mean` is close to `model.xopt`.

    New, simpler callback class.

    Sigma decrease saves (only) 30% on the 10-D ellipsoid.
    """
    def __init__(self, model):
        """sigma_distance_lower_threshold=0 means decrease never"""
        self.update_model = False  # do not when es.fit.fit is based on surrogate values itself
        self.model = model
        self.logger = Logger(self)
        self.settings = ModelInjectionCallbackSettings(locals(), 0, self)
    def __call__(self, es):
        if self.update_model:
            self.model.add_data(es.pop_sorted, es.fit.fit)
        if 11 < 3 and not self.model.types:  # no injection in linear case
            self.logger.add(0).push()
            return
        es.inject([self.model.xopt])
        xdist = es.mahalanobis_norm(self.model.xopt - es.mean)
        self.logger.add(self.settings.sigma_distance_lower_threshold * es.N**0.5 / es.sp.weights.mueff)
        if xdist < self.settings.sigma_distance_lower_threshold * es.N**0.5 / es.sp.weights.mueff:
            es.sigma *= self.settings.sigma_factor
            self.logger.add(xdist).push()
        else:
            self.logger.add(-xdist).push()

class Tau(object): "placeholder to store Kendall tau related things"

def _n_for_model_building(m):  # type: (LQModel) -> int
    """truncate worst solutions for model building"""
    n = int(max((m.current_complexity + 2,
                 m.settings.truncation_ratio * (m.size + 1))))
    return min((m.size, n))

class LQModelSettings(DefaultSettings):
    max_relative_size_init = None  # 1.5  # times self.max_df: initial limit archive size
    max_relative_size_end = 2  # times self.max_df: limit archive size including truncated data
    max_relative_size_factor = 1.05  # factor to increment max_relevative_size
    truncation_ratio = max((3/4, (3 - max_relative_size_end) / 2))  # use only truncation_ratio best in _n_for_model_building
    tau_threshold_for_model_increase = 0.5  # rarely in use
    min_relative_size = 1.1  # earliest when to switch to next model complexity
    max_absolute_size = 0  # limit archive size as max((max_absolute, df * max_relative))
    # to be removed remove_worse = lambda m: int(min((m.size - m.current_complexity - 2, m.size / 4)))
    n_for_model_building = _n_for_model_building
    max_weight = 20  # min weight is one
    disallowed_types = ()
    f_transformation = False  # a simultaneous transformation of all Y values

    def _checking(self):
        if not 0 < self.truncation_ratio <= 1:
            raise ValueError(
                'need: 0 < truncation_ratio <= 1, was: truncation_ratio=%f' %
                self.truncation_ratio)
        max_init = self.max_relative_size_init or self.max_relative_size_end
        if not 0 < self.min_relative_size / self.truncation_ratio <= max_init <= self.max_relative_size_end:
            raise ValueError(
                'need max_relative_size_end=%f >= max_relative_size_init=%s >= min_relative_size/self.truncation_ratio=%f/%f >= 0' %
                (self.max_relative_size_end, str(self.max_relative_size_init), self.min_relative_size, self.truncation_ratio))
        return self

class LQModel(object):
    """Up to a full quadratic model using the pseudo inverse to compute
    the model coefficients.

    The full model has 1 + 2n + n(n-1)/2 = n(n+3) + 1 parameters. Model
    building "works" with any number of data.

    Model size 1.0 doesn't work well on bbob-f10, 1.1 however works fine.

    TODO: change self.types: List[str] to self.type: str with only one entry

    >>> import numpy as np
    >>> import cma
    >>> import cma.fitness_models as fm
    >>> # fm.Logger, Logger = fm.LoggerDummy, fm.Logger
    >>> m = fm.LQModel()
    >>> for i in range(30):
    ...     x = np.random.randn(3)
    ...     y = cma.ff.elli(x - 1.2)
    ...     _ = m.add_data_row(x, y)
    >>> assert np.allclose(m.coefficients, [
    ...   1.44144144e+06,
    ...  -2.40000000e+00,  -2.40000000e+03,  -2.40000000e+06,
    ...   1.00000000e+00,   1.00000000e+03,   1.00000000e+06,
    ...  -4.65661287e-10,  -6.98491931e-10,   1.97906047e-09,
    ...   ], atol=1e-5)
    >>> assert np.allclose(m.xopt, [ 1.2,  1.2,  1.2])
    >>> assert np.allclose(m.xopt, [ 1.2,  1.2,  1.2])

    Check the same before the full model is build:

    >>> m = fm.LQModel()
    >>> m.settings.min_relative_size = 3 * m.settings.truncation_ratio
    >>> for i in range(30):
    ...     x = np.random.randn(4)
    ...     y = cma.ff.elli(x - 1.2)
    ...     _ = m.add_data_row(x, y)
    >>> print(m.types)
    ['quadratic']
    >>> assert np.allclose(m.coefficients, [
    ...   1.45454544e+06,
    ...  -2.40000000e+00,  -2.40000000e+02,  -2.40000000e+04, -2.40000000e+06,
    ...   1.00000000e+00,   1.00000000e+02,   1.00000000e+04,   1.00000000e+06,
    ...   ])
    >>> assert np.allclose(m.xopt, [ 1.2,  1.2,  1.2,  1.2])
    >>> assert np.allclose(m.xopt, [ 1.2,  1.2,  1.2,  1.2])

    Check the Hessian in the rotated case:

    >>> fitness = cma.fitness_transformations.Rotated(cma.ff.elli)
    >>> m = fm.LQModel(2, 2)
    >>> for i in range(30):
    ...     x = np.random.randn(4) - 5
    ...     y = fitness(x - 2.2)
    ...     _ = m.add_data_row(x, y)
    >>> R = fitness[1].dicMatrices[4]
    >>> H = np.dot(np.dot(R.T, np.diag([1, 1e2, 1e4, 1e6])), R)
    >>> assert np.all(np.isclose(H, m.hessian))
    >>> assert np.allclose(m.xopt, 4 * [2.2])
    >>> m.set_xoffset([2.335, 1.2, 2, 4])
    >>> assert np.all(np.isclose(H, m.hessian))
    >>> assert np.allclose(m.xopt, 4 * [2.2])

    Check a simple linear case, the optimum is not necessarily at the
    expected position (the Hessian matrix is chosen somewhat arbitrarily)

    >>> m = fm.LQModel()
    >>> m.settings.min_relative_size = 4
    >>> _ = m.add_data_row([1, 1, 1], 220 + 10)
    >>> _ = m.add_data_row([2, 1, 1], 220)
    >>> print(m.types)
    []
    >>> assert np.allclose(m.coefficients, [80, -10, 80, 80])
    >>> assert np.allclose(m.xopt, [22, -159, -159])  # [ 50,  -400, -400])  # depends on Hessian
    >>> # fm.Logger = Logger

    For results see:
    
    Hansen (2019). A Global Surrogate Model for CMA-ES. In Genetic and Evolutionary
    Computation Conference (GECCO 2019), Proceedings, ACM.
    
    lq-CMA-ES at http://lq-cma.gforge.inria.fr/ppdata-archives/pap-gecco2019/figure5/

    """
    _complexities = [  # must be ordered by complexity here
        ['quadratic', lambda d: 2 * d + 1],
        ['full', lambda d: d * (d + 3) / 2 + 1]]
    known_types = [c[0] for c in _complexities]
    complexity = dict(_complexities)

    @property
    def current_complexity(self):
        """degrees of freedom (nb of parameters) of the current model"""
        if self.types:
            return max(self.complexity[t](self.dim) for t in self.types)
        return self.dim + 1

    def __init__(self,
                 max_relative_size_init=None,  # when to prune, only applicable after last model switch
                 max_relative_size_end=None,
                 min_relative_size=None,  # when to switch to next model
                 max_absolute_size=None,  # maximum archive size
                 ):
        """

        Increase model complexity if the number of data exceeds
        ``max(min_relative_size * df_biggest_model_type, self.min_absolute_size)``.

        Limit the number of kept data
        ``max(max_absolute_size, max_relative_size * max_df)``.

        """
        self.settings = LQModelSettings(locals(), 4, self)._checking()
        self._fieldnames = ['X', 'F', 'Y', 'Z', 'counts', 'hashes']
        self.logger = Logger(self, ['logging_trace'],
                             labels=[# 'H(X[0]-X[1])', 'H(X[0]-Xopt)',
                                     '||X[0]-X[1]||^2', '||X[0]-Xopt||^2'])
        self.log_eigenvalues = Logger(self, ['eigenvalues'],
                                      name='Modeleigenvalues')
        self.reset()

    def reset(self):
        for name in self._fieldnames:
            setattr(self, name, [])
        self.types = []  # ['quadratic', 'full']
        self.type_updates = defaultdict(list)  # for the record only
        self._type = 'linear'  # not in use yet
        "the model can have several types, for the time being"
        self.count = 0  # number of overall data seen
        self.max_relative_size = self.settings.max_relative_size_init or (
                        self.settings.max_relative_size_end)
        self._coefficients_count = -1
        self._xopt_count = -1
        self._xoffset = 0
        self.number_of_data_last_added = 0  # sweep of data added
        self.tau = Tau()
        self.tau.tau, self.tau.n = 0, 0

    def sorted_weights(self, number=None):
        """regression weights in decreasing order"""
        return np.linspace(self.settings.max_weight, 1,
                           self.size if number is None or number > self.size
                           else number)

    @property
    def logging_trace(self):
        """some data of the current state which may be interesting to display"""
        if len(self.X) < 2:
            return [1, 1]
        trace = []
        d1 = np.asarray(self.X[0]) - self.X[1]
        d2 = np.asarray(self.X[0]) - self.xopt
        # trace += [self.mahalanobis_norm_squared(d1),
        #           self.mahalanobis_norm_squared(d2)]
        trace += [sum(d1**2), sum(d2**2)]
        return trace

    @property
    def max_size(self):
        return max((self.complexity[self.known_types[-1]](self.dim) *
                        self.max_relative_size,
                    self.settings.max_absolute_size))

    @property
    def max_df(self):
        return self.complexity[self.known_types[-1]](self.dim)

    @property
    def size(self):
        """number of data available to build the model"""
        return len(self.X)

    @property
    def dim(self):
        return len(self.X[0]) if len(self.X) else None

    def update_type(self):
        """model type/size depends on the number of observed data
        """
        if not len(self.X):
            if 11 < 3 and not self._current_type == 'linear':  # TODO: we could check that the model is linear
                warnings.warn('empty model is not linear')
            return
        n, d = len(self.X), len(self.X[0])
        # d + 1 affine linear coefficients are always computed
        for type in self.known_types[::-1]:  # most complex type first
            if (n * self.settings.truncation_ratio >= self.complexity[type](d) * self.settings.min_relative_size
                and type not in self.types
                and type not in self.settings.disallowed_types):
                self.types.append(type)
                self._current_type = type  # not in use (yet)
                self.reset_Z()
                self.type_updates[self.count] += [type]
                # print(self.count, self.types)

    def type(self):
        """one of the model `known_types`, depending on self.size.

        This may replace `types`, but is not in use yet.
        """
        d = len(self.X[0])
        for type in self.known_types[::-1]:  # most complex type first
            if (self.size * self.settings.truncation_ratio >= self.complexity[type](d) * self.settings.min_relative_size
                and type not in self.settings.disallowed_types):
                break
        else:
            type = 'linear'
        self._current_type = type  # here we could check whether the type changed
        return self._current_type

    def _prune(self):
        "deprecated"
        while (len(self.X) > self.max_size and
               len(self.X) * self.settings.truncation_ratio - 1 >= self.max_df * self.settings.min_relative_size):
            for name in self._fieldnames:
                getattr(self, name).pop()

    def prune(self):
        """prune data depending on size parameters"""
        remove = int(self.size - max((self.max_size,
                                      self.max_df * self.settings.min_relative_size / self.settings.truncation_ratio)))
        if remove <= 0:
            return
        for name in self._fieldnames:
            try:
                setattr(self, name, getattr(self, name)[:-remove])
            except TypeError:
                field = getattr(self, name)
                for _ in range(remove):
                    field.pop()

    def add_data_row(self, x, f, prune=True, force=False):
        """add `x` to `self` ``if `force` or x not in self``"""
        hash = self._hash(x)
        if hash in self.hashes:
            warnings.warn("x value already in Model, " + (
                "use `force` argument to force add" if force else "nothing added"))
            if not force:
                return
        if 11 < 3:
            # x = np.asarray(x)
            self.X.insert(0, x)
            self.Z.insert(0, self.expand_x(x))
            self.F.insert(0, f)
            self.Y.insert(0, f)
        else:  # in 10D reduces time in ﻿numpy.core.multiarray.array by a factor of five from 2nd to 8th largest consumer
            self.X = np.vstack([x] + ([self.X] if self.count > 0 else []))  # stack x on top
            self.Z = np.vstack([self.expand_x(x)] + ([self.Z] if self.count > 0 else []))
            self.F = np.hstack([f] + ([self.F] if self.count > 0 else []))
            self.Y = np.hstack([f] + ([self.Y] if self.count > 0 else []))
        self.count += 1
        self.counts.insert(0, self.count)
        self.hashes.insert(0, hash)
        self.number_of_data_last_added = 1
        self.update_type()
        if prune:
            self.prune()
        if self.settings.f_transformation:
            self.Y = self.settings.f_transformation(self.F)
        return self

    def add_data(self, X, Y, prune=True):
        """add a sequence of x- and y-data, sorted by y-data (best last)
        """
        if len(X) != len(Y):
            raise ValueError("input X and Y have different lengths %d!=%d" % (len(X), len(Y)))
        idx = np.argsort(Y)[::-1]
        for i in idx:  # insert smallest/best last
            self.add_data_row(X[i], Y[i], prune=prune)
        self.number_of_data_last_added = len(X)
        return self

    def _sort(self, number=None, argsort=np.argsort):
        """old? sort last `number` entries TODO: for some reason this seems not to pass the doctest"""
        assert self.size == len(self.X)
        if number is None:
            number = len(self.X)
        if number <= 0:
            return self
        # print(number, len(self.Y))
        number = min((number, len(self.Y)))
        idx = argsort([self.Y[i] for i in range(number)])  # [:number] doesn't work on deque's
        for name in self._fieldnames:
            field = getattr(self, name)
            tmp = [field[i] for i in idx]  # a sorted copy
            # tmp = [field[i] for i in range(number)]  # a copy
            for i in range(len(tmp)):
                field[i] = tmp[i]
        assert list(self.Y[:number]) == sorted(self.Y[:number])

    def sort(self, number=None, argsort=np.argsort):
        """sort last `number` entries"""
        # print(number)
        if number is None or number is True:  # remark that ``1 in (True,) is True`` and
            number = self.size                # True and 1 must be treated different here!
        number = min((number, self.size))
        if number <= 1:  # nothing to sort
            return self
        if number < self.size:
            idx = argsort(self.Y[:number])  # [:number] doesn't work on deque's
        else:
            idx = argsort(self.Y)
        for name in self._fieldnames:
            field = getattr(self, name)
            try:
                field[:number] = field[idx]
            except TypeError:
                s = [field[i] for i in idx]
                for i in range(number):
                    field[i] = s[i]
            # setattr(self, name, field)
        if 11 < 3:
            assert list(self.Y[:number]) == sorted(self.Y[:number])
            Y = list(self.Y)
            self._sort(number)
            assert Y == list(self.Y)

    def adapt_max_relative_size(self, tau):
        if len(self.types) == len(self.known_types) and tau < self.settings.tau_threshold_for_model_increase:
            self.max_relative_size = min((self.settings.max_relative_size_end,
                                self.settings.max_relative_size_factor * self.max_relative_size))

    def xmean(self):
        return np.mean(self.X, axis=0)

    def set_xoffset(self, offset):
        self._xoffset = np.asarray(offset)
        self.reset_Z()

    def reset_Z(self):
        """set x-values Z attribute"""
        self.Z = np.asarray([self.expand_x(x) for x in self.X])
        self._coefficients_count = -1
        self._xopt_count = -1

    def _list_expand_x(self, x):
        x = np.asarray(x) + self._xoffset
        z += [1] + list(x)  # or np.hstack([1, x])
        if 'quadratic' in self.types:
            z += list(np.square(x))
            if 'full' in self.types:
                # TODO: takes in 10-D about as much time as SVD (generator is slighly more expensive)
                # using np.array seems not to help either, array itself takes a considerable chunk of time
                z += [x[i] * x[j] for i in range(len(x)) for j in range(len(x)) if i < j]
        return z

    def expand_x(self, x):
        x = np.asarray(x) + self._xoffset
        z = np.hstack([1, x])
        if 'quadratic' in self.types:
            z = np.hstack([z, np.square(x)])
            if 'full' in self.types:
                # TODO: takes in 10-D about 65% of the time of SVD (generator is slighly more expensive)
                z = np.hstack([z, [x[i] * x[j] for i in range(len(x)) for j in range(len(x)) if i < j]])
        return z

    def eval_true(self, x, max_number=None):
        """never used, return true f-value if ``x in self.X[:max_number]``, else Model value.

        Not clear whether this is useful, because the return value is unpredictably
        incomparable.
        """
        try:
            idx = self.hashes.index(self._hash(x))
        except ValueError:
            assert self._hash(x) not in self.hashes
            return self.eval(x)
        return self.Y[idx]
        # using self.references.index(x) doesn't work
        if max_number is None:
            max_number = len(self.Y)
        max_number = min((len(self.Y), max_number))
        for idx, val in enumerate(self.references):
            if val is x or idx >= max_number:
                break
        return self.Y[idx] if idx < max_number else self.eval(x)

    def eval(self, x):
        """return Model value of `x`"""
        if self.count and len(x) != len(self.X[0]):
            raise ValueError("x = %s must be of len %d != %d"
                             % (str(x), len(self.X[0]), len(x)))
        if self.count <= 0:
            return 0
        # since Python 3.8: if (idx := self.index(x)) >= 0:
        try:  # if self.isin(x):  # makes in case two calls to isin/index
            z = self.Z[self.index(x)]
        except ValueError:  # index of x not available
            assert not self.isin(x)
            z = self.expand_x(x)
        return np.dot(self.coefficients, z)

    def evalpop(self, X):
        """never used, return Model values of ``x for x in X``"""
        return [0 if self.count == 0 else self.eval(x)
                for x in X]

    def optimize(self, fitness, x0, evals):
        """this works very poorly e.g. on Rosenbrock::

            x, m = Model().optimize(cma.ff.rosen, [0.1, -0.1], 13)

        TODO (implemented, next: test): account for xopt not changing.
        """
        self.add_data_row(x0, fitness(x0))
        while self.count < evals:
            xopt_old = self.xopt[:]
            self.add_data_row(list(self.xopt), fitness(self.xopt))
            if sum((xopt_old - self.xopt)**2) < 1e-3 * sum((np.asarray(self.X[1]) - self.X[0])**2):
                x_new = (np.asarray(self.X[1]) - self.X[0]) / 2
                self.add_data_row(x_new, fitness(x_new))
        return self.xopt, self

    def kendall(self, number, F_model=None):
        """return Kendall tau between true F-values (Y) and model values.
        """
        if F_model:
            raise NotImplementedError("save model evaluations if implemented")
        number = min((number, self.size))
        self.tau.n = number
        self.tau.count = self.count
        if self.tau.n < 3:
            self.tau.result = None
            self.tau.tau, self.tau.pvalue = 0, 0
            return 0
        self.tau.tau = kendall_tau(self.Y[:number],
                                   [self.eval(self.X[i]) for i in range(number)])
        return self.tau.tau

    def isin(self, x):
        """return False if `x` is not (anymore) in the model archive"""
        hash = self._hash(x)
        return self.hashes.index(hash) + 1 if hash in self.hashes else False

    def index(self, x):
        return self.hashes.index(self._hash(x))

    def _hash(self, x):
        return sum(x)  # with a tuple as hash ``self._hash(x) in self.hashes`` fails under Python 2.6 and 3.4

    def mahalanobis_norm_squared(self, dx):
        """caveat: this can be negative because hessian is not guarantied
        to be pos def.
        """
        return np.dot(dx, np.dot(self.hessian, dx))

    def old_weighted_array(self, Z):
        """return weighted Z, worst entries are clipped if possible.

        Z can be a vector or a matrix.
        """
        idx = np.argsort(self.Y)
        clip = max((0, self.settings.remove_worse(self)))
        if clip:
            idx = idx[:-clip]
            w = self.sorted_weights(self.size - clip)
        else:
            w = self.sorted_weights()
        return w * np.asarray(Z)[idx].T

    def weighted_array(self, Z):
        """return weighted Z, worst entries are clipped if possible.

        Z can be a vector or a matrix.
        """
        size = self.settings.n_for_model_building(self)
        idx = np.argsort(self.Y)
        if size < self.size:
            idx = idx[:size]
        return self.sorted_weights(size) * np.asarray(Z)[idx].T

    @property
    def pinv(self):
        """return Pseudoinverse, computed unconditionally (not lazy).

        `pinv` is usually not used directly but via the `coefficients` property.

        Should this depend on something and/or become lazy?
        """
        try:
            self._pinv = np.linalg.pinv(self.weighted_array(self.Z)).T
            # self._pinv = np.linalg.pinv(self.weights * np.asarray(self.Z).T).T
        except np.linalg.LinAlgError as laerror:
            warnings.warn('Model.pinv(d=%d,m=%d,n=%d): np.linalg.pinv'
                          ' raised an exception %s' % (
                        len(self.X[0]) if self.size else -1,
                        len(self._coefficients),
                        len(self.X),
                        str(laerror)))
        return self._pinv

    @property
    def coefficients(self):
        """model coefficients that are linear in self.expand(.)"""
        if self._coefficients_count < self.count:
            self._coefficients_count = self.count
            self._coefficients = np.dot(self.pinv, self.weighted_array(self.Y))
            self.logger.push()  # use logging_trace attribute and xopt
            self.log_eigenvalues.push()
        return self._coefficients

    @property
    def hessian(self):
        d = len(self.X[0])
        m = len(self.coefficients)
        # assert m in (d + 1, 2 * d + 1, d * (d + 3) / 2 + 1)
        assert m in [d + 1] + [self.complexity[type](d) for type in self.known_types]
        H = np.zeros((d, d))  # TODO: use (sparse) diagonal matrix if 'full' not in self.types
        k = 2 * d + 1
        for i in range(d):
            if m > d + 1:
                assert 'quadratic' in self.types
                H[i, i] = self.coefficients[d + i + 1]
            else:
                assert m == d + 1
                H[i, i] = self.coefficients[i + 1] / 100  # TODO arbitrary factor to make xopt finite, shouldn't this be 1 / coefficient or min(abs(coefficients))
                H[i, i] = min(np.abs(self.coefficients[1:])) / 100
            if m > 3 * d:
                assert 'full' in self.types
                assert m == self.complexity['full'](d)  # here the only possibility left
                for j in range(i + 1, d):
                    H[i, j] = H[j, i] = self.coefficients[k] / 2
                    k += 1
        return H

    @property
    def b(self):
        return self.coefficients[1:len(self.X[0]) + 1]

    @property
    def xopt(self):
        if self._xopt_count < self.count:
            self._xopt_count = self.count
            if not self.types:  # linear case
                # print(self.coefficients[1:], self.X[0])
                self._xopt = self.X[0] - 2 * self.coefficients[1:]  # TODO: don't need xoffset here!?
            else:
                try:
                    self._xopt = np.dot(np.linalg.pinv(self.hessian), self.b / -2.) - self._xoffset
                except np.linalg.LinAlgError as laerror:
                    warnings.warn('Model.xopt(d=%d,m=%d,n=%d): np.linalg.pinv'
                                  ' raised an exception %s' % (
                                self.dim or -1,
                                len(self._coefficients),
                                self.size,
                                str(laerror)))
                    if not hasattr(self, '_xopt') and self.dim:
                        # TODO: zeros is the right choice but is devistatingly good on test functions
                        self._opt = np.zeros(self.dim)
                        self._opt = np.random.randn(self.dim)
        return self._xopt

    @property
    def minY(self):
        """smallest f-values in data queue"""
        return min(self.F)

    @property
    def eigenvalues(self):
        """eigenvalues of the Hessian of the model"""
        return sorted(np.linalg.eigvals(self.hessian))

