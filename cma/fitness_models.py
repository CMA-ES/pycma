# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function  #, unicode_literals, with_statement
del absolute_import, division, print_function  #, unicode_literals, with_statement
___author__ = "Nikolaus Hansen"
__license__ = "BSD 3-clause"

import warnings
from collections import defaultdict  # since Python 2.5
import numpy as np
from scipy.stats import kendalltau as _kendalltau
from .utilities import utils


def kendall_tau(x, y):
    """return Kendall tau rank correlation coefficient.

    Implemented only to potentially remove dependency on `scipy.stats`.

    This

    >>> import numpy as np
    >>> from cma.fitness_models import kendall_tau
    >>> kendalltau = lambda x, y: (kendall_tau(x, y), 0)
    >>> from scipy.stats import kendalltau  # incomment if not available
    >>> for dim in np.random.randint(3, 22, 5):
    ...     x, y = np.random.randn(dim), np.random.randn(dim)
    ...     t1, t2 = kendall_tau(x, y), kendalltau(x, y)[0]
    ...     # print(t1, t2)
    ...     assert np.isclose(t1, t2)

    """
    equal_correction = 1 / 2
    assert len(x) == len(y)
    x, y = np.asarray(x), np.asarray(y)
    s = 0
    for i in range(len(x)):
        if 1 < 3:  # 20 times faster with len(x)=200
            dx = np.sign(x[i] - x[:i])
            dy = np.sign(y[i] - y[:i])
            s += sum(dx * dy)
            if equal_correction:
                s += equal_correction * sum((dx == 0) * (dy == 0))
        else:
            for j in range(i):
                s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    return s * 2. / (len(x) * (len(x) - 1))


class LoggerDummy:
    """use to fake a `Logger` in non-verbose setting"""
    def __init__(self, *args, **kwargs):
        self.count = 0
    def __call__(self, *args, **kwargs):
        self.push()
    def add(self, *args, **kwargs):
        return self
    def push(self, *args, **kwargs):
        self.count += 1
    def load(self, *args, **kwargs):
        return self
    def plot(self, *args, **kwargs):
        warnings.warn("loggers is in dummy (silent) mode,"
                      " there is nothing to plot")

class Logger:
    """log an arbitrary number of data (a data row) per "timestep".

    The `add` method can be called several times per timestep, the
    `push` method must be called once per timestep. `load` and `plot`
    will only work if each time the same number of data was pushed.

    For the time being, the data is saved to a file after each timestep.

    To append data, set `self.counter` > 0 before to call `push` the first
    time. ``len(self.load().data)`` is the number of current data.

    Useless example::

        >> es = cma.CMAEvolutionStrategy
        >> lg = Logger(es, ['countiter'])
        >> lg.push()  # add the counter

    """
    def __init__(self, obj_or_name, attributes=None, callables=None, name=None, labels=None):
        """obj can also be a name"""
        self.format = "%.19e"
        if obj_or_name == str(obj_or_name) and attributes is not None:
            raise ValueError('string obj %s has no attributes %s' % (
                str(obj_or_name), str(attributes)))
        self.obj = obj_or_name
        self.name = name
        self._autoname(obj_or_name)
        self.attributes = attributes or []
        self.callables = callables or []
        self.labels = labels or []
        self.count = 0
        self.current_data = []
        # print('Logger:', self.name, self._name)

    def _autoname(self, obj):
        """TODO: how to handle two loggers in the same class??"""
        if str(obj) == obj:
            self.name = obj
        if self.name is None:
            s = str(obj)
            s = s.split('class ')[-1]
            s = s.split('.')[-1]
            # print(s)
            if ' ' in s:
                s = s.split(' ')[0]
            if "'" in s:
                s = s.split("'")[-2]
            self.name = s
        self._name = self.name
        if '.' not in self._name:
            self._name = self._name + '.logdata'
        if not self._name.startswith(('._', '_')):
            self._name = '._' + self._name

    def _stack(self, data):
        """stack data into current row managing the different access...

        ... and type formats.
        """
        if isinstance(data, list):
            self.current_data += data
        else:
            try:  # works for numpy array
                self.current_data += [d for d in data]
            except TypeError:
                self.current_data += [data]

    def __call__(self, obj=None):
        """see also method `push`.

        TODO: replacing `obj` here is somewhat inconsistent, but maybe
        an effective hack.
        """
        if obj is not None:
            self.obj = obj
        return self.push()

    def add(self, data):
        """data may be a value, or a `list`, or a `numpy` array.

        See also `push` to complete the iteration.
        """
        # if data is not None:
        self._stack(data)
        return self

    def _add_defaults(self):
        for name in self.attributes:
            data = getattr(self.obj, name)
            self._stack(data)
        for callable in self.callables:
            self._stack(callable())
        return self

    def push(self, *args):
        """call ``stack()`` and finalize the current timestep, ignore
        input arguments."""
        self._add_defaults()
        if self.count == 0:
            self.push_header()
        with open(self._name, 'at') as file_:
            file_.write(' '.join(self.format % val
                                 for val in self.current_data) + '\n')
        self.current_data = []
        self.count += 1

    def push_header(self):
        mode = 'at' if self.count else 'wt'
        with open(self._name, mode) as file_:
            if self.labels:
                file_.write('# %s\n' % repr(self.labels))
            if self.attributes:
                file_.write('# %s\n' % repr(self.attributes))

    def load(self):
        import ast
        self.data = np.loadtxt(self._name)
        with open(self._name, 'rt') as file_:
            first_line = file_.readline()
        if first_line.startswith('#'):
            self.labels = ast.literal_eval((first_line[1:].lstrip()))
        return self

    def plot(self, plot=None):
        try:
            from matplotlib import pyplot as plt
        except ImportError: pass
        if plot is None:
            from matplotlib.pyplot import plot
        self.load()
        n = len(self.data)  # number of data rows
        try:
            m = len(self.data[0])  # number of "variables"
        except TypeError:
            m = 0
        plt.gca().clear()
        if not m or len(self.labels) == 1:  # data cannot be indexed like data[:,0]
            plot(range(1, n + 1), self.data,
                 label=self.labels[0] if self.labels else None)
            return
        color=iter(plt.cm.winter_r(np.linspace(0.15, 1, m)))
        for i in range(m):
            plot(range(1, n + 1), self.data[:, i],
                 label=self.labels[i] if i < len(self.labels) else None)
            plt.gca().get_lines()[0].set_color(next(color))
        plt.legend(framealpha=0.3)  # more opaque than not
        return self

_Logger = Logger  # to reset Logger in doctest

class DefaultSettings(object):
    """somewhat resembling `types.SimpleNamespace` from Python >=3.3
    but with instantiation and even more the `dataclass` decorator from
    Python >=3.7.

    ``MyClassSettings(DefaultSettings)`` is used like:

    >>> class MyClass:
    ...     def __init__(self, a, b=None, param1=None, c=3):
    ...         self.settings = MyClassSettings(locals(), 1, self)

    The `1` signals, purely for consistency checking, that one parameter
    defined in ``MyClassSettings`` is to be set. The settings may be
    defined like

    >>> from cma.fitness_models import DefaultSettings
    >>> class MyClassSettings(DefaultSettings):
    ...     param1 = 123
    ...     val2 = False
    ...     another_par = None  # we need to assign at least None always

    The main purpose is, with the least effort, (i) to separate
    parameters/settings of a class from its remaining attributes, and (ii) to be
    flexible as to which of these parameters are arguments to ``__init__``.
    Parameters can always be modified after instantiation. Further advantages
    are (a) no typing of ``self.`` to assign the default value or the passed
    parameter value (the latter do not even be assigned) and (b) no confusing
    name change between the passed option and attribute name.

    It is not possible to overwrite the default value with `None`.

    Usage: define a bunch of parameters in a derived parameter class:

    Now we assign a settings (or parameters) attribute in the ``__init__` of the
    target class, which should here use (only) one value from the input
    arguments list and doesn't use any names which are already defined in
    ``self.__dict__``:

    Now any of these parameters can be used or re-assigned like:

    >>> c = MyClass(0.1)
    >>> c.settings.param1 == 123
    True
    >>> c = MyClass(2, param1=False)
    >>> c.settings.param1 is False
    True

    """
    def __init__(self, params, number_of_params, obj):
        """Overwrite default settings in case.

        :param params: A dictionary containing the parameters to set/overwrite
        :param number_of_params: Number of parameters to set/overwrite
        :param obj: elements of obj.__dict__ are on the ignore list.
        """
        self.inparams = dict(params)
        self._number_of_params = number_of_params
        self.obj = obj
        self.inparams.pop('self', None)
        self._set_from_defaults()
        self._set_from_input()

    def __str__(self):
        # works with print:
        return ("{" +
                '\n'.join(r"%s: %s" % (str(k), str(v))
                          for k, v in self.items()) +
                "}")
        return str(self.__dict__)

    def _set_from_defaults(self):
        """defaults are taken from the class attributes"""
        self.__dict__.update(((key, val)
                              for (key, val) in type(self).__dict__.items()
                              if not key.startswith('_')))
    def _set_from_input(self):
        """Only existing parameters/attributes and non-None values are set.

        The number of parameters is cross-checked.

        Remark: we could select only the last arguments
        of obj.__init__.__func__.__code__.co_varnames
        which have defaults obj.__init__.__func__.__defaults__ (we do
        not need the defaults)
        """
        discarded = {}  # discard name if not in self.__dict__
        for key in list(self.inparams):
            if key not in self.__dict__ or key in self.obj.__dict__:
                discarded[key] = self.inparams.pop(key)
            elif self.inparams[key] is not None:
                setattr(self, key, self.inparams[key])
        if len(self.inparams) != self._number_of_params:
            warnings.warn("%s: %d parameters desired; remaining: %s; discarded: %s "
                          % (str(type(self)), self._number_of_params, str(self.inparams),
                             str(discarded)))
        # self.__dict__.update(self.inparams)
        delattr(self, 'obj')  # set only once

class SurrogatePopulationSettings(DefaultSettings):
    minimum_model_size = 3  # absolute minimum number of true evaluations before to build the model
    n_for_tau = lambda popsi, nevaluated: int(max((15, min((1.5 * nevaluated, 0.75 * popsi)))))
    model_max_size_factor = 3  # times popsize, 3 is big!?
    tau_truth_threshold = 0.85  # tau between model and ground truth
    min_evals_percent = 2  # eval int(1 + min_evals_percent / 100) unconditionally
    model_sort_globally = True
    return_true_fitnesses = True  # return true fitness if all solutions are evaluated
    # add_xopt_condition = False  # not in use
    change_threshold = -1.0     # not in use tau between previous and new model; was: 0.8

class SurrogatePopulation:
    """surrogate f-values for a population.

    What is new:

    - the model is built with >=3 evaluations (compared to LS-CMA and lmm-CMA)
    - the model is linear at first, then diagonal, then full
    - the model uses the fitness ranks as weights for weighted regression.

    >>> import cma
    >>> import cma.fitness_models as fm
    >>> from cma.fitness_transformations import Function as FFun  # adds evaluations attribute
    >>> fm.Logger = fm.LoggerDummy
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
    ...     print(fitfun.evaluations)  # was: 12, 161, 18 131 (and even smaller)
    ...     assert 'ftarget' in es.stop()
    18
    150

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
    >>> fm.Logger = fm._Logger

    """
    def __init__(self,
                 fitness,
                 model=None,
                 model_max_size_factor=None,
                 # model_min_size_factor=None,
                 tau_truth_threshold=None,
                 # add_xopt_condition=None,
                 ):
        """

        If ``model is None``, a default `Model` instance is used. By
        setting `self.model` to `None`, only the `fitness` for each
        population member is evaluated in each call.

        """
        self.fitness = fitness
        self.model = model if model else Model()
        # set 3 parameters of settings from locals() which are not attributes of self
        self.settings = SurrogatePopulationSettings(locals(), 2, self)  # not in use yet
        self.count = 0
        self.evaluations = 0
        self.logger = Logger(self, labels=['tau0', 'tau1', 'evaluated ratio'])
        self.logger_eigenvalues = Logger(self.model, ['eigenvalues'])

    class FContainer:
        """Manage incremental evaluation of a population of solutions.

        Evaluate solutions and add them into the model and keep track of
        evaluated solutions.

        Uses `model.add_data_row` and `model.eval`.
        """
        def __init__(self, X):
            self.X = X
            self.evaluated = len(X) * [False]
            self.fvalues = len(X) * [np.nan]
        def eval(self, i, fitness, model):
            self.fvalues[i] = fitness(self.X[i])
            self.evaluated[i] = True
            model.add_data_row(self.X[i], self.fvalues[i])
        def eval_sequence(self, idx, number, fitness, model):
            """evaluate unevaluated entries of X[idx] until overall
            `number` entries are evaluated.

            Assumes that set(X[idx]) == set(X).

            ``idx`` defines the evaluation sequence, post condition is
            ``sum(self.evaluated) == min(number, len(self.X))``.
            """
            assert len(idx) == len(self.evaluated)
            n = sum(self.evaluated)
            for i in idx:
                if n >= number:
                    break
                if not self.evaluated[i]:
                    self.eval(i, fitness, model)
                    n += 1
        def svalues(self, model, true_values_if_all_available=True):
            """assumes that model and `evaluated` is not empty"""
            if true_values_if_all_available and sum(self.evaluated) == len(self.X):
                return self.fvalues
            F_model = [model.eval(x) for x in self.X]
            offset = np.nanmin(self.fvalues) - np.nanmin(F_model)
            return [f + offset for f in F_model]
        @property
        def evaluations(self):
            return sum(self.evaluated)
        @property
        def remaining(self):
            """number of not yet evaluated solutions"""
            return len(self.X) - sum(self.evaluated)

    def __call__(self, X):
        """return population f-values.

        The smallest value is never smaller than a truly evaluated value.

        Uses: `model.settings.max_absolute_size`, `len(model.X)`, `model.sort`, `model.eval`

        """
        model = self.model
        if self.settings.model_max_size_factor * len(X) > 1.1 * model.settings.max_absolute_size:
            model.settings.max_absolute_size = self.settings.model_max_size_factor * len(X)
            model.reset()  # reset, because the population size changed
        evals = SurrogatePopulation.FContainer(X)
        self.evals = evals  # for the record

        # make minimum_model_size unconditional evals in the first call and quit
        if len(model.X) < self.settings.minimum_model_size:
            for i in range(self.settings.minimum_model_size - len(model.X)):
                evals.eval(i, self.fitness, model)
            self.evaluations += evals.evaluations
            if self.settings.model_sort_globally:
                model.sort()
            return evals.svalues(model, self.settings.return_true_fitnesses)

        number_evaluated = int(1 + len(X) * self.settings.min_evals_percent / 100)
        first_path = True
        while evals.remaining:
            idx = np.argsort([model.eval(x) for x in X])  # like previously, move down to recompute indices
            evals.eval_sequence(idx, number_evaluated, self.fitness, model)
            model.sort(number_evaluated)  # makes only a difference if elements of X are pushed out on later adds
            # model.sort()
            tau = model.kendall(self.settings.n_for_tau(len(X), evals.evaluations))
            if first_path:
                self.logger.add(tau)  # log first tau
                first_path = False
            if tau >= self.settings.tau_truth_threshold:
                break
            number_evaluated += int(np.ceil(0.5 * number_evaluated))
            """multiply with 1.5 and take ceil
                [1, 2, 3, 5, 8, 12, 18, 27, 41, 62, 93, 140, 210, 315, 473]
                +[1, 1, 2, 3, 4,  6,  9, 14, 21, 31, 47,  70, 105, 158]
            """

        if self.settings.model_sort_globally:
            model.sort()
        model.adapt_max_relative_size(tau)
        self.logger.add(tau)  # log last tau
        self.evaluations += evals.evaluations
        if self.evaluations == 0:  # can currently not happen
            # a hack to have some grasp on zero evaluations from outside
            self.evaluations = 1e-2  # hundred zero=iterations sum to one evaluation
        self.logger.add(evals.evaluations / len(X))
        self.logger.push()  # TODO: check that we do not miss anything below
        return evals.svalues(model, self.settings.return_true_fitnesses)

    def old__call__(self, X):
        """return population f-values.

        The smallest value is never smaller than a truly evaluated value.

        """
        if self.model is None:
            return [self.fitness(x) for x in X]
        model = self.model
        if self.settings.model_max_size_factor * len(X) > 1.1 * model.settings.max_absolute_size:
            model.settings.max_absolute_size = self.settings.model_max_size_factor * len(X)
            # model.min_absolute_size = max((model.min_absolute_size,
            #                                self.settings.model_min_size_factor * len(X)))
            model.reset()
        F_true = {}
        # need at least two evaluations for a non-flat linear model
        for k in range(len(X)):
            if len(model.X) >= self.settings.minimum_model_size:
                break
            F_true[k] = self.fitness(X[k])
            model.add_data_row(X[k], F_true[k], prune=False)
        model.sort(len(F_true))
        model.prune()
        F_model = [model.eval(x) for x in X]
        if F_true:  # go with minimum model size for one iteration
            self.evaluations += len(F_true)
            offset = min(F_true.values()) - min(F_model)
            return [f + offset for f in F_model]

        sidx0 = np.argsort(F_model)
        # making no evaluation at all should not save too many evaluations
        # if in some iterations many points need to be evaluated
        # hence we make always one evaluation unconditionally

        # TODO/done: proofread further from here

        # TODO: find a smarter order depending on F_model?
        # eidx = range(len(X))  # indices to be evaluated
        eidx = sidx0[:]
        iloop = 0
        i1 = 0  # i1-1 is last evaluation index
        for iloop in range(len(eidx)):
            i0 = i1
            i1 += int(np.ceil(0.5 * i1)) if i1 else int(1 + len(X) * self.settings.min_evals_percent / 100)
            """multiply with 1.5 and take ceil
                [1, 2, 3, 5, 8, 12, 18, 27, 41, 62, 93, 140, 210, 315, 473]
                +[1, 1, 2, 3, 4,  6,  9, 14, 21, 31, 47,  70, 105, 158]
            """
            for k in eidx[i0:i1]:
                F_true[k] = self.fitness(X[k])
                model.add_data_row(X[k], F_true[k], prune=False)
            # TODO: prevent this duplicate "finalize model.add_data" code?
            # print(len(F_true))
            model.sort(len(F_true))
            model.prune()
            if i1 >= len(eidx):
                assert len(F_true) == len(X)
                if not self.settings.return_true_fitnesses:
                    F_model = [model.eval(x) for x in X]
                break
            assert i1 < len(eidx)  # just a reminder
            # the model has changed, so we recompute surrogate f-values
            F_model = [model.eval(x) for x in X]
            sidx = np.argsort(F_model)
            # TODO: is it enough to just check whether sidx0 and sidx agree?
            # to find out we want to log the kendall of sidx0 vs sidx and
            # the tau computed below

            # kendall compares F_true[k] with model.eval(X[k]) ranks
            # TODO (minor): we would not need to recompute model.eval(X)
            # TODO: with large popsize we do not want all solutions in kendall, but only the best popsize/3 of this iteration?

            tau = model.kendall(self.settings.n_for_tau(len(X), evals.evaluations))
            if 11 < 3:  # TODO: remove soon (cross check performance though)
                print(tau, model._old_kendall(
                [F_true[k] for k in sorted(F_true)[:self.settings.maximum_n_for_tau(len(X), evals.evaluations)]],  # sorted is used to get k in a deterministic order
                [X[k] for k in sorted(F_true)],
                self.settings.minimum_n_for_tau - len(F_true),
                # [F_model[k] for k in sorted(F_true)  # check that this is correct
                ))  # take also last few
            if iloop == 0:
                self.logger.add(tau)
            if tau > self.settings.tau_truth_threshold:  # and _kendalltau(sidx0, sidx)[0] > self.settings.change_threshold
                break
            sidx0 = sidx  # is never used
            # TODO: we could also reconsider the order eidx which to compute next

        if self.settings.model_sort_globally:
            self.model.sort()
        self.logger.add(tau)
        self.evaluations += len(F_true)
        if 11 < 3 and self.settings.add_xopt_condition >= 1:  # this fails, because xopt may be very far astray
            evs = sorted(model.eigenvalues)
            if evs[0] > 0 and evs[-1] <= self.settings.add_xopt_condition * evs[0]:
                model.add_data_row(model.xopt, self.fitness(model.xopt))
                self.evaluations += 1
                F_model = [model.eval(x) for x in X]
        if self.evaluations == 0:  # can currently not happen
            # a hack to have some grasp on zero evaluations from outside
            self.evaluations = 1e-2  # hundred zero=iterations sum to one evaluation
        self.logger.add(len(F_true) / len(X))
        self.logger.push()  # TODO: check that we do not miss anything below
        if len(X) == len(F_true) and self.settings.return_true_fitnesses:
            # model.set_xoffset(model.xopt)
            return [F_true[i] for i in range(len(X))]

        # mix F_model and F_true?
        # TODO (depending on correlation threshold that rarely happens anyways):
        #     correct for false model ranks, but how?
        offset = min(F_true.values()) - min(F_model)  # such that no value is below F_true[i_min]
        return [F_model[i] + offset for i in range(len(X))]

class ModelInjectionCallbackSettings(DefaultSettings):
    sigma_distance_lower_threshold = 0  # 0 == never decrease sigma
    sigma_factor = 1 / 1.1

class ModelInjectionCallback:
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
        es.inject([self.model.xopt])
        xdist = es.mahalanobis_norm(self.model.xopt - es.mean)
        self.logger.add(self.settings.sigma_distance_lower_threshold * es.N**0.5 / es.sp.weights.mueff)
        if xdist < self.settings.sigma_distance_lower_threshold * es.N**0.5 / es.sp.weights.mueff:
            es.sigma *= self.settings.sigma_factor
            self.logger.add(xdist).push()
        else:
            self.logger.add(-xdist).push()

class Tau: "placeholder to store Kendall tau related things"

class ModelSettings(DefaultSettings):
    max_relative_size_init = 1.5  # times self.max_df: initial limit archive size
    max_relative_size_end = 3  # times self.max_df: limit archive size
    max_relative_size_factor = 1.1  # factor to increment max_relevative_size
    tau_threshold_for_model_increase = 0.5
    min_relative_size = 1.5  # earliest when to switch to next model complexity
    max_absolute_size = 0  # limit archive size as max((max_absolute, df * max_relative))
    max_weight = 20  # min weight is one
    disallowed_types = ()
    f_transformation = False  # a simultanious transformation of all Y values

    def _checking(self):
        if not 0 < self.min_relative_size <= self.max_relative_size_init <= self.max_relative_size_end:
            raise ValueError(
                'need max_relative_size_end=%f >= max_relative_size_init=%f >= min_relative_size=%f >= 0' %
                (self.max_relative_size_end, self.max_relative_size_init, self.min_relative_size))
        return self

class Model:
    """Up to a full quadratic model using the pseudo inverse to compute
    the model coefficients.

    The full model has 1 + 2n + n(n-1)/2 = n(n+3) + 1 parameters. Model
    building "works" with any number of data.

    Model size 1.0 doesn't work well on bbob-f10, 1.1 however works fine.

    TODO: change self.types: List[str] to self.type: str with only one entry

    >>> import numpy as np
    >>> import cma
    >>> import cma.fitness_models as fm
    >>> fm.Logger = fm.LoggerDummy
    >>> m = fm.Model()
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

    >>> m = fm.Model()
    >>> m.settings.min_relative_size = 3
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
    >>> m = fm.Model(2)
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

    >>> m = fm.Model()
    >>> m.settings.min_relative_size = 4
    >>> _ = m.add_data_row([1, 1, 1], 220 + 10)
    >>> _ = m.add_data_row([2, 1, 1], 220)
    >>> print(m.types)
    []
    >>> assert np.allclose(m.coefficients, [80, -10, 80, 80])
    >>> assert np.allclose(m.xopt, [ 50,  -400, -400])  # depends on Hessian
    >>> fm.Logger = fm._Logger

    """
    complexity = [  # must be ordered by complexity here
        ['quadratic', lambda d: 2 * d + 1],
        ['full', lambda d: d * (d + 3) / 2 + 1]]
    known_types = [c[0] for c in complexity]
    complexity = dict(complexity)

    @property
    def current_complexity(self):
        raise NotImplementedError

    def __init__(self,
                 max_relative_size_init=None,    # when to prune, only applicable after last model switch
                 max_relative_size_end=None,
                 min_relative_size=None,  # when to switch to next model
                 max_absolute_size=None, # maximum archive size
                 ):
        """

        Increase model complexity if the number of data exceeds
        ``max(min_relative_size * df_biggest_model_type, self.min_absolute_size)``.

        Limit the number of kept data
        ``max(max_absolute_size, max_relative_size * max_df)``.

        """
        self.settings = ModelSettings(locals(), 4, self)._checking()
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
        self.max_relative_size = self.settings.max_relative_size_init
        self._coefficients_count = -1
        self._xopt_count = -1
        self._xoffset = 0
        self.number_of_data_last_added = 0  # sweep of data added
        self.tau = Tau()
        self.tau.tau, self.tau.n = 0, 0

    def sorted_weights(self, number=None):
        return np.linspace(self.settings.max_weight, 1,
                           self.size if number is None else number)

    @property
    def logging_trace(self):
        if len(self.X) < 2:
            return [1, 1, 1, 1]
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
            return
        n, d = len(self.X), len(self.X[0])
        # d + 1 affine linear coefficients are always computed
        for type in self.known_types[::-1]:  # most complex type first
            if (n >= self.complexity[type](d) * self.settings.min_relative_size
                and type not in self.types
                and type not in self.settings.disallowed_types):
                self.types.append(type)
                self.reset_Z()
                self.type_updates[self.count] += [type]
                # print(self.count, self.types)

    def _prune(self):
        while (len(self.X) > self.max_size and
               len(self.X) - 1 >= self.max_df * self.settings.min_relative_size):
            for name in self._fieldnames:
                getattr(self, name).pop()

    def prune(self):
        remove = int(self.size - max((self.max_size,
                                      # self.min_absolute_size,
                                      self.max_df * self.settings.min_relative_size)))
        if remove <= 0:
            return
        for name in self._fieldnames:
            try:
                setattr(self, name, getattr(self, name)[:-remove])
            except TypeError:
                field = getattr(self, name)
                for _ in range(remove):
                    field.pop()

    def add_data_row(self, x, f, prune=True):
        hash = self._hash(x)
        if hash in self.hashes:
            warnings.warn("x value already in Model")
            return
        self.count += 1
        if 11 < 3:
            # x = np.asarray(x)
            self.X.insert(0, x)
            self.Z.insert(0, self.expand_x(x))
            self.F.insert(0, f)
            self.Y.insert(0, f)
        else:  # in 10D reduces time in ﻿numpy.core.multiarray.array by a factor of five from 2nd to 8th largest consumer
            self.X = np.vstack([x] + ([self.X] if self.count > 1 else []))
            self.Z = np.vstack([self.expand_x(x)] + ([self.Z] if self.count > 1 else []))
            self.F = np.hstack([f] + ([self.F] if self.count > 1 else []))
            self.Y = np.hstack([f] + ([self.Y] if self.count > 1 else []))
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
        """sort last `number` entries TODO: for some reason this seems not to pass the doctest"""
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
        if number is None:
            number = self.size
        if number <= 0:
            return self
        number = min((number, self.size))
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
        """return true value if ``x in self.X[:max_number]``, else Model value"""
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
        return 0 if self.count == 0 else np.dot(self.coefficients, self.expand_x(x))

    def evalpop(self, X):
        """return Model values of ``x for x in X``"""
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
        self.tau.result = _kendalltau(self.Y[:number],
                                      [self.eval(self.X[i]) for i in range(number)])
        try:
            self.tau.tau, self.tau.pvalue = self.tau.result[:2]
        except TypeError:  # kendalltau([3,3,3], [3,3,3]) == 1
            self.tau.tau, self.tau.pvalue = 0, 0
        if not np.isfinite(self.tau.tau):
            self.tau.tau = 0
        return self.tau.tau


    def _old_kendall(self, F_true, X, more=0, F_model=None):
        """return Kendall tau.

        TODO: possibly better interface, see _new_kendall

        `F_true` is the list of true f-values of the solutions in `X`.

        Computes tau between ``F_true + self.Y[:more]`` and
        ``self.eval(X + self.X[:more])``.

        Store correlation coefficient in ``self.tau.tau``.

        Argument `F_model` is only for computational efficiency.

        I "simple" usecase after the model update testing the first 15
        Y[i] (true values) versus model(X[i]) values::

            model._old_kendall([], [], 15)

        """
        # TODO:
        more = min((len(self.Y), more))
        F_true = F_true + [self.Y[i] for i in range(more)]
        F_model = [self.eval(X[i]) if F_model is None or F_model[i] is None
                       else F_model[i] for i in range(len(X))
                   ] + self.evalpop(self.X[i] for i in range(more))
        self.tau.n = len(F_true)
        self.tau.count = self.count
        if self.tau.n < 3:
            self.tau.result = None
            self.tau.tau, self.tau.pvalue = 0, 0
        else:
            self.tau.result = _kendalltau(F_true, F_model)
            try:
                self.tau.tau, self.tau.pvalue = self.tau.result[:2]
            except TypeError:  # kendalltau([3,3,3], [3,3,3]) == 1
                self.tau.tau, self.tau.pvalue = 0, 0
        if not np.isfinite(self.tau.tau):
            self.tau.tau = 0
        return self.tau.tau

    def isin(self, x):
        """return False if `x` is not (anymore) in the model archive"""
        hash = self._hash(x)
        return self.hashes.index(hash) + 1 if hash in self.hashes else False

    def _hash(self, x):
        return x[0], sum(x[1:])

    def mahalanobis_norm_squared(self, dx):
        """caveat: this can be negative because hessian is not guarantied
        to be pos def.
        """
        return np.dot(dx, np.dot(self.hessian, dx))

    @property
    def pinv(self):
        """should depend on something?"""
        try:
            self._pinv = np.linalg.pinv(self.weights * np.asarray(self.Z).T).T
        except np.linalg.LinAlgError as laerror:
            warnings.warn('Model.pinv(d=%d,m=%d,n=%d): np.linalg.pinv'
                          ' raised an exception %s' % (
                        len(self.X[0]) if self.X else -1,
                        len(self._coefficients),
                        len(self.X),
                        str(laerror)))
        return self._pinv

    @property
    def coefficients(self):
        if self._coefficients_count < self.count:
            self._coefficients_count = self.count
            self._coefficients = np.dot(self.pinv, self._weights * self.Y)
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
    def weights(self):
        self._weights = np.zeros(len(self.Y))
        idx = np.argsort(self.Y)
        self._weights[idx] = self.sorted_weights()
        assert np.all(np.argsort(self._weights) == idx[::-1])
        # self._weights = 1
        # self._weights = idx
        # ymax = max(self.Y)
        # dy = ymax - min(self.Y)
        # self._weights =  ((ymax - np.asarray(self.Y)) / dy if dy else 0)**self.wexpo + 1 / self.wdecay
        return self._weights

    @property
    def xopt(self):
        if self._xopt_count < self.count:
            self._xopt_count = self.count
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
        return min(self.F)

    @property
    def eigenvalues(self):
        return sorted(np.linalg.eigvals(self.hessian))

