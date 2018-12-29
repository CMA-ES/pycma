# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function  #, unicode_literals, with_statement
del absolute_import, division, print_function  #, unicode_literals, with_statement
___author__ = "Nikolaus Hansen"
__license__ = "BSD 3-clause"

import warnings
# from collections import deque
deque = list
import numpy as np
from scipy.stats import kendalltau as _kendalltau

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

    The `stack` method can be called several times per timestep, the
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
    def __init__(self, obj_or_name, attributes=None, callables=None, name=None):
        """obj can also be a name"""
        if obj_or_name == str(obj_or_name) and attributes is not None:
            raise ValueError('string obj %s has no attributes %s' % (
                str(obj_or_name), str(attributes)))
        self.name = name
        self._autoname(obj_or_name)
        self.obj = obj_or_name
        self.attributes = attributes or []
        self.callables = callables or []
        self.format = "%.19e"
        self.count = 0
        self.current_data = []
        # print('Logger:', self.name, self._name)

    def _autoname(self, obj):
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
        effective.
        """
        if obj is not None:
            self.obj = obj
        return self.push()

    def add(self, data):
        """data may be a value, or a `list`, or a `numpy` array, or...

        ...an object with attributes `self.attributes` iff self.attributes
        is not `None`.
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
            if self.attributes:
                file_.write('# %s\n' % repr(self.attributes))

    def load(self):
        self.data = np.loadtxt(self._name)
        return self

    def plot(self, plot=None):
        try:
            from matplotlib import pyplot as plt
        except ImportError: pass
        if plot is None:
            from matplotlib.pyplot import plot
        self.load()
        n = len(self.data)
        try:
            m = len(self.data[0])
        except TypeError:
            m = 0
        plt.gca().clear()
        if not m:
            plot(range(1, n + 1), self.data)
            return
        color=iter(plt.cm.winter_r(np.linspace(0.15, 1, m)))
        for i in range(m):
            plot(range(1, n + 1), self.data[:, i])
            plt.gca().get_lines()[0].set_color(next(color))
        return self

class SurrogatePopulation:
    """surrogate f-values for a population.

    What is new:

    - the model is global (compared to lmm-CMA)
    - the model has a linear component
    - the model is in the beginning linear and diagonal quadratic
    - the model uses the fitness ranks as weights.

    >>> import cma
    >>> import cma.fitness_models as fm
    >>> from cma.fitness_transformations import Function as FFun  # adds evaluations attribute
    >>> fm.Logger = fm.LoggerDummy
    >>> surrogate = fm.SurrogatePopulation(cma.ff.elli)

    Example using the ask-and-tell interface:

    >>> for fitfun, evals in [[FFun(cma.ff.elli), 21],
    ...                      [FFun(cma.ff.sectorsphere), 122]]:
    ...     es = cma.CMAEvolutionStrategy(5 * [1], 2.2,
    ...                    {'CMA_injections_threshold_keep_len': 1,
    ...                     'ftarget':1e-9, 'verbose': -9, 'seed':5})
    ...     surrogate = fm.SurrogatePopulation(fitfun)
    ...     while not es.stop():
    ...         X = es.ask()
    ...         es.tell(X, surrogate(X))  # surrogate evaluation
    ...         es.inject([surrogate.model.xopt])
    ...         # es.disp(); es.logger.add()  # ineffective with verbose=-9
    ...     assert 'ftarget' in es.stop()
    ...     assert fitfun.evaluations <= evals
    ...     # print(fitfun.evaluations)

    Example using the ``parallel_objective`` interface to `cma.fmin`:

    >>> for fitfun, evals in [[cma.ff.elli, 22], [cma.ff.ellirot, 40]]:
    ...     surrogate = fm.SurrogatePopulation(fitfun)
    ...     inject_xopt = fm.ModelInjectionCallback(surrogate.model)  # must use the same model
    ...     xopt, es = cma.fmin2(None, 5 * [1], 2.2,
    ...                      {'CMA_injections_threshold_keep_len': 1,
    ...                       'ftarget':1e-12, 'verbose': -9},
    ...                      parallel_objective=surrogate,
    ...                      callback=inject_xopt)
    ...     assert es.result[1] < 1e-12
    ...     assert es.result[2] < evals
    ...     # print(fitfun.evaluations)

    """
    def __init__(self,
                 fitness,
                 model=None,
                 model_size_factor=3,
                 tau_truth_threshold=0.9,
                 eval_xopt_condition=False):
        """

        :param fitness:
        :param model: fitness function model
        :param model_size_factor: population size multiplier to possibly increase the maximal model size
        :param tau_truth_threshold:
        :param eval_xopt_condition:

        If ``model is None``, a default `Model` instance is used. By
        setting `self.model` to `None`, only the `fitness` for each
        population member is evaluated in each call.

        """
        self.minimum_n_for_tau = 5
        self.fitness = fitness
        self.model = model if model else Model()
        self.model_size_factor = model_size_factor
        self.eval_xopt_condition = eval_xopt_condition
        self.change_threshold = -1.0  # tau between previous and new model; was: 0.8
        self.truth_threshold = tau_truth_threshold  # tau between model and ground truth
        self.count = 0
        self.last_evaluations = 0
        self.evaluations = 0
        self.logger = Logger(self, ['evaluations'])

    def _number_of_evaluations_to_do_now(self, iloop):
        """return ``(iloop - 1)**2 + 2``.

        ::
            n = arange(1, 8)
            n1 = n**2
            n2 = hstack([1, n**2 + 1])
            n3 = hstack([1, (n - 1)**2 + 2])  # current choice = sign(n - 1) * (n - 1)**2 + 2
            n4 = hstack([1, (n - 1)**2 + 1])
            # n1, cumsum(n1), aa(100 * n1[1:] / cumsum(n1)[:-1], dtype=int), n2, cumsum(n2), aa(100 * n2[1:] / cumsum(n2)[:-1], dtype=int)
            print("current_evaluations_starting_from_one + number_of_additional_evaluations=n(loop=1,2,3,...)")
            for nn in [n1, n2, n3, n4]:
                for n, s in zip(nn[1:], cumsum(nn)[:-1]):
                    s = '%d+%d' % (s, n)
                    s = (5 - len(s)) * ' ' + s
                    print(s, end=' ')
                print()

              1+4   5+9 14+16 30+25 55+36 91+49
              1+2   3+5  8+10 18+17 35+26 61+37 98+50
              1+2   3+3   6+6 12+11 23+18 41+27 68+38
              1+1   2+2   4+5  9+10 19+17 36+26 62+37

        """
        # already_done + do_now = 0+1 1+2 3+3 6+6 12+11 23+18 41+27 68+38 106+51
        return (iloop - 1)**2 + 2 if iloop > 0 else 1
        # already_done + do_now = 0+1 1+2 3+5 8+10 18+17 35+26 61+37 98+50
        return 1 + iloop**2

    def __call__(self, X):
        """return population f-values.

        The smallest value is never smaller than a truly evaluated value.

        """
        if self.model is None:
            return [self.fitness(x) for x in X]
        model = self.model
        if self.model_size_factor * len(X) > 1.1 * self.model.max_absolute_size:
            model.max_absolute_size = self.model_size_factor * len(X)
            model.reset()
        F_true = {}
        # need at least two evaluations for a non-flat linear model
        for k in range(len(X)):
            if len(model.X) > 1:
                break
            F_true[k] = self.fitness(X[k])
            model.add_data_row(X[k], F_true[k])
        F_model = [model.eval(x) for x in X]
        if F_true:
            offset = min(model.Y) - min(F_model)
            return [f + offset for f in F_model]

        sidx0 = np.argsort(F_model)
        # making no evaluation at all should not save too many evaluations
        # if in some iterations many points need to be evaluated

        # TODO: find a smarter order depending on F_model?
        # eidx = range(len(X))  # indices to be evaluated
        eidx = sidx0[:]
        iloop = 0
        i1 = 0  # i1-1 is last evaluation index
        for iloop in range(len(eidx)):
            i0 = i1
            i1 += self._number_of_evaluations_to_do_now(iloop)
            for k in eidx[i0:i1]:
                F_true[k] = self.fitness(X[k])
                model.add_data_row(X[k], F_true[k])
            if i1 >= len(eidx):
                break
            assert i1 < len(eidx)

            # the model has changed, so we recompute surrogate f-values
            F_model = [model.eval(x) for x in X]
            sidx = np.argsort(F_model)
            # TODO: is it enough to just check whether sidx0 and sidx agree?
            # to find out we want to log the kendall of sidx0 vs sidx and
            # the tau computed below

            # kendall compares F_true[k] with model.eval(X[k]) ranks
            # TODO: we would not need to recompute model.eval(X)
            # TODO: with large popsize we do not want all solutions in kendall, but only the best popsize/3 of this iteration?
            tau = model.kendall(
                [F_true[k] for k in sorted(F_true)],  # sorted is used to get k in a deterministic order
                [X[k] for k in sorted(F_true)],
                self.minimum_n_for_tau - len(F_true),
                # [F_model[k] for k in sorted(F_true)  # check that this is correct
            )  # take also last few
            if iloop == 0:
                self.logger.add(tau)
            if _kendalltau(sidx0, sidx)[0] > self.change_threshold and tau > self.truth_threshold:
                break
            sidx0 = sidx
            # TODO: we could also reconsider the order eidx which to compute next

        self.logger.add(tau)
        self.last_evaluations = len(F_true)
        self.evaluations += self.last_evaluations
        if self.eval_xopt_condition >= 1:  # this fails, because xopt may be very far astray
            evs = sorted(model.eigenvalues)
            if evs[0] > 0 and evs[-1] <= self.eval_xopt_condition * evs[0]:
                model.add_data_row(model.xopt, self.fitness(model.xopt))
                self.evaluations += 1
                F_model = [model.eval(x) for x in X]
        # TODO: is global sorting better?
        model.sort_(self.evaluations)  # crop worse solutions first, but keep iteration order
        if self.evaluations == 0:  # can currently not happen
            # a hack to have some grasp on zero evaluations from outside
            self.evaluations = 1e-2  # hundred zero=iterations sum to one evaluation
        # self.logs['evaluations'].push()
        self.logger.push()  # TODO: check that we do not miss anything below
        if len(X) == len(F_true):
            # model.set_xoffset(model.xopt)
            return [F_true[i] for i in range(len(X))]

        # get argmin(F_true)
        # i_min, f_min = min(F_true.items(), key=lambda x: x[1])  # is about 1.5 times slower
        f_min = np.inf
        for i in F_true:
            if F_true[i] < f_min:
                i_min, f_min = i, F_true[i]

        # mix F_model and F_true
        # TODO (depending on correlation threshold that rarely happens anyways):
        #     correct for false model ranks, but how?
        offset = F_true[i_min] - min(F_model)  # such that no value is below F_true[i_min]
        return [F_model[i] + offset for i in range(len(X))]

class ModelInjectionCallback:
    """inject `model.xopt` and decrease `sigma` if `mean` is close to `model.xopt`.

    New, simpler callback class.

    Sigma decrease saves (only) 30% on the 10-D ellipsoid.
    """
    def __init__(self, model, sigma_distance_lower_threshold=0, sigma_factor=1/1.1):
        """sigma_distance_lower_threshold=0 means decrease never"""
        self.model = model
        self.sigma_distance_threshold = sigma_distance_lower_threshold
        self.sigma_factor = sigma_factor
        self.logger = Logger(self)
    def __call__(self, es):
        es.inject([self.model.xopt])
        xdist = es.mahalanobis_norm(self.model.xopt - es.mean)
        self.logger.add(self.sigma_distance_threshold * es.N**0.5 / es.sp.weights.mueff)
        if xdist < self.sigma_distance_threshold * es.N**0.5 / es.sp.weights.mueff:
            es.sigma *= self.sigma_factor
            self.logger.add(xdist).push()
        else:
            self.logger.add(-xdist).push()

class Tau: "placeholder to store Kendall tau related things"

# TODO: check that `xopt` gives OK value if hessian is negative definite
class Model:
    """Full quadratic model heavily using the pseudo inverse.

    The full model has 2n + n(n-1)/2 + 1 = n(n+3) + 1 parameters. Model
    building "works" with any number of data.

    Model size 1 doesn't work well on bbob-f10, 1.5 is the minimum that
    works.

    >>> import numpy as np
    >>> import cma
    >>> import cma.fitness_models as fm
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
    ...   ])
    >>> assert np.allclose(m.xopt, [ 1.2,  1.2,  1.2])
    >>> assert np.allclose(m.xopt, [ 1.2,  1.2,  1.2])

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

    """
    @staticmethod
    def sorted_weights(len_):
        return np.linspace(20, 1, len_)

    def __init__(self,
                 max_relative_size=3,
                 min_relative_size=1.5,
                 max_absolute_size=None,
                 ):
        """

        :param max_relative_size:
        :param max_absolute_size:
        :param min_relative_size:


        Increase model complexity if the number of data exceeds
        ``min_relative_size * df_bigger_model_type``.

        Limit the number of kept data
        ``max(max_absolute_size, max_relative_size * max_df)``.

        """
        self.max_relative_size = max_relative_size
        self.min_relative_size = min_relative_size
        self.max_absolute_size = max_absolute_size \
                                   if max_absolute_size is not None else 0
        if not 1 <= min_relative_size <= max_relative_size:
            raise ValueError(
                'need max_relative_size=%f >= min_relative_size=%f >= 1' %
                (max_relative_size, min_relative_size))
        self.reset()

    def reset(self):
        self.type = {}  # ['quadratic', 'full']
        self.X = deque()
        self.Y = deque()
        self.Z = deque()
        self.counts = deque()
        """time stamp for data row"""
        self.hashes = deque()
        """reference to x-value quick-and-dirty instead of a hash"""
        self._fieldnames = ['X', 'Y', 'Z', 'counts', 'hashes']
        self.count = 0  # number of overall data seen
        self._coefficients_count = -1
        self._xopt_count = -1
        self._xoffset = 0
        self.number_of_data_last_added = 0  # sweep of data added
        self.tau = Tau()
        self.tau.tau, self.tau.n = 0, 0
        # self.logger = ModelLogger()

    @property
    def logging_trace(self):
        if len(self.X) < 2:
            return [1, 1, 1, 1]
        trace = []
        d1 = self.X[0] - self.X[1]
        d2 = self.X[0] - self.xopt
        trace += [self.mahalanobis_norm(d1),
                  self.mahalanobis_norm(d2)]
        trace += [sum(d1**2)**0.5, sum(d2**2)**0.5]
        return trace

    @property
    def max_size(self):
        def df(d):
            return d * (d + 3) / 2 + 1
        return max((df(len(self.X[0])) * self.max_relative_size,
                    self.max_absolute_size))

    @property
    def max_df(self):
        d = len(self.X[0])
        return d * (d + 3) / 2 + 1

    def update_type(self):
        """depending on the number of observed data"""
        if not len(self.X):
            return
        n, d = len(self.X), len(self.X[0])
        # d + 1 affine linear coefficients are always computed
        def dfquadratic(d): return 2 * d + 1
        def dffull(d): return d * (d + 3) / 2 + 1

        if n >= dfquadratic(d) * self.min_relative_size:
            if 'quadratic' not in self.type:
                self.type.update({'quadratic': dfquadratic})
                self.reset_Z()
            if n >= dffull(d) * self.min_relative_size:
                if 'full' not in self.type:
                    self.type.update({'full': dffull})
                    self.reset_Z()

    def add_data_row(self, x, f):
        hash = self._hash(x)
        if hash in self.hashes:
            warnings.warn("x value already in Model")
            return
        self.count += 1
        x = np.asarray(x)
        self.X.insert(0, x)
        self.Y.insert(0, f)
        self.Z.insert(0, self.expand_x(x))
        self.counts.insert(0, self.count)
        self.hashes.insert(0, hash)
        # n = len(self.X[0])
        # m = n * (n + 3) / 2 + 1
        while len(self.X) > self.max_size:
            for name in self._fieldnames:
                getattr(self, name).pop()
        self.number_of_data_last_added = 1
        self.update_type()
        # self.logger.add(self)
        return self

    def add_data(self, X, Y):
        """a sequence of x- and y-data"""
        if len(X) != len(Y):
            raise ValueError("X and Y have different lengths %d!=%d", (len(X), len(Y)))
        idx = np.argsort(Y)[::-1]
        for i in idx:  # insert smallest/best last
            self.add_data_row(X[i], Y[i])
        self.number_of_data_last_added = len(X)
        return self

    def sort_(self, number=None, argsort=np.argsort):
        """sort last `number` entries"""
        if number is None:
            number = len(self.X)
        if number <= 0:
            return self
        number = min((number, len(self.Y)))
        idx = argsort([self.Y[i] for i in range(number)])  # [:number] doesn't work on deque's
        for name in self._fieldnames:
            field = getattr(self, name)
            tmp = [field[i] for i in range(number)]
            for i in range(len(idx)):
                field[i] = tmp[idx[i]]

    def xmean(self):
        return np.mean(self.X, axis=0)

    def set_xoffset(self, offset):
        self._xoffset = np.asarray(offset)
        self.reset_Z()

    def reset_Z(self):
        """set x-values Z attribute"""
        self.Z = deque(self.expand_x(x) for x in self.X)
        self._coefficients_count = -1
        self._xopt_count = -1

    def expand_x(self, x):
        x = np.asarray(x) + self._xoffset
        z = [1]
        z += list(x)
        if 'quadratic' in self.type:
            z += list(np.square(x))
            if 'full' in self.type:
                z += (x[i] * x[j] for i in range(len(x)) for j in range(len(x)) if i < j)
        return z

    def eval_true(self, x, max_number=None):
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
            if sum((xopt_old - self.xopt)**2) < 1e-3 * sum((self.X[1] - self.X[0])**2):
                x_new = (self.X[1] - self.X[0]) / 2
                self.add_data_row(x_new, fitness(x_new))
        return self.xopt, self

    def _new_kendall(self, number, F_true=None, F_model=None, X=None):
        """return Kendall tau."""

    def kendall(self, F_true, X, more=0, F_model=None):
        """return Kendall tau.

        TODO: possibly better interface, see _new_kendall

        `F_true` is the list of true f-values of the solutions in `X`.

        Computes tau between ``F_true + self.Y[:more]`` and
        ``self.eval(X + self.X[:more])``.

        Store correlation coefficient in ``self.tau.tau``.

        Argument `F_model` is only for computational efficiency.

        I "simple" usecase after the model update testing the first 15
        Y[i] (true values) versus model(X[i]) values::

            model.kendall([], [], 15)

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

    def _hash(self, x):
        return x[0], sum(x[1:])

    def mahalanobis_norm(self, dx):
        return np.sqrt(np.dot(dx, np.dot(self.hessian, dx)))

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
        return self._coefficients

    @property
    def hessian(self):
        d = len(self.X[0])
        m = len(self.coefficients)
        assert m in (d + 1, 2 * d + 1, d * (d + 3) / 2 + 1)
        assert m in [d + 1] + [complexity(d) for complexity in self.type.values()]
        H = np.zeros((d, d))  # TODO: use (sparse) diagonal matrix if 'full' not in self.type
        k = 2 * d + 1
        for i in range(d):
            if m > d + 1:
                assert 'quadratic' in self.type
                H[i, i] = self.coefficients[d + i + 1]
            else:
                assert m == d + 1
                H[i, i] = self.coefficients[i + 1] / 100  # arbitrary factor to make xopt finite
            if m > 2 * d + 1:
                assert m == self.type['full'](d)
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
        self._weights[idx] = Model.sorted_weights(len(self.Y))
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
            self._xopt = np.dot(np.linalg.pinv(self.hessian), self.b / -2.) - self._xoffset
        return self._xopt

    @property
    def eigenvalues(self):
        return sorted(np.linalg.eigvals(self.hessian))

