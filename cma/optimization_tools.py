"""Utility classes and functionalities loosely related to optimization
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
import numpy as np
try: from matplotlib import pyplot as plt
except: pass
from .utilities.utils import BlancClass as _BlancClass
from .utilities.math import Mh
# from .transformations import BoundTransform  # only to make it visible but gives circular import anyways
from .utilities.python3for2 import range
del absolute_import, division, print_function  #, unicode_literals

def semilogy_signed(x=None, y=None, yoffset=0, minabsy=None, iabscissa=1,
                    **kwargs):
    """signed semilogy plot.

    `y` (or `x` if `y` is `None`) is a data array, by default read from
    `outcmaesxmean.dat` or (first) from the default logger output file
    like::

        xy = cma.logger.CMADataLogger().load().data['xmean']
        x, y = xy[:, iabscissa], xy[:, 5:]
        semilogy_signed(x, y)

    Plotted is `y - yoffset` vs `x` for positive values as a semilogy plot
    and for negative values as a semilogy plot of absolute values with
    inverted axis.

    `minabsy` controls the minimum shown value away from zero, which can
    be useful if extremely small non-zero values occur in the data.

    """
    if y is None:
        if x is not None:
            x, y = y, x
        else:
            try:
                from . import logger
                xy = logger.CMADataLogger().load().data['xmean']
            except:
                xy = np.loadtxt('outcmaesxmean.dat', comments=('%',))
            x, y = xy[:, iabscissa], xy[:, 5:]
    y = np.array(y, copy=True)  # not always necessary, but sometimes?
    if yoffset not in (None, 0):
        try:
            y -= yoffset
        except:  # recycle last entry of yoffset
            yoffset = [yoffset[i if i < len(yoffset) else -1] for i in range(y.shape[1])]
            y -= yoffset
    elif 11 < 3:
        pass  # TODO: subtract optionally last x!? (not smallest which is done anyways)
    min_log = np.log10(minabsy) if minabsy else \
              int(np.floor(np.min(np.log10(np.abs(y[y!=0])))))

    idx_zeros = np.abs(y) < 10**min_log
    idx_pos = y >= 10**min_log
    idx_neg = y <= -10**min_log
    y[idx_pos] = np.log10(y[idx_pos]) - min_log
    y[idx_neg] = -(np.log10(-y[idx_neg]) - min_log)
    y[idx_zeros] = 0

    if x is None:
        x = range(1, y.shape[0] + 1)
    plt.plot(x, y, **kwargs)

    # the remainder is changing y-labels
    ax = plt.gca()
    ticks, labels = [], []
    for val in ax.get_yticks():
        s = (r"$10^{%.2f}$") % (val + min_log)
        if val < 0:
            s = (r"$-10^{%.2f}$") % (-val + min_log)
        elif val == 0:
            s = (r"$\pm10^{%.2f}$") % min_log
        if '.' in s:
            while s[-3] == '0':  # remove trailing zeros
                s = s[:-3] + s[-2:]
            if s[-3] == '.':  # remove trailing dot
                s = s[:-3] + s[-2:]
        labels += [s]
        ticks += [val]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    plt.grid(True)

def contour_data(fct, x_range, y_range=None):
    """generate x,y,z-data for contour plot.

    `fct` is a 2-D function.
    `x`- and `y_range` are `iterable`s (e.g. `list`s or arrays)
    to define the meshgrid.

    CAVEAT: this function calls `fct` ``len(list(x_range)) * len(list(y_range))``
    times. Hence using `Sections` may be the better first choice to
    investigate an expensive function.

    Example:

    >>> import numpy as np
    ...
    >>> def example():  # def avoids doctest execution
    ...     from matplotlib import pyplot as plt
    ...
    ...     X, Y, Z = contour_data(lambda x: sum([xi**2 for xi in x]),
    ...                            np.arange(0.90, 1.10, 0.02),
    ...                            np.arange(-0.10, 0.10, 0.02))
    ...     CS = plt.contour(X, Y, Z)
    ...     plt.axes().set_aspect('equal')
    ...     plt.clabel(CS)

    See `cma.fitness_transformations.FixVariables` to create a 2-D
    function from a d-D function, e.g. like

    >>> import cma
    ...
    >>> fd = cma.ff.elli
    >>> x0 = np.zeros(22)
    >>> indices_to_vary = [2, 4]
    >>> f2 = cma.fitness_transformations.FixVariables(fd,
    ...          dict((i, x0[i]) for i in range(len(x0))
    ...                          if i not in indices_to_vary))
    >>> isinstance(f2, cma.fitness_transformations.FixVariables)
    True
    >>> isinstance(f2, cma.fitness_transformations.ComposedFunction)
    True
    >>> f2[0] is fd, len(f2) == 2
    (True, True)

    """
    if y_range is None:
        y_range = x_range
    X, Y = np.meshgrid(x_range, y_range)
    Z = X.copy()
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i][j] = fct(np.asarray([X[i][j], Y[i][j]]))
    return X, Y, Z

# ecdf_data
def step_data(data, smooth_corners=0.1):

    """return x, y ECDF data for ECDF plot. Smoothing may look strange
    in a semilogx plot.
    """
    x = np.asarray(sorted(data))
    y = np.linspace(0, 1, len(x) + 1, endpoint=True)
    if smooth_corners:
        x = np.array([x - smooth_corners * np.hstack([[0], np.diff(x)]),
                      x, x, x + smooth_corners * np.hstack([np.diff(x), [0]])])
    else:
        x = np.array([x, x])
    x = x.reshape(x.size, order='F')
    if smooth_corners:
        y = np.array([y[:-1], (1 - smooth_corners) * y[:-1] + smooth_corners * y[1:],
                      smooth_corners * y[:-1] + (1 - smooth_corners) * y[1:], y[1:]])
    else:
            y = np.array([y[:-1], y[1:]])
    y = y.reshape(y.size, order='F')
    # y = np.linspace(0, 1, len(x), endpoint=True)
    return x, y

class BestSolution(object):
    """container to keep track of the best solution seen.

    Keeps also track of the genotype, if available.
    """
    def __init__(self, x=None, f=np.inf, evals=None):
        """initialize the best solution with ``x``, ``f``, and ``evals``.

        Better solutions have smaller ``f``-values.
        """
        self.x = x
        self.x_geno = None
        self.f = f if f is not None and f is not np.nan else np.inf
        self.evals = evals
        self.evalsall = evals
        self.last = _BlancClass()
        self.last.x = x
        self.last.f = f
    def update(self, arx, xarchive=None, arf=None, evals=None):
        """checks for better solutions in list ``arx``.

        Based on the smallest corresponding value in ``arf``,
        alternatively, `update` may be called with a `BestSolution`
        instance like ``update(another_best_solution)`` in which case
        the better solution becomes the current best.

        ``xarchive`` is used to retrieve the genotype of a solution.
        """
        if isinstance(arx, BestSolution):
            if self.evalsall is None:
                self.evalsall = arx.evalsall
            elif arx.evalsall is not None:
                self.evalsall = max((self.evalsall, arx.evalsall))
            if arx.f is not None and arx.f < np.inf:
                self.update([arx.x], xarchive, [arx.f], arx.evals)
            return self
        assert arf is not None
        # find failsave minimum
        try:
            minidx = np.nanargmin(arf)
        except ValueError:
            return
        if minidx is np.nan:
            return
        minarf = arf[minidx]
        # minarf = reduce(lambda x, y: y if y and y is not np.nan
        #                   and y < x else x, arf, np.inf)
        if minarf < np.inf and (minarf < self.f or self.f is None):
            self.x, self.f = arx[minidx], arf[minidx]
            if xarchive is not None and xarchive.get(self.x) is not None:
                self.x_geno = xarchive[self.x].get('geno')
            else:
                self.x_geno = None
            self.evals = None if not evals else evals - len(arf) + minidx + 1
            self.evalsall = evals
        elif evals:
            self.evalsall = evals
        self.last.x = arx[minidx]
        self.last.f = minarf
    def get(self):
        """return ``(x, f, evals)`` """
        return self.x, self.f, self.evals  # , self.x_geno

class EvolutionPath(object):
    """not in use (yet)

    A variance-neutral exponentially smoothened vector.
    """
    def __init__(self, p0, time_constant=None):
        self.path = np.asarray(p0)
        self.count = 0
        self.time_constant = time_constant
        if time_constant is None:
            self.time_constant = 1 + len(p0)**0.5
    def update(self, v):
        self.count += 1
        c = max((1 / self.count, 1. / self.time_constant))
        self.path *= 1 - c
        self.path += (c * (2 - c))**0.5 * np.asarray(v)

class NoiseHandler(object):
    """Noise handling according to [Hansen et al 2009, A Method for
    Handling Uncertainty in Evolutionary Optimization...]

    The interface of this class is yet versatile and subject to changes.

    The noise handling follows closely [Hansen et al 2009] in the
    measurement part, but the implemented treatment is slightly
    different: for ``noiseS > 0``, ``evaluations`` (time) and sigma are
    increased by ``alpha``. For ``noiseS < 0``, ``evaluations`` (time)
    is decreased by ``alpha**(1/4)``.

    The (second) parameter ``evaluations`` defines the maximal number
    of evaluations for a single fitness computation. If it is a list,
    the smallest element defines the minimal number and if the list has
    three elements, the median value is the start value for
    ``evaluations``.

    `NoiseHandler` serves to control the noise via steps-size
    increase and number of re-evaluations, for example via `fmin` or
    with `ask_and_eval`.

    Examples
    --------
    Minimal example together with `fmin` on a non-noisy function:

    >>> import cma
    >>> res = cma.fmin(cma.ff.elli, 7 * [1], 1, noise_handler=cma.NoiseHandler(7))  #doctest: +ELLIPSIS
    (4_w,9)-aCMA-ES (mu_w=2.8,...
    >>> assert res[1] < 1e-8
    >>> res = cma.fmin(cma.ff.elli, 6 * [1], 1, {'AdaptSigma':cma.sigma_adaptation.CMAAdaptSigmaTPA},
    ...          noise_handler=cma.NoiseHandler(6))  #doctest: +ELLIPSIS
    (4_w,...
    >>> assert res[1] < 1e-8

    in dimension 7 (which needs to be given tice). More verbose example
    in the optimization loop with a noisy function defined in ``func``:

    >>> import cma, numpy as np
    >>> func = lambda x: cma.ff.sphere(x) * (1 + 4 * np.random.randn() / len(x))  # cma.ff.noisysphere
    >>> es = cma.CMAEvolutionStrategy(np.ones(10), 1)  #doctest: +ELLIPSIS
    (5_w,10)-aCMA-ES (mu_w=3.2,...
    >>> nh = cma.NoiseHandler(es.N, maxevals=[1, 1, 30])
    >>> while not es.stop():
    ...     X, fit_vals = es.ask_and_eval(func, evaluations=nh.evaluations)
    ...     es.tell(X, fit_vals)  # prepare for next iteration
    ...     es.sigma *= nh(X, fit_vals, func, es.ask)  # see method __call__
    ...     es.countevals += nh.evaluations_just_done  # this is a hack, not important though
    ...     es.logger.add(more_data = [nh.evaluations, nh.noiseS])  # add a data point
    ...     es.disp()
    ...     # nh.maxevals = ...  it might be useful to start with smaller values and then increase
    ...                # doctest: +ELLIPSIS
    Iterat...
    >>> print(es.stop())
    ...                # doctest: +ELLIPSIS
    {...
    >>> print(es.result[-2])  # take mean value, the best solution is totally off
    ...                # doctest: +ELLIPSIS
    [...
    >>> assert sum(es.result[-2]**2) < 1e-9
    >>> print(X[np.argmin(fit_vals)])  # not bad, but probably worse than the mean
    ...                # doctest: +ELLIPSIS
    [...

    >>> # es.logger.plot()


    The command ``logger.plot()`` will plot the logged data.

    The noise options of fmin` control a `NoiseHandler` instance
    similar to this example. The command ``cma.CMAOptions('noise')``
    lists in effect the parameters of `__init__` apart from
    ``aggregate``.

    Details
    -------
    The parameters reevals, theta, c_s, and alpha_t are set differently
    than in the original publication, see method `__init__`. For a
    very small population size, say popsize <= 5, the measurement
    technique based on rank changes is likely to fail.

    Missing Features
    ----------------
    In case no noise is found, ``self.lam_reeval`` should be adaptive
    and get at least as low as 1 (however the possible savings from this
    are rather limited). Another option might be to decide during the
    first call by a quantitative analysis of fitness values whether
    ``lam_reeval`` is set to zero. More generally, an automatic noise
    mode detection might also set the covariance matrix learning rates
    to smaller values.

    :See also: `fmin`, `CMAEvolutionStrategy.ask_and_eval`

    """
    # TODO: for const additive noise a better version might be with alphasigma also used for sigma-increment,
    # while all other variance changing sources are removed (because they are intrinsically biased). Then
    # using kappa to get convergence (with unit sphere samples): noiseS=0 leads to a certain kappa increasing rate?
    def __init__(self, N, maxevals=[1, 1, 1], aggregate=np.median,
                 reevals=None, epsilon=1e-7, parallel=False):
        """Parameters are:

        ``N``
            dimension, (only) necessary to adjust the internal
            "alpha"-parameters
        ``maxevals``
            maximal value for ``self.evaluations``, where
            ``self.evaluations`` function calls are aggregated for
            noise treatment. With ``maxevals == 0`` the noise
            handler is (temporarily) "switched off". If `maxevals`
            is a list, min value and (for >2 elements) median are
            used to define minimal and initial value of
            ``self.evaluations``. Choosing ``maxevals > 1`` is only
            reasonable, if also the original ``fit`` values (that
            are passed to `__call__`) are computed by aggregation of
            ``self.evaluations`` values (otherwise the values are
            not comparable), as it is done within `fmin`.
        ``aggregate``
            function to aggregate single f-values to a 'fitness', e.g.
            ``np.median``.
        ``reevals``
            number of solutions to be reevaluated for noise
            measurement, can be a float, by default set to ``2 +
            popsize/20``, where ``popsize = len(fit)`` in
            ``__call__``. zero switches noise handling off.
        ``epsilon``
            multiplier for perturbation of the reevaluated solutions
        ``parallel``
            a single f-call with all resampled solutions

        :See also: `fmin`, `CMAOptions`, `CMAEvolutionStrategy.ask_and_eval`

        """
        self.lam_reeval = reevals  # 2 + popsize/20, see method indices(), originally 2 + popsize/10
        self.epsilon = epsilon
        self.parallel = parallel
        ## meta_parameters.noise_theta == 0.5
        self.theta = 0.5  # 0.5  # originally 0.2
        self.cum = 0.3  # originally 1, 0.3 allows one disagreement of current point with resulting noiseS
        ## meta_parameters.noise_alphasigma == 2.0
        self.alphasigma = 1 + 2.0 / (N + 10) # 2, unit sphere sampling: 1 + 1 / (N + 10)
        ## meta_parameters.noise_alphaevals == 2.0
        self.alphaevals = 1 + 2.0 / (N + 10)  # 2, originally 1.5
        ## meta_parameters.noise_alphaevalsdown_exponent == -0.25
        self.alphaevalsdown = self.alphaevals** -0.25  # originally 1/1.5
        # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        if 11 < 3 and maxevals[2] > 1e18:  # for testing purpose
            self.alphaevals = 1.5
            self.alphaevalsdown = self.alphaevals**-0.999  # originally 1/1.5

        self.evaluations = 1
        """number of f-evaluations to get a single measurement by aggregation"""
        self.minevals = 1
        self.maxevals = int(np.max(maxevals))
        if hasattr(maxevals, '__contains__'):  # i.e. can deal with ``in``
            if len(maxevals) > 1:
                self.minevals = min(maxevals)
                self.evaluations = self.minevals
            if len(maxevals) > 2:
                self.evaluations = np.median(maxevals)
        ## meta_parameters.noise_aggregate == None
        self.f_aggregate = aggregate if not None else {1: np.median, 2: np.mean}[ None ]
        self.evaluations_just_done = 0  # actually conducted evals, only for documentation
        self.noiseS = 0

    def __call__(self, X, fit, func, ask=None, args=()):
        """proceed with noise measurement, set anew attributes ``evaluations``
        (proposed number of evaluations to "treat" noise) and ``evaluations_just_done``
        and return a factor for increasing sigma.

        Parameters
        ----------
        ``X``
            a list/sequence/vector of solutions
        ``fit``
            the respective list of function values
        ``func``
            the objective function, ``fit[i]`` corresponds to
            ``func(X[i], *args)``
        ``ask``
            a method to generate a new, slightly disturbed solution. The
            argument is (only) mandatory if ``epsilon`` is not zero, see
            `__init__`.
        ``args``
            optional additional arguments to ``func``

        Details
        -------
        Calls the methods `reeval`, `update_measure` and ``treat` in
        this order. ``self.evaluations`` is adapted within the method
        `treat`.

        """
        self.evaluations_just_done = 0
        if not self.maxevals or self.lam_reeval == 0:
            return 1.0
        res = self.reeval(X, fit, func, ask, args)
        if not len(res):
            return 1.0
        self.update_measure()
        return self.treat()

    def treat(self):
        """adapt self.evaluations depending on the current measurement
        value and return ``sigma_fac in (1.0, self.alphasigma)``

        """
        if self.noiseS > 0:
            self.evaluations = min((self.evaluations * self.alphaevals, self.maxevals))
            return self.alphasigma
        else:
            self.evaluations = max((self.evaluations * self.alphaevalsdown, self.minevals))
            return 1.0  # / self.alphasigma

    def reeval(self, X, fit, func, ask, args=()):
        """store two fitness lists, `fit` and ``fitre`` reevaluating some
        solutions in `X`.
        ``self.evaluations`` evaluations are done for each reevaluated
        fitness value.
        See `__call__`, where `reeval` is called.

        """
        self.fit = list(fit)
        self.fitre = list(fit)
        self.idx = self.indices(fit)
        if not len(self.idx):
            return self.idx
        evals = int(self.evaluations) if self.f_aggregate else 1
        fagg = np.median if self.f_aggregate is None else self.f_aggregate
        for i in self.idx:
            X_i = X[i]
            if self.epsilon:
                if self.parallel:
                    self.fitre[i] = fagg(func(ask(evals, X_i, self.epsilon), *args))
                else:
                    self.fitre[i] = fagg([func(ask(1, X_i, self.epsilon)[0], *args)
                                            for _k in range(evals)])
            else:
                self.fitre[i] = fagg([func(X_i, *args) for _k in range(evals)])
        self.evaluations_just_done = evals * len(self.idx)
        return self.fit, self.fitre, self.idx

    def update_measure(self):
        """updated noise level measure using two fitness lists ``self.fit`` and
        ``self.fitre``, return ``self.noiseS, all_individual_measures``.

        Assumes that ``self.idx`` contains the indices where the fitness
        lists differ.

        """
        lam = len(self.fit)
        idx = np.argsort(self.fit + self.fitre)
        ranks = np.argsort(idx).reshape((2, lam))
        rankDelta = ranks[0] - ranks[1] - np.sign(ranks[0] - ranks[1])

        # compute rank change limits using both ranks[0] and ranks[1]
        r = np.arange(1, 2 * lam)  # 2 * lam - 2 elements
        limits = [0.5 * (Mh.prctile(np.abs(r - (ranks[0, i] + 1 - (ranks[0, i] > ranks[1, i]))),
                                      self.theta * 50) +
                         Mh.prctile(np.abs(r - (ranks[1, i] + 1 - (ranks[1, i] > ranks[0, i]))),
                                      self.theta * 50))
                    for i in self.idx]
        # compute measurement
        #                               max: 1 rankchange in 2*lambda is always fine
        s = np.abs(rankDelta[self.idx]) - Mh.amax(limits, 1)  # lives roughly in 0..2*lambda
        self.noiseS += self.cum * (np.mean(s) - self.noiseS)
        return self.noiseS, s

    def indices(self, fit):
        """return the set of indices to be reevaluated for noise
        measurement.

        Given the first values are the earliest, this is a useful policy
        also with a time changing objective.

        """
        ## meta_parameters.noise_reeval_multiplier == 1.0
        lam_reev = 1.0 * (self.lam_reeval if self.lam_reeval
                            else 2 + len(fit) / 20)
        lam_reev = int(lam_reev) + ((lam_reev % 1) > np.random.rand())
        ## meta_parameters.noise_choose_reeval == 1
        choice = 1
        if choice == 1:
            # take n_first first and reev - n_first best of the remaining
            n_first = lam_reev - lam_reev // 2
            sort_idx = np.argsort(np.array(fit, copy=False)[n_first:]) + n_first
            return np.array(list(range(0, n_first)) +
                            list(sort_idx[0:lam_reev - n_first]), copy=False)
        elif choice == 2:
            idx_sorted = np.argsort(np.array(fit, copy=False))
            # take lam_reev equally spaced, starting with best
            linsp = np.linspace(0, len(fit) - len(fit) / lam_reev, lam_reev)
            return idx_sorted[[int(i) for i in linsp]]
        # take the ``lam_reeval`` best from the first ``2 * lam_reeval + 2`` values.
        elif choice == 3:
            return np.argsort(np.array(fit, copy=False)[:2 * (lam_reev + 1)])[:lam_reev]
        else:
            raise ValueError('unrecognized choice value %d for noise reev'
                             % choice)

class Sections(object):
    """plot sections through an objective function.

    A first rational thing to do, when facing an (expensive)
    application. By default 6 points in each coordinate are evaluated.
    This class is still experimental.

    Examples
    --------
    ::

        import cma, numpy as np
        s = cma.Sections(cma.ff.rosen, np.zeros(3)).do(plot=False)
        s.do(plot=False)  # evaluate the same points again, i.e. check for noise
        try:
            s.plot()
        except:
            print('plotting failed: matplotlib.pyplot package missing?')

    Details
    -------
    Data are saved after each function call during `do`. The filename
    is attribute ``name`` and by default ``str(func)``, see `__init__`.

    A random (orthogonal) basis can be generated with
    ``cma.Rotation()(np.eye(3))``.

    CAVEAT: The default name is unique in the function name, but it
    should be unique in all parameters of `__init__` but `plot_cmd`
    and `load`. If, for example, a different basis is chosen, either
    the name must be changed or the ``.pkl`` file containing the
    previous data must first be renamed or deleted.

    ``s.res`` is a dictionary with an entry for each "coordinate" ``i``
    and with an entry ``'x'``, the middle point. Each entry ``i`` is
    again a dictionary with keys being different dx values and the
    value being a sequence of f-values. For example ``s.res[2][0.1] ==
    [0.01, 0.01]``, which is generated using the difference vector ``s
    .basis[2]`` like

    ``s.res[2][dx] += func(s.res['x'] + dx * s.basis[2])``.

    :See also: `__init__`

    """
    def __init__(self, func, x, args=(), basis=None, name=None,
                 plot_cmd=None, load=True):
        """
        Parameters
        ----------
        ``func``
            objective function
        ``x``
            point in search space, middle point of the sections
        ``args``
            arguments passed to `func`
        ``basis``
            evaluated points are ``func(x + locations[j] * basis[i])
            for i in len(basis) for j in len(locations)``,
            see `do()`
        ``name``
            filename where to save the result
        ``plot_cmd``
            command used to plot the data, typically matplotlib pyplots
            `plot` or `semilogy`
        ``load``
            load previous data from file ``str(func) + '.pkl'``

        """
        if plot_cmd is None:
            from matplotlib.pyplot import plot as plot_cmd
        self.func = func
        self.args = args
        self.x = x
        self.name = name if name else str(func).replace(' ', '_').replace('>', '').replace('<', '')
        self.plot_cmd = plot_cmd  # or semilogy
        self.basis = np.eye(len(x)) if basis is None else basis

        try:
            self.load()
            if any(self.res['x'] != x):
                self.res = {}
                self.res['x'] = x  # TODO: res['x'] does not look perfect
            else:
                print(self.name + ' loaded')
        except:
            self.res = {}
            self.res['x'] = x

    def do(self, repetitions=1, locations=np.arange(-0.5, 0.6, 0.2), plot=True):
        """generates, plots and saves function values ``func(y)``,
        where ``y`` is 'close' to `x` (see `__init__()`). The data are stored in
        the ``res`` attribute and the class instance is saved in a file
        with (the weired) name ``str(func)``.

        Parameters
        ----------
        ``repetitions``
            for each point, only for noisy functions is >1 useful. For
            ``repetitions==0`` only already generated data are plotted.
        ``locations``
            coordinated wise deviations from the middle point given in
            `__init__`

        """
        if not repetitions:
            self.plot()
            return

        res = self.res
        for i in range(len(self.basis)):  # i-th coordinate
            if i not in res:
                res[i] = {}
            # xx = np.array(self.x)
            # TODO: store res[i]['dx'] = self.basis[i] here?
            for dx in locations:
                xx = self.x + dx * self.basis[i]
                xkey = dx  # xx[i] if (self.basis == np.eye(len(self.basis))).all() else dx
                if xkey not in res[i]:
                    res[i][xkey] = []
                n = repetitions
                while n > 0:
                    n -= 1
                    res[i][xkey].append(self.func(xx, *self.args))
                    if plot:
                        self.plot()
                    self.save()
        return self

    def plot(self, plot_cmd=None, tf=lambda y: y):
        """plot the data we have, return ``self``"""
        from matplotlib import pyplot
        if not plot_cmd:
            plot_cmd = self.plot_cmd
        colors = 'bgrcmyk'
        pyplot.gcf().clear()
        res = self.res

        flatx, flatf = self.flattened()
        minf = np.inf
        for i in flatf:
            minf = min((minf, min(flatf[i])))
        addf = 1e-9 - minf if minf <= 1e-9 else 0
        for i in sorted(k for k in res.keys() if isinstance(k, int)):  # we plot not all values here
            color = colors[i % len(colors)]
            arx = sorted(res[i].keys())
            plot_cmd(arx, [tf(np.median(res[i][x]) + addf) for x in arx], color + '-')
            pyplot.text(arx[-1], tf(np.median(res[i][arx[-1]])), i)
            if len(flatx[i]) < 11:
                plot_cmd(flatx[i], tf(np.array(flatf[i]) + addf), color + 'o')
        pyplot.ylabel('f + ' + str(addf))
        pyplot.draw()
        pyplot.ion()
        pyplot.show()
        return self

    def flattened(self):
        """return flattened data ``(x, f)`` such that for the sweep
        through coordinate ``i`` we have for data point ``j`` that
        ``f[i][j] == func(x[i][j])``

        """
        flatx = {}
        flatf = {}
        for i in self.res:
            if isinstance(i, int):
                flatx[i] = []
                flatf[i] = []
                for x in sorted(self.res[i]):
                    for d in sorted(self.res[i][x]):
                        flatx[i].append(x)
                        flatf[i].append(d)
        return flatx, flatf

    def save(self, name=None):
        """save to file"""
        import pickle
        name = name if name else self.name
        fun = self.func
        del self.func  # instance method produces error
        pickle.dump(self, open(name + '.pkl', "wb"))
        self.func = fun
        return self

    def load(self, name=None):
        """load from file"""
        import pickle
        name = name if name else self.name
        s = pickle.load(open(name + '.pkl', 'rb'))
        self.res = s.res  # disregard the class
        return self

