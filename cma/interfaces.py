"""Very few interface defining base class definitions"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
del absolute_import, division, print_function  #, unicode_literals

class OOOptimizer(object):
    """abstract base class for an Object Oriented Optimizer interface.

    Relevant methods are `__init__`, `ask`, `tell`, `optimize` and `stop`,
    and property `result`. Only `optimize` is fully implemented in this
    base class.

    Examples
    --------
    All examples minimize the function `elli`, the output is not shown.
    (A preferred environment to execute all examples is ``ipython``.)

    First we need::

        # CMAEvolutionStrategy derives from the OOOptimizer class
        from cma import CMAEvolutionStrategy
        from cma.fitness_functions import elli

    The shortest example uses the inherited method
    `OOOptimizer.optimize`::

        es = CMAEvolutionStrategy(8 * [0.1], 0.5).optimize(elli)

    The input parameters to `CMAEvolutionStrategy` are specific to this
    inherited class. The remaining functionality is based on interface
    defined by `OOOptimizer`. We might have a look at the result::

        print(es.result[0])  # best solution and
        print(es.result[1])  # its function value

    Virtually the same example can be written with an explicit loop
    instead of using `optimize`. This gives the necessary insight into
    the `OOOptimizer` class interface and entire control over the
    iteration loop::

        # a new CMAEvolutionStrategy instance
        optim = CMAEvolutionStrategy(9 * [0.5], 0.3)

        # this loop resembles optimize()
        while not optim.stop():  # iterate
            X = optim.ask()      # get candidate solutions
            f = [elli(x) for x in X]  # evaluate solutions
            #  in case do something else that needs to be done
            optim.tell(X, f)     # do all the real "update" work
            optim.disp(20)       # display info every 20th iteration
            optim.logger.add()   # log another "data line", non-standard

        # final output
        print('termination by', optim.stop())
        print('best f-value =', optim.result[1])
        print('best solution =', optim.result[0])
        optim.logger.plot()  # if matplotlib is available

    Details
    -------
    Most of the work is done in the methods `tell` or `ask`. The property
    `result` provides more useful output.

    """
    def __init__(self, xstart, *more_mandatory_args, **optional_kwargs):
        """``xstart`` is a mandatory argument"""
        self.xstart = xstart
        self.more_mandatory_args = more_mandatory_args
        self.optional_kwargs = optional_kwargs
        self.initialize()
    def initialize(self):
        """(re-)set to the initial state"""
        raise NotImplementedError('method initialize() must be implemented in derived class')
        self.countiter = 0
        self.xcurrent = [xi for xi in self.xstart]
    def ask(self, **optional_kwargs):
        """abstract method, AKA "get" or "sample_distribution", deliver
        new candidate solution(s), a list of "vectors"
        """
        raise NotImplementedError('method ask() must be implemented in derived class')
    def tell(self, solutions, function_values):
        """abstract method, AKA "update", pass f-values and prepare for
        next iteration
        """
        self.countiter += 1
        raise NotImplementedError('method tell() must be implemented in derived class')
    def stop(self):
        """abstract method, return satisfied termination conditions in a
        dictionary like ``{'termination reason': value, ...}`` or ``{}``.

        For example ``{'tolfun': 1e-12}``, or the empty dictionary ``{}``.

        TODO: this should rather be a property!? Unfortunately, a change
        would break backwards compatibility.
        """
        raise NotImplementedError('method stop() is not implemented')
    def disp(self, modulo=None):
        """abstract method, display some iteration info when
        ``self.iteration_counter % modulo < 1``, using a reasonable
        default for `modulo` if ``modulo is None``.
        """
    @property
    def result(self):
        """abstract property, contain ``(x, f(x), ...)``, that is, the
        minimizer, its function value, ...
        """
        raise NotImplementedError('result property is not implemented')
        return [self.xcurrent]

    def optimize(self, objective_fct,
                 maxfun=None, iterations=None, min_iterations=1,
                 args=(),
                 verb_disp=None,
                 callback=None):
        """find minimizer of ``objective_fct``.

        CAVEAT: the return value for `optimize` has changed to ``self``,
        allowing for a call like::

            solver = OOOptimizer(x0).optimize(f)

        and investigate the state of the solver.

        Arguments
        ---------

        ``objective_fct``: f(x: array_like) -> float
            function be to minimized
        ``maxfun``: number
            maximal number of function evaluations
        ``iterations``: number
            number of (maximal) iterations, while ``not self.stop()``,
            it can be useful to conduct only one iteration at a time.
        ``min_iterations``: number
            minimal number of iterations, even if ``not self.stop()``
        ``args``: sequence_like
            arguments passed to ``objective_fct``
        ``verb_disp``: number
            print to screen every ``verb_disp`` iteration, if `None`
            the value from ``self.logger`` is "inherited", if
            available.
        ``callback``: callable or list of callables
            callback function called like ``callback(self)`` or
            a list of call back functions called in the same way. If
            available, ``self.logger.add`` is added to this list.
            TODO: currently there is no way to prevent this other than
            changing the code of `_prepare_callback_list`.

        ``return self``, that is, the `OOOptimizer` instance.

        Example
        -------
        >>> import cma
        >>> es = cma.CMAEvolutionStrategy(7 * [0.1], 0.1
        ...              ).optimize(cma.ff.rosen, verb_disp=100)
        ...                   #doctest: +ELLIPSIS
        (4_w,9)-aCMA-ES (mu_w=2.8,w_1=49%) in dimension 7 (seed=...)
        Iterat #Fevals   function value  axis ratio  sigma ...
            1      9 ...
            2     18 ...
            3     27 ...
          100    900 ...
        >>> cma.s.Mh.vequals_approximately(es.result[0], 7 * [1], 1e-5)
        True

        """
        if iterations is not None and min_iterations > iterations:
            print("doing min_iterations = %d > %d = iterations"
                  % (min_iterations, iterations))
            iterations = min_iterations

        callback = self._prepare_callback_list(callback)

        citer, cevals = 0, 0
        while not self.stop() or citer < min_iterations:
            if (maxfun and cevals >= maxfun) or (
                  iterations and citer >= iterations):
                return self
            citer += 1

            X = self.ask()  # deliver candidate solutions
            fitvals = [objective_fct(x, *args) for x in X]
            cevals += len(fitvals)
            self.tell(X, fitvals)  # all the work is done here
            for f in callback:
                f(self)
            self.disp(verb_disp)  # disp does nothing if not overwritten

        # final output
        self._force_final_logging()

        if verb_disp:  # do not print by default to allow silent verbosity
            self.disp(1)
            print('termination by', self.stop())
            print('best f-value =', self.result[1])
            print('solution =', self.result[0])

        return self

    def _prepare_callback_list(self, callback):  # helper function
        """return a list of callbacks including ``self.logger.add``.

        ``callback`` can be a `callable` or a `list` (or iterable) of
        callables. Otherwise a `ValueError` exception is raised.
        """
        if callback is None:
            callback = []
        if callable(callback):
            callback = [callback]
        try:
            callback = list(callback) + [self.logger.add]
        except AttributeError:
            pass
        try:
            for c in callback:
                if not callable(c):
                    raise ValueError("""callback argument %s is not
                        callable""" % str(c))
        except TypeError:
            raise ValueError("""callback argument must be a `callable` or
                an iterable (e.g. a list) of callables, after some
                processing it was %s""" % str(callback))
        return callback

    def _force_final_logging(self):  # helper function
        """try force the logger to log NOW"""
        try:
            if not self.logger:
                return
        except AttributeError:
            return
        # the idea: modulo == 0 means never log, 1 or True means log now
        try:
            modulo = bool(self.logger.modulo)
        except AttributeError:
            modulo = True  # could also be named force
        try:
            self.logger.add(self, modulo=modulo)
        except AttributeError:
            pass
        except TypeError:
            try:
                self.logger.add(self)
            except Exception as e:
                print('  The final call of the logger in'
                      ' OOOptimizer._force_final_logging from'
                      ' OOOptimizer.optimize did not succeed: %s'
                      % str(e))

class StatisticalModelSamplerWithZeroMeanBaseClass(object):
    """yet versatile base class to replace a sampler namely in
    `CMAEvolutionStrategy`
    """
    def __init__(self, std_vec, **kwargs):
        """pass the vector of initial standard deviations or dimension of
        the underlying sample space.

        Ideally catch the case when `std_vec` is a scalar and then
        interpreted as dimension.
        """
        try:
            dimension = len(std_vec)
        except TypeError:  # std_vec has no len
            dimension = std_vec
            std_vec = dimension * [1]
        raise NotImplementedError

    def sample(self, number, update=None):
        """return list of i.i.d. samples.

        :param number: is the number of samples.
        :param update: controls a possibly lazy update of the sampler.
        """
        raise NotImplementedError

    def update(self, vectors, weights):
        """``vectors`` is a list of samples, ``weights`` a corrsponding
        list of learning rates
        """
        raise NotImplementedError

    def parameters(self, mueff=None, lam=None):
        """return `dict` with (default) parameters, e.g., `c1` and `cmu`.

        :See also: `RecombinationWeights`"""
        if (hasattr(self, '_mueff') and hasattr(self, '_lam') and
            (mueff == self._mueff or mueff is None) and
            (lam == self._lam or lam is None)):
            return self._parameters
        self._mueff = mueff
        lower_lam = 6  # for setting c1
        if lam is None:
            lam = lower_lam
        self._lam = lam
        # todo: put here rather generic formula with degrees of freedom
        # todo: replace these base class computations with the appropriate
        c1 = min((1, lam / lower_lam)) * 2 / ((self.dimension + 1.3)**2.0 + mueff)
        alpha = 2
        self._parameters = dict(
            c1=c1,
            cmu=min((1 - c1,
                     # or alpha * (mueff - 0.9) with relative min and
                     # max value of about 1: 0.4, 1.75: 1.5
                     alpha * (0.25 + mueff - 2 + 1 / mueff) /
                     ((self.dimension + 2)**2 + alpha * mueff / 2)))
        )
        return self._parameters

    def norm(self, x):
        """return Mahalanobis norm of `x` w.r.t. the statistical model"""
        return sum(self.transform_inverse(x)**2)**0.5
    @property
    def condition_number(self):
        raise NotImplementedError
    @property
    def covariance_matrix(self):
        raise NotImplementedError
    @property
    def variances(self):
        """vector of coordinate-wise (marginal) variances"""
        raise NotImplementedError

    def transform(self, x):
        """transform ``x`` as implied from the distribution parameters"""
        raise NotImplementedError

    def transform_inverse(self, x):
        raise NotImplementedError

    def to_linear_transformation_inverse(self, reset=False):
        """return inverse of associated linear transformation"""
        raise NotImplementedError

    def to_linear_transformation(self, reset=False):
        """return associated linear transformation"""
        raise NotImplementedError

    def inverse_hessian_scalar_correction(self, mean, X, f):
        """return scalar correction ``alpha`` such that ``X`` and ``f``
        fit to ``f(x) = (x-mean) (alpha * C)**-1 (x-mean)``
        """
        raise NotImplementedError

    def __imul__(self, factor):
        raise NotImplementedError

class BaseDataLogger(object):
    """abstract base class for a data logger that can be used with an
    `OOOptimizer`.

    Details: attribute `modulo` is used in `OOOptimizer.optimize`.
    """

    def __init__(self):
        self.optim = None
        """object instance to be logging data from"""
        self._data = None
        """`dict` of logged data"""
        self.filename = "_BaseDataLogger_datadict.py"
        """file to save to or load from unless specified otherwise"""

    def register(self, optim, *args, **kwargs):
        """register an optimizer ``optim``, only needed if method `add` is
        called without passing the ``optim`` argument
        """
        self.optim = optim
        return self

    def add(self, optim=None, more_data=None, **kwargs):
        """abstract method, add a "data point" from the state of ``optim``
        into the logger.

        The argument ``optim`` can be omitted if ``optim`` was
        ``register`` ()-ed before, acts like an event handler
        """
        raise NotImplementedError

    def disp(self, *args, **kwargs):
        """abstract method, display some data trace"""
        print('method BaseDataLogger.disp() not implemented, to be done in subclass ' + str(type(self)))

    def plot(self, *args, **kwargs):
        """abstract method, plot data"""
        print('method BaseDataLogger.plot() is not implemented, to be done in subclass ' + str(type(self)))

    def save(self, name=None):
        """save data to file `name` or `self.filename`"""
        with open(name or self.filename, 'w') as f:
            f.write(repr(self._data))

    def load(self, name=None):
        """load data from file `name` or `self.filename`"""
        from ast import literal_eval
        with open(name or self.filename, 'r') as f:
            self._data = literal_eval(f.read())
        return self
    @property
    def data(self):
        """logged data in a dictionary"""
        return self._data
