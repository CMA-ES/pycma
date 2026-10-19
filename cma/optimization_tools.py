"""Utility classes and functionalities loosely related to optimization
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
import sys
import warnings
import math
from copy import copy
import numpy as np
from multiprocessing import Pool as ProcessingPool
# from pathos.multiprocessing import ProcessingPool
from .utilities.python3for2 import abc as _abc
from .utilities.utils import BlancClass as _BlancClass
from .utilities import math as _um
# from .transformations import BoundTransform  # only to make it visible but gives circular import anyways
from .utilities.python3for2 import range
del absolute_import, division, print_function  #, unicode_literals

def semilogy_signed(x=None, y=None, yoffset=0, minabsy=None, iabscissa=1,
                    **kwargs):
    """signed semilogy plot.

    ``plt.yscale('symlog', linthreshy=min(abs(data[data != 0])))`` should
    do the same job at least as good.

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
    from matplotlib import pyplot as plt
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
    if 'labels' in kwargs:
        kwargs_labels = kwargs.pop('labels')
        for i, yi in enumerate(np.asarray(y).T):
            plt.plot(x, yi, label=kwargs_labels[i] if i < len(kwargs_labels) else None, **kwargs)
        plt.legend(framealpha=0.1)  # more opaque than not
    else:
        plt.plot(x, y, **kwargs)
        if 'label' in kwargs:
            plt.legend(framealpha=0.1)  # more opaque than not

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
    `x`- and `y_range` are `iterable` (e.g. `list` or arrays)
    to define the meshgrid.

    CAVEAT: this function calls `fct` ``len(list(x_range)) * len(list(y_range))``
    times. Hence using `Sections` may be the better first choice to
    investigate an expensive function.

    Examples::

        from cma import optimization_tools
        import numpy as np
       
        def plt_contour():  # def avoids doctest execution
            from matplotlib import pyplot as plt

            X, Y, Z = optimization_tools.contour_data(
                          lambda x: sum([xi**2 for xi in x]),
                          np.arange(0.90, 1.10, 0.02),
                          np.arange(-0.10, 0.10, 0.02))
            CS = plt.contour(X, Y, Z)
            plt.gca().set_aspect('equal')
            plt.clabel(CS)
        def plt_surface():  # def avoids doctest execution
            from matplotlib import pyplot as plt
            from mpl_toolkits import mplot3d

            X, Y, Z = optimization_tools.contour_data(
                          lambda x: sum([xi**2 for xi in x]),
                          np.arange(-1, 1.1, 0.02))
            ax = plt.axes(projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    See `cma.fitness_transformations.FixVariables` to create a 2-D
    function from a d-D function, e.g. like

    >>> import cma
    ...
    >>> fd = cma.ff.elli
    >>> x0 = 22 * [0]
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

def id(*args):
    return args
def ei_copy_arrays(evals, fval, successes):
    """Needed when the array `evals` or `successes` is kept for later use, because

    internally the arrays change *in place* over the iterations.

    ``return_filter=ei_copy_arrays`` is the default for `EvaluationIterator` to
    avoid unexpected results in that::

        ert1 = [ei_ert(d) for d in list(EvaluationIterator(data))]
        ert2 = [ei_ert(d) for d in EvaluationIterator(data)]

    give the same result, as to be desired, whereas::

        ert3 = [ei_ert(d) for d in list(EvaluationIterator(data, return_filter=None))]

    is wrong unless the `list` call is omitted.

    For efficiency, we should use ``return_filter=None`` with a for loop like::

        for evals, fval, successes in ot.EvaluationIterator(data, return_filter=None)

    when ``evals`` and ``successes`` are never used *after* the iterator advances
    (unless copied).
    """
    return np.array(evals, copy=True), fval, np.array(successes, copy=True)
def ei_ert(evals, fval, successes):
    """expected runtime = average runtime over all runs / success_rate from EI data line"""
    return float(sum(evals) / sum(successes))
def ei_sp1(evals, fval, successes):
    """like ERT but disregarding the runtime values of unsuccessful runs by

    replacing them with the average runtime of successful runs. When
    unsuccessful runs terminate on a timeout max budget, SP1 is usually a better
    performance indicator than ERT.
    """
    return float(np.mean(evals[successes]) * len(successes) / sum(successes))
def ei_median(evals, fval, successes):
    """return median evaluations to reach `fval`.

    Unsuccessful runs are sorted to the end and the median is taken over all
    runs. Hence, return `nan` when half or more runs were unsuccessful and hence
    `fval` doesn't map to any median evaluations.

    When plotting `fval` versus this median, the graph stops after the last
    improvement, even when the true median run continues (flat).

    Details: the median of only successful trials is not monotonuous in `fval`,
    hence seems semantically not always meaningful.
    """
    n = len(successes)
    if 2 * sum(successes) <= n:
        return np.nan
    ev = sorted(evals[successes])
    n2 = int(n / 2)
    if n2 == n // 2:
        return (ev[n2 - 1] + ev[n2]) / 2
    return ev[n2]
def ei_xy_data(xvalues=(ei_ert,)):
    """return a function that creates xy-values from a single `EvaluationsIterator` element.

    Argument `xvalues` is a `list` or sequence of functions, each used to create
    one "x-column".

    The return value of `ei_xy_data` can be passed to `EvaluationsIterator` as
    ``return_filter`` argument, thereby internally computing the given `xvalues`
    and the f-value as last column.

    Example::

        import numpy as np
        import cma.optimization_tools as ot

        xy = np.asarray(list(ot.EvaluationsIterator(data, return_filter=ot.ei_xy_data())))
        assert xy.shape[1] == 2  # x=ERT and y=fval

    """
    def return_xy(evals, fval, successes):
        return tuple(fun(evals, fval, successes) for fun in xvalues) + (fval,)
    return return_xy

class EvaluationsIterator(_abc.Iterator):  # the parent class provides the __iter__ method
    """return iteratively ``evals, fval, successes``

    where ``evals`` and ``successes`` are `numpy` arrays and ``fval`` is a
    `float`.

    The input argument ``data`` is an array where the first column,
    ``data[:, 0]``, contains the monotonuously increasing evaluations and
    the remaining columns, ``data[:,1:]``, are the recorded respective
    f-values (which do not need to be monotonuous) for several runs.

    Each output ``fval`` value reflects an entry in the input
    ``data[:,1:]``. When the f-values are monotonously decreasing in the
    ``data`` row index, each value appears exactly once in the iterator
    sequence (in decreasing order). Increased f-values are in effect
    ignored.

    The first output element, ``evals``, gives for each run the number of
    evaluations when ``fval`` was reached for the first time or, in case
    ``fval`` was not reached in the run, the last evaluations with a finite
    (exisiting) f-value of this run.

    The third output element, ``successes``, is a `bool` array indicating
    whether the ``fval`` was reached.

    Details: this class is an iterator mainly to allow for more efficient
    memory use, because we usually do not need to keep all evaluations (of
    each run) for each ``fval``. When ``evals`` and ``successes`` are
    directly consumed and not stored, passing ``return_filter=None``
    additionally avoids the unnecessary copy of both.

    Examples
    --------

    >>> import cma.optimization_tools as ot
    >>> data = [[1, 44.2, 33.3, 22.2],
    ...         [2, 32,   22.1, 33],
    ...         [3, 11,   22,   66]]

    The shortest usage example is:

    >>> list_of_evals_fval_succs = list(ot.EvaluationsIterator(data))

    which creates a `list` of about ``n x m`` length where ``n =
    data.shape[0]`` and ``m = data.shape[1] - 1`` (usually n>>m). Each list
    element is a three-`tuple` with ``(m evaluations, 1 f-value, m bools)``.

    A typical use case computes the ERT-value in the first "column" (as by
    default argument ``xvalues=(ei_ert,)`` of `ei_xy_data`) and has (by
    default) f-values in the second (last) "column":

    >>> xy = list(ot.EvaluationsIterator(data, return_filter=ot.ei_xy_data()))
    >>> print(xy)  # doctest: +ELLIPSIS
    [(1.0, 44.2), (1.33333...

    Now we can plot the ERT as the x-value (with evaluations as unit)::

        plt.plot([d[0] for d in xy],
                    [d[1] for d in xy], label='ERT')
        plt.xlabel('evaluations'), plt.ylabel('f-value');

    An example collecting ERT, SP1, the median, the f-value, and the number
    of successes without creating the full new data array otherwise:

    >>> list_of_xs_y_s = [(ot.ei_ert(*d),     # ERT
    ...                    ot.ei_sp1(*d),     # SP1
    ...                    ot.ei_median(*d),  # median
    ...                    d[1],              # f-value
    ...                    sum(d[2])          # number of runs that reached d[1]
    ...                   ) for d in ot.EvaluationsIterator(data, return_filter=None)]

    Passing ``return_filter=None`` avoids an unnecessary internal copy of
    ``d[0]`` and ``d[2]``.

    We can again plot the ERT as an x-value::

        plt.plot([d[0] for d in list_of_xs_y_s],
                    [d[3] for d in list_of_xs_y_s], label='ERT')

    More conveniently, we can use arrays for plotting, here with annotation
    of the graph end and of successes too::

        from matplotlib import pyplot as plt

        xy = np.asarray(list_of_xs_y_s)
        p = plt.plot(xy[:, 0], xy[:, -2], '-', label='ERT')
        plt.plot(xy[-1, 0], xy[-1, -2], 'o', color=p[-1].get_color())
        p = plt.plot(xy[:, 2], xy[:, -2], '-', label='median')
        i = np.nonzero(np.isfinite(xy[:, 2]))[0][-1]
        plt.plot(xy[i, 2], xy[i, -2], 'o', color=p[-1].get_color())
        idx = np.where(np.diff(xy[:, -1], prepend=xy[0, -1]))[0]  # indices where successes changed
        for j, i in enumerate(idx):          # annotate:
            if (j == 0 or xy[i, -1] <= 5 or  # first failure and 5..1 successes
                xy[i, -1] <= xy[0, -1] / 2 < xy[idx[j-1], -1]):  # and 50% failures
                plt.text(xy[i, 0], xy[i, -2], int(xy[i, -1]))
        # plt.gca().set_xscale('log')
        plt.xlabel('evaluations'), plt.ylabel('f-value'), plt.legend();

    A similar example as above but with an explicit for-loop instead of a
    list comprehension and hence longer but not more useful here:

    >>> import cma.optimization_tools as ot
    >>> fvals, erts, med = [], [], []
    >>> xy_succ = {s: None for s in [2, 5, 9]}  # number for successes to annotate
    >>> for evals, fval, succ in ot.EvaluationsIterator(data):
    ...     erts.append(ot.ei_ert(evals, fval, succ))  # compress all runs to one column
    ...     med.append(ot.ei_median(evals, fval, succ))  # compress all runs to one column
    ...     fvals.append(fval)  # y-value column
    ...     if sum(succ) < len(succ):
    ...         for k in xy_succ:
    ...             if xy_succ[k] is None and sum(succ) <= k:
    ...                 xy_succ[k] = erts[-1], fvals[-1]

    Plotting the above::

        plt.plot(data[:, 0], data[:, 1:])  # plot all data lines
        plt.plot(erts, fvals)  # plot an ert line
        for k, xy in xy_succ.items():
            plt.text(xy[0], xy[1], k)
        plt.xlabel('evaluations'), plt.ylabel('f-value');

    Testing:

    >>> ot.EvaluationsIterator._no_asserts = False
    >>> x = [ot.ei_ert(*d) for d in ot.EvaluationsIterator(data)]
    >>> assert x == sorted(x), x
    >>> for d in zip(x, [1.0, 1.333333333333333, 1.666666666666667, 2.0, 4.0, 4.5, 9.0]):
    ...     assert -1e-7 < d[0] - d[1] < 1e-7, (x, d)
    >>> y = [d[1] for d in ot.EvaluationsIterator(data, return_filter=None)]
    >>> assert y == [44.2, 33.3, 32., 22.2, 22.1, 22., 11.], y
    >>> xy = list(ot.EvaluationsIterator(data, return_filter=ot.ei_xy_data()))
    >>> for i in range(len(xy)):
    ...     assert xy[i] == (x[i], y[i]), (xy[i], x[i], y[i])
    >>> import numpy as np
    >>> n, m = 22, 11
    >>> a = np.log(np.abs(np.random.randn(n, m+1) / np.random.randn(n, m+1)))
    >>> a.sort(0)
    >>> a[:,:] = a[-1::-1, :]
    >>> a[:,0] = 2 * np.arange(a.shape[0]) + 1
    >>> for agg in [ot.ei_ert, ot.ei_sp1, ot.ei_median]:
    ...     x = np.asarray([agg(*d) for d in ot.EvaluationsIterator(a)])
    ...     if agg == ot.ei_median:
    ...         x = x[np.isfinite(x)]
    ...         assert np.all(x == np.array(x, dtype=int)), x  # works because all evals are odd
    ...     assert np.all(np.diff(x) >= 0), x

    """
    _no_asserts = True
    evals_type = float

    def __init__(self, data, return_filter=ei_copy_arrays):
        """``data[:, 0]`` are evaluations and ``data[:, 1:]`` are f-values.

        `return_filter` is a function applied to the iterator return value
        ``evals, fval, successes`` on the fly. ``None`` and ``'id'`` are the
        identity.
        """
        self.data = np.asarray(data)
        self.return_filter = return_filter
        if return_filter in (None, 'id'):
            self.return_filter = id
        # if evals_type in (None, int, 'int'):
        #     evals_are_int = all(d == d // 1 for d in self.data[:, 0])
        #     if evals_type is None:
        #         evals_type = int if evals_are_int else float
        #     elif not evals_are_int:
        #         warnings.warn("Non-int evaluation values found. "
        #                       "``evals_type=int`` parameter will remove fractions.")
        self.count = 0
        self.row_indices = np.zeros(self.data.shape[1] - 1, dtype=int)
        '''unsuccessful data have negative indices (which is used in `fval`)'''
        self._current_data = self.data[0, 0] * np.ones(
                    self.data.shape[1] - 1, dtype=EvaluationsIterator.evals_type)
        '''current evaluations, changing these in place saves 35% CPU!?'''
        self.successes = np.ones(self.data.shape[1] - 1, dtype=bool)
        '''success bool array, equals to ``row_indices >= 0``'''

    @property
    def current_data(self):
        """return array of evals either reaching self.fval or the last finite evals"""
        assert self._no_asserts or np.all(
            self.data[self.row_indices, 0] == self._current_data), (
            self.data[self.row_indices, 0], self._current_data)
        return self.data[self.row_indices, 0]  # _current_data changes in place
    @property
    def fval(self):
        """current f-value, bails when no data are available anymore"""
        return float(max(self.data[i, j+1]
                            for j, i in enumerate(self.row_indices)
                            if i >= 0))  # (i >= 0) == self.successes[j]

    def __next__(self):  # in Python 2 we need def next(self)
        """increment ``i = self.row_indices[j]`` for which ``self.data[i, j] >= fval``

        until it surpasses ``fval`` and ``return evals, fval, successes``
        such that ``evals[j] == self.data[self.row_indices[j], 0]``.
        """
        self.count += 1
        if self.count == 1:
            return self.return_filter(self._current_data, self.fval, self.successes)
        # advance row index for each column until fval decreased
        fval = self.fval  # based on the current row_indices
        for j in range(1, 1 + len(self.row_indices)):  # j = column in data
            i = self.row_indices[j-1]
            if i < 0 or self.data[i, j] < fval:  # column did not reach fval previously
                continue                         # or i is still good, i.e. below fval
            i += 1
            while i < self.data.shape[0] and (
                np.isfinite(self.data[i, j]) and
                self.data[i, j] >= fval):
                i += 1
            if i >= self.data.shape[0] or not np.isfinite(self.data[i, j]):
                self.successes[j-1] = False
                i -= 1
                assert self._no_asserts or np.isfinite(self.data[i, j]), (i, self.data[i-1:i+2, j])
                assert self.data[i, j] == self.data[i - self.data.shape[0], j], (i, j, self.data.shape)
                i = i - self.data.shape[0]  # same but negative
            else:  # index i is fine
                assert self.data[i, j] < fval, (fval, i, j, self.data[i, j])
            self.row_indices[j-1] = i
            self._current_data[j-1] = self.data[i, 0]
        assert self._no_asserts or np.all(self.successes == (self.row_indices >= 0)), (
                self.row_indices >= 0, self.successes)  # takes 10% of execution time
        assert self._no_asserts or np.all(
                self.data[self.row_indices, 0] == self._current_data), (
                self.data[self.row_indices, 0], self._current_data)
        if np.any(self.successes):
            return self.return_filter(self._current_data, self.fval, self.successes)
        raise StopIteration

    if sys.version_info[0] == 2:  # noqa: UP036
        next = __next__  # in Python 2 we need next(self) and __iter__

class EvalParallel2(object):  # noqa: UP004
    """A class and context manager for parallel evaluations.

    This class is based on the ``Pool`` class of the `multiprocessing` module.

    The interface in v2 changed, such that the fitness function can be
    given once in the constructor. Hence the number of processes has
    become the second (optional) argument of `__init__` and the function
    has become the second and optional argument of `__call__`.

    To be used with the `with` statement (otherwise `terminate` needs to
    be called to free resources)::

        with EvalParallel2(fitness_function) as eval_all:
            fvals = eval_all(solutions)

    assigns a callable `EvalParallel2` class instance to ``eval_all``.
    The instance can be called with a `list` (or `tuple` or any
    sequence) of solutions and returns their fitness values. That is::

        eval_all(solutions) == [fitness_function(x) for x in solutions]

    `EvalParallel2.__call__` may take three additional optional arguments,
    namely `fitness_function` (like this the function may change from call
    to call), `args` passed to ``fitness`` and `timeout` passed to the
    `multiprocessing.pool.ApplyResult.get` method which raises
    `multiprocessing.TimeoutError` in case.

    ``eval_all = EvalParallel2(fitness_function, 0)`` bypasses
    `multiprocessing`, hence the construct can be used even when
    `multiprocessing` fails on this `fitness_function` instantiation.

    Examples:

    >>> from cma.optimization_tools import EvalParallel2
    >>> for n_jobs in [None, -1, 0, 1, 2, 4]:
    ...     with EvalParallel2(cma.fitness_functions.elli, n_jobs) as eval_all:
    ...         res = eval_all([[1,2], [3,4]])
    >>> # class usage, don't forget to call terminate
    >>> ep = EvalParallel2(cma.fitness_functions.elli, 4)
    >>> [float(v) for v in ep([[1,2], [3,4], [4, 5]])]  # doctest:+ELLIPSIS
    [4000000.944...
    >>> ep.terminate()
    ...
    >>> # use with `with` statement (context manager)
    >>> es = cma.CMAEvolutionStrategy(3 * [1], 1, dict(verbose=-9, ftarget=1e-3))
    >>> with EvalParallel2(cma.fitness_functions.elli,
    ...                    number_of_processes=12) as eval_all:
    ...     while not es.stop():
    ...         X = es.ask()
    ...         es.tell(X, eval_all(X, args=(1e1,)))  # `eval_all` also accepts
    ...                                               # `fitness_function` as
    ...                                               # (optional) keyword argument
    >>> assert es.result[1] < 1e-3 and es.result[2] < 1500

    Parameters: the `EvalParallel2` constructor takes the number of
    processes as optional input argument, which is by default
    ``multiprocessing.cpu_count()``. If ``number_of_processes <= 0``, no
    `multiprocessing` is invoked and the fitness is computed directly in a
    regular loop.

    Limitations: the `multiprocessing` module, on which this class is based
    upon, may not work with certain class instance methods or Cython
    instances, or class instances that contain modules as it uses `pickle`.

    Details: in some cases the execution may be considerably slowed down,
    as for example in previous tests done with test suites from coco/bbob.

    Comparing setting ``number_of_processes = 0`` with
    ``number_of_processes = 1`` evaluates the overhead introduced by
    ``multiprocessing.Pool.apply_async``.
"""
    def __init__(self, fitness_function=None, number_of_processes=None):
        self.fitness_function = fitness_function
        self.processes = number_of_processes  # for the record
        if self.processes is None or self.processes > 0:
            self.pool = ProcessingPool(self.processes)
        else:
            self.pool = None

    def __call__(self, solutions, fitness_function=None, args=(), timeout=None):
        """evaluate a list/sequence of solution-"vectors", return a list
        of corresponding f-values.

        `args` must be a tuple and is passed to `fitness_function` like
        ``fitness_function(solutions[0], *args)``. For example, a single
        argument, say `a1`, should be passed like ``args=(a1, )``.

        Raises `multiprocessing.TimeoutError` if `timeout` is given and
        exceeded.
        """
        fitness_function = fitness_function or self.fitness_function
        if fitness_function is None:
            raise ValueError("`fitness_function` was never given, must be"
                             " passed in `__init__` or `__call__`")
        if not self.pool:
            return [fitness_function(x, *args) for x in solutions]
        warning_str = ("`fitness_function` must be a function, not a"
                       " `lambda` or an instancemethod, in order to work with"
                       " `multiprocessing` under Python 2")
        if sys.version_info[0] == 2:
            if isinstance(fitness_function, type(self.__init__)):
                warnings.warn(warning_str)
        jobs = [self.pool.apply_async(fitness_function, (x,) + args)
                for x in solutions]
        try:
            return [job.get(timeout) for job in jobs]
        except:
            sys.version_info[0] == 2 and warnings.warn(warning_str)
            raise

    def terminate(self):
        """free allocated processing pool"""
        if not self.pool:
            return
        # self.pool.close()  # would wait for job termination
        self.pool.terminate()  # terminate jobs regardless
        self.pool.join()  # end spawning

    def __enter__(self):
        # we could assign self.pool here, but then `EvalParallel2` would
        # *only* work when using the `with` statement
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def __del__(self):
        """though generally not recommended `__del__` should be OK here"""
        self.terminate()

class BestSolution(object):
    """container to keep track of the best solution seen.

    Keeps also track of the genotype, if available. Stored solutions are
    copied so that callers can reuse their evaluation buffers.

    >>> from cma.optimization_tools import BestSolution
    >>> x = np.array([1., 2.])
    >>> best = BestSolution(x, 5.)
    >>> x[:] = 9
    >>> assert np.array_equal(best.x, [1, 2])
    >>> assert np.array_equal(best.last.x, [1, 2])
    >>> population = np.array([[3., 4.], [0., 1.]])
    >>> best.update(population, arf=[25., 1.])
    >>> population[:] = 9
    >>> assert np.array_equal(best.x, [0, 1]) and best.f == 1
    >>> assert np.array_equal(best.last.x, [0, 1])
    >>> other = BestSolution()
    >>> _ = other.update(best)
    >>> best.x[:] = 9
    >>> assert np.array_equal(other.x, [0, 1])
    >>> population = [[3., 4.], [2., 2.]]
    >>> other.update(population, arf=[25., 8.])
    >>> population[1][:] = [9., 9.]
    >>> assert other.last.x == [2., 2.] and other.f == 1
    >>> from cma.utilities.utils import SolutionDict
    >>> archive, genotype = SolutionDict(), np.array([0., 1.])
    >>> archive[(0., 1.)] = {'geno': genotype}
    >>> other.update([[0., 1.]], xarchive=archive, arf=[0.5])
    >>> genotype[:] = 9
    >>> assert np.array_equal(other.x_geno, [0, 1])

    """
    def __init__(self, x=None, f=math.inf, evals=None):
        """initialize the best solution with ``x``, ``f``, and ``evals``.

        Better solutions have smaller ``f``-values.
        """
        self.x = copy(x)
        self.x_geno = None
        self.f = f if f is not None and f is not np.nan else math.inf
        self.evals = evals
        self.evalsall = evals
        self.compared = 0
        "  number of overall compared values, posterior hack"
        self.last = _BlancClass()
        self.last.x = self.x
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
            if arx.f is not None and arx.f < math.inf:
                self.update([arx.x], xarchive, [arx.f], arx.evals)
            self.compared += arx.compared
            return self
        assert arf is not None
        self.compared += len(arf)
        # find failsave minimum
        try:
            minidx = np.nanargmin(arf)
        except ValueError:
            return
        if minidx is np.nan:
            return
        minarf = arf[minidx] if isinstance(arf[minidx], int) else float(arf[minidx])
        # minarf = reduce(lambda x, y: y if y and y is not np.nan
        #                   and y < x else x, arf, math.inf)
        self.last.x = copy(arx[minidx])
        if minarf < math.inf and (minarf < self.f or self.f is None):
            self.x = self.last.x
            self.f = minarf
            if xarchive is not None and xarchive.get(self.x) is not None:
                self.x_geno = copy(xarchive[self.x].get('geno'))
            else:
                self.x_geno = None
            self.evals = None if not evals else evals - len(arf) + int(minidx) + 1  # minidx was np.int
            self.evalsall = evals
        elif evals:
            self.evalsall = evals
        self.last.f = minarf
    def get(self):
        """return ``(x, f, evals)`` """
        return self.x, self.f, self.evals  # , self.x_geno

class BestSolution2(object):
    """minimal tracker of a smallest f-value with variable meta-info"""
    def __init__(self):
        self.f = math.inf
        self.x = None
        self.info = None
        self.count_saved = None
        self.count = 0
        "  number of overall compared values"
        self.previous = None
    def update(self, f, x=None, info=None, info_construct=None):
        """`info` may be a dictionary with everything we want to know,
        `info_construct` may be used to finalize versatile elements of
        `info`, like make a copy of an array within the info dictionary
        """
        self.count += 1
        if self.count == 1 or (np.isfinite(f) and (not np.isfinite(self.f) or f < self.f)):
            self.previous = dict(self.__dict__)
            del self.previous['previous']  # otherwise we get a linked list of all previous entries
            self.f = _um.ifloat(f)
            self.x = x
            self.info = info_construct(info) if info_construct else info
            self.count_saved = self.count
        return self
    def __str__(self):
        return str(self.__dict__)
class BestFeasibleSolution(BestSolution2):
    def __init__(self):
        BestSolution2.__init__(self)
        self.g = None
        self.al_penalties = None
    def update(self, f, g, al_penalties=None, x=None, info=None, info_construct=None):
        """g and al_penalties are lists of constraint and penalty values"""
        if any(gi > 0 for gi in g):
            return self  # not a feasible solution, do nothing
        BestSolution2.update(self, f, x, info, info_construct)
        if self.count_saved == self.count:
            self.g = _um.lifloat(g)
            self.al_penalties = _um.lifloat(al_penalties)
        return self
class ExponentialSmoothing(object):
    """not in use (yet)

    Exponentially smoothened vector, new data are added via
    calling the class instance. The `normalizer` is applied to
    the weight ``1 / time_constant`` used for the new data.

    """
    def __init__(self, time_constant=None, normalizer=lambda x: x):
        self.time_constant = time_constant
        if self.time_constant is not None and self.time_constant < 1:
            raise ValueError("time_constant = %d must be >=1" % self.time_constant)
        self.normalizer = normalizer
        self.values = None
        self.count = 0

    def _init_(self, v):
        self.values = np.array(v, dtype=float)
        if self.time_constant is None:
            self.time_constant = 1 + len(v)**0.5

    def __getitem__(self, i):
        return self.values[i]

    def __call__(self, v):
        if self.values is None:
            self._init_(v)
        self.count += 1
        tc = np.min((self.count, self.time_constant))
        self.values *= 1 - 1 / tc
        self.values += self.normalizer(1 / tc) * np.asarray(v)
        return self

class EvolutionPath(ExponentialSmoothing):
    """not in use (yet)

    A variance-neutral exponentially smoothened vector.
    """
    def __init__(self, time_constant=None):
        super(EvolutionPath, self).__init__(
            time_constant, lambda x: np.sqrt(x * (2 - x)))

    @property
    def path(self):
        return self.values

class BinaryEvolutionPath(EvolutionPath):

    @property
    def probability_larger_than_one_from_binary(self):
        """propability of path entries to be larger than one,

        given the input is ``sign(randn())``. Check out::

            n = int(1e4)
            greater_than_one = []
            ar_tc = [1.2, 1.5, 1.9, 2, 4, 8, 16, 32, 100]
            for tc in ar_tc:
                p = cma.optimization_tools.EvolutionPath(tc)
                for i in range(int(10 * tc)):
                    p(np.sign(np.random.randn(n)))
                # plot(*step_data(p.path))
                greater_than_one += [(np.mean(p.path > 1) + np.mean(p.path < -1)) / 2]

        """
        return np.minimum(0.25, 0.15865525393145707  # these come from the math for tc=1 and tc=infty
                          + 0.2 / np.asarray(self.time_constant)**1.9)  # empirical fit to the data

    @property
    def raw_binary_s(self):
        """return one of two possible values with expectation of zero.

        the maximum for the larger value is 1 - 0.15865525393145707 for tc to infty.
        """
        # p * (I - p) + (1 - p) * (0 - p) = p - p^2  - p + p^2 = 0
        return (np.abs(self.values) > 1) - self.probability_larger_than_one_from_binary

    def binary_s(self, odds_of_increment=1):
        """how many increments for one decrement in stationary state"""
        s = self.raw_binary_s
        s[s > 0] /= odds_of_increment
        return s

class OldEvolutionPath(object):
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
    >>> x, es = cma.fmin2(cma.ff.elli, 3 * [1], 1, noise_handler=cma.NoiseHandler)  #doctest: +ELLIPSIS
    (3_w,7)-aCMA-ES (mu_w=2...
    >>> assert es.result[1] < 1e-8
    >>> x, es = cma.fmin2(cma.ff.elli, 2 * [1], 1, {'AdaptSigma':cma.sigma_adaptation.CMAAdaptSigmaTPA},
    ...          noise_handler=cma.NoiseHandler)  #doctest: +ELLIPSIS
    (3_w,...
    >>> assert es.result[1] < 1e-8

    A more verbose example in the optimization loop with a noisy function
    defined in ``func``:

    >>> import cma, numpy as np
    >>> func = lambda x: cma.ff.sphere(x) * (1 + 4 * np.random.randn() / len(x))  # cma.ff.noisysphere
    >>> es = cma.CMAEvolutionStrategy(np.ones(4), 1)  #doctest: +ELLIPSIS
    (4_w,8)-aCMA-ES (mu_w=2...
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

        A parallel evaluator still receives a batch when perturbation is off:

        >>> import cma, numpy as np
        >>> points = [np.arange(3.) + i for i in range(8)]
        >>> handler = cma.NoiseHandler(3, maxevals=[2, 2, 2], reevals=2,
        ...                            epsilon=0, parallel=True)
        >>> with EvalParallel2(cma.ff.sphere, number_of_processes=0) as evaluate:
        ...     values = evaluate(points)
        ...     _ = handler(points, values, evaluate)
        >>> assert np.array_equal(handler.fitre, values)
        >>> assert handler.evaluations_just_done == 4

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
            elif self.parallel:
                self.fitre[i] = fagg(func([X_i] * evals, *args))
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
        limits = [0.5 * (_um.Mh.prctile(np.abs(r - (ranks[0, i] + 1 - (ranks[0, i] > ranks[1, i]))),
                                      self.theta * 50) +
                         _um.Mh.prctile(np.abs(r - (ranks[1, i] + 1 - (ranks[1, i] > ranks[0, i]))),
                                      self.theta * 50))
                    for i in self.idx]
        # compute measurement
        #                               max: 1 rankchange in 2*lambda is always fine
        s = np.abs(rankDelta[self.idx]) - _um.Mh.amax(limits, 1)  # lives roughly in 0..2*lambda
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
            sort_idx = np.argsort(np.asarray(fit)[n_first:]) + n_first
            return np.asarray(list(range(0, n_first)) +
                            list(sort_idx[0:lam_reev - n_first]))
        elif choice == 2:
            idx_sorted = np.argsort(np.asarray(fit))
            # take lam_reev equally spaced, starting with best
            linsp = np.linspace(0, len(fit) - len(fit) / lam_reev, lam_reev)
            return idx_sorted[[int(i) for i in linsp]]
        # take the ``lam_reeval`` best from the first ``2 * lam_reeval + 2`` values.
        elif choice == 3:
            return np.argsort(np.asarray(fit)[:2 * (lam_reev + 1)])[:lam_reev]
        else:
            raise ValueError('unrecognized choice value %d for noise reev'
                             % choice)

class Sections(object):
    """plot sections through an objective function.

    A first rational thing to do, when facing an (expensive) application.
    By default 6 points in each coordinate are evaluated. The data is saved
    and reloaded by default. A change of basis will invalide the loaded
    result. This class is still experimental.

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
    ``cma.transformations.Rotation()(np.eye(3))``.

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
            load and self.load()
            if any(self.res['x'] != x):
                self.res = {}
                self.res['x'] = x  # TODO: res['x'] does not look perfect
            else:
                print(self.name + ' loaded')
        except:
            self.res = {}
            self.res['x'] = x

    def do(self, repetitions=1, locations=tuple((i - 2.5) / 3 for i in range(6)), plot=True):
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
        minf = math.inf
        for i in flatf:
            minf = min((minf, min(flatf[i])))
        addf = 1e-9 - minf if minf <= 1e-9 else 0
        for i in sorted(k for k in res.keys() if isinstance(k, int)):  # we plot not all values here
            color = colors[i % len(colors)]
            arx = sorted(res[i].keys())
            plot_cmd(arx, [tf(np.median(res[i][x]) + addf) for x in arx], color + '-')
            pyplot.text(arx[-1], tf(np.median(res[i][arx[-1]])), i)
            if len(flatx[i]) < 33:
                plot_cmd(flatx[i], tf(np.array(flatf[i]) + addf), color + ('o' if len(flatx[i]) < 11 else '.'))
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
