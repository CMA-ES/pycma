#!/usr/bin/env python
"""A minimalistic implemention of CMA-ES without using `numpy`.

The Covariance Matrix Adaptation Evolution Strategy, CMA-ES, serves for
nonlinear function minimization.

The **main functionality** is implemented in

1. class `CMAES`, and

2. function `fmin` which is a single-line-usage wrapper around `CMAES`.

This code has two **purposes**:

1. it might be used for READING and UNDERSTANDING the basic flow and
   the details of the CMA-ES *algorithm*. The source code is meant to
   be read. For short, study the class `CMAES`, in particular its doc
   string and the code of the method `CMAES.tell`, where all the real
   work is done in about 20 lines (see "def tell" in the source).
   Otherwise, reading from the top is a feasible option.

2. it might be used when the python module `numpy` is not available.
   When `numpy` is available, rather use `cma.fmin` or
   `cma.CMAEvolutionStrategy` to run serious simulations.
   The latter code has many more lines, but executes faster
   (roughly ten times), offers a richer user interface, better
   termination options, supposedly quite useful output, boundary and
   noise handling, injection, automatic restarts...

Dependencies: `math.exp`, `math.log` and `random.normalvariate` (modules
`matplotlib.pylab` and `sys` are optional).

Testing: call ``python purecma.py`` at the OS shell. Tested with
Python 2.6, 2.7, 3.3, 3.5, 3.6.

URL: http://github.com/CMA-ES/pycma

Last change: August, 2017, version 3.0.0

:Author: Nikolaus Hansen, 2010-2011, 2017

This code is released into the public domain (that is, you may
use and modify it however you like).

"""
from __future__ import division  # such that 1/2 != 0
from __future__ import print_function  # available since 2.6, not needed

from sys import stdout  # not strictly necessary
from math import log, exp
from random import normalvariate as random_normalvariate

try:
    from .interfaces import OOOptimizer, BaseDataLogger
except (ImportError, ValueError):
    OOOptimizer, BaseDataLogger = object, object
try:
    from .recombination_weights import RecombinationWeights
except (ImportError, ValueError):
    RecombinationWeights = None

# Optional imports, can be out-commented, if not available

__version__ = '3.0.0'
__author__ = 'Nikolaus Hansen'
__docformat__ = 'reStructuredText'


def fmin(objective_fct, xstart, sigma, args=(),
         maxfevals='1e3 * N**2', ftarget=None,
         verb_disp=100, verb_log=1, verb_save=1000):
    """non-linear non-convex minimization procedure, a functional
    interface to CMA-ES.

    Parameters
    ==========
        `objective_fct`: `callable`
            a function that takes as input a `list` of floats (like
            [3.0, 2.2, 1.1]) and returns a single `float` (a scalar).
            The objective is to find ``x`` with ``objective_fct(x)``
            to be as small as possible.
        `xstart`: `list` or sequence
            list of numbers (like `[3.2, 2, 1]`), initial solution vector
        `sigma`: `float`
            initial step-size, standard deviation in any coordinate
        `args`: `tuple` or sequence
            arguments to `objective_fct`
        `ftarget`: `float`
            target function value
        `maxfevals`: `int` or `str`
            maximal number of function evaluations, a string
            is evaluated with ``N`` being the search space dimension
        `verb_disp`: `int`
            display on console every `verb_disp` iteration, 0 for never
        `verb_log`: `int`
            data logging every `verb_log` iteration, 0 for never
        `verb_save`: `int`
            save logged data every ``verb_save * verb_log`` iteration

    Return
    ======
    A `CMAES` instance with additional attribute `logger` which is a
    `CMAESDataLogger` instance.

    Example
    =======
    The following example minimizes the function `ff.elli`::

        >> from cma import purecma, ff
        >> es = purecma.fmin(ff.elli, 10 * [0.5], 0.3, verb_disp=100)
        evals: ax-ratio max(std)   f-value
           10:     1.0  2.8e-01  198003.585517
         1000:     8.4  5.5e-02  95.9162313173
         2000:    40.5  3.6e-02  5.07618122556
         3000:   149.1  8.5e-03  0.271537247667
         4000:   302.2  4.2e-03  0.0623570374451
         5000:   681.1  5.9e-03  0.000485971681802
         6000:  1146.4  9.5e-06  5.26919100476e-10
         6510:  1009.1  2.3e-07  3.34128914738e-13
        termination by {'tolfun': 1e-12}
        best f-value = 3.34128914738e-13

        >> print(es.result[0])
        [2.1187532328944602e-07, 6.893386424102321e-08, -2.008255256456535e-09, 4.472078873398156e-09, -9.421306741003398e-09, 7.331265238205156e-09, 2.4804701814730273e-10, -6.030651566971234e-10, -6.063921614755129e-10, -1.066906137937511e-10]

        >> print(es.result[1])
        3.34128914738e-13

        >> es.logger.plot()  # needs pylab/matplotlib to be installed

    Details
    =======
    This call::

        >> import cma.purecma as pcma
        >> pcma.fmin(pcma.ff.elli, 10 * [0.5], 0.3, verb_save=0)

    and these lines::

        >> import cma.purecma as pcma
        >> es.logger = pcma.CMAESDataLogger()
        >> pcma.CMAES(10 * [0.5], 0.3).optimize(pcma.ff.elli,
        ...                                     callback=es.logger.add)

    do pretty much the same. The `verb_save` parameter to `fmin` adds
    the possibility to plot the saved data *during* the execution from a
    different Python shell like ``pcma.CMAESDataLogger().load().plot()``.

    :See: `CMAES`, `OOOptimizer`.
    """
    es = CMAES(xstart, sigma, maxfevals, ftarget)  # new optimizer instance
    es.logger = CMAESDataLogger(verb_log).add(es, force=True)  # add data row
    while not es.stop():
        X = es.ask()  # get a list of sampled candidate solutions
        fit = [objective_fct(x, *args) for x in X]  # evaluate each candidate
        es.tell(X, fit)  # update distribution parameters

    # all the remainder is only managing output behavior
        es.disp(verb_disp)
        es.logger.add(es) if verb_log and es.counteval / es.params.lam % verb_log < 1 else None
        es.logger.save() if verb_save * verb_log and es.counteval / es.params.lam % (verb_save * verb_log) < 1 else None
    if verb_disp:  # do not print by default to allow silent verbosity
        es.disp(1)
        print('termination by', es.stop())
        print('best f-value =', es.result[1])
        print('solution =', es.result[0])
    if verb_log:
        es.logger.add(es, force=True)
        es.logger.save() if verb_save else None
    return es


class CMAESParameters(object):
    """static "internal" parameter setting for `CMAES`"""
    def __init__(self, N, popsize, RecombinationWeights=None):
        """set static, fixed "strategy" parameters once and for all.

        Input parameter ``RecombinationWeights`` may be set to the class
        `RecombinationWeights`.
        """
        # Strategy parameter setting: Selection
        self.lam = eval(safe_str(popsize, ['N', 'int', 'log']))  # population size, offspring number
        self.mu = int(self.lam / 2)  # number of parents/points for recombination
        if RecombinationWeights:
            self.weights = RecombinationWeights(self.lam)
            self.mueff = self.weights.mueff
        else:  # set recombination weights "manually"
            _weights = [log(self.mu+0.5) - log(i+1) if i < self.mu else 0  # non-negative
                        for i in range(self.lam)]
            self.weights = [w / sum(_weights[:self.mu]) for w in _weights]  # sum is one
            self.mueff = sum(self.weights[:self.mu])**2 / sum(w**2 for w in self.weights[:self.mu])  # variance-effectiveness of sum w_i x_i
        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff/N) / (N+4 + 2 * self.mueff/N)  # time constant for cumulation for C
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)  # time constant for cumulation for sigma control
        self.c1 = 2 / ((N + 1.3)**2 + self.mueff)  # learning rate for rank-one update of C
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((N + 2)**2 + self.mueff)])  # and for rank-mu update
        self.damps = 2 * self.mueff/self.lam + 0.3 + self.cs  # damping for sigma, usually close to 1
        if RecombinationWeights:
            self.weights.finalize_negative_weights(N, self.c1, self.cmu)

class CMAES(OOOptimizer):  # could also inherit from object
    """class for non-linear non-convex numerical minimization with CMA-ES.

    The class implements the interface define in `OOOptimizer`, namely
    the methods `__init__`, `ask`, `tell`, `stop`, `disp` and property
    `result`.

    Examples
    --------
    All examples minimize the function `elli`, the output is not shown.
    (A preferred environment to execute all examples is
    ``ipython`` in ``%pylab`` mode.)

    First we need to import the modules we want to use::

        from cma import purecma as pcma, ff

    The shortest example uses the inherited method
    `OOOptimizer.optimize`::

        es = pcma.CMAES(8 * [0.1], 0.5).optimize(ff.elli)

    See method `CMAES.__init__` for a documentation of the input
    parameters to `CMAES`. We might have a look at the result::

        print(es.result[0])  # best solution and
        print(es.result[1])  # its function value

    `result` is a property of `CMAES.results`. In order to display more
    exciting output we do::

        logger = pcma.CMAESDataLogger()
        es = pcma.CMAES(9 * [0.5], 0.3).optimize(pcma.ff.elli, callback=logger.add)
        logger.plot()  # if matplotlib is available

    Virtually the same example can be written with an explicit loop
    instead of using `optimize()`. This gives the necessary insight
    into the `CMAES` class interface and gives entire control over the
    iteration loop ::

        optim = pcma.CMAES(9 * [0.5], 0.3)  # calls CMAES.__init__()
        logger = pcma.CMAESDataLogger().register(optim)  # logger instance

        # this loop resembles optimize()
        while not optim.stop(): # iterate
            X = optim.ask()     # get candidate solutions
            #  do whatever needs to be done, however rather don't
            #  change X unless like for example X[2] = optim.ask()[0]
            f = [pcma.ff.elli(x) for x in X]  # evaluate solutions
            optim.tell(X, f)    # do all the real work
            optim.disp(20)      # display info every 20th iteration
            logger.add()        # log another "data line"

        # final output
        print('termination by', optim.stop())
        print('best f-value =', optim.result[1])
        print('best solution =', optim.result[0])

        print('potentially better solution xmean =', optim.result[5])
        print("let's check f(xmean) = ", pcma.ff.elli(optim.result[5]))
        logger.plot()  # if matplotlib is available

    A slightly longer example, which may also save data within the loop,
    is the implementation of function `fmin`.

    Details
    -------
    Most of the work is done in the method `tell`. The property
    `result` contains more useful output.

    :See: `fmin`, `OOOptimizer.optimize`

    """
    def __init__(self, xstart, sigma,  # mandatory
                 maxfevals='1e3 * N**2',
                 ftarget=None,
                 popsize="4 + int(3 * log(N))",
                 randn=random_normalvariate):
        """Instantiate `CMAES` object instance, the first two arguments
        are mandatory.

        Parameters
        ----------
            `xstart`: `list`
                of numbers (like ``[3, 2, 1.2]``), initial
                solution vector
            `sigma`: `float`
                initial step-size (standard deviation in each coordinate)
            `maxfevals`: `int` or `str`
                maximal number of function evaluations, a string is
                evaluated with ``N`` being the search space dimension
            `ftarget`: `float`
                target function value
            `popsize`: `int`
                population size, number of candidate samples per iteration
            `randn`: `callable`
                normal random number generator, by default
                `random.normalvariate`

        Details: this method initializes the dynamic state variables and
        creates a `CMAESParameters` instance for static parameters.
        """
        # process some input parameters and set static parameters
        N = len(xstart)  # number of objective variables/problem dimension
        self.ftarget = ftarget  # stop if fitness <= ftarget
        self.maxfevals = eval(safe_str(maxfevals, ['N']))  # eval a string
        self.randn = randn
        self.params = CMAESParameters(N, popsize)

        # initializing dynamic state variables
        self.xmean = xstart[:]  # initial point, distribution mean, a copy
        self.sigma = sigma
        self.pc = N * [0]  # evolution path for C
        self.ps = N * [0]  # and for sigma
        self.C = Matrix(N)  # covariance matrix
        self.B = eye(N)  # B defines the coordinate system
        self.D = N * [1]  # diagonal D defines the scaling
        self.invsqrtC = eye(N)  # C^-1/2
        self.eigeneval = 0      # tracking the update of B and D from C
        self.counteval = 0  # countiter should be equal to counteval / lam
        self.fitvals = []   # for bookkeeping output and termination
        self.best = BestSolution()

    def ask(self):
        """return a list of lambda candidate solutions,

        distributed according to::

            m + sigma * Normal(0,C) = m + sigma * B * D * Normal(0,I)
                                    = m + B * D * sigma * Normal(0,I)
        """
        self._updateBD()  # update B, D and invsqrtC from C
        candidate_solutions = []
        for k in range(self.params.lam):  # repeat lam times
            z = [di * self.sigma * self.randn(0, 1) for di in self.D]
            candidate_solutions.append(plus(self.xmean, dot(self.B, z)))
        return candidate_solutions

    def tell(self, arx, fitvals):
        """update the evolution paths and the distribution parameters m,
        sigma, and C within CMA-ES.

        Parameters
        ----------
            `arx`: `list` of "row vectors" (arx[k][i] is the i-th element of vector k)
                a list of solutions, presumably from calling `ask`
            `fitvals`: `list`
                the corresponding objective function values
        """
        ### bookkeeping and convenience short cuts
        self.counteval += len(fitvals)  # used evaluations for updating
        N = len(self.xmean)
        iN = range(N)  # iterate from 0 to N-1
        par = self.params
        xold = self.xmean

        ### Sort by fitness
        arx = [arx[k] for k in argsort(fitvals)]  # sorted arx
        self.fitvals = sorted(fitvals)  # used for termination and display only
        self.best.update(arx[0], self.fitvals[0], self.counteval)

        ### recombination, compute new weighted mean value
        self.xmean = dot(arx[0:par.mu], par.weights[:par.mu], transpose=True)
        #          = [sum(self.weights[k] * arx[k][i] for k in range(self.mu))
        #                                             for i in iN]

        ### Cumulation: update evolution paths
        y = minus(self.xmean, xold)
        z = dot(self.invsqrtC, y)  # == C**(-1/2) * (xnew - xold)
        csn = (par.cs * (2 - par.cs) * par.mueff)**0.5 / self.sigma
        for i in iN:  # do the update
            self.ps[i] = (1 - par.cs) * self.ps[i] + csn * z[i]
        # turn off rank-one update if sigma should increase quickly
        hsig = (sum(x**2 for x in self.ps)  # squared length of ps
                / (1-(1-par.cs)**(2*self.counteval/par.lam)) / N
                < 2 + 4./(N+1))
        ccn = (par.cc * (2 - par.cc) * par.mueff)**0.5 / self.sigma
        for i in iN:  # do the update
            self.pc[i] = (1 - par.cc) * self.pc[i] + ccn * hsig * y[i]

        ### Adapt covariance matrix C
        # minor adjustment for the variance loss by hsig
        c1a = par.c1 * (1 - (1-hsig**2) * par.cc * (2-par.cc))
        self.C.multiply_with(1 - c1a - par.cmu * sum(par.weights))  # C *= 1 - c1 - cmu
        self.C.addouter(self.pc, par.c1)  # C += c1 * pc * pc^T
        for k in range(par.lam):  # so-called rank-mu update
            # guaranty positive definiteness given appropriate negative weights
            w2 = 1 if par.weights[k] >= 0 else N / self.mahalanobis_norm(minus(arx[k], xold))**2
            self.C.addouter(minus(arx[k], xold),  # C += w * cmu * dx * dx^T
                            par.weights[k] * w2 * par.cmu / self.sigma**2)

        ### Adapt step-size sigma
        mean_square_ps = sum(x**2 for x in self.ps) / len(self.ps)
        self.sigma *= exp(min(1, (par.cs / par.damps) * (mean_square_ps - 1) / 2))
        # chiN = (1-1./(4.*N)+1./(21.*N**2))
        # self.sigma *= exp(min(1, (self.cs/self.damps) *
        #     (sum(x**2 for x in self.ps)**0.5 / chiN - 1)))

    def stop(self):
        """return satisfied termination conditions in a dictionary like
        {'termination reason':value, ...}, for example {'tolfun':1e-12},
        or the empty dict {}"""
        res = {}
        if self.counteval <= 0:
            return res
        if self.counteval >= self.maxfevals:
            res['evals'] = self.maxfevals
        if self.ftarget is not None and len(self.fitvals) > 0 \
                and self.fitvals[0] <= self.ftarget:
            res['ftarget'] = self.ftarget
        if max(self.D) > 1e7 * min(self.D):
            res['condition'] = 1e7
        if len(self.fitvals) > 1 \
                and self.fitvals[-1] - self.fitvals[0] < 1e-12:
            res['tolfun'] = 1e-12
        if self.sigma * max(self.D) < 1e-11:
            # remark: max(D) >= max(diag(C))**0.5
            res['tolx'] = 1e-11
        return res

    def _updateBD(self):
        """execute eigendecomposition of C if necessary"""
        # postpone in case to achieve O(N**2)
        # divisor 2 is chosen such that in dimension 30 eig and tell take about the same time
        if self.counteval - self.eigeneval > \
                self.params.lam / (self.params.c1 + self.params.cmu) / len(self.C) / 2:
            self.eigeneval = self.counteval
            self.D, self.B = eig(self.C)  # eigen decomposition, B==normalized eigenvectors, O(N**3)
            self.D = [d**0.5 for d in self.D]  # D contains standard deviations now
            rg = range(len(self.invsqrtC))
            # this is O(n^3) and takes about half the time of eig
            for i in rg:  # compute invsqrtC = C**(-1/2) = B D**(-1/2) B'
                for j in rg:
                    self.invsqrtC[i][j] = sum(self.B[i][k] * self.B[j][k]
                                              / self.D[k] for k in rg)
    @property
    def result(self):
        """the `tuple` (xbest, f(xbest), evaluations_xbest, evaluations,
        iterations, xmean, final_stds)
        """
        return (self.best.x,
                self.best.f,
                self.best.evals,
                self.counteval,
                int(self.counteval / self.params.lam),
                self.xmean,
                [self.sigma * self.C[i][i]**0.5 for i in range(len(self.C))])

    def disp(self, verb_modulo=1):
        """display (`print`) some iteration info to `stdout`"""
        if verb_modulo is None:
            verb_modulo = 20
        if not verb_modulo:
            return
        iteration = self.counteval / self.params.lam

        if iteration == 1 or iteration % (10 * verb_modulo) < 1:
            print('evals: ax-ratio max(std)   f-value')
        if iteration <= 2 or iteration % verb_modulo < 1:
            max_std = max([self.C[i][i] for i in range(len(self.C))])**0.5
            print(repr(self.counteval).rjust(5) + ': ' +
                  ' %6.1f %8.1e  ' % (max(self.D) / min(self.D),
                                      self.sigma * max_std) +
                  str(self.fitvals[0]))
            stdout.flush()

    def mahalanobis_norm(self, dx):
        """return ``sigma^-1 * (dx^T * C^-1 * dx)**0.5``
        """
        return sum(xi**2 for xi in dot(self.invsqrtC, dx))**0.5 / self.sigma

# -----------------------------------------------
class CMAESDataLogger(BaseDataLogger):  # could also inherit from object
    """data logger for class `CMAES`, that can record and plot data.

    TODO: optionally clear the `_data` attribute when writing the data
    to a file (to prevent memory overload), and optionally downsample
    to prevent plotting of long runs to take forever.
    """

    plotted = 0
    """plot count for all instances"""

    def __init__(self, verb_modulo=1):
        """`verb_modulo` controls whether and when logging takes place
        for each call to the method `add`

        """
        # BaseDataLogger.__init__(self)  # not necessary
        self.filename = "_CMAESDataLogger_datadict.py"
        self.optim = None
        self.modulo = verb_modulo
        self._data = {'eval': [], 'iter': [], 'stds': [], 'D': [],
                      'sig': [], 'fit': [], 'xm': [], 'more_data': []}
        self.counter = 0  # number of calls of add

    def add(self, es=None, force=False, more_data=None):
        """append some logging data from CMAES class instance `es`,
        if ``number_of_times_called modulo verb_modulo`` equals zero
        """
        es = es or self.optim
        if not isinstance(es, CMAES):
            raise RuntimeWarning('logged object must be a CMAES instance,'
                                 ' was %s' % type(es))
        dat = self._data  # a convenient alias
        self.counter += 1
        if force and self.counter == 1:
            self.counter = 0
        if (self.modulo
                and (len(dat['eval']) == 0
                     or es.counteval != dat['eval'][-1])
                and (self.counter < 4 or force
                     or int(self.counter) % self.modulo == 0)):
            dat['eval'].append(es.counteval)
            dat['iter'].append(es.counteval / es.params.lam)
            dat['stds'].append([es.C[i][i]**0.5
                                for i in range(len(es.C))])
            dat['D'].append(sorted(es.D))
            dat['sig'].append(es.sigma)
            dat['fit'].append(es.fitvals[0] if hasattr(es, 'fitvals')
                              and es.fitvals
                              else None)
            dat['xm'].append([x for x in es.xmean])
            if more_data is not None:
                dat['more_data'].append(more_data)
        return self

    def plot(self, fig_number=322):
        """plot the stored data in figure fig_number.

        Dependencies: `matlabplotlib.pylab`

        Example
        =======
        ::

            >> import cma.purecma as pcma
            >> es = pcma.CMAES(3 * [0.1], 1)
            >> logger = pcma.CMAESDataLogger().register(es)
            >> while not es.stop():
            >>     X = es.ask()
            >>     es.tell(X, [pcma.ff.elli(x) for x in X])
            >>     logger.add()
            >> logger.plot()

        """
        from matplotlib.pylab import gca, figure, plot, ylabel, grid, \
            semilogy, text, xlabel, draw, show, subplot
        figure(fig_number)

        dat = self._data  # dictionary with entries as given in __init__
        if not dat:
            return
        try:  # a hack to get the presumable population size lambda
            strpopsize = ' (popsize~' + str(dat['eval'][-2] -
                                            dat['eval'][-3]) + ')'
        except IndexError:
            strpopsize = ''

        # plot fit, Delta fit, sigma
        subplot(221)
        gca().clear()
        if dat['fit'][0] is None:
            dat['fit'][0] = dat['fit'][1]
            # should be reverted later, but let's be lazy
        assert dat['fit'].count(None) == 0
        dmin = min(dat['fit'])
        i = dat['fit'].index(dmin)
        dat['fit'][i] = max(dat['fit']) + 1
        dmin2 = min(dat['fit'])
        dat['fit'][i] = dmin
        semilogy(dat['iter'], [d - dmin if d - dmin > 1e-19 else None
                               for d in dat['fit']],
                 'c', linewidth=1)
        semilogy(dat['iter'], [max((dmin2 - dmin, 1e-19)) if d - dmin <= 1e-19 else None
                               for d in dat['fit']], 'C1*')

        semilogy(dat['iter'], [abs(d) for d in dat['fit']], 'b')
        semilogy(dat['iter'][i], abs(dmin), 'r*')
        semilogy(dat['iter'], dat['sig'], 'g')
        ylabel('f-value, Delta-f-value, sigma')
        if dat['more_data']:
            gca().twinx()
            plot(dat['iter'], dat['more_data'])
        grid(True)

        # plot xmean
        subplot(222)
        gca().clear()
        plot(dat['iter'], dat['xm'])
        for i in range(len(dat['xm'][-1])):
            text(dat['iter'][0], dat['xm'][0][i], str(i))
            text(dat['iter'][-1], dat['xm'][-1][i], str(i))
        ylabel('mean solution', ha='center')
        grid(True)

        # plot D
        subplot(223)
        gca().clear()
        semilogy(dat['iter'], dat['D'], 'm')
        xlabel('iterations' + strpopsize)
        ylabel('axes lengths')
        grid(True)

        # plot stds
        subplot(224)
        # if len(gcf().axes) > 1:
        #     sca(pylab.gcf().axes[1])
        # else:
        #     twinx()
        gca().clear()
        semilogy(dat['iter'], dat['stds'])
        for i in range(len(dat['stds'][-1])):
            text(dat['iter'][-1], dat['stds'][-1][i], str(i))
        ylabel('coordinate stds disregarding sigma', ha='center')
        grid(True)
        xlabel('iterations' + strpopsize)
        stdout.flush()
        draw()
        show()
        CMAESDataLogger.plotted += 1

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

#_____________________________________________________________________
#_________________ Fitness (Objective) Functions _____________________

class ff(object):  # instead of a submodule
    """versatile collection of test functions in static methods"""

    @staticmethod  # syntax available since 2.4
    def elli(x):
        """ellipsoid test objective function"""
        n = len(x)
        aratio = 1e3
        return sum(x[i]**2 * aratio**(2.*i/(n-1)) for i in range(n))

    @staticmethod
    def sphere(x):
        """sphere, ``sum(x**2)``, test objective function"""
        return sum(x[i]**2 for i in range(len(x)))

    @staticmethod
    def tablet(x):
        """discus test objective function"""
        return sum(xi**2 for xi in x) + (1e6-1) * x[0]**2

    @staticmethod
    def rosenbrock(x):
        """Rosenbrock test objective function"""
        n = len(x)
        if n < 2:
            raise ValueError('dimension must be greater one')
        return sum(100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2 for i
                   in range(n-1))

#_____________________________________________________________________
#_______________________ Helper Class&Functions ______________________
#
class BestSolution(object):
    """container to keep track of the best solution seen"""
    def __init__(self, x=None, f=None, evals=None):
        """take `x`, `f`, and `evals` to initialize the best solution
        """
        self.x, self.f, self.evals = x, f, evals

    def update(self, x, f, evals=None):
        """update the best solution if ``f < self.f``
        """
        if self.f is None or f < self.f:
            self.x = x
            self.f = f
            self.evals = evals
        return self
    @property
    def all(self):
        """``(x, f, evals)`` of the best seen solution"""
        return self.x, self.f, self.evals

class Matrix(list):
    """Rudimental square matrix class"""
    def __init__(self, dimension):
        """initialize with identity matrix"""
        for i in range(dimension):
            self.append(dimension * [0])
            self[i][i] = 1

    def multiply_with(self, factor):
        """multiply matrix in place with `factor`"""
        for row in self:
            for j in range(len(row)):
                row[j] *= factor
        return self

    def addouter(self, b, factor=1):
        """add in place `factor` times outer product of vector b,

        without any dimensional consistency checks done.
        """
        for i, row in enumerate(self):
            for j in range(len(row)):
                row[j] += factor * b[i] * b[j]
        return self

def eye(dimension):
    """return identity matrix as `list` of "vectors" (lists themselves)"""
    m = [dimension * [0] for i in range(dimension)]
    # m = N * [N * [0]] fails because it gives N times the same reference
    for i in range(dimension):
        m[i][i] = 1
    return m

def dot(A, b, transpose=False):
    """ usual dot product of "matrix" A with "vector" b.

    ``A[i]`` is the i-th row of A. With ``transpose=True``, A transposed
    is used.
    """
    n = len(b)
    if not transpose:
        m = len(A)  # number of rows, like printed by pprint
        v = m * [0]
        for i in range(m):
            v[i] = sum(b[j] * A[i][j] for j in range(n))
    else:
        m = len(A[0])  # number of columns
        v = m * [0]
        for i in range(m):
            v[i] = sum(b[j] * A[j][i] for j in range(n))
    return v

def plus(a, b):
    """add vectors, return a + b """
    return [a[i] + b[i] for i in range(len(a))]

def minus(a, b):
    """subtract vectors, return a - b"""
    try:
        return [a[i] - b[i] for i in range(len(a))]
    except:
        return [minus(a[i], b[i]) for i in range(len(a))]

def argsort(a):
    """return index list to get `a` in order, ie
    ``a[argsort(a)[i]] == sorted(a)[i]``
    """
    return sorted(range(len(a)), key=a.__getitem__)  # a.__getitem__(i) is a[i]

def safe_str(s, known_words=None):
    """return a `str` safe to `eval` or raise an exception.

    Strings in the `list` `known_words` (or combinations thereof!) are
    considered safe to evaluate and separated with spaces on return
    unless contained in *any* other known word (which may be considered
    as an issue).
    """
    safe_chars = ' 0123456789.,+-*()[]e'
    known_words = sorted(known_words if known_words else [],
                         key=len, reverse=True)  # long words first
    s = str(s)
    stest = s[:]  # test this string
    sret = s[:]  # return this string
    for word in known_words:
        stest = stest.replace(word, '  ')
        if not any(word in w for w in known_words if w != word):
            sret = sret.replace(word, " %s " % word)
    for c in stest:
        if c not in safe_chars:
            raise ValueError('"%s" is not a safe string'
                             ' (known words are %s)' % (s, str(known_words)))
    return sret

#____________________________________________________________
#____________________________________________________________
#
# C and B are arrays rather than matrices, because they are
# addressed via B[i][j], matrices can only be addressed via B[i,j]

# tred2(N, B, diagD, offdiag);
# tql2(N, diagD, offdiag, B);

# Symmetric Householder reduction to tridiagonal form, translated from
#   JAMA package.

def eig(C):
    """eigendecomposition of a symmetric matrix.

    Return the eigenvalues and an orthonormal basis
    of the corresponding eigenvectors, ``(EVals, Basis)``, where

    - ``Basis[i]``: `list`, is the i-th row of ``Basis``
    - the i-th column of ``Basis``, ie ``[Basis[j][i] for j in range(len(Basis))]``
      is the i-th eigenvector with eigenvalue ``EVals[i]``

    Details: much slower than `numpy.linalg.eigh`.
    """
    # class eig(object):
    #     def __call__(self, C):

    # Householder transformation of a symmetric matrix V into tridiagonal
    #   form.
    # -> n             : dimension
    # -> V             : symmetric nxn-matrix
    # <- V             : orthogonal transformation matrix:
    #                    tridiag matrix == V * V_in * V^t
    # <- d             : diagonal
    # <- e[0..n-1]     : off diagonal (elements 1..n-1)

    # Symmetric tridiagonal QL algorithm, iterative
    # Computes the eigensystem from a tridiagonal matrix in roughtly 3N^3
    #    operations
    # -> n     : Dimension.
    # -> d     : Diagonale of tridiagonal matrix.
    # -> e[1..n-1] : off-diagonal, output from Householder
    # -> V     : matrix output von Householder
    # <- d     : eigenvalues
    # <- e     : garbage?
    # <- V     : basis of eigenvectors, according to d

    #  tred2(N, B, diagD, offdiag); B=C on input
    #  tql2(N, diagD, offdiag, B);

    #import numpy as np
    #return np.linalg.eigh(C)  # return sorted EVs
    try:
        num_opt = False  # doesn't work
        if num_opt:
            import numpy as np
    except:
        num_opt = False

    #  private void tred2 (int n, double V[][], double d[], double e[]) {
    def tred2(n, V, d, e):
        #  This is derived from the Algol procedures tred2 by
        #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        #  Fortran subroutine in EISPACK.

        # num_opt = False  # factor 1.5 in 30-D

        d[:] = V[n-1][:]  # d is output argument
        if num_opt:
            # V = np.asarray(V, dtype=float)
            e = np.asarray(e, dtype=float)

        # Householder reduction to tridiagonal form.

        for i in range(n-1, 0, -1):
            # Scale to avoid under/overflow.
            h = 0.0
            if not num_opt:
                scale = 0.0
                for k in range(i):
                    scale = scale + abs(d[k])
            else:
                scale = sum(np.abs(d[0:i]))

            if scale == 0.0:
                e[i] = d[i-1]
                for j in range(i):
                    d[j] = V[i-1][j]
                    V[i][j] = 0.0
                    V[j][i] = 0.0
            else:

                # Generate Householder vector.
                if not num_opt:
                    for k in range(i):
                        d[k] /= scale
                        h += d[k] * d[k]
                else:
                    d[:i] /= scale
                    h = np.dot(d[:i], d[:i])

                f = d[i-1]
                g = h**0.5

                if f > 0:
                    g = -g

                e[i] = scale * g
                h -= f * g
                d[i-1] = f - g
                if not num_opt:
                    for j in range(i):
                        e[j] = 0.0
                else:
                    e[:i] = 0.0

                # Apply similarity transformation to remaining columns.

                for j in range(i):
                    f = d[j]
                    V[j][i] = f
                    g = e[j] + V[j][j] * f
                    if not num_opt:
                        for k in range(j+1, i):
                            g += V[k][j] * d[k]
                            e[k] += V[k][j] * f
                        e[j] = g
                    else:
                        e[j+1:i] += V.T[j][j+1:i] * f
                        e[j] = g + np.dot(V.T[j][j+1:i], d[j+1:i])

                f = 0.0
                if not num_opt:
                    for j in range(i):
                        e[j] /= h
                        f += e[j] * d[j]
                else:
                    e[:i] /= h
                    f += np.dot(e[:i], d[:i])

                hh = f / (h + h)
                if not num_opt:
                    for j in range(i):
                        e[j] -= hh * d[j]
                else:
                    e[:i] -= hh * d[:i]

                for j in range(i):
                    f = d[j]
                    g = e[j]
                    if not num_opt:
                        for k in range(j, i):
                            V[k][j] -= (f * e[k] + g * d[k])
                    else:
                        V.T[j][j:i] -= (f * e[j:i] + g * d[j:i])

                    d[j] = V[i-1][j]
                    V[i][j] = 0.0

            d[i] = h
        # end for i--

        # Accumulate transformations.

        for i in range(n-1):
            V[n-1][i] = V[i][i]
            V[i][i] = 1.0
            h = d[i+1]
            if h != 0.0:
                if not num_opt:
                    for k in range(i+1):
                        d[k] = V[k][i+1] / h
                else:
                    d[:i+1] = V.T[i+1][:i+1] / h

                for j in range(i+1):
                    if not num_opt:
                        g = 0.0
                        for k in range(i+1):
                            g += V[k][i+1] * V[k][j]
                        for k in range(i+1):
                            V[k][j] -= g * d[k]
                    else:
                        g = np.dot(V.T[i+1][0:i+1], V.T[j][0:i+1])
                        V.T[j][:i+1] -= g * d[:i+1]

            if not num_opt:
                for k in range(i+1):
                    V[k][i+1] = 0.0
            else:
                V.T[i+1][:i+1] = 0.0

        if not num_opt:
            for j in range(n):
                d[j] = V[n-1][j]
                V[n-1][j] = 0.0
        else:
            d[:n] = V[n-1][:n]
            V[n-1][:n] = 0.0

        V[n-1][n-1] = 1.0
        e[0] = 0.0

    # Symmetric tridiagonal QL algorithm, taken from JAMA package.
    # private void tql2 (int n, double d[], double e[], double V[][]) {
    # needs roughly 3N^3 operations
    def tql2(n, d, e, V):
        #  This is derived from the Algol procedures tql2, by
        #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        #  Fortran subroutine in EISPACK.

        # num_opt = False  # True doesn't work

        if not num_opt:
            for i in range(1, n):  # (int i = 1; i < n; i++):
                e[i-1] = e[i]
        else:
            e[0:n-1] = e[1:n]
        e[n-1] = 0.0

        f = 0.0
        tst1 = 0.0
        eps = 2.0**-52.0
        for l in range(n):  # (int l = 0; l < n; l++) {

            # Find small subdiagonal element

            tst1 = max(tst1, abs(d[l]) + abs(e[l]))
            m = l
            while m < n:
                if abs(e[m]) <= eps*tst1:
                    break
                m += 1

            # If m == l, d[l] is an eigenvalue,
            # otherwise, iterate.

            if m > l:
                iiter = 0
                while 1:  # do {
                    iiter += 1  # (Could check iteration count here.)

                    # Compute implicit shift

                    g = d[l]
                    p = (d[l+1] - g) / (2.0 * e[l])
                    r = (p**2 + 1)**0.5  # hypot(p, 1.0)
                    if p < 0:
                        r = -r

                    d[l] = e[l] / (p + r)
                    d[l+1] = e[l] * (p + r)
                    dl1 = d[l+1]
                    h = g - d[l]
                    if not num_opt:
                        for i in range(l+2, n):
                            d[i] -= h
                    else:
                        d[l+2:n] -= h

                    f = f + h

                    # Implicit QL transformation.

                    p = d[m]
                    c = 1.0
                    c2 = c
                    c3 = c
                    el1 = e[l+1]
                    s = 0.0
                    s2 = 0.0

                    # hh = V.T[0].copy()  # only with num_opt
                    for i in range(m-1, l-1, -1):
                        # (int i = m-1; i >= l; i--) {
                        c3 = c2
                        c2 = c
                        s2 = s
                        g = c * e[i]
                        h = c * p
                        r = (p**2 + e[i]**2)**0.5  # hypot(p,e[i])
                        e[i+1] = s * r
                        s = e[i] / r
                        c = p / r
                        p = c * d[i] - s * g
                        d[i+1] = h + s * (c * g + s * d[i])

                        # Accumulate transformation.

                        if not num_opt:  # overall factor 3 in 30-D
                            for k in range(n):  # (int k = 0; k < n; k++){
                                h = V[k][i+1]
                                V[k][i+1] = s * V[k][i] + c * h
                                V[k][i] = c * V[k][i] - s * h
                        else:  # about 20% faster in 10-D
                            hh = V.T[i+1].copy()
                            # hh[:] = V.T[i+1][:]
                            V.T[i+1] = s * V.T[i] + c * hh
                            V.T[i] = c * V.T[i] - s * hh
                            # V.T[i] *= c
                            # V.T[i] -= s * hh

                    p = -s * s2 * c3 * el1 * e[l] / dl1
                    e[l] = s * p
                    d[l] = c * p

                    # Check for convergence.
                    if abs(e[l]) <= eps*tst1:
                        break
                # } while (Math.abs(e[l]) > eps*tst1);

            d[l] += f
            e[l] = 0.0

        # Sort eigenvalues and corresponding vectors.
        if 11 < 3:
            for i in range(n-1):  # (int i = 0; i < n-1; i++) {
                k = i
                p = d[i]
                for j in range(i+1, n):  # (int j = i+1; j < n; j++) {
                    if d[j] < p:  # NH find smallest k>i
                        k = j
                        p = d[j]

                if k != i:
                    d[k] = d[i]  # swap k and i
                    d[i] = p
                    for j in range(n):  # (int j = 0; j < n; j++) {
                        p = V[j][i]
                        V[j][i] = V[j][k]
                        V[j][k] = p
    # tql2
    N = len(C[0])
    V = [C[i][:] for i in range(N)]
    d = N * [0]
    e = N * [0]
    tred2(N, V, d, e)
    tql2(N, d, e, V)
    return d, V  # sorting of V-columns in place is non-trivial

def test():
    """test of the `purecma` module, called ``if __name__ == "__main__"``.

    Currently only based on `doctest`::

        >>> import cma.purecma as pcma
        >>> import random
        >>> random.seed(3)
        >>> es = pcma.fmin(pcma.ff.rosenbrock, 4 * [0.5], 0.5, verb_disp=0, verb_log=1)
        >>> print(es.counteval)
        1680
        >>> print(es.best.evals)
        1664
        >>> assert es.best.f < 1e-12

    Large population size::

        >>> import cma.purecma as pcma
        >>> import random
        >>> random.seed(4)
        >>> es = pcma.CMAES(3 * [1], 1, popsize=300)
        >>> es.logger = pcma.CMAESDataLogger()
        >>> es = es.optimize(pcma.ff.elli, verb_disp=0)
        >>> assert es.result[1] < 1e13
        >>> print(es.result[2])
        9000

    """
    import doctest
    print('launching doctest...')
    print(doctest.testmod(report=True, verbose=0))  # module test

#_____________________________________________________________________
#_____________________________________________________________________
#
if __name__ == "__main__":

    test()

    # fmin(ff.rosenbrock, 10 * [0.5], 0.5)
