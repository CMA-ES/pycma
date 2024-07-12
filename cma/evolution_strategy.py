# -*- coding: utf-8 -*-
"""CMA-ES (evolution strategy), the main sub-module of `cma` implementing
in particular `CMAEvolutionStrategy`, `fmin2` and further ``fmin_*``
functions.
"""

# TODO (mainly done): remove separable CMA within the code (keep as sampler only)
# TODO (low): implement a (deep enough) copy-constructor for class
#       CMAEvolutionStrategy to repeat the same step in different
#       configurations for online-adaptation of meta parameters
# TODO (complex): reconsider geno-pheno transformation. Can it be a
#       separate module that operates inbetween optimizer and objective?
#       Can we still propagate a repair of solutions to the optimizer?
#       A repair-only internal geno-pheno transformation is less
#       problematic, given the repair is idempotent. In any case, consider
#       passing a repair function in the interface instead.
#       How about gradients (should be fine)?
#       Must be *thoroughly* explored before to switch, in particular the
#       order of application of repair and other transformations, as the
#       internal repair can only operate on the internal representation.
# TODO: split tell into a variable transformation part and the "pure"
#       functionality
#       usecase: es.tell_geno(X, [func(es.pheno(x)) for x in X])
#       genotypic repair is not part of tell_geno
# TODO: self.opts['mindx'] is checked without sigma_vec, which is a little
#       inconcise. Cheap solution: project sigma_vec on smallest eigenvector?
# TODO: class _CMAStopDict implementation looks way too complicated,
#       design generically from scratch?
# TODO: separate display and logging options, those CMAEvolutionStrategy
#       instances don't use themselves (probably all?)
# TODO: check scitools.easyviz and how big the adaptation would be
# TODO: separate initialize==reset_state from __init__
# TODO: keep best ten solutions
# TODO: implement constraints handling
# TODO: eigh(): thorough testing would not hurt
# TODO: (partly done) apply style guide
# WON'T FIX ANYTIME SOON (done within fmin): implement bipop in a separate
#       algorithm as meta portfolio algorithm of IPOP and a local restart
#       option to be implemented
#       in fmin (e.g. option restart_mode in [IPOP, local])
# DONE: extend function unitdoctest, or use unittest?
# DONE: copy_always optional parameter does not make much sense,
#       as one can always copy the input argument first. Similar,
#       copy_if_changed should be keep_arg_unchanged or just copy
# DONE: expand/generalize to dynamically read "signals" from a file
#       see import ConfigParser, DictFromTagsInString,
#       function read_properties, or myproperties.py (to be called after
#       tell()), signals_filename, if given, is parsed in stop()
# DONE: switch to np.loadtxt
#
# typical parameters in scipy.optimize: disp, xtol, ftol, maxiter, maxfun,
#         callback=None
#         maxfev, diag (A sequency of N positive entries that serve as
#                 scale factors for the variables.)
#           full_output -- non-zero to return all optional outputs.
#   If xtol < 0.0, xtol is set to sqrt(machine_precision)
#    'infot -- a dictionary of optional outputs with the keys:
#                      'nfev': the number of function calls...
#
#    see eg fmin_powell
# typical returns
#        x, f, dictionary d
#        (xopt, {fopt, gopt, Hopt, func_calls, grad_calls, warnflag},
#         <allvecs>)
#

# changes:
# 20/04/xx: no negative weights for injected solutions
# 16/10/xx: versatile options are read from signals_filename
#           RecombinationWeights refined and work without numpy
#           new options: recombination_weights, timeout,
#           integer_variable with basic integer handling
#           step size parameters removed from CMAEvolutionStrategy class
#           ComposedFunction class implements function composition
# 16/10/02: copy_always parameter is gone everywhere, use
#           np.array(., copy=True)
# 16/xx/xx: revised doctests with doctest: +ELLIPSIS option, test call(s)
#           moved all test related to test.py, is quite clean now
#           "python -m cma.test" is how it works now
# 16/xx/xx: cleaning up, all kind of larger changes.
# 16/xx/xx: single file cma.py broken into pieces such that cma has now
#           become a package.
# 15/02/xx: (v1.2) sampling from the distribution sampling refactorized
#           in class Sampler which also does the (natural gradient)
#           update. New AdaptiveDecoding class for sigma_vec.
# 15/01/26: bug fix in multiplyC with sep/diagonal option
# 15/01/20: larger condition numbers for C realized by using tf_pheno
#           of GenoPheno attribute gp.
# 15/01/19: injection method, first implementation, short injections
#           and long injections with good fitness need to be addressed yet
# 15/01/xx: _prepare_injection_directions to simplify/centralize injected
#           solutions from mirroring and TPA
# 14/12/26: bug fix in correlation_matrix computation if np.diag is a view
# 14/12/06: meta_parameters now only as annotations in ## comments
# 14/12/03: unified use of base class constructor call, now always
#         super(ThisClass, self).__init__(args_for_base_class_constructor)
# 14/11/29: termination via "stop now" in file cmaes_signals.par
# 14/11/28: bug fix initialization of C took place before setting the
#           seed. Now in some dimensions (e.g. 10) results are (still) not
#           determistic due to np.linalg.eigh, in some dimensions (<9, 12)
#           they seem to be deterministic.
# 14/11/23: bipop option integration, contributed by Petr Baudis
# 14/09/30: initial_elitism option added to fmin
# 14/08/1x: developing fitness wrappers in FFWrappers class
# 14/08/xx: return value of OOOptimizer.optimize changed to self.
#           CMAOptions now need to uniquely match an *initial substring*
#           only (via method corrected_key).
#           Bug fix in CMAEvolutionStrategy.stop: termination conditions
#           are now recomputed iff check and self.countiter > 0.
#           Doc corrected that self.gp.geno _is_ applied to x0
#           Vaste reorganization/modularization/improvements of plotting
# 14/08/01: bug fix to guaranty pos. def. in active CMA
# 14/06/04: gradient of f can now be used with fmin and/or ask
# 14/05/11: global rcParams['font.size'] not permanently changed anymore,
#           a little nicer annotations for the plots
# 14/05/07: added method result_pretty to pretty print optimization result
# 14/05/06: associated show() everywhere with ion() which should solve the
#           blocked terminal problem
# 14/05/05: all instances of "unicode" removed (was incompatible to 3.x)
# 14/05/05: replaced type(x) == y with isinstance(x, y), reorganized the
#           comments before the code starts
# 14/05/xx: change the order of kwargs of OOOptimizer.optimize,
#           remove prepare method in AdaptSigma classes, various changes/cleaning
# 14/03/01: bug fix BoundaryHandlerBase.has_bounds didn't check lower bounds correctly
#           bug fix in BoundPenalty.repair len(bounds[0]) was used instead of len(bounds[1])
#           bug fix in GenoPheno.pheno, where x was not copied when only boundary-repair was applied
# 14/02/27: bug fixed when BoundPenalty was combined with fixed variables.
# 13/xx/xx: step-size adaptation becomes a class derived from CMAAdaptSigmaBase,
#           to make testing different adaptation rules (much) easier
# 12/12/14: separated CMAOptions and arguments to fmin
# 12/10/25: removed useless check_points from fmin interface
# 12/10/17: bug fix printing number of infeasible samples, moved not-in-use methods
#           timesCroot and divCroot to the right class
# 12/10/16 (0.92.00): various changes commit: bug bound[0] -> bounds[0], more_to_write fixed,
#   sigma_vec introduced, restart from elitist, trace normalization, max(mu,popsize/2)
#   is used for weight calculation.
# 12/07/23: (bug:) BoundPenalty.update respects now genotype-phenotype transformation
# 12/07/21: convert value True for noisehandling into 1 making the output compatible
# 12/01/30: class Solution and more old stuff removed r3101
# 12/01/29: class Solution is depreciated, GenoPheno and SolutionDict do the job (v0.91.00, r3100)
# 12/01/06: CMA_eigenmethod option now takes a function (integer still works)
# 11/09/30: flat fitness termination checks also history length
# 11/09/30: elitist option (using method clip_or_fit_solutions)
# 11/09/xx: method clip_or_fit_solutions for check_points option for all sorts of
#           injected or modified solutions and even reliable adaptive encoding
# 11/08/19: fixed: scaling and typical_x type clashes 1 vs array(1) vs ones(dim) vs dim * [1]
# 11/07/25: fixed: fmin wrote first and last line even with verb_log==0
#           fixed: method settableOptionsList, also renamed to versatileOptions
#           default seed depends on time now
# 11/07/xx (0.9.92): added: active CMA, selective mirrored sampling, noise/uncertainty handling
#           fixed: output argument ordering in fmin, print now only used as function
#           removed: parallel option in fmin
# 11/07/01: another try to get rid of the memory leak by replacing self.unrepaired = self[:]
# 11/07/01: major clean-up and reworking of abstract base classes and of the documentation,
#           also the return value of fmin changed and attribute stop is now a method.
# 11/04/22: bug-fix: option fixed_variables in combination with scaling
# 11/04/21: stopdict is not a copy anymore
# 11/04/15: option fixed_variables implemented
# 11/03/23: bug-fix boundary update was computed even without boundaries
# 11/03/12: bug-fix of variable annotation in plots
# 11/02/05: work around a memory leak in numpy
# 11/02/05: plotting routines improved
# 10/10/17: cleaning up, now version 0.9.30
# 10/10/17: bug-fix: return values of fmin now use phenotyp (relevant
#           if input scaling_of_variables is given)
# 08/10/01: option evalparallel introduced,
#           bug-fix for scaling being a vector
# 08/09/26: option CMAseparable becomes CMA_diagonal
# 08/10/18: some names change, test functions go into a class
# 08/10/24: more refactorizing
# 10/03/09: upper bound np.exp(min(1,...)) for step-size control

from __future__ import (absolute_import, division, print_function,
                        )  # unicode_literals, with_statement)
import collections  # deque since Python 2.4, defaultdict since 2.5, namedtuple() since 2.6
# from builtins import ...
from .utilities.python3for2 import range  # redefine range in Python 2

import sys
import os
import time  # not really essential
import warnings  # catch numpy warnings
import ast  # for literal_eval
import numpy as np
# arange, cos, size, eye, inf, dot, floor, outer, zeros, linalg.eigh,
# sort, argsort, random, ones,...
from numpy import inf, array
# to access the built-in sum fct:  ``__builtins__.sum`` or ``del sum``
# removes the imported sum and recovers the shadowed build-in

# import logging
# logging.basicConfig(level=logging.INFO)  # only works before logging is used
# logging.info('message')  # prints INFO:root:message on red background
# logger = logging.getLogger(__name__)  # should not be done during import
# logger.info('message')  # prints INFO:cma...:message on red background
# see https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/

from . import interfaces
from . import options_parameters
from . import transformations
from . import optimization_tools as ot
from . import sampler
from .utilities import utils
from .options_parameters import CMAOptions, cma_default_options
from . import constraints_handler as _constraints_handler
from cma import fitness_models as _fitness_models
from .constraints_handler import BoundNone, BoundPenalty, BoundTransform, AugmentedLagrangian
from .integer_centering import IntegerCentering
from .logger import CMADataLogger  # , disp, plot
from .utilities.utils import BlancClass as _BlancClass
from .utilities.utils import rglen  #, global_verbosity
from .utilities.utils import seval as eval
from .utilities.utils import SolutionDict as _SolutionDict
from .utilities.math import Mh
from .sigma_adaptation import *
from . import restricted_gaussian_sampler as _rgs

_where = np.nonzero  # to make pypy work, this is how where is used here anyway
del division, print_function, absolute_import  #, unicode_literals, with_statement

class InjectionWarning(UserWarning):
    """Injected solutions are not passed to tell as expected"""

def _callable_to_list(c):
    """A `callable` is wrapped and `None` defaults to the empty list.

    All other values, in particular `False`, remain unmodified on return.
    """
    if c is None:
        return []
    elif callable(c):
        return [c]
    return c

def _pass(*args, **kwargs):
    """a callable that does nothing and return args[0] in case"""
    return args[0] if args else None

# use_archives uses collections
use_archives = sys.version_info[0] >= 3 or sys.version_info[1] >= 6
# use_archives = False  # on False some unit tests fail
"""speed up for very large population size. `use_archives` prevents the
need for an inverse gp-transformation, relies on collections module,
not sure what happens if set to ``False``. """

class _CMASolutionDict_functional(_SolutionDict):
    def __init__(self, *args, **kwargs):
        # _SolutionDict.__init__(self, *args, **kwargs)
        super(_CMASolutionDict, self).__init__(*args, **kwargs)
        self.last_solution_index = 0

    # TODO: insert takes 30% of the overall CPU time, mostly in def key()
    #       with about 15% of the overall CPU time
    def insert(self, key, geno=None, iteration=None, fitness=None,
                value=None, cma_norm=None):
        """insert an entry with key ``key`` and value
        ``value if value is not None else {'geno':key}`` and
        ``self[key]['kwarg'] = kwarg if kwarg is not None`` for the further kwargs.

        """
        # archive returned solutions, first clean up archive
        if iteration is not None and iteration > self.last_iteration and (iteration % 10) < 1:
            self.truncate(300, iteration - 3)
        elif value is not None and value.get('iteration'):
            iteration = value['iteration']
            if (iteration % 10) < 1:
                self.truncate(300, iteration - 3)

        self.last_solution_index += 1
        if value is not None:
            try:
                iteration = value['iteration']
            except:
                pass
        if iteration is not None:
            if iteration > self.last_iteration:
                self.last_solution_index = 0
            self.last_iteration = iteration
        else:
            iteration = self.last_iteration + 0.5  # a hack to get a somewhat reasonable value
        if value is not None:
            self[key] = value
        else:
            self[key] = {'pheno': key}
        if geno is not None:
            self[key]['geno'] = geno
        if iteration is not None:
            self[key]['iteration'] = iteration
        if fitness is not None:
            self[key]['fitness'] = fitness
        if cma_norm is not None:
            self[key]['cma_norm'] = cma_norm
        return self[key]

class _CMASolutionDict_empty(dict):
    """a hack to get most code examples running"""
    def insert(self, *args, **kwargs):
        pass
    def get(self, key):
        return None
    def __getitem__(self, key):
        return None
    def __setitem__(self, key, value):
        pass

_CMASolutionDict = _CMASolutionDict_functional if use_archives else _CMASolutionDict_empty
# _CMASolutionDict = _CMASolutionDict_empty

# ____________________________________________________________
# ____________________________________________________________
# check out built-in package abc: class ABCMeta, abstractmethod, abstractproperty...
# see http://docs.python.org/whatsnew/2.6.html PEP 3119 abstract base classes
#

_debugging = False  # not in use
_new_injections = True
_assertions_quadratic = True  # issue warnings
_assertions_cubic = True
_depreciated = True

class CMAEvolutionStrategyResult(collections.namedtuple(
    'CMAEvolutionStrategyResult', [
        'xbest',
        'fbest',
        'evals_best',
        'evaluations',
        'iterations',
        'xfavorite',
        'stds',
        'stop',
    ])):
    """A results tuple from `CMAEvolutionStrategy` property ``result``.

    This tuple contains in the given position and as attribute

    - 0 ``xbest`` best solution evaluated
    - 1 ``fbest`` objective function value of best solution
    - 2 ``evals_best`` evaluation count when ``xbest`` was evaluated
    - 3 ``evaluations`` evaluations overall done
    - 4 ``iterations``
    - 5 ``xfavorite`` distribution mean in "phenotype" space, to be
      considered as current best estimate of the optimum
    - 6 ``stds`` effective standard deviations, can be used to
      compute a lower bound on the expected coordinate-wise distance
      to the true optimum, which is (very) approximately stds[i] *
      dimension**0.5 / min(mueff, dimension) / 1.5 / 5 ~ std_i *
      dimension**0.5 / min(popsize / 2, dimension) / 5, where
      dimension = CMAEvolutionStrategy.N and mueff =
      CMAEvolutionStrategy.sp.weights.mueff ~ 0.3 * popsize.
    - 7 ``stop`` termination conditions in a dictionary

    The penalized best solution of the last completed iteration can be
    accessed via attribute ``pop_sorted[0]`` of `CMAEvolutionStrategy`
    and the respective objective function value via ``fit.fit[0]``.

    Details:

    - This class is of purely declarative nature and for providing
      this docstring. It does not provide any further functionality.
    - ``list(fit.fit).find(0)`` is the index of the first sampled
      solution of the last completed iteration in ``pop_sorted``.

    """

class _CMAEvolutionStrategyResult(tuple):
    """A results tuple from `CMAEvolutionStrategy` property ``result``.

    This tuple contains in the given position

    - 0 best solution evaluated, ``xbest``
    - 1 objective function value of best solution, ``f(xbest)``
    - 2 evaluation count when ``xbest`` was evaluated
    - 3 evaluations overall done
    - 4 iterations
    - 5 distribution mean in "phenotype" space, to be considered as
      current best estimate of the optimum
    - 6 effective standard deviations, give a lower bound on the expected
      coordinate-wise distance to the true optimum of (very) approximately
      std_i * dimension**0.5 / min(mueff, dimension) / 1.2 / 5
      ~ std_i * dimension**0.5 / min(popsize / 0.4, dimension) / 5, where
      mueff = CMAEvolutionStrategy.sp.weights.mueff ~ 0.3 * popsize.

    The penalized best solution of the last completed iteration can be
    accessed via attribute ``pop_sorted[0]`` of `CMAEvolutionStrategy`
    and the respective objective function value via ``fit.fit[0]``.

    Details:

    - This class is of purely declarative nature and for providing this
      docstring. It does not provide any further functionality.
    - ``list(fit.fit).find(0)`` is the index of the first sampled solution
      of the last completed iteration in ``pop_sorted``.

"""  # here starts the code: (beating the code folding glitch)
    # remark: a tuple is immutable, hence we cannot change it anymore
    # in __init__. This would work if we inherited from a `list`.
    @staticmethod
    def _generate(self):
        """return a results tuple of type `CMAEvolutionStrategyResult`.

        `_generate` is a surrogate for the ``__init__`` method, which
        cannot be used to initialize the immutable `tuple` super class.
        """
        return _CMAEvolutionStrategyResult(
            self.best.get() + (  # (x, f, evals) triple
            self.countevals,
            self.countiter,
            self.gp.pheno(self.mean[:], into_bounds=self.boundary_handler.repair),
            self.stds))  # 

class CMAEvolutionStrategy(interfaces.OOOptimizer):
    """CMA-ES stochastic optimizer class with ask-and-tell interface.

    Calling Sequences
    =================

    - ``es = CMAEvolutionStrategy(x0, sigma0)``

    - ``es = CMAEvolutionStrategy(x0, sigma0, opts)``

    - ``es = CMAEvolutionStrategy(x0, sigma0).optimize(objective_fct)``

    - ::

        res = CMAEvolutionStrategy(x0, sigma0,
                                opts).optimize(objective_fct).result

    Arguments
    =========
    `x0`
        initial solution, starting point. `x0` is given as "phenotype"
        which means, if::

            opts = {'transformation': [transform, inverse]}

        is given and ``inverse is None``, the initial mean is not
        consistent with `x0` in that ``transform(mean)`` does not
        equal to `x0` unless ``transform(mean)`` equals ``mean``.
    `sigma0`
        initial standard deviation.  The problem variables should
        have been scaled, such that a single standard deviation
        on all variables is useful and the optimum is expected to
        lie within about `x0` +- ``3*sigma0``. Often one wants to
        check for solutions close to the initial point. This allows,
        for example, for an easier check of consistency of the
        objective function and its interfacing with the optimizer.
        In this case, a much smaller `sigma0` is advisable.
    `opts`
        options, a dictionary with optional settings,
        see class `CMAOptions`.

    Main interface / usage
    ======================
    The interface is inherited from the generic `OOOptimizer`
    class (see also there). An object instance is generated from::

        es = cma.CMAEvolutionStrategy(8 * [0.5], 0.2)

    The least verbose interface is via the optimize method::

        es.optimize(objective_func)
        res = es.result

    More verbosely, the optimization is done using the
    methods `stop`, `ask`, and `tell`::

        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [cma.ff.rosen(s) for s in solutions])
            es.disp()
        es.result_pretty()


    where `ask` delivers new candidate solutions and `tell` updates
    the `optim` instance by passing the respective function values
    (the objective function `cma.ff.rosen` can be replaced by any
    properly defined objective function, see `cma.ff` for more
    examples).

    To change an option, for example a termination condition to
    continue the optimization, call::

        es.opts.set({'tolfacupx': 1e4})

    The class `CMAEvolutionStrategy` also provides::

        (solutions, func_values) = es.ask_and_eval(objective_func)

    and an entire optimization can also be written like::

        while not es.stop():
            es.tell(*es.ask_and_eval(objective_func))

    Besides for termination criteria, in CMA-ES only the ranks of the
    `func_values` are relevant.

    Attributes and Properties
    =========================
    - `inputargs`: passed input arguments
    - `inopts`: passed options
    - `opts`: actually used options, some of them can be changed any
      time via ``opts.set``, see class `CMAOptions`
    - `popsize`: population size lambda, number of candidate
      solutions returned by `ask` ()
    - `logger`: a `CMADataLogger` instance utilized by `optimize`

    Examples
    ========
    Super-short example, with output shown:

    >>> import cma
    >>> # construct an object instance in 4-D, sigma0=1:
    >>> es = cma.CMAEvolutionStrategy(4 * [1], 1, {'seed':234})
    ...      # doctest: +ELLIPSIS
    (4_w,8)-aCMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=234...)

    and optimize the ellipsoid function

    >>> es.optimize(cma.ff.elli, verb_disp=1)  # doctest: +ELLIPSIS
    Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
        1      8 2.09...
    >>> assert len(es.result) == 8, es.result
    >>> assert es.result[1] < 1e-9, es.result

    The optimization loop can also be written explicitly:

    >>> es = cma.CMAEvolutionStrategy(4 * [1], 1)  # doctest: +ELLIPSIS
    (4_w,8)-aCMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=...
    >>> while not es.stop():
    ...    X = es.ask()
    ...    es.tell(X, [cma.ff.elli(x) for x in X])
    ...    es.disp()  # doctest: +ELLIPSIS
    Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
        1      8 ...

    achieving the same result as above.

    An example with lower bounds (at zero) and handling infeasible
    solutions:

    >>> import numpy as np
    >>> es = cma.CMAEvolutionStrategy(10 * [0.2], 0.5,
    ...         {'bounds': [0, np.inf]})  #doctest: +ELLIPSIS
    (5_w,...
    >>> while not es.stop():
    ...     fit, X = [], []
    ...     while len(X) < es.popsize:
    ...         curr_fit = None
    ...         while curr_fit in (None, np.nan):
    ...             x = es.ask(1)[0]
    ...             curr_fit = cma.ff.somenan(x, cma.ff.elli) # might return np.nan
    ...         X.append(x)
    ...         fit.append(curr_fit)
    ...     es.tell(X, fit)
    ...     es.logger.add()
    ...     es.disp()  #doctest: +ELLIPSIS
    Itera...
    >>>
    >>> assert es.result[1] < 1e-9, es.result
    >>> assert es.result[2] < 9000, es.result  # by internal termination
    >>> # es.logger.plot()  # will plot data
    >>> # cma.s.figshow()  # display plot window

    An example with user-defined transformation, in this case to realize
    a lower bound of 2.

    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as warns:
    ...     es = cma.CMAEvolutionStrategy(5 * [3], 0.1,
    ...                 {"transformation": [lambda x: x**2+1.2, None],
    ...                  "ftarget": 1e-7 + 5.54781521192,
    ...                  "verbose": -2,})
    >>> warns[0].message  # doctest:+ELLIPSIS
    UserWarning('in class GenoPheno: user defined transformations have not been tested thoroughly (...
    >>> warns[1].message  # doctest:+ELLIPSIS
    UserWarning('computed initial point...
    >>> es.optimize(cma.ff.rosen, verb_disp=0)  #doctest: +ELLIPSIS
    <cma...
    >>> assert cma.ff.rosen(es.result[0]) < 1e-7 + 5.54781521192, es.result
    >>> assert es.result[2] < 3300, es.result

    The inverse transformation is (only) necessary if the `BoundPenalty`
    boundary handler is used at the same time.

    The `CMAEvolutionStrategy` class also provides a default logger
    (cave: files are overwritten when the logger is used with the same
    filename prefix):

    >>> es = cma.CMAEvolutionStrategy(4 * [0.2], 0.5, {'verb_disp': 0})
    >>> es.logger.disp_header()  # annotate the print of disp
    Iterat Nfevals  function value    axis ratio maxstd  minstd
    >>> while not es.stop():
    ...     X = es.ask()
    ...     es.tell(X, [cma.ff.sphere(x) for x in X])
    ...     es.logger.add()  # log current iteration
    ...     es.logger.disp([-1])  # display info for last iteration   #doctest: +ELLIPSIS
        1  ...
    >>> es.logger.disp_header()
    Iterat Nfevals  function value    axis ratio maxstd  minstd
    >>> # es.logger.plot() # will make a plot

    Example implementing restarts with increasing popsize (IPOP):

    >>> bestever = cma.optimization_tools.BestSolution()
    >>> for lam in 10 * 2**np.arange(8):  # 10, 20, 40, 80, ..., 10 * 2**7
    ...     es = cma.CMAEvolutionStrategy(6 - 8 * np.random.rand(4),  # 4-D
    ...                                   5,  # initial std sigma0
    ...                                   {'popsize': lam,  # options
    ...                                    'verb_append': bestever.evalsall})
    ...     # logger = cma.CMADataLogger().register(es, append=bestever.evalsall)
    ...     while not es.stop():
    ...         X = es.ask()    # get list of new solutions
    ...         fit = [cma.ff.rastrigin(x) for x in X]  # evaluate each solution
    ...         es.tell(X, fit) # besides for termination only the ranking in fit is used
    ...
    ...         # display some output
    ...         # logger.add()  # add a "data point" to the log, writing in files
    ...         es.disp()  # uses option verb_disp with default 100
    ...
    ...     print('termination:', es.stop())
    ...     cma.s.pprint(es.best.__dict__)
    ...
    ...     bestever.update(es.best)
    ...
    ...     # show a plot
    ...     # logger.plot();
    ...     if bestever.f < 1e-8:  # global optimum was hit
    ...         break  #doctest: +ELLIPSIS
    (5_w,...
    >>> assert es.result[1] < 1e-8, es.result

    On the Rastrigin function, usually after five restarts the global
    optimum is located.

    Using the `multiprocessing` module, we can evaluate the function in
    parallel with a simple modification of the example (however
    multiprocessing seems not always reliable):

    >>> from cma.fitness_functions import elli  # cannot be an instance method
    >>> from cma.optimization_tools import EvalParallel2
    >>> es = cma.CMAEvolutionStrategy(22 * [0.0], 1.0, {'maxiter':10})  # doctest:+ELLIPSIS
    (6_w,13)-aCMA-ES (mu_w=...
    >>> with EvalParallel2(elli, es.popsize + 1) as eval_all:
    ...     while not es.stop():
    ...         X = es.ask()
    ...         es.tell(X, eval_all(X))
    ...         es.disp()
    ...         # es.logger.add()  # doctest:+ELLIPSIS
    Iterat...

    The final example shows how to resume:

    >>> import pickle
    >>>
    >>> es0 = cma.CMAEvolutionStrategy(12 * [0.1],  # a new instance, 12-D
    ...                                0.12)         # initial std sigma0
    ...   #doctest: +ELLIPSIS
    (5_w,...
    >>> es0.optimize(cma.ff.rosen, iterations=100)  #doctest: +ELLIPSIS
    I...
    >>> s = es0.pickle_dumps()  # return pickle.dumps(es) with safeguards
    >>> # save string s to file like open(filename, 'wb').write(s)
    >>> del es0  # let's start fresh
    >>> # s = open(filename, 'rb').read()  # load string s from file
    >>> es = pickle.loads(s)  # read back es instance from string
    >>> # resuming
    >>> es.optimize(cma.ff.rosen, verb_disp=200)  #doctest: +ELLIPSIS
      200 ...
    >>> assert es.result[2] < 15000, es.result
    >>> assert cma.s.Mh.vequals_approximately(es.result[0], 12 * [1], 1e-5), es.result
    >>> assert len(es.result) == 8, es.result

    Details
    =======
    The following two enhancements are implemented, the latter is only
    turned on by default for very small population sizes.

    *Active CMA* is implemented with option ``CMA_active`` and
    conducts an update of the covariance matrix with negative weights.
    The negative update is implemented, such that positive definiteness
    is guarantied. A typical speed up factor (number of f-evaluations)
    is between 1.1 and two.

    References: Jastrebski and Arnold, Improving evolution strategies
    through active covariance matrix adaptation, CEC 2006.
    Hansen, The CMA evolution strategy: a tutorial, arXiv 2016.

    *Selective mirroring* is implemented with option ``CMA_mirrors``
    in the method `get_mirror` and `get_selective_mirrors`.
    The method `ask_and_eval` (used by `fmin`) will then sample
    selectively mirrored vectors within the iteration
    (``CMA_mirrormethod==1``). Otherwise, or if ``CMA_mirromethod==2``,
    selective mirrors are injected for the next iteration.
    In selective mirroring, only the worst solutions are mirrored. With
    the default small number of mirrors, *pairwise selection* (where at
    most one of the two mirrors contribute to the update of the
    distribution mean) is implicitly guarantied under selective
    mirroring and therefore not explicitly implemented.

    Update: pairwise selection for injected mirrors is also applied in the
    covariance matrix update: for all injected solutions, as for those from
    TPA, this is now implemented in that the recombination weights are
    constrained to be nonnegative for injected solutions in the covariance
    matrix (otherwise recombination weights are anyway nonnegative). This
    is a precaution to prevent failure when injected solutions are
    systematically bad (see e.g. https://github.com/CMA-ES/pycma/issues/124),
    but may not be "optimal" for mirrors.

    References: Brockhoff et al, PPSN 2010, Auger et al, GECCO 2011.

    :See also: `fmin` (), `OOOptimizer`, `CMAOptions`, `plot` (), `ask` (),
        `tell` (), `ask_and_eval` ()

"""  # here starts the code: (beating the code folding glitch)
    @property  # read only attribute decorator for a method
    def popsize(self):
        """number of samples by default returned by `ask` ()
        """
        return self.sp.popsize

    # this is not compatible with python2.5:
    #     @popsize.setter
    #     def popsize(self, p):
    #         """popsize cannot be set (this might change in future)
    #         """
    #         raise RuntimeError("popsize cannot be changed")

    def stop(self, check=True, ignore_list=(), check_in_same_iteration=False,
             get_value=None):
        """return the termination status as dictionary.

        With ``check == False``, the termination conditions are not checked
        and the status might not reflect the current situation.
        ``check_on_same_iteration == False`` (new) does not re-check during
        the same iteration. When termination options are manually changed,
        it must be set to `True` to advance afterwards.
        ``stop().clear()`` removes the currently active termination
        conditions.

        As a convenience feature, keywords in `ignore_list` are removed from
        the conditions.

        If `get_value` is set to a condition name (not the empty string),
        `stop` does not update the termination dictionary but returns the
        measured value that would be compared to the threshold. This only
        works for some conditions, like 'tolx'. If the condition name is
        not known or cannot be computed, `None` is returned and no warning
        is issued.

        Testing `get_value` functionality:

        >>> import cma
        >>> es = cma.CMAEvolutionStrategy(2 * [1], 1e4, {'verbose': -9})
        >>> with warnings.catch_warnings(record=True) as w:
        ...     es.stop(get_value='tolx')  # triggers zero iteration warning
        ...     assert len(w) == 1, [str(wi) for wi in w]
        >>> es = es.optimize(cma.ff.sphere, iterations=4)
        >>> assert 1e3 < es.stop(get_value='tolx') < 1e4, es.stop(get_value='tolx')
        >>> assert es.stop() == {}
        >>> assert es.stop(get_value='catch 22') is None

    """
        if (check and self.countiter > 0 and self.opts['termination_callback'] and
                self.opts['termination_callback'] != str(self.opts['termination_callback'])):
            self.callbackstop = utils.ListOfCallables(self.opts['termination_callback'])(self)

        self._stopdict._get_value = get_value  # a hack to avoid passing arguments down to _add_stop and back
        # check_on_same_iteration == False makes como code much faster
        res = self._stopdict(self, check_in_same_iteration or get_value or (  # update the stopdict and return a Dict (self)
                                   check and self.countiter != self._stopdict.lastiter))
        if ignore_list:
            for key in ignore_list:
                res.pop(key, None)
        if get_value:  # deliver _value and reset
            res, self._stopdict._value = self._stopdict._value, None
        return res

    def __init__(self, x0, sigma0, inopts=None, options=None):
        """see class `CMAEvolutionStrategy`

        `options` is for consistency with `fmin2` options and is only
        in effect if ``inopts is None``.
        """
        if options and inopts is None:
            inopts = options
        del options
        self.inputargs = dict(locals())  # for the record
        del self.inputargs['self']  # otherwise the instance self has a cyclic reference
        if inopts is None:
            inopts = {}
        self.inopts = inopts
        opts = CMAOptions(inopts).complement()  # CMAOptions() == fmin([],[]) == defaultOptions()
        if opts.eval('verbose') is None:
            opts['verbose'] = CMAOptions()['verbose']
        utils.global_verbosity = global_verbosity = opts.eval('verbose')
        if global_verbosity < -8:
            opts['verb_disp'] = 0
            opts['verb_log'] = 0
            opts['verb_plot'] = 0

        if 'noise_handling' in opts and opts.eval('noise_handling'):
            raise ValueError('noise_handling not available with class CMAEvolutionStrategy, use function fmin')
        if 'restarts' in opts and opts.eval('restarts'):
            raise ValueError('restarts not available with class CMAEvolutionStrategy, use function fmin')

        self._set_x0(x0)  # manage weird shapes, set self.x0
        self.N_pheno = len(self.x0)

        self.sigma0 = sigma0
        if utils.is_str(sigma0):
            raise ValueError("sigma0 must be a scalar, a string is no longer permitted")
            # self.sigma0 = eval(sigma0)  # like '1./N' or 'np.random.rand(1)[0]+1e-2'
        if np.size(self.sigma0) != 1 or np.shape(self.sigma0):
            raise ValueError('input argument sigma0 must be (or evaluate to) a scalar,'
                             ' use `cma.ScaleCoordinates` or option `"CMA_stds"` when'
                             ' different sigmas in each coordinate are in order.')
        self.sigma = self.sigma0  # goes to inialize

        # extract/expand options
        N = self.N_pheno
        if utils.is_str(opts['fixed_variables']):
            opts['fixed_variables'] = ast.literal_eval(
                    opts['fixed_variables'])
        assert (isinstance(opts['fixed_variables'], dict)
            or opts['fixed_variables'] is None)
        if isinstance(opts['fixed_variables'], dict):
            N = self.N_pheno - len(opts['fixed_variables'])
        opts.evalall(locals())  # using only N
        if np.isinf(opts['CMA_diagonal']):
            opts['CMA_diagonal'] = True
        self.opts = opts
        if not utils.is_nan(opts['seed']):
            if self.opts['randn'] is np.random.randn:
                if not opts['seed'] or opts['seed'] is time or str(opts['seed']).startswith('time'):
                    np.random.seed()
                    six_decimals = (time.time() - 1e6 * (time.time() // 1e6))
                    opts['seed'] = int(1e5 * np.random.rand() + six_decimals
                                       + 1e5 * (time.time() % 1))
                np.random.seed(opts['seed'])  # a printable seed
            elif opts['seed'] not in (None, time):
                utils.print_warning("seed=%s will never be used (seed is only used if option 'randn' is np.random.randn)"
                                    % str(opts['seed']))
        self.gp = transformations.GenoPheno(self.N_pheno,
                        opts['scaling_of_variables'],
                        opts['typical_x'],
                        opts['fixed_variables'],
                        opts['transformation'])
        if not self.gp.isidentity:
            warnings.warn("genotype-phenotype transformations induced by {0}\n may not"
                    " be compatible with more recently introduced code features (like integer handling) and are deprecated."
                    "\nRather use an objective function wrapper instead, see e.g."
                    "\n`ScaleCoordinates` or `FixVariables` in `cma.fitness_transformations`."
                    .format("""
                        opts['scaling_of_variables'],
                        opts['typical_x'],
                        opts['fixed_variables'],
                        opts['transformation'])"""),
                        DeprecationWarning
                          )

        self.boundary_handler = opts['BoundaryHandler']
        if isinstance(self.boundary_handler, type):
            self.boundary_handler = self.boundary_handler(opts['bounds'])
        elif opts['bounds'] not in (None, False, [], ()) or (
                opts['bounds'][0] is None and opts['bounds'][1] is None):
            warnings.warn("""
                Option 'bounds' ignored because a BoundaryHandler *instance* was found.
                Consider to pass only the desired BoundaryHandler class. """)
        if not self.boundary_handler.has_bounds():
            self.boundary_handler = BoundNone()  # just a little faster and well defined
        else:
            # check that x0 is in bounds
            if not self.boundary_handler.is_in_bounds(self.x0):
                if opts['verbose'] >= 0:
                    idxs = self.boundary_handler.idx_out_of_bounds(self.x0)
                    warnings.warn("""
                Initial solution is out of the domain boundaries
                in ind%s %s:
                    x0   = %s
                    ldom = %s
                    udom = %s
                THIS MIGHT LEAD TO AN EXCEPTION RAISED LATER ON.
                """ % ('ices' if len(idxs) > 1 else 'ex',
                        str(idxs),
                        str(self.gp.pheno(self.x0)),
                        str(self.boundary_handler.bounds[0]),
                        str(self.boundary_handler.bounds[1])))
            # set maxstd "in bounds" unless given explicitly
            if opts['maxstd'] is None:
                # set maxstd according to boundary range
                opts['maxstd'] = (self.boundary_handler.get_bounds('upper', self.N_pheno) -
                                  self.boundary_handler.get_bounds('lower', self.N_pheno)
                                  ) * opts['maxstd_boundrange']
            # fix corner case with integer variables and bounds mod 1 at 0.5
            self.boundary_handler.amend_bounds_for_integer_variables(
                    opts['integer_variables'])

        # set self.mean to geno(x0)
        tf_geno_backup = self.gp.tf_geno
        if self.gp.tf_pheno and self.gp.tf_geno is None:
            def identity(x):
                return x
            self.gp.tf_geno = identity  # a hack to avoid an exception
            warnings.warn(
                "computed initial point may well be wrong, because no\n"
                "inverse for the user provided phenotype transformation "
                "was given")
        self.mean = self.gp.geno(np.array(self.x0, copy=True),
                            from_bounds=self.boundary_handler.inverse,
                            copy=False)
        self.mean_after_tell = np.array(self.mean, copy=True)  # to separate any after-iteration change
        self.mean0 = array(self.mean, copy=True)  # relevant for initial injection
        self.gp.tf_geno = tf_geno_backup
        # without copy_always interface:
        # self.mean = self.gp.geno(array(self.x0, copy=True), copy_if_changed=False)
        self.N = len(self.mean)
        assert N == self.N
        # self.fmean = np.nan  # TODO name should change? prints nan in output files (OK with matlab&octave)
        # self.fmean_noise_free = 0.  # for output only

        opts.amend_integer_options(N, inopts)
        self.sp = options_parameters.CMAParameters(N, opts, verbose=opts['verbose'] > 0)
        self.sp0 = self.sp  # looks useless, as it is not a copy

        def instantiate_adapt_sigma(adapt_sigma, self):
            """return instantiated sigma adaptation object"""
            if adapt_sigma is None:
                utils.print_warning(
                    "Value `None` for option 'AdaptSigma' is ambiguous and\n"
                    "hence deprecated. AdaptSigma can be set to `True` or\n"
                    "`False` or a class or class instance which inherited from\n"
                    "`cma.sigma_adaptation.CMAAdaptSigmaBase`")
                adapt_sigma = CMAAdaptSigmaCSA
            elif adapt_sigma is True:
                if self.opts['CMA_diagonal'] is True and self.N > 299:
                    adapt_sigma = CMAAdaptSigmaTPA
                else:
                    adapt_sigma = CMAAdaptSigmaCSA
            elif adapt_sigma is False:
                adapt_sigma = CMAAdaptSigmaNone()
            if isinstance(adapt_sigma, type):  # is a class?
                # then we want the instance
                adapt_sigma = adapt_sigma(dimension=self.N, popsize=self.sp.popsize)
            return adapt_sigma
        self.adapt_sigma = instantiate_adapt_sigma(opts['AdaptSigma'], self)

        self.mean_shift_samples = True if (isinstance(self.adapt_sigma, CMAAdaptSigmaTPA) or
            opts['mean_shift_line_samples']) else False

        def eval_vector(in_, opts, N, default_value=1.0):
            """return `default_value` as scalar or `in_` after removing
            fixed variables if ``len(in_) == N``
            """
            if in_ is None:
                return default_value
            if np.size(in_) == 1:  # return scalar value
                try:
                    res = float(in_[0])
                except TypeError:
                    res = float(in_)
            elif opts['fixed_variables'] and np.size(in_) > N:
                res = array([in_[i] for i in range(len(in_))
                                    if i not in opts['fixed_variables']],
                            dtype=float)
                if len(res) != N:
                    utils.print_warning(
                        "resulting len %d != N = %d" % (len(res), N),
                        'eval_vector', iteration=self.countiter)
            elif N and len(in_) < N:  # recycle last entry
                res = np.concatenate((in_, (N - len(in_)) * [in_[-1]]),
                                     dtype=float)
            else:
                res = array(in_, dtype=float)
            if np.size(res) not in (1, N):
                raise ValueError(
                    "vector (like CMA_stds or minstd) must have "
                    "dimension %d instead of %d" % (N, np.size(res)))
            return res

        opts['minstd'] = eval_vector(opts['minstd'], opts, N, 0)
        opts['maxstd'] = eval_vector(opts['maxstd'], opts, N, np.inf)

        # iiinteger handling currently as LB-IC, see cma.integer_centering:
        if len(opts['integer_variables']):
            opts.set_integer_min_std(N, self.sp.weights.mueff)
            self.integer_centering = IntegerCentering(self)  # read opts['integer_variables'] and use boundary handler
        else:
            self.integer_centering = _pass  # do nothing by default

        if 11 < 3 and len(opts['integer_variables']):
            try:
                from . import integer
                s = utils.format_message(
                    "Option 'integer_variables' is discouraged. "
                    "Use class `cma.integer.CMAIntMixed` or function "
                    "`cma.integer.fmin_int` instead.")
                warnings.warn(s, category=DeprecationWarning)  # TODO: doesn't show up
            except ImportError:
                pass

        # initialization of state variables
        self.countiter = 0
        self._isotropic_mean_shift_iteration = -1
        self.countevals = max((0, opts['verb_append'])) \
            if not isinstance(opts['verb_append'], bool) else 0
        self.pc = np.zeros(N)
        self.pc2 = np.zeros(N)
        self.pc_neg = np.zeros(N)
        if 1 < 3:  # new version with class
            self.sigma_vec0 = eval_vector(self.opts['CMA_stds'], opts, N)  # may be a scalar
            if np.size(self.sigma_vec0) == 1 and self.opts['CMA_diagonal_decoding']:
                self.sigma_vec0 *= np.ones(N)
            self.sigma_vec = transformations.DiagonalDecoding(self.sigma_vec0)
            if np.isfinite(self.opts['CMA_dampsvec_fac']):
                self.sigma_vec *= np.ones(N)  # make sure to get a vector
        else:
            self.sigma_vec = eval_vector(self.opts['CMA_stds'], opts, N)
            if np.isfinite(self.opts['CMA_dampsvec_fac']):
                self.sigma_vec *= np.ones(N)  # make sure to get a vector
            self.sigma_vec0 = self.sigma_vec if np.isscalar(self.sigma_vec) \
                                            else self.sigma_vec.copy()
        if self.opts['CMA_diagonal']:  # is True or > 0
            # linear time and space complexity
            self.sm = sampler.GaussStandardConstant(N, randn=self.opts['randn'])
            self._updateBDfromSM(self.sm)
            if self.opts['CMA_diagonal'] is True:
                self.sp.weights.finalize_negative_weights(N,
                                                      self.sp.c1_sep,
                                                      self.sp.cmu_sep,
                                                      pos_def=False)
            elif self.opts['CMA_diagonal'] == 1:
                raise ValueError("""Option 'CMA_diagonal' == 1 is disallowed.
                Use either `True` or an iteration number > 1 up to which C should be diagonal.
                Only `True` has linear memory demand.""")
            else:  # would ideally be done when switching
                self.sp.weights.finalize_negative_weights(N,
                                                      self.sp.c1,
                                                      self.sp.cmu)
        else:
            stds = eval_vector(self.opts['CMA_teststds'], opts, N)
            if 11 < 3:
                if hasattr(self.opts['vv'], '__getitem__') and \
                        'sweep_ccov' in self.opts['vv']:
                    self.opts['CMA_const_trace'] = True
            if self.opts['CMA_sampler'] is None:
                self.sm = sampler.GaussFullSampler(stds * np.ones(N),
                    lazy_update_gap=(
                        1. / (self.sp.c1 + self.sp.cmu + 1e-23) / self.N / 10
                        if self.opts['updatecovwait'] is None
                        else self.opts['updatecovwait']),
                    constant_trace=self.opts['CMA_const_trace'],
                    randn=self.opts['randn'],
                    eigenmethod=self.opts['CMA_eigenmethod'],
                    )
                p = self.sm.parameters(mueff=self.sp.weights.mueff,
                                       lam=self.sp.weights.lambda_)
                self.sp.weights.finalize_negative_weights(N, p['c1'], p['cmu'])
            elif isinstance(self.opts['CMA_sampler'], type):
                try:
                    self.sm = self.opts['CMA_sampler'](
                                stds * np.ones(N),
                                **self.opts['CMA_sampler_options'])
                except:
                    if max(stds) > min(stds):
                        utils.print_warning("different initial standard"
                            " deviations are not supported by the current"
                            " sampler and hence ignored")
                    elif stds[0] != 1:
                        utils.print_warning("""ignoring scaling factor %f
    for sample distribution""" % stds[0])
                    self.sm = self.opts['CMA_sampler'](N,
                                **self.opts['CMA_sampler_options'])
            else:  # CMA_sampler is already initialized as class instance
                self.sm = self.opts['CMA_sampler']
            if not isinstance(self.sm, interfaces.StatisticalModelSamplerWithZeroMeanBaseClass):
                utils.print_warning("""statistical model sampler did
    not evaluate to the expected type `%s` but to type `%s`. This is
    likely to lead to an exception later on. """ % (
                    str(type(interfaces.StatisticalModelSamplerWithZeroMeanBaseClass)),
                    str(type(self.sm))))
            self._updateBDfromSM(self.sm)
        self._stds_into_limits(warn=global_verbosity > 0)  # put stds into [minstd, maxstd]
        self.dC = self.sm.variances
        self.D = self.dC**0.5  # we assume that the initial C is diagonal
        self.pop_injection_solutions = []
        self.pop_injection_directions = []
        self.number_of_solutions_asked = 0
        self.number_of_injections_delivered = 0  # used/delivered in asked

        # self.gp.pheno adds fixed variables
        relative_stds = ((self.gp.pheno(self.mean + self.sigma * self.sigma_vec * self.D)
                          - self.gp.pheno(self.mean - self.sigma * self.sigma_vec * self.D)) / 2.001  # .001 fixes warning due to initial variation in self.D
                         / (self.boundary_handler.get_bounds('upper', self.N_pheno)
                            - self.boundary_handler.get_bounds('lower', self.N_pheno)))
        if np.any(relative_stds > 1):
            idx = np.nonzero(relative_stds > 1)[0]
            s = (
            "ValueWarning:\n\n"
            "  Initial standard deviation%s larger than the bounded domain size in variable%s.\n"
            "  Consider using `cma.ScaleCoordinates` if the bounded domain sizes differ significantly. "
            "\n" % (("s sigma0 x stds are", 's %s' % str(idx))
                    if len(idx) > 1 else (" sigma0 x stds is",
                                          ' %s' % str(idx[0]))))
            warnings.warn(s)
        self._flgtelldone = True
        self.itereigenupdated = self.countiter
        self.count_eigen = 0
        self.noiseS = 0  # noise "signal"
        self.hsiglist = []

        self.sent_solutions = _CMASolutionDict()
        self.archive = _CMASolutionDict()
        self._injected_solutions_archive = _SolutionDict()
        self.best = ot.BestSolution()

        self.const = _BlancClass()
        self.const.chiN = N**0.5 * (1 - 1. / (4.*N) + 1. / (21.*N**2))  # expectation of norm(randn(N,1))

        self.logger = CMADataLogger(opts['verb_filenameprefix'],
                                    modulo=opts['verb_log'],
                                    expensive_modulo=opts['verb_log_expensive']).register(self)

        self._stopdict = _CMAStopDict()
        "    attribute for stopping criteria in function stop"
        self._stoptolxstagnation = _StopTolXStagnation(self.mean)
        self.callbackstop = ()
        "    return values of callbacks, used like ``if any(callbackstop)``"
        self.fit = _BlancClass()
        self.fit.fit = None  # objective function values sorted
        self.fit.bndpen = None  # boundary penalty values
        self.fit.fit_plus_pen = None # obj fct values + bndpen (not sorted)
        self.fit.idx = None  # sort index from fit_plus_pen
        self.fit.hist = []  # short history of best
        self.fit.histbest = list()  # long history of best
        self.fit.histmedian = list()  # long history of median
        self.fit.median = None
        self.fit.median0 = None
        self.fit.median_min = np.inf
        self.fit.median_previous = np.inf
        self.fit.median_got_worse = 0
        self.fit.flatfit_iterations = 0

        self.more_to_write = utils.MoreToWrite()  # [1, 1, 1, 1]  #  N*[1]  # needed when writing takes place before setting

        # say hello
        if opts['verb_disp'] > 0 and opts['verbose'] >= 0:
            sweighted = '_w' if self.sp.weights.mu > 1 else ''
            smirr = 'mirr%d' % (self.sp.lam_mirr) if self.sp.lam_mirr else ''
            print('(%d' % (self.sp.weights.mu) + sweighted + ',%d' % (self.sp.popsize) + smirr +
                  ')-' + ('a' if opts['CMA_active'] else '') + 'CMA-ES' +
                  ' (mu_w=%2.1f,w_1=%d%%)' % (self.sp.weights.mueff, int(100 * self.sp.weights[0])) +
                  ' in dimension %d (seed=%s, %s)' % (N, str(opts['seed']), time.asctime()))  # + func.__name__
            if opts['CMA_diagonal'] and self.sp.CMA_on:
                s = ''
                if opts['CMA_diagonal'] is not True:
                    s = ' for '
                    if opts['CMA_diagonal'] < np.inf:
                        s += str(int(opts['CMA_diagonal']))
                    else:
                        s += str(np.floor(opts['CMA_diagonal']))
                    s += ' iterations'
                    s += ' (1/ccov=' + str(round(1. / (self.sp.c1 + self.sp.cmu))) + ')'
                print('   Covariance matrix is diagonal' + s)

    def _set_x0(self, x0):
        """Assign `self.x0` from argument `x0`.

        Input `x0` may be a `callable` or a `list` or `numpy.ndarray` of
        the desired length.

        Below an artificial example is given, where calling `x0`
        delivers in the first two calls ``dimension * [5]`` and in
        succeeding calls``dimension * [0.01]``. Only the initial value of
        0.01 solves the Rastrigin function:

        >>> import cma
        >>> class X0:
        ...     def __init__(self, dimension):
        ...         self.irun = 0
        ...         self.dimension = dimension
        ...     def __call__(self):
        ...         """"""
        ...         self.irun += 1
        ...         return (self.dimension * [5] if self.irun < 3
        ...                 else self.dimension * [0.01])
        >>> xopt, es = cma.fmin2(cma.ff.rastrigin, X0(3), 0.01,
        ...                      {'verbose':-9}, restarts=1)
        >>> assert es.result.fbest > 1e-5
        >>> xopt, es = cma.fmin2(cma.ff.rastrigin, X0(3), 0.01,
        ...                      {'verbose':-9}, restarts=2)
        >>> assert es.result.fbest < 1e-5  # third run succeeds due to x0

        """
        try:
            x0 = x0()
        except TypeError:
            if utils.is_str(x0):
                raise ValueError("x0 may be a callable, but a string is no longer permitted")
                # x0 = eval(x0)
        self.x0 = array(x0, dtype=float, copy=True)  # should not have column or row, is just 1-D
        if self.x0.ndim == 2 and 1 in self.x0.shape:
            utils.print_warning('input x0 should be a list or 1-D array, trying to flatten ' +
                                str(self.x0.shape) + '-array')
            if self.x0.shape[0] == 1:
                self.x0 = self.x0[0]
            elif self.x0.shape[1] == 1:
                self.x0 = array([x[0] for x in self.x0])
        if self.x0.ndim != 1:
            raise ValueError('x0 must be 1-D array')
        if len(self.x0) <= 1:
            utils.print_warning('Initial solution argument x0=%s.\n'
                                'CAVEAT: Optimization in 1-D is poorly tested.'
                                % str(self.x0))
        try:
            self.x0.resize(self.x0.shape[0])  # 1-D array, not really necessary?!
        except NotImplementedError:
            pass

    def _stds_into_limits(self, warn=False):
        """set ``self.sigma_vec.scaling`` to respect ``opts['max/minstd']``
        """
        is_min = np.any(self.opts['minstd'] > 0)  # accepts also a scalar
        is_max = np.any(np.isfinite(self.opts['maxstd']))
        initial_linalg_fix = np.exp(1e-4) if self.countiter <= 2 else 1
        """prevent warning from equal eigenvals prevention hack"""
        if not is_min and not is_max:
            return
        def get_i(bnds, i):
            if np.isscalar(bnds):
                return bnds
            return bnds[i]
        for i, s in enumerate(self.stds):
            found = False
            if is_min:
                sb = get_i(self.opts['minstd'], i)
                found = s < sb
            if is_max and not found:
                sb = get_i(self.opts['maxstd'], i)
                found = s > sb * initial_linalg_fix
            if found:
                self.sigma_vec._init_(self.N)
                self.sigma_vec.set_i(i, self.sigma_vec.scaling[i] * sb / s)
                if warn:
                    warnings.warn("Sampling standard deviation i={0} at iteration {1}"
                                  " change by {2} to stds[{3}]={4}"
                                  .format(i, self.countiter, sb / s, i, self.stds[i]))

    def _copy_light(self, sigma=None, inopts=None):
        """tentative copy of self, versatile (interface and functionalities may change).

        `sigma` overwrites the original initial `sigma`.
        `inopts` allows to overwrite any of the original options.

        This copy may not work as expected depending on the used sampler.

        Copy mean and sample distribution parameters and input options. Do
        not copy evolution paths, termination status or other state variables.

        >>> import cma
        >>> es = cma.CMAEvolutionStrategy(3 * [1], 0.1,
        ...          {'bounds':[0,9], 'verbose':-9}).optimize(cma.ff.elli, iterations=10)
        >>> es2 = es._copy_light()
        >>> assert es2.sigma == es.sigma
        >>> assert not sum((es.sm.C - es2.sm.C).flat > 1e-12), (es.sm.C, es2.sm.C)
        >>> assert not sum((es.sm.C - es2.sm.C).flat < -1e-12), (es.sm.C, es2.sm.C)
        >>> es3 = es._copy_light(sigma=3)
        >>> assert es3.sigma == es3.sigma0 == 3
        >>> es.mean[0] = -11
        >>> es4 = es._copy_light(inopts={'CMA_on': False})
        >>> assert es4.sp.c1 == es4.sp.cmu == 0
        >>> assert es.mean[0] == -11 and es4.mean[0] >= -4.5

        """
        if sigma is None:
            sigma = self.sigma
        opts = dict(self.inopts)
        if inopts is not None:
            opts.update(inopts)
        es = type(self)(self.gp.pheno(self.mean[:],
                                      into_bounds=self.boundary_handler.repair),
                        sigma, opts)
        es._set_C_from(self)
        return es

    def _set_C_from(self, es, scaling=True):
        """set the current covariance matrix from another class instance.

        If `scaling`, also set the diagonal decoding scaling from `es.sigma_vec`.

        This method may not work as expected unless the default sampler is
        used.
        """
        if es.N != self.N:
            warnings.warn("setting C with dimension {0} != {1} (the current "
                          "dimension). This is likely to fail."
                          .format(es.N, self.N))
        if scaling:
            self.sigma_vec = transformations.DiagonalDecoding(es.sigma_vec.scaling)
        try:
            self.sm.C = es.sm.C.copy()
        except Exception:
            warnings.warn("`self.sm.C = es.sm.C.copy()` failed")
        self.sm.update_now(-1)  # make B and D consistent with C
        self._updateBDfromSM()

    # ____________________________________________________________
    # ____________________________________________________________
    def ask(self, number=None, xmean=None, sigma_fac=1,
            gradf=None, args=(), **kwargs):
        """get/sample new candidate solutions.

        Solutions are sampled from a multi-variate
        normal distribution and transformed to f-representation
        (phenotype) to be evaluated.

        Arguments
        ---------
            `number`
                number of returned solutions, by default the
                population size ``popsize`` (AKA ``lambda``).
            `xmean`
                distribution mean, phenotyp?
            `sigma_fac`
                multiplier for internal sample width (standard
                deviation)
            `gradf`
                gradient, ``len(gradf(x)) == len(x)``, if
                ``gradf is not None`` the third solution in the
                returned list is "sampled" in supposedly Newton
                direction ``np.dot(C, gradf(xmean, *args))``.
            `args`
                additional arguments passed to gradf

        Return
        ------
        A list of N-dimensional candidate solutions to be evaluated

        Example
        -------
        >>> import cma
        >>> es = cma.CMAEvolutionStrategy([0,0,0,0], 0.3)  #doctest: +ELLIPSIS
        (4_w,...
        >>> while not es.stop() and es.best.f > 1e-6:
        ...     X = es.ask()  # get list of new solutions
        ...     fit = [cma.ff.rosen(x) for x in X]  # call fct with each solution
        ...     es.tell(X, fit)  # feed values

        :See: `ask_and_eval`, `ask_geno`, `tell`
    """
        assert self.countiter >= 0
        if kwargs:
            utils.print_warning("""Optional argument%s \n\n  %s\n\nignored""" % (
                                    '(s)' if len(kwargs) > 1 else '', str(kwargs)),
                                "ask", "CMAEvolutionStrategy",
                                self.countiter, maxwarns=1)
        if self.countiter == 0:
            self.timer = utils.ElapsedWCTime()
        else:
            self.timer.tic
        pop_geno = self.ask_geno(number, xmean, sigma_fac)

        # N,lambda=20,200: overall CPU 7s vs 5s == 40% overhead, even without bounds!
        #                  new data: 11.5s vs 9.5s == 20%
        # TODO: check here, whether this is necessary?
        # return [self.gp.pheno(x, copy=False, into_bounds=self.boundary_handler.repair) for x in pop]  # probably fine
        # return [Solution(self.gp.pheno(x, copy=False), copy=False) for x in pop]  # here comes the memory leak, now solved
        pop_pheno = [self.gp.pheno(x, copy=True,
                                into_bounds=self.boundary_handler.repair)
                     for x in pop_geno]

        if gradf is not None:
            if not isinstance(self.sm, sampler.GaussFullSampler):
                utils.print_warning("Gradient injection may fail, because\n"
                                    "sampler attributes `B` and `D` are not present",
                                    "ask", "CMAEvolutionStrategy",
                                    self.countiter, maxwarns=1)
            try:
                # see Hansen (2011), Injecting external solutions into CMA-ES
                if not self.gp.islinear:
                    utils.print_warning("""
                    using the gradient (option ``gradf``) with a non-linear
                    coordinate-wise transformation (option ``transformation``)
                    has never been tested.""")
                    # TODO: check this out
                def grad_numerical_of_coordinate_map(x, map, epsilon=None):
                    """map is a coordinate-wise independent map, return
                    the estimated diagonal of the Jacobian.
                    """
                    eps = 1e-8 * (1 + abs(x)) if epsilon is None else epsilon
                    return (map(x + eps) - map(x - eps)) / (2 * eps)
                def grad_numerical_sym(x, func, epsilon=None):
                    """return symmetric numerical gradient of func : R^n -> R.
                    """
                    eps = 1e-8 * (1 + abs(x)) if epsilon is None else epsilon
                    grad = np.zeros(len(x))
                    ei = np.zeros(len(x))  # float is 1.6 times faster than int
                    for i in rglen(x):
                        ei[i] = eps[i]
                        grad[i] = (func(x + ei) - func(x - ei)) / (2*eps[i])
                        ei[i] = 0
                    return grad
                try:
                    if self.last_iteration_with_gradient == self.countiter:
                        utils.print_warning('gradient is used several times in ' +
                                'this iteration', iteration=self.countiter,
                                    verbose=self.opts['verbose'])
                    self.last_iteration_with_gradient = self.countiter
                except AttributeError:
                    pass
                index_for_gradient = min((2, len(pop_pheno)-1))
                if xmean is None:
                    xmean = self.mean
                xpheno = self.gp.pheno(xmean, copy=True,
                                    into_bounds=self.boundary_handler.repair)
                grad_at_mean = gradf(xpheno, *args)
                # lift gradient into geno-space
                if not self.gp.isidentity or (self.boundary_handler is not None
                        and self.boundary_handler.has_bounds()):
                    boundary_repair = None
                    gradpen = 0
                    if isinstance(self.boundary_handler, BoundTransform):
                        boundary_repair = self.boundary_handler.repair
                    elif isinstance(self.boundary_handler,
                                    BoundPenalty):
                        def fpenalty(x):
                            return self.boundary_handler.__call__(
                                x, _SolutionDict({tuple(x): {'geno': x}}), self.gp)
                        gradpen = grad_numerical_sym(
                            xmean, fpenalty)
                    elif self.boundary_handler is None or \
                            isinstance(self.boundary_handler,
                                       BoundNone):
                        pass
                    else:
                        raise NotImplementedError(
                            "unknown boundary handling method" +
                            str(self.boundary_handler) +
                            " when using gradf")
                    def _gp_for_num_grad(x):
                        return self.gp.pheno(x, into_bounds=boundary_repair)
                    gradgp = grad_numerical_of_coordinate_map(xmean, _gp_for_num_grad)
                    grad_at_mean = grad_at_mean * gradgp + gradpen

                # TODO: frozen variables brake the code (e.g. at grad of map)
                if len(grad_at_mean) != self.N or self.opts['fixed_variables']:
                    NotImplementedError("""
                    gradient with fixed variables is not (yet) implemented,
                    implement a simple transformation of the objective instead""")
                v = self.sm.D * np.dot(self.sm.B.T, self.sigma_vec * grad_at_mean)
                # newton_direction = sv * B * D * D * B^T * sv * gradient = sv * B * D * v
                # v = D^-1 * B^T * sv^-1 * newton_direction = D * B^T * sv * gradient
                q = sum(v**2)
                if q:
                    # Newton direction
                    pop_geno[index_for_gradient] = xmean - self.sigma \
                                * (self.N / q)**0.5 \
                                * (self.sigma_vec * np.dot(self.sm.B, self.sm.D * v))
                    if 11 < 3 and self.opts['vv']:
                        # gradient direction
                        q = sum((np.dot(self.sm.B.T, self.sigma_vec**-1 * grad_at_mean) / self.sm.D)**2)
                        pop_geno[index_for_gradient] = xmean - self.sigma \
                                        * (self.N / q)**0.5 * grad_at_mean \
                            if q else xmean
                else:
                    pop_geno[index_for_gradient] = xmean
                    utils.print_warning('gradient zero observed',
                                        iteration=self.countiter)
                # test "pure" gradient:
                # pop_geno[index_for_gradient] = -0.52 * grad_at_mean
                pop_pheno[index_for_gradient] = self.gp.pheno(
                    pop_geno[index_for_gradient], copy=True,
                    into_bounds=self.boundary_handler.repair)
                if 11 < 3:
                    print("x/m", pop_pheno[index_for_gradient] / self.mean)
                    print("  x-m=",
                          pop_pheno[index_for_gradient] - self.mean)
                    print("    g=", grad_at_mean)
                    print("      (x-m-g)/||g||=", (pop_pheno[index_for_gradient] - self.mean - grad_at_mean) / sum(grad_at_mean**2)**0.5
                          )
            except AttributeError:
                warnings.warn("Gradient injection failed presumably due\n"
                              "to missing attribute ``self.sm.B or self.sm.D``")

        # insert solutions, this could also (better?) be done in self.gp.pheno
        for i in rglen((pop_geno)):
            self.sent_solutions.insert(pop_pheno[i], geno=pop_geno[i],
                                       iteration=self.countiter)
        ### iiinteger handling could come here
        return pop_pheno

    # ____________________________________________________________
    # ____________________________________________________________
    def ask_geno(self, number=None, xmean=None, sigma_fac=1):
        """get new candidate solutions in genotyp.

        Solutions are sampled from a multi-variate normal distribution.

        Arguments are
            `number`
                number of returned solutions, by default the
                population size `popsize` (AKA lambda).
            `xmean`
                distribution mean
            `sigma_fac`
                multiplier for internal sample width (standard
                deviation)

        `ask_geno` returns a list of N-dimensional candidate solutions
        in genotyp representation and is called by `ask`.

        Details: updates the sample distribution if needed and might
        change the geno-pheno transformation during this update.

        :See: `ask`, `ask_and_eval`
    """
        # TODO: return one line samples depending on a switch
        #       loosely akin to the mean_shift_samples part of
        #       _prepare_injection_directions
        if number is None or number < 1:
            number = self.sp.popsize
        if self.number_of_solutions_asked == 0:
            self.number_of_injections = (
                len(self.pop_injection_directions) +
                len(self.pop_injection_solutions))

            # update distribution, might change self.mean

        # if not self.opts['tolconditioncov'] or not np.isfinite(self.opts['tolconditioncov']):
        if self.opts['conditioncov_alleviate']:
            self.alleviate_conditioning_in_coordinates(self.opts['conditioncov_alleviate'][0])
            self.alleviate_conditioning(self.opts['conditioncov_alleviate'][-1])

        xmean_arg = xmean
        if xmean is None:
            xmean = self.mean
        else:
            try:
                xmean = self.archive[xmean]['geno']
                # noise handling after call of tell
            except KeyError:
                try:
                    xmean = self.sent_solutions[xmean]['geno']
                    # noise handling before calling tell
                except KeyError:
                    pass

        if 11 < 3:
            if self.opts['CMA_AII']:
                if self.countiter == 0:
                    # self.aii = AII(self.x0, self.sigma0)
                    pass
                self._flgtelldone = False
                pop = self.aii.ask(number)
                return pop

        sigma = sigma_fac * self.sigma

        # update parameters for sampling the distribution
        #        fac  0      1      10
        # 150-D cigar:
        #           50749  50464   50787
        # 200-D elli:               == 6.9
        #                  99900   101160
        #                 100995   103275 == 2% loss
        # 100-D elli:               == 6.9
        #                 363052   369325  < 2% loss
        #                 365075   365755

        # sample distribution
        if self._flgtelldone:  # could be done in tell()!?
            self._flgtelldone = False
            self.ary = []

        # check injections from pop_injection_directions
        arinj = []
        # a hack: do not use injection when only a single solution is asked for or a solution with a specific mean
        if number > 1 and (xmean_arg is None or Mh.vequals_approximately(xmean_arg, self.mean)):
            if self.countiter < 4 and \
                    len(self.pop_injection_directions) > self.popsize - 2:
                utils.print_warning('  %d special injected samples with popsize %d, '
                                    % (len(self.pop_injection_directions), self.popsize)
                                    + "popsize %d will be used" % (len(self.pop_injection_directions) + 2)
                                    + (" and the warning is suppressed in the following" if self.countiter == 3 else ""))
            # directions must come first because of mean_shift_samples/TPA
            while self.pop_injection_directions:
                if len(arinj) >= number:
                    break
                # TODO: if len(arinj) > number, ask doesn't fulfill the contract
                y = self.pop_injection_directions.pop(0)
                # sigma_vec _is_ taken into account here
                # this may be done again in tell
                if self.mahalanobis_norm(y) > self.N**0.5 * self.opts['CMA_injections_threshold_keep_len']:
                    nominator = self._random_rescaling_factor_to_mahalanobis_size(y)
                else:
                    nominator = 1
                y *= nominator / self.sigma
                arinj.append(y)
            while self.pop_injection_solutions:
                arinj.append((self.pop_injection_solutions.pop(0) - self.mean) / self.sigma)
            if self.mean_shift_samples and self.countiter > 1:
                # TPA is implemented by injection of the Delta mean
                if len(arinj) < 2:
                    raise RuntimeError(
                        "Mean shift samples are expected but missing.\n"
                        "This happens if, for example, `ask` is called"
                        "  more than once, without calling `tell`\n"
                        "(because the first call removes the samples from"
                        " the injection list).\n"
                        "`cma.sigma_adaptation.CMAAdaptSigmaTPA`"
                        " step-size adaptation generates mean shift\n"
                        "samples and relies on them. \n"
                        "Using ``ask(1)`` for any subsequent calls of"
                        " `ask` works OK and TPA works if the\n"
                        "first two samples from the"
                        " first call are retained as first samples when"
                        " calling `tell`. \n"
                        "EXAMPLE: \n"
                        "    X = es.ask()\n"
                        "    X.append(es.ask(1)[0])\n"
                        "    ...\n"
                        "    es.tell(X, ...)"
                    )
                # for TPA, set both vectors to the same length and don't
                # ever keep the original length
                arinj[0] *= self._random_rescaling_factor_to_mahalanobis_size(arinj[0]) / self.sigma
                arinj[1] *= (np.sum(arinj[0]**2) / np.sum(arinj[1]**2))**0.5
                if not Mh.vequals_approximately(arinj[0], -arinj[1]):
                    m = utils.format_warning(
                            "mean_shift_samples, but the first two solutions"
                            " are not mirrors.",
                            "ask_geno", "CMAEvolutionStrategy",
                            self.countiter); m and warnings.warn(m)
                    # arinj[1] /= sum(arinj[0]**2)**0.5 / s1  # revert change
            self.number_of_injections_delivered += len(arinj)
            assert (self.countiter < 2 or not self.mean_shift_samples
                    or self.number_of_injections_delivered >= 2)

        Niid = number - len(arinj) # each row is a solution
        # compute ary
        if Niid >= 0:  # should better be true
            ary = self.sigma_vec * np.asarray(self.sm.sample(Niid))
            self._updateBDfromSM(self.sm)  # sm.sample invoked lazy update
            # unconditional mirroring
            if self.sp.lam_mirr and self.opts['CMA_mirrormethod'] == 0:
                for i in range(Mh.sround(self.sp.lam_mirr * number / self.popsize)):
                    if 2 * (i + 1) > len(ary):
                        utils.print_warning("fewer mirrors generated than given in parameter setting (%d<%d)"
                                            % (i, self.sp.lam_mirr),
                                       "ask_geno", "CMAEvolutionStrategy",
                                            iteration=self.countiter,
                                            maxwarns=4)
                        break
                    ary[-1 - 2 * i] = -ary[-2 - 2 * i]
            if len(arinj):
                ary = np.vstack((arinj, ary))
        else:
            ary = array(arinj)
            assert number == len(arinj)

        if (self.opts['verbose'] > 4 and self.countiter < 3 and len(arinj) and
                self.adapt_sigma is not CMAAdaptSigmaTPA):
            utils.print_message('   %d pre-injected solutions will be used (popsize=%d)' %
                                (len(arinj), len(ary)))

        pop = xmean + sigma * ary
        for i, x in enumerate(pop[:len(arinj)]):
            self._injected_solutions_archive[x] = {
                'iteration': self.countiter,  # values are currently never used
                'index': i,
                'counter': len(self._injected_solutions_archive)
                }
            # pprint(dict(self._injected_solutions_archive))
        self.evaluations_per_f_value = 1
        self.ary = ary
        self.number_of_solutions_asked += len(pop)
        return pop

    def random_rescale_to_mahalanobis(self, x):
        """change `x` like for injection, all on genotypic level"""
        x = x - self.mean  # -= fails if dtypes don't agree
        if any(x):  # let's not divide by zero
            x *= sum(self.opts['randn'](1, len(x))[0]**2)**0.5 / self.mahalanobis_norm(x)
        x += self.mean
        return x
    def _random_rescaling_factor_to_mahalanobis_size(self, y):
        """``self.mean + self._random_rescaling_factor_to_mahalanobis_size(y) * y``
        is guarantied to appear like from the sample distribution.
        """
        if len(y) != self.N:
            raise ValueError('len(y)=%d != %d=dimension' % (len(y), self.N))
        if not any(y):
            utils.print_warning("input was all-zeros, which is probably a bug",
                           "_random_rescaling_factor_to_mahalanobis_size",
                                iteration=self.countiter)
            return 1.0
        return np.sum(self.opts['randn'](1, len(y))[0]**2)**0.5 / self.mahalanobis_norm(y)


    def get_mirror(self, x, preserve_length=False):
        """return ``pheno(self.mean - (geno(x) - self.mean))``.

        >>> import numpy as np, cma
        >>> es = cma.CMAEvolutionStrategy(np.random.randn(3), 1)  #doctest: +ELLIPSIS
        (3_w,...
        >>> x = np.random.randn(3)
        >>> assert cma.utilities.math.Mh.vequals_approximately(es.mean - (x - es.mean), es.get_mirror(x, preserve_length=True))
        >>> x = es.ask(1)[0]
        >>> vals = (es.get_mirror(x) - es.mean) / (x - es.mean)
        >>> assert cma.utilities.math.Mh.equals_approximately(sum(vals), len(vals) * vals[0])

        TODO: this implementation is yet experimental.

        TODO: this implementation includes geno-pheno transformation,
        however in general GP-transformation should be separated from
        specific code.

        Selectively mirrored sampling improves to a moderate extend but
        overadditively with active CMA for quite understandable reasons.

        Optimal number of mirrors are suprisingly small: 1,2,3 for
        maxlam=7,13,20 where 3,6,10 are the respective maximal possible
        mirrors that must be clearly suboptimal.

        """
        try:
            dx = self.sent_solutions[x]['geno'] - self.mean
        except:  # can only happen with injected solutions?!
            dx = self.gp.geno(x, from_bounds=self.boundary_handler.inverse,
                              copy=True) - self.mean

        if not preserve_length:
            # dx *= sum(self.opts['randn'](1, self.N)[0]**2)**0.5 / self.mahalanobis_norm(dx)
            dx *= self._random_rescaling_factor_to_mahalanobis_size(dx)
        x = self.mean - dx
        y = self.gp.pheno(x, into_bounds=self.boundary_handler.repair)
        # old measure: costs 25% in CPU performance with N,lambda=20,200
        self.sent_solutions.insert(y, geno=x, iteration=self.countiter)
        return y

    # ____________________________________________________________
    # ____________________________________________________________
    #
    def ask_and_eval(self, func, args=(), gradf=None, number=None, xmean=None, sigma_fac=1,
                     evaluations=1, aggregation=np.median, kappa=1, parallel_mode=False):
        """sample `number` solutions and evaluate them on `func`.

        Each solution ``s`` is resampled until
        ``self.is_feasible(s, func(s)) is True``.

        Arguments
        ---------
        `func`:
            objective function, ``func(x)`` accepts a `numpy.ndarray`
            and returns a scalar ``if not parallel_mode``. Else returns a
            `list` of scalars from a `list` of `numpy.ndarray`.
        `args`:
            additional parameters for `func`
        `gradf`:
            gradient of objective function, ``g = gradf(x, *args)``
            must satisfy ``len(g) == len(x)``
        `number`:
            number of solutions to be sampled, by default
            population size ``popsize`` (AKA lambda)
        `xmean`:
            mean for sampling the solutions, by default ``self.mean``.
        `sigma_fac`:
            multiplier for sampling width, standard deviation, for example
            to get a small perturbation of solution `xmean`
        `evaluations`:
            number of evaluations for each sampled solution
        `aggregation`:
            function that aggregates `evaluations` values to
            as single value.
        `kappa`:
            multiplier used for the evaluation of the solutions, in
            that ``func(m + kappa*(x - m))`` is the f-value for ``x``.

        Return
        ------
        ``(X, fit)``, where

        - `X`: list of solutions
        - `fit`: list of respective function values

        Details
        -------
        While ``not self.is_feasible(x, func(x))`` new solutions are
        sampled. By default
        ``self.is_feasible == cma.feasible == lambda x, f: f not in (None, np.nan)``.
        The argument to `func` can be freely modified within `func`.

        Depending on the ``CMA_mirrors`` option, some solutions are not
        sampled independently but as mirrors of other bad solutions. This
        is a simple derandomization that can save 10-30% of the
        evaluations in particular with small populations, for example on
        the cigar function.

        Example
        -------
        >>> import cma
        >>> x0, sigma0 = 8 * [10], 1  # 8-D
        >>> es = cma.CMAEvolutionStrategy(x0, sigma0)  #doctest: +ELLIPSIS
        (5_w,...
        >>> while not es.stop():
        ...     X, fit = es.ask_and_eval(cma.ff.elli)  # handles NaN with resampling
        ...     es.tell(X, fit)  # pass on fitness values
        ...     es.disp(20) # print every 20-th iteration  #doctest: +ELLIPSIS
        Iterat #Fevals...
        >>> print('terminated on ' + str(es.stop()))  #doctest: +ELLIPSIS
        terminated on ...

        A single iteration step can be expressed in one line, such that
        an entire optimization after initialization becomes::

            while not es.stop():
                es.tell(*es.ask_and_eval(cma.ff.elli))

        """
        # initialize
        popsize = self.sp.popsize
        if number is not None:
            popsize = int(number)

        if self.opts['CMA_mirrormethod'] == 1:  # direct selective mirrors
            nmirrors = Mh.sround(self.sp.lam_mirr * popsize / self.sp.popsize)
            self._mirrormethod1_done = self.countiter
        else:
            # method==0 unconditional mirrors are done in ask_geno
            # method==2 delayed selective mirrors are done via injection
            nmirrors = 0
        assert nmirrors <= popsize // 2
        self.mirrors_idx = np.arange(nmirrors)  # might never be used
        is_feasible = self.opts['is_feasible']

        # do the work
        fit = []  # or np.nan * np.empty(number)
        X_first = self.ask(popsize, xmean=xmean, gradf=gradf, args=args)
        if xmean is None:
            xmean = self.mean  # might have changed in self.ask
        X = []
        if parallel_mode:
            if hasattr(func, 'evaluations'):
                evals0 = func.evaluations
            fit_first = func(X_first, *args)
            # the rest is only book keeping and warnings spitting
            if hasattr(func, 'evaluations'):
                self.countevals += func.evaluations - evals0 - self.popsize  # why not .sp.popsize ?
            if nmirrors and self.opts['CMA_mirrormethod'] > 0 and self.countiter < 2:
                utils.print_warning(
                    "selective mirrors will not work in parallel mode",
                    "ask_and_eval", "CMAEvolutionStrategy")
            if evaluations > 1 and self.countiter < 2:
                utils.print_warning(
                    "aggregating evaluations will not work in parallel mode",
                    "ask_and_eval", "CMAEvolutionStrategy")
        else:
            fit_first = len(X_first) * [None]
        for k in range(popsize):
            x, f = X_first.pop(0), fit_first.pop(0)
            rejected = -1
            while f is None or not is_feasible(x, f):  # rejection sampling
                if parallel_mode:
                    utils.print_warning(
                        "rejection sampling will not work in parallel mode"
                        " unless the parallel_objective makes a distinction\n"
                        "between called with a numpy array vs a list (of"
                        " numpy arrays) as first argument.",
                        "ask_and_eval", "CMAEvolutionStrategy")
                rejected += 1
                if rejected:  # resample
                    x = self.ask(1, xmean, sigma_fac)[0]
                elif k >= popsize - nmirrors:  # selective mirrors
                    if k == popsize - nmirrors:
                        self.mirrors_idx = np.argsort(fit)[-1:-1 - nmirrors:-1]
                    x = self.get_mirror(X[self.mirrors_idx[popsize - 1 - k]])

                # constraints handling test hardwired ccccccccccc

                length_normalizer = 1
                # zzzzzzzzzzzzzzzzzzzzzzzzz
                if 11 < 3:
                    # for some unclear reason, this normalization does not work as expected: the step-size
                    # becomes sometimes too large and overall the mean might diverge. Is the reason that
                    # we observe random fluctuations, because the length is not selection relevant?
                    # However sigma-adaptation should mainly work on the correlation, not the length?
                    # Or is the reason the deviation of the direction introduced by using the original
                    # length, which also can effect the measured correlation?
                    # Update: if the length of z in CSA is clipped at chiN+1, it works, but only sometimes?
                    length_normalizer = self.N**0.5 / self.mahalanobis_norm(x - xmean)  # self.const.chiN < N**0.5, the constant here is irrelevant (absorbed by kappa)
                    # print(self.N**0.5 / self.mahalanobis_norm(x - xmean))
                    # self.more_to_write += [length_normalizer * 1e-3, length_normalizer * self.mahalanobis_norm(x - xmean) * 1e2]

                f = func(x, *args) if kappa == 1 else \
                    func(xmean + kappa * length_normalizer * (x - xmean),
                         *args)
                if is_feasible(x, f) and evaluations > 1:
                    f = aggregation([f] + [(func(x, *args) if kappa == 1 else
                                            func(xmean + kappa * length_normalizer * (x - xmean), *args))
                                           for _i in range(int(evaluations - 1))])
                if (rejected + 1) % 1000 == 0:
                    utils.print_warning('  %d solutions rejected (f-value NaN or None) at iteration %d' %
                          (rejected, self.countiter))
            fit.append(f)
            X.append(x)
        self.evaluations_per_f_value = int(evaluations)
        if any(f is None or utils.is_nan(f) for f in fit):
            idxs = [i for i in range(len(fit))
                    if fit[i] is None or utils.is_nan(fit[i])]
            utils.print_warning("f-values %s contain None or NaN at indices %s"
                                % (str(fit[:30]) + ('...' if len(fit) > 30 else ''),
                                   str(idxs)),
                                'ask_and_tell',
                                'CMAEvolutionStrategy',
                                self.countiter)
        return X, fit

    def _prepare_injection_directions(self):
        """provide genotypic directions for TPA and selective mirroring,
        with no specific length normalization, to be used in the
        coming iteration.

        Details:
        This method is called in the end of `tell`. The result is
        assigned to ``self.pop_injection_directions`` and used in
        `ask_geno`.

        """
        # self.pop_injection_directions is supposed to be empty here
        if self.pop_injection_directions or self.pop_injection_solutions:
            raise ValueError("""Found unused injected direction/solutions.
                This could be a bug in the calling order/logics or due to
                a too small popsize used in `ask()` or when only using
                `ask(1)` repeatedly. """)
        ary = []
        if self.mean_shift_samples:
            ary = [self.mean - self.mean_old]
            ary.append(self.mean_old - self.mean)  # another copy!
            if np.all(ary[-1] == 0.0):
                utils.print_warning('zero mean shift encountered',
                               '_prepare_injection_directions',
                               'CMAEvolutionStrategy', self.countiter)
        if self.opts['pc_line_samples']: # caveat: before, two samples were used
            ary.append(self.pc.copy())
        if self.sp.lam_mirr and (
                self.opts['CMA_mirrormethod'] == 2 or (
                    self.opts['CMA_mirrormethod'] == 1 and ( # replacement for direct selective mirrors
                        not hasattr(self, '_mirrormethod1_done') or
                        self._mirrormethod1_done < self.countiter - 1))):
            i0 = len(ary)
            ary += self.get_selective_mirrors()
            self._indices_of_selective_mirrors = range(i0, len(ary))
        self.pop_injection_directions = ary
        return ary

    def get_selective_mirrors(self, number=None):
        """get mirror genotypic directions from worst solutions.

        Details:

        To be called after the mean has been updated.

        Takes the last ``number=sp.lam_mirr`` entries in the
        ``self.pop[self.fit.idx]`` as solutions to be mirrored.

        Do not take a mirror if it is suspected to stem from a
        previous mirror in order to not go endlessly back and forth.
        """
        if number is None:
            number = self.sp.lam_mirr
        if not hasattr(self, '_indices_of_selective_mirrors'):
            self._indices_of_selective_mirrors = []
        res = []
        for i in range(1, number + 1):
            if 'all-selective-mirrors' in self.opts['vv'] or self.fit.idx[-i] not in self._indices_of_selective_mirrors:
                res.append(self.mean_old - self.pop[self.fit.idx[-i]])
        assert len(res) >= number - len(self._indices_of_selective_mirrors)
        return res

    def limit_integer_relative_deltas(self, dX, threshold=None,
                                      recombination_weight_condition=None):
        """versatile: limit absolute values of int-coordinates in vector list `dX`

         relative to the current sample standard deviations and by default
         only when the respective recombination weight is negative.

        This function is currently not in effect (called with threshold=inf)
        and not guarantied to stay as is.

        ``dX == pop_sorted - mold`` where ``pop_sorted`` is a genotype.

        ``threshold=2.3`` by default.
        
        A 2.3-sigma threshold affects 2 x 1.1% of the unmodified
        (nonsorted) normal samples.
        """
        if not self.opts['integer_variables'] or not np.isfinite(threshold):
            return dX
        if threshold is None:  # TODO: how interpret negative thresholds?
            threshold = 2.3
        if recombination_weight_condition is None:
            def recombination_weight_condition(w):
                return w < 0
        elif recombination_weight_condition is True:
            def recombination_weight_condition(w):
                return True
        stds = self.sigma * self.sigma_vec.scaling * np.sqrt(self.sm.variances)
        for w, dx in zip(self.sp.weights, dX):
            if recombination_weight_condition(w):
                for i in self.opts['integer_variables']:
                    if np.abs(dx[i]) > threshold * stds[i]:  # ==> |dx[i]| > 0
                        # print('fixing dx={} sigma={}'.format(dx[i], stds[i]))
                        dx[i] *= threshold * stds[i] / np.abs(dx[i])
        return dX

    # ____________________________________________________________
    def tell(self, solutions, function_values, check_points=None,
             copy=False):
        """pass objective function values to prepare for next
        iteration. This core procedure of the CMA-ES algorithm updates
        all state variables, in particular the two evolution paths, the
        distribution mean, the covariance matrix and a step-size.

        Arguments
        ---------
        `solutions`
            list or array of candidate solution points (of
            type `numpy.ndarray`), most presumably before
            delivered by method `ask()` or `ask_and_eval()`.
        `function_values`
            list or array of objective function values
            corresponding to the respective points. Beside for termination
            decisions, only the ranking of values in `function_values`
            is used.
        `check_points`
            If ``check_points is None``, only solutions that are not generated
            by `ask()` are possibly clipped (recommended). ``False`` does not clip
            any solution (not recommended).
            If ``True``, clips solutions that realize long steps (i.e. also
            those that are unlikely to be generated with `ask()`). `check_points`
            can be a list of indices to be checked in solutions.
        `copy`
            ``solutions`` can be modified in this routine, if ``copy is False``

        Details
        -------
        `tell()` updates the parameters of the multivariate
        normal search distribution, namely covariance matrix and
        step-size and updates also the attributes ``countiter`` and
        ``countevals``. To check the points for consistency is quadratic
        in the dimension (like sampling points).

        Bugs
        ----
        The effect of changing the solutions delivered by `ask()`
        depends on whether boundary handling is applied. With boundary
        handling, modifications are disregarded. This is necessary to
        apply the default boundary handling that uses unrepaired
        solutions but might change in future.

        Example
        -------

        >>> import cma
        >>> func = cma.ff.sphere  # choose objective function
        >>> es = cma.CMAEvolutionStrategy(np.random.rand(2) / 3, 1.5)
        ... # doctest:+ELLIPSIS
        (3_...
        >>> while not es.stop():
        ...    X = es.ask()
        ...    es.tell(X, [func(x) for x in X])
        >>> es.result  # result is a `namedtuple` # doctest:+ELLIPSIS
        CMAEvolutionStrategyResult(xbest=array([...

        :See: class `CMAEvolutionStrategy`, `ask`, `ask_and_eval`, `fmin`
    """
        if self._flgtelldone:
            raise RuntimeError('tell should only be called once per iteration')

        lam = len(solutions)
        if lam != len(function_values):
            raise ValueError('#f-values = %d must equal #solutions = %d'
                             % (len(function_values), lam))
        if lam + self.sp.lam_mirr < 3:
            raise ValueError('population size ' + str(lam) +
                             ' is too small with option ' +
                             'CMA_mirrors * popsize < 0.5')
        if not np.isscalar(function_values[0]):
            try:
                if np.isscalar(function_values[0][0]):
                    if self.countiter <= 1:
                        m = utils.format_warning('''function_values is not a list of scalars,
                        the first element equals %s with non-scalar type %s.
                        Using now ``[v[0] for v in function_values]`` instead (further warnings are suppressed)'''
                                            % (str(function_values[0]), str(type(function_values[0])))
                                ); m and warnings.warn(m)
                    function_values = [val[0] for val in function_values]
                else:
                    raise ValueError('objective function values must be a list of scalars')
            except:
                utils.print_message("function values=%s" % function_values,
                                    method_name='tell', class_name='CMAEvolutionStrategy',
                                    verbose=9, iteration=self.countiter)
                raise
        if any(f is None or utils.is_nan(f) for f in function_values):
            idx_none = [i for i, f in enumerate(function_values) if f is None]
            idx_nan = [i for i, f in enumerate(function_values) if f is not None and utils.is_nan(f)]
            m = np.median([f for f in function_values
                           if f is not None and not utils.is_nan(f)])
            utils.print_warning("function values with index %s/%s are nan/None and will be set to the median value %s"
                                % (str(idx_nan), str(idx_none), str(m)), 'ask',
                                'CMAEvolutionStrategy', self.countiter)
            for i in idx_nan + idx_none:
                function_values[i] = m
        if not all(np.isfinite(float(val)) for val in function_values):
            idx = [i for i, f in enumerate(function_values)
                   if not np.isfinite(float(f))]
            utils.print_warning("function values with index %s are not finite but %s."
                                % (str(idx), str([function_values[i] for i in idx])), 'ask',
                                'CMAEvolutionStrategy', self.countiter)
        if self.number_of_solutions_asked <= self.number_of_injections:
            utils.print_warning("""no independent samples generated because the
                number of injected solutions, %d, equals the number of
                solutions asked, %d, where %d solutions remain to be injected
                """ % (self.number_of_injections,
                       self.number_of_solutions_asked,
                       len(self.pop_injection_directions) + len(self.pop_injection_solutions)),
                "ask_geno", "CMAEvolutionStrategy", self.countiter)
        self.number_of_solutions_asked = 0

        # ## prepare
        N = self.N
        sp = self.sp
        if 11 < 3 and lam != sp.popsize:  # turned off, because mu should stay constant, still not desastrous
            utils.print_warning('population size has changed, recomputing parameters')
            self.sp.set(self.opts, lam)  # not really tested
        if 1 < 3 and (lam > sp.popsize + 2 or lam < sp.popsize - 2 or (
            lam < sp.popsize and lam < 5)):  # see above
            m = "The number of solutions passed to `tell` should"
            utils.print_warning("{0} generally be the same as (or close to) the population size,"
                                "\n  was: len(solutions)={1} != {2}=popsize."
                                "\n  To suppress this warning execute"
                                "\nwarnings.filterwarnings('ignore', message='{3}.*')"
                                "\n".format(m, len(solutions), sp.popsize, m),
                                'tell', 'CMAEvolutionStrategy', self.countiter+1)
        if lam < sp.weights.mu:  # rather decrease cmean instead of having mu > lambda//2
            raise ValueError('not enough solutions passed to function tell'
                             ' (passed solutions={0} < mu={1})'
                             .format(lam, sp.weights.mu))

        self.countiter += 1  # >= 1 now
        self.countevals += lam * self.evaluations_per_f_value
        self.best.update(solutions,  # caveat: these solutions may be out-of-bounds
                         self.sent_solutions, function_values, self.countevals)
        flg_diagonal = self.opts['CMA_diagonal'] is True \
                       or self.countiter <= self.opts['CMA_diagonal']
        if not flg_diagonal and isinstance(self.sm, sampler.GaussStandardConstant):
            # switching from diagonal to full covariance learning
            self.sm = sampler.GaussFullSampler(N)
            self._updateBDfromSM(self.sm)

        self._record_rankings(function_values[:2], function_values[2:])  # for analysis

        # ## manage fitness
        fit = self.fit  # make short cut

        # CPU for N,lam=20,200: this takes 10s vs 7s
        fit.bndpen = self.boundary_handler.update(function_values, self)(solutions, self.sent_solutions, self.gp)
        # for testing:
        # fit.bndpen = self.boundary_handler.update(function_values, self)([s.unrepaired for s in solutions])
        fit.fit_plus_pen = np.asarray(fit.bndpen) + function_values
        fit.idx = np.argsort(fit.fit_plus_pen)
        fit.fit = sorted(function_values)  # was: array(function_values)[fit.idx] which can falsely trigger tolflatfitness

        # update output data TODO: this is obsolete!? However: need communicate current best x-value?
        # old: out['recent_x'] = self.gp.pheno(pop[0])
        # self.out['recent_x'] = array(solutions[fit.idx[0]])  # TODO: change in a data structure(?) and use current as identify
        # self.out['recent_f'] = fit.fit[0]

        # fitness histories
        fit.hist.insert(0, fit.fit[0])  # FIXED caveat: this may neither be the best nor the best in-bound fitness
        fit.median = (fit.fit[len(fit.fit) // 2] if len(fit.fit) % 2
                      else np.mean(fit.fit[len(fit.fit) // 2 - 1: len(fit.fit) // 2 + 1]))
        # if len(self.fit.histbest) < 120+30*N/sp.popsize or  # does not help, as tablet in the beginning is the critical counter-case
        if ((self.countiter % 5) == 0):  # 20 percent of 1e5 gen.
            fit.histbest.insert(0, fit.fit[0])
            fit.histmedian.insert(0, fit.median)
        if len(fit.histbest) > 2e4:  # 10 + 30*N/sp.popsize:
            fit.histbest.pop()
            fit.histmedian.pop()
        if len(fit.hist) > 10 + 30 * N / sp.popsize:
            fit.hist.pop()
        if fit.median0 is None:
            fit.median0 = fit.median
        if fit.median_min > fit.median:
            fit.median_min = fit.median
        if fit.median <= fit.median_previous:
            fit.median_got_worse = 0  # don't keep a high number when median is constant
        else:  # if fit.median > fit.median_previous:
            fit.median_got_worse += 1
        fit.median_previous = fit.median

        ### line 2665

        # TODO: clean up inconsistency when an unrepaired solution is available and used
        # now get the genotypes
        self.pop_sorted = None
        pop = []  # create pop from input argument solutions
        for k, s in enumerate(solutions):  # use phenotype before Solution.repair()
            if 1 < 3:
                pop += [self.gp.geno(s,
                            from_bounds=self.boundary_handler.inverse,
                            repair=(self.repair_genotype if check_points not in (False, 0, [], ()) else None),
                            archive=self.sent_solutions)]  # takes genotype from sent_solutions, if available
                try:
                    self.archive.insert(s, value=self.sent_solutions.pop(s), fitness=function_values[k])
                    # self.sent_solutions.pop(s)
                except KeyError:
                    pass
        # check that TPA mirrors are available
        self.pop = pop  # used in check_consistency of CMAAdaptSigmaTPA
        self.adapt_sigma.check_consistency(self)

        if self.countiter > 1:
            self.mean_old_old = self.mean_old
        self.mean_old = self.mean
        mold = self.mean_old  # just an alias

        # check and normalize each x - m
        # check_points is a flag (None is default: check non-known solutions) or an index list
        # should also a number possible (first check_points points)?
        if check_points not in (None, False, 0, [], ()):
            # useful in case of injected solutions and/or adaptive encoding, however is automatic with use_sent_solutions
            # by default this is not executed
            try:
                if len(check_points):
                    idx = check_points
            except:
                idx = range(sp.popsize)

            for k in idx:
                self.repair_genotype(pop[k])

        # sort pop for practicability, now pop != self.pop, which is unsorted
        pop = np.asarray(pop)[fit.idx]  # array is used for weighted recombination

        # prepend best-ever solution to population, in case
        # note that pop and fit.fit do not agree anymore in this case
        if self.opts['CMA_elitist'] == 'initial':
            if not hasattr(self, 'f0'):
                utils.print_warning(
                    'Set attribute `es.f0` to make initial elitism\n' +
                    'available or use cma.fmin.',
                    'tell', 'CMAEvolutionStrategy', self.countiter)
            elif fit.fit[0] > self.f0:
                x_elit = self.mean0.copy()
                # self.clip_or_fit_solutions([x_elit], [0]) # just calls repair_genotype
                self.random_rescale_to_mahalanobis(x_elit)
                pop = np.asarray([x_elit] + list(pop))
                utils.print_message('initial solution injected %f<%f' %
                                    (self.f0, fit.fit[0]),
                               'tell', 'CMAEvolutionStrategy',
                                    self.countiter, verbose=self.opts['verbose'])
        elif self.opts['CMA_elitist'] and self.best.f < fit.fit[0]:
            if self.best.x_geno is not None:
                xp = [self.best.x_geno]
                # xp = [self.best.xdict['geno']]
                # xp = [self.gp.geno(self.best.x[:])]  # TODO: remove
                # print self.mahalanobis_norm(xp[0]-self.mean)
            else:
                xp = [self.gp.geno(array(self.best.x, copy=True),
                                   self.boundary_handler.inverse,
                                   copy=False)]
                utils.print_warning('genotype for elitist not found', 'tell')
            # self.clip_or_fit_solutions(xp, [0])
            self.random_rescale_to_mahalanobis(xp[0])
            pop = np.asarray([xp[0]] + list(pop))

        self.pop_sorted = pop
        self.integer_centering(pop[:sp.weights.mu], self.mean)

        # compute new mean
        self.mean = np.dot(sp.weights.positive_weights, pop[:sp.weights.mu])
        if sp.cmean != 1:
            self.mean *= sp.cmean
            self.mean += (1 - sp.cmean) * mold

        # check Delta m (this is not default, but could become at some point)
        # CAVE: upper_length=sqrt(2)+2 is too restrictive, test upper_length = sqrt(2*N) thoroughly.
        # replaced by repair_geno?
        # simple test case injecting self.mean:
        # self.mean = 1e-4 * self.sigma * np.random.randn(N)
        if 11 < 3 and self.opts['vv'] and check_points:  # CAVEAT: check_points might be an index-list
            cmean = self.sp.cmean / min(1, ((self.opts['vv'] * N)**0.5 + 2) / (# abuse of cmean
                (self.sp.weights.mueff**0.5 / self.sp.cmean) *
                self.mahalanobis_norm(self.mean - mold)))
        else:
            cmean = self.sp.cmean

        # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        if 11 < 3:
            self.more_to_write += [sum(self.mean**2)]
        if 11 < 3:  # plot length of mean - mold
            self.more_to_write += [self.sp.weights.mueff**0.5 *
                sum(((1. / self.D) * np.dot(self.B.T, self.mean - mold))**2)**0.5 /
                       self.sigma / N**0.5 / cmean]
        ### line 2799

        # get learning rate constants
        cc = sp.cc
        c1 = self.opts['CMA_on'] * self.opts['CMA_rankone'] * self.sm.parameters(
            mueff=sp.weights.mueff, lam=sp.weights.lambda_).get('c1', sp.c1)  # mueff and lambda_ should not be necessary here
        cmu = self.opts['CMA_on'] * self.opts['CMA_rankmu'] * self.sm.parameters().get('cmu', sp.cmu)
        if flg_diagonal:
            cc, c1, cmu = sp.cc_sep, sp.c1_sep, sp.cmu_sep

        # now the real work can start

        # _update_ps must be called before the distribution is changed,
        # hsig() calls _update_ps
        hsig = self.adapt_sigma.hsig(self)

        if 11 < 3:
            # hsig = 1
            # sp.cc = 4 / (N + 4)
            # sp.cs = 4 / (N + 4)
            # sp.cc = 1
            # sp.damps = 2  #
            # sp.CMA_on = False
            # c1 = 0  # 2 / ((N + 1.3)**2 + 0 * sp.weights.mu) # 1 / N**2
            # cmu = min([1 - c1, cmu])
            if self.countiter == 1:
                print('parameters modified')
        # hsig = sum(self.ps**2) / self.N < 2 + 4./(N+1)

        if 11 < 3:  # diagnostic data
            # self.out['hsigcount'] += 1 - hsig
            if not hsig:
                self.hsiglist.append(self.countiter)
        if 11 < 3:  # diagnostic message
            if not hsig:
                print(str(self.countiter) + ': hsig-stall')
        if not CMAOptions._hsig:  # for testing purpose
            hsig = 1  # TODO:
            #       put correction term, but how?
            if self.countiter == 1:
                print('hsig=1')

        # adjust missing variance due to hsig, in 4-D with damps=1e99 and sig0 small
        #       hsig leads to premature convergence of C otherwise
        # hsiga = (1-hsig**2) * c1 * cc * (2-cc)  # to be removed in future
        c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))  # adjust for variance loss

        self.pc = (1 - cc) * self.pc + hsig * (
                    (cc * (2 - cc) * self.sp.weights.mueff)**0.5 / self.sigma
                        / cmean) * (self.mean - mold) / self.sigma_vec.scaling
        dd_params = self.sigma_vec.parameters(self.sp.weights.mueff,
                                        c1_factor=self.opts['CMA_rankone'],
                                        cmu_factor=self.opts['CMA_rankmu']
                                        )
        cc2 = dd_params['cc']
        self.pc2 = (1 - cc2) * self.pc2 + hsig * (
                    (cc2 * (2 - cc2) * self.sp.weights.mueff)**0.5 / self.sigma
                        / cmean) * (self.mean - mold)

        try:
            self.isotropic_mean_shift  # compute before sigma_vec or C are updated
        except AttributeError:
            pass  # without CSA we may not need the mean_shift

        # covariance matrix adaptation/udpate
        pop_zero = self.limit_integer_relative_deltas(  # does by default nothing
                        pop - mold,
                        options_parameters.integer_active_limit_std,
                        options_parameters.integer_active_limit_recombination_weight_condition)
        if c1a + cmu > 0:
            # TODO: make sure cc is 1 / N**0.5 rather than 1 / N
            # TODO: simplify code: split the c1 and cmu update and call self.sm.update twice
            #       caveat: for this the decay factor ``c1_times_delta_hsigma - sum(weights)`` should be zero in the second update
            _weights = sp.weights(len(pop_zero))
            sampler_weights = [c1a] + [cmu * w for w in _weights]
            sampler_weights_dd = [dd_params['c1']] + [
                                  dd_params['cmu'] * w for w in _weights]

            if len(pop_zero) > len(sp.weights):  # TODO: can be removed
                _sampler_weights = [c1a] + [cmu * w for w in sp.weights]
                _sampler_weights = (
                        _sampler_weights[:1+sp.weights.mu] +
                        (len(pop_zero) - len(sp.weights)) * [0] +
                        _sampler_weights[1+sp.weights.mu:])
                assert sampler_weights == _sampler_weights

            if 'inc_cmu_pos' in self.opts['vv']:
                sampler_weights = np.asarray(sampler_weights)
                sampler_weights[sampler_weights > 0] *= 1 + self.opts['vv']['inc_cmu_pos']
            # logger = logging.getLogger(__name__)  # "global" level needs to be DEBUG
            # logger.debug("w[0,1]=%f,%f", sampler_weights[0],
            #               sampler_weights[1]) if self.countiter < 2 else None
            # print(' injected solutions', tuple(self._injected_solutions_archive.values()))
            for i, x in enumerate(pop):
                try:
                    self._injected_solutions_archive.pop(x)
                    # self.gp.repaired_solutions.pop(x)
                except KeyError:
                    pass  # print(i)
                else:
                    # apply active_injected multiplier to non-TPA injections
                    if i > 1 or not isinstance(self.adapt_sigma, CMAAdaptSigmaTPA):
                        if sampler_weights[i + 1] < 0:  # weight index 0 is for pc
                            sampler_weights[i + 1] *= self.opts['CMA_active_injected']
                        if sampler_weights_dd[i + 1] < 0:
                            sampler_weights_dd[i + 1] *= self.opts['CMA_active_injected']
            for k, s in list(self._injected_solutions_archive.items()):
                if s['iteration'] < self.countiter - 2:
                    # warn unless TPA injections were messed up by integer centering
                    if (not isinstance(self.adapt_sigma, CMAAdaptSigmaTPA)
                            # self.integer_centering and
                            # self.integer_centering is not _pass and
                        or not isinstance(self.integer_centering, IntegerCentering)
                        or s['index'] > 1):
                        warnings.warn("""orphanated injected solution %s
                            This could be a bug in the calling order/logics or due to
                            a too small popsize used in `ask()` or when only using
                            `ask(1)` repeatedly. Please check carefully.
                            In case this is desired, the warning can be surpressed with
                            ``warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)``
                            """ % str(s), InjectionWarning)
                    self._injected_solutions_archive.pop(k)
            assert len(sampler_weights) == len(pop_zero) + 1
            if flg_diagonal:
                self.sigma_vec.update(
                    [self.sm.transform_inverse(self.pc)] +
                    list(self.sm.transform_inverse(pop_zero /
                                        (self.sigma * self.sigma_vec.scaling))),
                    np.log(2) * np.asarray(sampler_weights))  # log(2) is here for historical reasons
            else:
                pop_zero_encoded = pop_zero / (self.sigma * self.sigma_vec.scaling)
                if self.opts['CMA_diagonal_decoding'] and hasattr(self.sm, 'beta_diagonal_acceleration'):
                    ws = [self.opts['CMA_on'] * self.opts['CMA_diagonal_decoding'] /
                          self.sm.beta_diagonal_acceleration * w for w in sampler_weights_dd]
                    self.sigma_vec.update(
                        [self.sm.transform_inverse(self.pc2 / self.sigma_vec.scaling)] +
                            [self.sm.transform_inverse(z) for z in pop_zero_encoded],
                        ws)
                # TODO: recompute population after adaptation (see transformations.DD.update)?
                if 11 < 3:  # may be better but needs to be checked
                    pop_zero_encoded = pop_zero / (self.sigma * self.sigma_vec.scaling)
                    # pc is already good
                pc = self.pc
                if CMAOptions._ps_for_pc:  # experimental
                    try:
                        # avoid a large update when ||ps|| is large
                        fac = self.N**0.5 / np.linalg.norm(self.adapt_sigma.ps)
                        # fac = min((fac, 1/fac))  # this is biased
                        # the step-size increases iff the norm is large
                        # fac = 1
                        pc =  fac * self.sm.transform(
                            self.adapt_sigma.ps)  # ps is not yet updated
                    except: raise
                self.sm.update([(c1 / (c1a + 1e-23))**0.5 * pc] +  # c1a * pc**2 gets c1 * pc**2
                              list(pop_zero_encoded),
                              sampler_weights)
            if any(np.asarray(self.sm.variances) < 0):
                raise RuntimeError("A sampler variance has become negative "
                                   "after the update, this must be considered as a bug.\n"
                                   "Variances `self.sm.variances`=%s" % str(self.sm.variances))
        self._updateBDfromSM(self.sm)

        # step-size adaptation, adapt sigma
        # in case of TPA, function_values[0] and [1] must reflect samples colinear to xmean - xmean_old
        self._sigma_old = self.sigma
        try:
            self.sigma *= self.adapt_sigma.update2(self,
                                        function_values=function_values)
        except (NotImplementedError, AttributeError):
            self.adapt_sigma.update(self, function_values=function_values)

        # this is not sufficiently effective with CSA_squared option:
        if self.opts['stall_sigma_change_on_divergence_iterations'] and (
                self.fit.median_got_worse >= self.opts['stall_sigma_change_on_divergence_iterations']):
            # for the record only
            if not hasattr(self, '_stall_sigma_change_on_divergence_events'):
                self._stall_sigma_change_on_divergence_events = 30 * [None]
            self._stall_sigma_change_on_divergence_events[
                    self.countiter % len(self._stall_sigma_change_on_divergence_events)] = (
                self.countiter, self.fit.median_got_worse, self._sigma_old, self.sigma)
            # keep sigma constant, then sooner or later the median will improve again
            self.sigma = self._sigma_old  # min((es.sigma, es._sigma_old))

        self._stds_into_limits()

        # setting limits not coordinate wise works quite badly, fixed in late 2022
        if 11 < 3:  # old min/maxstd code
            if any(self.sigma * self.sigma_vec.scaling * self.dC**0.5 <
                        np.asarray(self.opts['minstd'])):
                self.sigma = max(np.asarray(self.opts['minstd']) /
                                    (self.sigma_vec * self.dC**0.5))
                assert all(self.sigma * self.sigma_vec * self.dC**0.5 >=
                        (1-1e-9) * np.asarray(self.opts['minstd']))
            elif any(self.sigma * self.sigma_vec.scaling * self.dC**0.5 >
                        np.asarray(self.opts['maxstd'])):
                self.sigma = min(np.asarray(self.opts['maxstd']) /
                                (self.sigma_vec * self.dC**0.5))
        # g = self.countiter
        # N = self.N
        # mindx = eval(self.opts['mindx'])
        #  if utils.is_str(self.opts['mindx']) else self.opts['mindx']
        if self.sigma * min(self.D) < self.opts['mindx']:  # TODO: sigma_vec is missing here
            self.sigma = self.opts['mindx'] / min(self.D)

        if self.sigma > 1e9 * self.sigma0:
            alpha = self.sigma / max(self.sm.variances)**0.5
            if alpha > 1:
                try:
                    self.sm *= alpha
                except:
                    pass
                else:
                    self.sigma /= alpha**0.5  # adjust only half
                    self.opts['tolupsigma'] /= alpha**0.5  # to be compared with sigma
                    self._updateBDfromSM()

        # TODO increase sigma in case of a plateau?

        # Uncertainty noise measurement is done on an upper level

        if CMAOptions._stationary_sphere:
            if callable(CMAOptions._stationary_sphere):
                self.mean *= (CMAOptions._stationary_sphere(self.mean_old) /
                              CMAOptions._stationary_sphere(self.mean))
            else:
                self.mean *= np.sqrt(sum(np.square(self.mean_old)) /
                                     sum(np.square(self.mean)))

        # move mean into "feasible preimage", leads to weird behavior on
        # 40-D tablet with bound 0.1, not quite explained (constant
        # dragging is problematic, but why doesn't it settle), still a bug?
        if 11 < 3 and isinstance(self.boundary_handler, BoundTransform) \
                and not self.boundary_handler.is_in_bounds(self.mean):
            self.mean = array(self.boundary_handler.inverse(
                self.boundary_handler.repair(self.mean, copy_if_changed=False),
                    copy_if_changed=False), copy=False)
        if _new_injections:
            self.pop_injection_directions = self._prepare_injection_directions()
            if (self.opts['verbose'] > 4 and self.countiter < 3 and
                not isinstance(self.adapt_sigma, CMAAdaptSigmaTPA) and
                len(self.pop_injection_directions)):
                utils.print_message('   %d directions prepared for injection %s' %
                                    (len(self.pop_injection_directions),
                                     "(no more messages will be shown)" if
                                     self.countiter == 2 else ""))
            self.number_of_injections_delivered = 0
        self.pop = []  # remove this in case pop is still needed
        # self.pop_sorted = []
        self.mean_after_tell[:] = self.mean
        self._stoptolxstagnation.set_params(self.opts['tolxstagnation']).update(self.mean)
        self._flgtelldone = True
        try:  # shouldn't fail, but let's be nice to code abuse
            self.timer.pause()
        except AttributeError:
            warnings.warn('CMAEvolutionStrategy.tell(countiter=%d): "timer" attribute '
                          'not found, probably because `ask` was never called. \n'
                          'Timing is likely to work only until `tell` is called (again), '
                          'because `tic` will never be called again afterwards.'
                          % self.countiter)
            self.timer = utils.ElapsedWCTime()

        self.more_to_write.check()
    # end tell()

    def _record_rankings(self, vals, function_values):
        "do nothing by default, otherwise assign to `_record_rankings_` after instantiation"
    def _record_rankings_(self, vals, function_values):
        """compute ranks of `vals` in `function_values` and

        in `self.fit.fit` and store the results in `_recorded_rankings`.
        The ranking differences between two solutions appear to be similar
        in the current and last iteration.
        """
        vals = list(vals)
        r0 = utils.ranks(vals + list(self.fit.fit))
        r1 = utils.ranks(vals + list(function_values))
        self._recorded_rankings = [r0[:2], r1[:2]]
        return self._recorded_rankings

    def inject(self, solutions, force=None):
        """inject list of one or several genotypic solution(s).

        This is the preferable way to pass outside proposal solutions
        into `CMAEvolutionStrategy`. Passing (bad) solutions directly
        via `tell` is likely to fail when ``CMA_active is True`` as by
        default.

        Unless ``force is True``, the `solutions` are used as direction
        relative to the distribution mean to compute a new candidate
        solution returned in method `ask_geno` which in turn is used in
        method `ask`. Even when ``force is True``, the update in `tell`
        takes later care of possibly trimming the update vector.

        `inject` is to be called before `ask` or after `tell` and can be
        called repeatedly.

        >>> import cma
        >>> es = cma.CMAEvolutionStrategy(4 * [1], 2)  #doctest: +ELLIPSIS
        (4_w,...
        >>> while not es.stop():
        ...     es.inject([4 * [0.0]])
        ...     X = es.ask()
        ...     if es.countiter == 0:
        ...         assert X[0][0] == X[0][1]  # injected sol. is on the diagonal
        ...     es.tell(X, [cma.ff.sphere(x) for x in X])

        Details: injected solutions are not used in the "active" update which
        would decrease variance in the covariance matrix in this direction.
        """
        for solution in solutions:
            if solution is None:
                continue
            if len(solution) != self.N:
                raise ValueError('method `inject` needs a list or array'
                    + (' each el with dimension (`len`) %d' % self.N))
            solution = np.asarray(solution, dtype=float)
            if force:
                self.pop_injection_solutions.append(solution)
            else:
                self.pop_injection_directions.append(solution - self.mean)

    @property
    def stds(self):
        """return array of coordinate-wise standard deviations (phenotypic).

        Takes into account geno-phenotype transformation, step-size,
        diagonal decoding, and the covariance matrix. Only the latter three
        apply to `self.mean`.
        """
        return ((self.sigma * self.gp.scales) *
                (self.sigma_vec.scaling * np.sqrt(self.sm.variances)))
    @property
    def _stds_geno(self):
        """return array of coordinate-wise standard deviations (genotypic).

        Takes into account step-size, diagonal decoding, and the covariance
        matrix but not the geno-phenotype transformation. Only the former
        three apply to `self.mean`.
        """
        return self.sigma * (self.sigma_vec.scaling * np.sqrt(self.sm.variances))

    @property
    def result(self):
        """return a `CMAEvolutionStrategyResult` `namedtuple`.

        :See: `cma.evolution_strategy.CMAEvolutionStrategyResult`
            or try ``help(...result)`` on the ``result`` property
            of an `CMAEvolutionStrategy` instance or on the
            `CMAEvolutionStrategyResult` instance itself.

        """
        # TODO: how about xcurrent?
        # return CMAEvolutionStrategyResult._generate(self)
        x, f, evals = self.best.get()
        return CMAEvolutionStrategyResult(
            x,
            f,
            evals,
            self.countevals,
            self.countiter,
            self.gp.pheno(self.mean[:], into_bounds=self.boundary_handler.repair),
            self.stds,
            self.stop()
        )

    def result_pretty(self, number_of_runs=0, time_str=None,
                      fbestever=None):
        """pretty print result.

        Returns `result` of ``self``.

        """
        if fbestever is None:
            fbestever = self.best.f
        s = (' after %i restart' + ('s' if number_of_runs > 1 else '')) \
            % number_of_runs if number_of_runs else ''
        for k, v in self.stop().items():
            print('termination on %s=%s%s' % (k, str(v), s +
                  (' (%s)' % time_str if time_str else '')))

        print('final/bestever f-value = %e %e after %d/%d evaluations' % (
            self.best.last.f, fbestever, self.countevals, self.best.evals))
        if self.N < 9:
            print('incumbent solution: ' + str(list(self.gp.pheno(self.mean, into_bounds=self.boundary_handler.repair))))
            print('std deviation: ' + str(list(self.stds)))
        else:
            print('incumbent solution: %s ...]' % (str(self.gp.pheno(self.mean, into_bounds=self.boundary_handler.repair)[:8])[:-1]))
            print('std deviations: %s ...]' % (str(self.stds[:8])[:-1]))
        return self.result

    def pickle_dumps(self):
        """return ``pickle.dumps(self)``,

        if necessary remove unpickleable (and also unnecessary) local
        function reference beforehand.

        The resulting `bytes` string-object can be saved to a file like::

            import cma
            es = cma.CMAEvolutionStrategy(3 * [1], 1)
            es.optimize(cma.ff.elli, iterations=22)
            filename = 'es-pickle-test'
            open(filename, 'wb').write(es.pickle_dumps())

        and recovered like::

            import pickle
            es = pickle.load(open(filename, 'rb'))

        or::

            es = pickle.loads(open(filename, 'rb').read())
            es.optimize(cma.ff.elli, iterations=22)  # continue optimizing

        """
        import pickle
        try:  # fine if local function self.objective_function was not assigned
            s = pickle.dumps(self)
        except:
            self.objective_function, fun = None, self.objective_function
            try:
                s = pickle.dumps(self)
            except: raise  # didn't work out
            finally:  # reset changed attribute either way
                self.objective_function = fun
        return s

    def repair_genotype(self, x, copy_if_changed=False):
        """make sure that solutions fit to the sample distribution.

        This interface is versatile and likely to change.

        The Mahalanobis distance ``x - self.mean`` is clipping at
        ``N**0.5 + 2 * N / (N + 2)``, but the specific repair
        mechanism may change in future.
        """
        x = np.asarray(x)
        mold = np.asarray(self.mean)
        if 1 < 3:  # hard clip at upper_length
            upper_length = self.N**0.5 + 2 * self.N / (self.N + 2)
            # should become an Option, but how? e.g. [0, 2, 2]
            fac = self.mahalanobis_norm(x - mold) / upper_length

            if fac > 1:
                if copy_if_changed:
                    x = (x - mold) / fac + mold
                else:  # should be 25% faster:
                    x -= mold
                    x /= fac
                    x += mold
                # print self.countiter, k, fac, self.mahalanobis_norm(pop[k] - mold)
                # adapt also sigma: which are the trust-worthy/injected solutions?
            elif 11 < 3:
                return np.exp(np.tanh(((upper_length * fac)**2 / self.N - 1) / 2) / 2)

        return x

    def manage_plateaus(self, sigma_fac=1.5, sample_fraction=0.5):
        """increase `sigma` by `sigma_fac` in case of a plateau.

        A plateau is assumed to be present if the best sample and
        ``popsize * sample_fraction``-th best sample have the same
        fitness.

        Example:

        >>> import cma
        >>> def f(X):
        ...     return (len(X) - 1) * [1] + [2]
        >>> es = cma.CMAEvolutionStrategy(4 * [0], 5, {'verbose':-9, 'tolflatfitness':1e4})
        >>> while not es.stop():
        ...     X = es.ask()
        ...     es.tell(X, f(X))
        ...     es.manage_plateaus()
        >>> if es.sigma < 1.5**es.countiter: print((es.sigma, 1.5**es.countiter, es.stop()))

        """
        if not self._flgtelldone:
            utils.print_warning("Inbetween `ask` and `tell` plateaus cannot" +
            " be managed, because `sigma` should not change.",
                           "manage_plateaus", "CMAEvolutionStrategy",
                                self.countiter)
            return
        f0, fm = Mh.prctile(self.fit.fit, [0, sample_fraction * 100])
        if f0 == fm:
            self.sigma *= sigma_fac

    @property
    def condition_number(self):
        """condition number of the statistical-model sampler.

        Details: neither encoding/decoding from `sigma_vec`-scaling nor
        `gp`-transformation are taken into account for this computation.
        """
        try:
            return self.sm.condition_number
        except (AttributeError):
            return (max(self.D) / min(self.D))**2
        except (NotImplementedError):
            return (max(self.D) / min(self.D))**2

    def alleviate_conditioning_in_coordinates(self, condition=1e8):
        """pass scaling from `C` to `sigma_vec`.

        As a result, `C` is a correlation matrix, i.e., all diagonal
        entries of `C` are `1`.
        """
        if (not condition or not np.isfinite(condition)
                or np.max(self.dC) / np.min(self.dC) < condition):
            return
        # allows for much larger condition numbers, if axis-parallel
        if hasattr(self, 'sm') and isinstance(self.sm, sampler.GaussFullSampler):
            old_coordinate_condition = np.max(self.dC) / np.min(self.dC)
            old_condition = self.sm.condition_number
            factors = self.sm.to_correlation_matrix()
            self.sigma_vec *= factors
            self.pc /= factors
            # self.pc2 /= factors
            self._updateBDfromSM(self.sm)
            utils.print_message('\ncondition in coordinate system exceeded'
                                ' %.1e, rescaled to %.1e, '
                                '\ncondition changed from %.1e to %.1e'
                    % (old_coordinate_condition, np.max(self.dC) / np.min(self.dC),
                       old_condition, self.sm.condition_number),
                    iteration=self.countiter)

    def _tfp(self, x):
        return np.dot(self.gp._tf_matrix, x)
    def _tfg(self, x):
        return np.dot(self.gp._tf_matrix_inv, x)

    def alleviate_conditioning(self, condition=1e12):
        """pass conditioning of `C` to linear transformation in `self.gp`.

        Argument `condition` defines the limit condition number above
        which the action is taken.

        >>> import cma
        >>> for dd in [0, 1]:
        ...     es = cma.CMA(2 * [1], 0.1, {'CMA_diagonal_decoding' : dd, 'verbose':-9})
        ...     es = es.optimize(cma.ff.elli, iterations=4)
        ...     es.alleviate_conditioning(1.01)  # check that alleviation_conditioning "works"
        ...     assert all(es.sigma_vec.scaling == [1, 1]), es.sigma_vec.scaling
        ...     assert es.sm.condition_number <= 1.01, es.sm.C

        Details: the action applies only if `self.gp.isidentity`. Then,
        the covariance matrix `C` is set (back) to identity and a
        respective linear transformation is "added" to `self.gp`.
        """
        # new interface: if sm.condition_number > condition ...
        if not self.gp.isidentity or not condition or self.condition_number < condition:
            return
        if len(self.opts['integer_variables']):
            utils.print_warning(
            'geno-pheno transformation not implemented with int-variables',
                'alleviate_conditioning', 'CMAEvolutionStrategy',
                self.countiter, maxwarns=1)
            return
        try:
            old_condition_number = self.condition_number
            tf_inv = self.sm.to_linear_transformation_inverse()
            tf = self.sm.to_linear_transformation(reset=True)
            self.pc = np.dot(tf_inv, self.pc)
            # self.pc2 = np.dot(tf_inv, self.pc2)
            old_C_scales = self.dC**0.5
            self._updateBDfromSM(self.sm)
        except NotImplementedError:
            utils.print_warning("Not Implemented",
                                method_name="alleviate_conditioning")
            return
        self._updateBDfromSM(self.sm)
        # dmean_prev = np.dot(self.B, (1. / self.D) * np.dot(self.B.T, (self.mean - 0*self.mean_old) / self.sigma_vec))

        # we may like to traverse tf through sigma_vec such that
        #       gp.tf * sigma_vec == sigma_vec * tf
        # but including sigma_vec shouldn't pose a problem even for
        # a large scaling which essentially just changes the exponents
        # uniformly in the rows of tf
        self.gp._tf_matrix = (self.sigma_vec * tf.T).T  # sig*tf.T .*-multiplies each column of tf with sig
        self.gp._tf_matrix_inv = tf_inv / self.sigma_vec  # here was the bug
        self.sigma_vec = transformations.DiagonalDecoding(self.sigma_vec.scaling**0)

        # TODO: refactor old_scales * old_sigma_vec into sigma_vec0 to prevent tolfacupx stopping

        self.gp.tf_pheno = self._tfp  # lambda x: np.dot(self.gp._tf_matrix, x)
        self.gp.tf_geno = self._tfg  # lambda x: np.dot(self.gp._tf_matrix_inv, x)  # not really necessary
        self.gp.isidentity = False
        assert self.mean is not self.mean_old

        # transform current mean and injected solutions accordingly
        self.mean = self.gp.tf_geno(self.mean)  # same as gp.geno()
        for i, x in enumerate(self.pop_injection_solutions):
            self.pop_injection_solutions[i] = self.gp.tf_geno(x)
        for i, x in enumerate(self.pop_injection_directions):
            self.pop_injection_directions[i] = self.gp.tf_geno(x)

        self.mean_old = self.gp.tf_geno(self.mean_old)  # not needed!?

        # self.pc = self.gp.geno(self.pc)  # now done above, which is a better place

        # dmean_now = np.dot(self.B, (1. / self.D) * np.dot(self.B.T, (self.mean - 0*self.mean_old) / self.sigma_vec))
        # assert Mh.vequals_approximately(dmean_now, dmean_prev)
        utils.print_warning('''
        geno-pheno transformation introduced based on the
        current covariance matrix with condition %.1e -> %.1e,
        injected solutions become "invalid" in this iteration'''
                        % (old_condition_number, self.condition_number),
            'alleviate_conditioning', 'CMAEvolutionStrategy',
            self.countiter)

    def _updateBDfromSM(self, sm_=None):
        """helper function for a smooth transition to sampling classes.

        By now all tests run through without this method in effect.
        Gradient injection and noeffectaxis however rely on the
        non-documented attributes B and D in the sampler. """
        # return  # should be outcommented soon, but self.D is still in use (for some tests)!
        if sm_ is None:
            sm_ = self.sm
        if isinstance(sm_, sampler.GaussStandardConstant):
            self.B = array(1)
            self.D = sm_.variances**0.5
            self.C = array(1)
            self.dC = self.D
        elif isinstance(sm_, (_rgs.GaussVDSampler, _rgs.GaussVkDSampler)):
            self.dC = sm_.variances
        else:
            self.C = self.sm.covariance_matrix  # TODO: this should go away
            try:
                self.B = self.sm.B
                self.D = self.sm.D
            except AttributeError:
                if self.C is not None:
                    self.D, self.B = self.opts['CMA_eigenmethod'](self.C)
                    self.D **= 0.5
                elif 11 < 3:  # would retain consistency but fails
                    self.B = None
                    self.D = None
            if self.C is not None:
                self.dC = np.diag(self.C)

    # ____________________________________________________________
    # ____________________________________________________________
    def feed_for_resume(self, X, function_values):
        """Resume a run using the solution history.

        CAVEAT: this hasn't been thoroughly tested or in intensive use.

        Given all "previous" candidate solutions and their respective
        function values, the state of a `CMAEvolutionStrategy` object
        can be reconstructed from this history. This is the purpose of
        function `feed_for_resume`.

        Arguments
        ---------
        `X`:
          (all) solution points in chronological order, phenotypic
          representation. The number of points must be a multiple
          of popsize.
        `function_values`:
          respective objective function values

        Details
        -------
        `feed_for_resume` can be called repeatedly with only parts of
        the history. The part must have the length of a multiple
        of the population size.
        `feed_for_resume` feeds the history in popsize-chunks into `tell`.
        The state of the random number generator might not be
        reconstructed, but this would be only relevant for the future.

        Example
        -------
        ::

            import cma

            # prepare
            (x0, sigma0) = ... # initial values from previous trial
            X = ... # list of generated solutions from a previous trial
            f = ... # respective list of f-values

            # resume
            es = cma.CMAEvolutionStrategy(x0, sigma0)
            es.feed_for_resume(X, f)

            # continue with func as objective function
            while not es.stop():
                X = es.ask()
                es.tell(X, [func(x) for x in X])


        Credits to Dirk Bueche and Fabrice Marchal for the feeding idea.

        :See also: class `CMAEvolutionStrategy` for a simple dump/load
            to resume.

        """
        if self.countiter > 0:
            utils.print_warning('feed should generally be used with a new object instance')
        if len(X) != len(function_values):
            raise ValueError('number of solutions ' + str(len(X)) +
                ' and number function values ' +
                str(len(function_values)) + ' must not differ')
        popsize = self.sp.popsize
        if (len(X) % popsize) != 0:
            raise ValueError('number of solutions ' + str(len(X)) +
                    ' must be a multiple of popsize (lambda) ' +
                    str(popsize))
        for i in range(len(X) // popsize):
            # feed in chunks of size popsize
            self.ask()  # a fake ask, mainly for a conditioned calling of
                        # updateBD and secondary to get possibly the same
                        # random state
            self.tell(X[i * popsize:(i + 1) * popsize],
                      function_values[i * popsize:(i + 1) * popsize])

    # ____________________________________________________________
    # ____________________________________________________________
    # ____________________________________________________________
    # ____________________________________________________________
    def _try_update_sm_now(self):
        """call sm.update_now like sm.sample would do.

        This avoids a bias when using
        `_random_rescaling_factor_to_mahalanobis_size` which was visible
        with TPA line samples.
        """
        try:  # make model reasonably uptodate
            self.sm.update_now()  # Why not just call sample?
        except AttributeError:
            self.sm.sample(1)
        try:
            if self.sm.last_update == self.sm.count_tell:
                self._updateBDfromSM(self.sm)  # should be cheap
        except AttributeError:
            self._updateBDfromSM(self.sm)  # should be cheap

    def mahalanobis_norm(self, dx):
        """return Mahalanobis norm based on the current sample
        distribution.

        The norm is based on Covariance matrix ``C`` times ``sigma**2``,
        and includes ``sigma_vec``. The expected Mahalanobis distance to
        the sample mean is about ``sqrt(dimension)``.

        Argument
        --------
        A *genotype* difference `dx`.

        Example
        -------
        >>> import cma, numpy
        >>> es = cma.CMAEvolutionStrategy(numpy.ones(10), 1)  #doctest: +ELLIPSIS
        (5_w,...
        >>> xx = numpy.random.randn(2, 10)
        >>> d = es.mahalanobis_norm(es.gp.geno(xx[0]-xx[1]))

        `d` is the distance "in" the true sample distribution,
        sampled points have a typical distance of ``sqrt(2*es.N)``,
        where ``es.N`` is the dimension, and an expected distance of
        close to ``sqrt(N)`` to the sample mean. In the example,
        `d` is the Euclidean distance, because C = I and sigma = 1.

        """
        self._try_update_sm_now()
        return self.sm.norm(np.asarray(dx) / self.sigma_vec.scaling) / self.sigma

    @property
    def isotropic_mean_shift(self):
        """normalized last mean shift, under random selection N(0,I)

        distributed.

        Caveat: while it is finite and close to sqrt(n) under random
        selection, the length of the normalized mean shift under
        *systematic* selection (e.g. on a linear function) tends to
        infinity for mueff -> infty. Hence it must be used with great
        care for large mueff.
        """
        if self._isotropic_mean_shift_iteration != self.countiter:
            self._isotropic_mean_shift = self.sm.transform_inverse(
                    (self.mean - self.mean_old) / self.sigma_vec.scaling)
            self._isotropic_mean_shift *= (self.sp.weights.mueff**0.5 / self.sigma
                                           / self.sp.cmean)
            self._isotropic_mean_shift_iteration = self.countiter
            # TODO:
            # works unless a re-parametrisation has been done, and otherwise also?
            # assert Mh.vequals_approximately(self._isotropic_mean_shift,
            #         np.dot(es.B, (1. / es.D) *
            #         np.dot(es.B.T, (es.mean - es.mean_old) / es.sigma_vec)))
        return self._isotropic_mean_shift

    def disp_annotation(self):
        """print annotation line for `disp` ()"""
        print('Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]')
        sys.stdout.flush()

    def disp(self, modulo=None, overwrite=None):
        """print current state variables in a single-line.

        Prints only if ``iteration_counter % modulo == 0``.
        Overwrites the line after iteration `overwrite`.

        :See also: `disp_annotation`.
        """
        if modulo is None:
            modulo = self.opts['verb_disp']

        def do_overwrite():
            if overwrite is None:
                iters = self.opts.get('verb_disp_overwrite', float('inf'))
            else:
                iters = overwrite
            return not self.stop() and iters > 0 and self.countiter > iters

        # console display

        if modulo:
            if (self.countiter - 1) % (10 * modulo) < 1 and not do_overwrite():
                self.disp_annotation()
            if not hasattr(self, 'times_displayed'):
                self.time_last_displayed = 0
                self.times_displayed = 0

            if self.countiter > 0 and (self.stop() or self.countiter < 4
                              or self.countiter % modulo < 1
                              or self.timer.elapsed - self.time_last_displayed > self.times_displayed):
                self.time_last_displayed = self.timer.elapsed
                self.times_displayed += 1
                if self.opts['verb_time']:
                    toc = self.timer.elapsed
                    stime = str(int(toc // 60)) + ':' + ("%2.1f" % (toc % 60)).rjust(4, '0')
                else:
                    stime = ''
                print(' '.join((repr(self.countiter).rjust(5),
                                repr(self.countevals).rjust(6),
                                '%.15e' % (min(self.fit.fit)),
                                '%4.1e' % (self.D.max() / self.D.min()
                                           if not self.opts['CMA_diagonal'] or self.countiter > self.opts['CMA_diagonal']
                                           else max(self.sigma_vec*1) / min(self.sigma_vec*1)),
                                '%6.2e' % self.sigma,
                                '%6.0e' % (self.sigma * min(self.sigma_vec * self.dC**0.5)),
                                '%6.0e' % (self.sigma * max(self.sigma_vec * self.dC**0.5)),
                                stime)),
                      end='\r' if do_overwrite() else '\n')
                # if self.countiter < 4:
                sys.stdout.flush()
        return self
    def plot(self, *args, **kwargs):
        """plot current state variables using `matplotlib`.

        Details: calls `self.logger.plot`.
        """
        if not hasattr(self.logger, 'es') or self.logger.es is None:
            self.logger.es = self  # let logger extract es.stop()
        try:
            self.logger.plot(*args, **kwargs)
        except AttributeError:
            utils.print_warning('plotting failed, no logger attribute found')
        except:
            utils.print_warning('plotting failed with: {0}'.format(sys.exc_info()),
                           'plot', 'CMAEvolutionStrategy')
        return self

class _StopTolXStagnation(object):
    """Provide a termination signal depending on how much a vector has changed,

    typically applied to the distribution mean over several iterations.

    The `stop` property is the boolean termination signal, the `update`
    method needs to be called in each iteration with the (changing) vector.

    ``self.stop is True`` iff
    - ``delta t > threshold`` and
    - ``Delta x < delta * max(1, sqrt(delta t / threshold))`` for at least delta t iterations.

    The (iteration) threshold is computed in property `time_threshold`
    based on the iteration count and three parameters as ``p1 + p2 * count**p3``.

    """
    def __init__(self, x=None):
        """`x` is the initial vector (optional), default settings are taken from `CMAOptions`"""
        vals = CMAOptions().eval('tolxstagnation')  # get default values
        self.delta = vals[0]
        self.time_delta_offset = vals[1]  # should depend on dimension because of adaptation delays?
        self.time_delta_frac = vals[2]
        self.time_delta_expo = 1
        self.count = 0
        self.count_x = 0
        self.x = x
    def set_params(self, param_values, names=(
                    'delta', 'time_delta_offset', 'time_delta_frac', 'time_delta_expo')):
        """`param_values` is a `list` conforming to ``CMAOptions['tolxstagnation']``.

        Do nothing if ``param_values in (None, True)``, set ``delta = -1``
        if ``param_values is False``.

        `None` entries in `param_values` don't change the respective
        parameter and ``[0.12]`` is the same as ``[0.12, None, None, None]``.

        Details: In principle, `'delta'` should be propto sqrt(mu/dimension).
        """
        if param_values in (None, True):
            return self  # use default or current values
        if param_values is False:
            self.delta = -1
            return self
        try: iter(param_values)
        except TypeError:
            self.delta = param_values
            return self
        for name, value in zip(names, param_values):
            assert hasattr(self, name), name
            if value is not None:
                setattr(self, name, value)
        return self
    def update(self, x):
        """caveat: this stores x as a reference"""
        self.count += 1
        # allow for a larger delta when self.x is older than the time_threshold
        delta = self.delta * max((1, (self.count - self.count_x) / self.time_threshold))**0.5
        if delta < 0 or self.x is None or np.sum((self.x - x)**2)**0.5 > delta:
            # reset stagnation measure
            self.x = np.asarray(x)
            self.count_x = self.count
        return self
    @property
    def time_threshold(self):
        return (self.time_delta_offset +
                self.time_delta_frac * self.count**self.time_delta_expo)
    @property
    def stop(self):
        return self.count - self.count_x > self.time_threshold

class _CMAStopDict(dict):
    """keep and update a termination condition dictionary.

    The dictionary is "usually" empty and returned by
    `CMAEvolutionStrategy.stop()`. The class methods entirely depend on
    `CMAEvolutionStrategy` class attributes.

    Details
    -------
    This class is not relevant for the end-user and could be a nested
    class, but nested classes cannot be serialized.

    Example
    -------
    >>> import cma
    >>> es = cma.CMAEvolutionStrategy(4 * [1], 1, {'verbose':-9})  #doctest: +ELLIPSIS
    >>> print(es.stop())
    {}
    >>> es.optimize(cma.ff.sphere, verb_disp=0)  #doctest: +ELLIPSIS
    <...
    >>> es.stop()['tolfun'] == 1e-11
    True

    :See: `OOOptimizer.stop()`, `CMAEvolutionStrategy.stop()`

    """
    def __init__(self, d={}):
        update = isinstance(d, CMAEvolutionStrategy)
        super(_CMAStopDict, self).__init__({} if update else d)
        self.stoplist = []  # to keep multiple entries
        self.lastiter = 0  # probably not necessary
        self._get_value = None  # a hack to pass some value
        self._value = None  # in iteration zero value is always None
        try:
            self.stoplist = d.stoplist  # multiple entries
        except:
            pass
        try:
            self.lastiter = d.lastiter    # probably not necessary
        except:
            pass
        if update:
            self._update(d)

    def __call__(self, es=None, check=True):
        """update and return the termination conditions dictionary

        """
        if not check:
            return self
        if es is None and self.es is None:
            raise ValueError('termination conditions need an optimizer to act upon')
        self._update(es)
        return self

    def _update(self, es):
        """Test termination criteria and update dictionary

        """
        if es is None:
            es = self.es
        assert es is not None

        if es.countiter == 0:  # in this case termination tests fail
            if self._get_value:
                warnings.warn("Cannot get stop value before the first iteration")
            self.__init__()
            return self

        if 11 < 3:  # options might have changed, so countiter ==
            if es.countiter == self.lastiter:  # lastiter doesn't help
                try:
                    if es == self.es:
                        return self
                except:  # self.es not yet assigned
                    pass

        self.lastiter = es.countiter
        self.es = es

        self._get_value or self.clear()  # compute conditions from scratch

        N = es.N
        opts = es.opts
        self.opts = opts  # a hack to get _addstop going

        # check user versatile options from signals file

        # in 5-D: adds 0% if file does not exist and 25% = 0.2ms per iteration if it exists and verbose=-9
        # old measure: adds about 40% time in 5-D, 15% if file is not present
        # to avoid any file checking set signals_filename to None or ''
        if opts['verbose'] >= -9 and opts['signals_filename'] and os.path.isfile(self.opts['signals_filename']):
            with open(opts['signals_filename'], 'r') as f:
                s = f.read()
            try:
                d = dict(ast.literal_eval(s.strip()))
            except SyntaxError:
                warnings.warn("SyntaxError when parsing the following expression with `ast.literal_eval`:"
                            "\n\n%s\n(contents of file %s)" % (s, str(self.opts['signals_filename'])))
            else:
                for key in list(d):
                    if key not in opts.versatile_options():
                        utils.print_warning("        unkown or non-versatile option '%s' found in file %s.\n"
                                            "        Check out the #v annotation in ``cma.CMAOptions()``."
                                            % (key, self.opts['signals_filename']))
                        d.pop(key)
                opts.update(d)
                for key in d:
                    opts.eval(key, {'N': N, 'dim': N})

        # fitness: generic criterion, user defined w/o default
        self._addstop('ftarget',
                      es.best.f <= opts['ftarget'])
        # maxiter, maxfevals: generic criteria
        self._addstop('maxfevals',
                      es.countevals - 1 >= opts['maxfevals'])
        self._addstop('maxiter',
                      ## meta_parameters.maxiter_multiplier == 1.0
                      es.countiter >= 1.0 * opts['maxiter'])
        # tolx, tolfacupx: generic criteria
        # tolfun, tolfunhist (CEC:tolfun includes hist)

        sigma_x_sigma_vec_x_sqrtdC = es.sigma * (es.sigma_vec.scaling * np.sqrt(es.dC))
        self._addstop('tolfacupx',
                      any(sigma_x_sigma_vec_x_sqrtdC >
                          es.sigma0 * es.sigma_vec0 * opts['tolfacupx']))
        self._addstop('tolx',
                      all(sigma_x_sigma_vec_x_sqrtdC < opts['tolx']) and
                      all(es.sigma * (es.sigma_vec.scaling * es.pc) < opts['tolx']),
                      max(sigma_x_sigma_vec_x_sqrtdC) if self._get_value else None)
                      # None only to be backwards compatible for the time being

        current_fitness_range = max(es.fit.fit) - min(es.fit.fit)
        historic_fitness_range = max(es.fit.hist) - min(es.fit.hist)
        self._addstop('tolfun',
                      current_fitness_range < opts['tolfun'] and  # fit.fit is sorted including bound penalties
                      historic_fitness_range < opts['tolfun'])
        self._addstop('tolfunrel',
                      current_fitness_range < opts['tolfunrel'] * (es.fit.median0 - es.fit.median_min),
                      current_fitness_range if self._get_value else None)
        self._addstop('tolfunhist',
                      len(es.fit.hist) > 9 and
                      historic_fitness_range < opts['tolfunhist'])
        if opts['tolstagnation']:
            # worst seen false positive: table N=80,lam=80, getting worse for fevals=35e3 \approx 50 * N**1.5
            # but the median is not so much getting worse
            # / 5 reflects the sparsity of histbest/median
            # / 2 reflects the left and right part to be compared
            ## meta_parameters.tolstagnation_multiplier == 1.0
            l = max(( 1.0 * opts['tolstagnation'] / 5. / 2, len(es.fit.histbest) / 10))
            # TODO: why max(..., len(histbest)/10) ???
            # TODO: the problem in the beginning is only with best ==> ???
            if 11 < 3:  # print for debugging
                print(es.countiter, (opts['tolstagnation'], es.countiter > N * (5 + 100 / es.popsize),
                    len(es.fit.histbest) > 100,
                    np.median(es.fit.histmedian[:l]) >= np.median(es.fit.histmedian[l:2 * l]),
                    np.median(es.fit.histbest[:l]) >= np.median(es.fit.histbest[l:2 * l])))
            # equality should handle flat fitness
            if l <= es.countiter:
                l = int(l)  # doesn't work for infinite l which can never happen anyways
                self._addstop('tolstagnation',  # leads sometimes early stop on ftablet, fcigtab, N>=50?
                        1 < 3 and opts['tolstagnation'] and es.countiter > N * (5 + 100 / es.popsize) and
                        (es.countevals - es.best.evals) / es.popsize > es.opts['tolstagnation'] / 2 and  # recorded best is in the first half of stagnation period
                        len(es.fit.histbest) > 100 and 2 * l < len(es.fit.histbest) and
                        np.median(es.fit.histmedian[:l]) >= np.median(es.fit.histmedian[l:2 * l]) and
                        np.median(es.fit.histbest[:l]) >= np.median(es.fit.histbest[l:2 * l]))
            # iiinteger: stagnation termination can prevent to find the optimum
        self._addstop('tolxstagnation', es._stoptolxstagnation.stop)

        s = es.sigma / es.D.max()
        self._addstop('tolupsigma', opts['tolupsigma'] and
                      s > es.sigma0 * opts['tolupsigma'],
                      s if self._get_value else None)
        try:
            self._addstop('timeout',
                          es.timer.elapsed > opts['timeout'],
                          es.timer.elapsed if self._get_value else None)
        except AttributeError:
            if es.countiter <= 0:
                pass
            # else: raise

        if 11 < 3 and 2 * l < len(es.fit.histbest):  # TODO: this might go wrong, because the nb of written columns changes
            tmp = (-np.median(es.fit.histmedian[:l]) + np.median(es.fit.histmedian[l:2 * l]),
                   - np.median(es.fit.histbest[:l]) + np.median(es.fit.histbest[l:2 * l]))
            es.more_to_write += [(10**t if t < 0 else t + 1) for t in tmp]  # the latter to get monotonicy

        if 1 < 3:
            # non-user defined, method specific
            # noeffectaxis (CEC: 0.1sigma), noeffectcoord (CEC:0.2sigma), conditioncov
            idx = (es.mean == es.mean + 0.2 * sigma_x_sigma_vec_x_sqrtdC).nonzero()[0]
            self._addstop('noeffectcoord', any(idx), list(idx))
#                         any([es.mean[i] == es.mean[i] + 0.2 * es.sigma *
#                                                         (es.sigma_vec if np.isscalar(es.sigma_vec) else es.sigma_vec[i]) *
#                                                         sqrt(es.dC[i])
#                              for i in range(N)])
#                )
            if (opts['CMA_diagonal'] is not True and es.countiter > opts['CMA_diagonal'] and
                (es.countiter % 1) == 0):  # save another factor of two?
                i = es.countiter % N
                try:
                    self._addstop('noeffectaxis',
                                  all(es.mean == es.mean + 0.1 * es.sigma *
                                     es.sm.D[i] * es.sigma_vec.scaling *
                                     (es.sm.B[:, i] if len(es.sm.B.shape) > 1 else es.sm.B[0])))
                except AttributeError:
                    pass
            self._addstop('tolconditioncov',
                          opts['tolconditioncov'] and
                          es.D[-1] > opts['tolconditioncov']**0.5 * es.D[0], opts['tolconditioncov'])

            self._addstop('callback', any(es.callbackstop), es.callbackstop)  # termination_callback

        if 1 < 3 or len(self): # only if another termination criterion is satisfied
            if 1 < 3:
                if es.fit.fit[0] < es.fit.fit[int(0.75 * len(es.fit.fit))]:
                    es.fit.flatfit_iterations = 0
                else:
                    # print(es.fit.fit)
                    es.fit.flatfit_iterations += 1
                    if (es.fit.flatfit_iterations > opts['tolflatfitness'] # or
                        # mainly for historical reasons:
                        # max(es.fit.hist[:1 + int(opts['tolflatfitness'])]) == min(es.fit.hist[:1 + int(opts['tolflatfitness'])])
                       ):
                        self._addstop('tolflatfitness')
                        if 11 < 3 and max(es.fit.fit) == min(es.fit.fit) == es.best.last.f:  # keep warning for historical reasons for the time being
                            utils.print_warning(
                                "flat fitness (f=%f, sigma=%.2e). "
                                "For small sigma, this could indicate numerical convergence. \n"
                                "Otherwise, please (re)consider how to compute the fitness more elaborately." %
                                (es.fit.fit[0], es.sigma), iteration=es.countiter)
            if 11 < 3:  # add stop condition, in case, replaced by above, subject to removal
                self._addstop('flat fitness',  # message via stopdict
                         len(es.fit.hist) > 9 and
                         max(es.fit.hist) == min(es.fit.hist) and
                              max(es.fit.fit) == min(es.fit.fit),
                         "please (re)consider how to compute the fitness more elaborately if sigma=%.2e is large" % es.sigma)
        if 11 < 3 and opts['vv'] == 321:
            self._addstop('||xmean||^2<ftarget', sum(es.mean**2) <= opts['ftarget'])

        return self

    def _addstop(self, key, cond=True, val=None):
        if key == self._get_value:
            self._value = val
            self._get_value = None
        elif cond:
            self.stoplist.append(key)  # can have the same key twice
            self[key] = val if val is not None \
                            else self.opts.get(key, None)

    def clear(self):
        """empty the stopdict"""
        for k in list(self):
            self.pop(k)
        self.stoplist = []

def fmin_lq_surr(objective_function, x0, sigma0, options=None, **kwargs):
    """minimize `objective_function` with lq-CMA-ES.

    See ``help(cma.fmin)`` for the input parameter descriptions where
    `parallel_objective` is not available and noise-related options may
    fail.

    Returns the tuple ``xbest, es`` similar to `fmin2`, however `xbest`
    takes into account only some of the recent history and not all
    evaluations. `es.result` is partly based on surrogate f-values and may
    hence be confusing. In particular, `es.best` contains the solution with
    the best _surrogate_ value (which is usually of little interest). See
    `fmin_lq_surr2` for a fix.

    As in general, `es.result.xfavorite` is considered the best available
    estimate of the optimal solution.

    Example code
    ------------

    >>> import cma
    >>> x, es = cma.fmin_lq_surr(cma.ff.rosen, 2 * [0], 0.1,
    ...                          {'verbose':-9,  # verbosity for doctesting
    ...                           'ftarget':1e-2, 'seed':11})
    >>> assert 'ftarget' in es.stop(), (es.stop(), es.result_pretty())
    >>> assert es.result.evaluations < 90, es.result.evaluations  # can be 137 depending on seed

    Details
    -------
    lq-CMA-ES builds a linear or quadratic (global) model as a surrogate to
    try to circumvent evaluations of the objective function, see link below.

    This function calls `fmin2` with a surrogate as ``parallel_objective``
    argument. The model is kept the same for each restart. Use
    `fmin_lq_surr2` if this is not desirable.

    ``kwargs['callback']`` is modified by appending a callable that injects
    ``model.xopt``. This can be prevented by passing `callback=False` or
    adding `False` as an element of the callback list (see also `cma.fmin`).

    ``parallel_objective`` is assigned to a surrogate model instance of
    ``cma.fitness_models.SurrogatePopulation``.

    `es.countevals` is updated from the `evaluations` attribute of the
    constructed surrogate to count only "true" evaluations.

    See https://cma-es.github.io/lq-cma for references and details about
    the algorithm.
    """
    surrogate = _fitness_models.SurrogatePopulation(objective_function)
    def inject_xopt(es):
        es.inject([surrogate.model.xopt])  # not strictly necessary
    def callback_in_kwargs(kwargs):
        """append `inject_xopt` to kwargs['callback']"""
        cb = _callable_to_list(kwargs.get('callback', None))
        if cb is False:
            kwargs['callback'] = []
        elif inject_xopt not in cb:  # don't inject twice
            cb.append(inject_xopt)
            kwargs['callback'] = cb
        return kwargs

    _, es = fmin2(None, x0, sigma0, options=options, parallel_objective=surrogate,
                  **callback_in_kwargs(kwargs))
    es.surrogate = surrogate
    return surrogate.model.X[np.argmin(surrogate.model.F)], es

def fmin_lq_surr2(objective_function, x0, sigma0, options=None,
                  inject=True, restarts=0, incpopsize=2,
                  keep_model=False, not_evaluated=np.isnan,
                  callback=None):
    """minimize `objective_function` with lq-CMA-ES.

    `x0` is the initial solution or can be a callable that returns an
    initial solution (different for each restarted run). See ``cma.fmin``
    for further input documentations and ``cma.CMAOptions()`` for the
    available options.

    `inject` determines whether the best solution of the model is
    reinjected in each iteration. By default, a new surrogate model is used
    after each restart (``keep_model=False``) and the population size is
    multiplied by a factor of two (``incpopsize=2``) like in IPOP-CMA-ES
    (see also ``help(cma.fmin)``).

    Returns the tuple ``xbest, es`` like `fmin2`. As in general,
    `es.result.xfavorite` (and `es.mean` as genotype) is considered the
    best available estimate of the optimal solution.

    Example code
    ------------

    >>> import cma
    >>> x, es = cma.fmin_lq_surr2(cma.ff.rosen, 2 * [0], 0.1,
    ...                           {'verbose':-9,  # verbosity for doctesting
    ...                            'ftarget':1e-2, 'seed':3})
    >>> assert 'ftarget' in es.stop(), (es.stop(), es.result_pretty())
    >>> assert es.result.evaluations < 90, es.result.evaluations  # can be >130? depending on seed
    >>> assert es.countiter < 60, es.countiter

    Details
    -------
    lq-CMA-ES builds a linear or quadratic (global) model as a surrogate to
    circumvent evaluations of the objective function, see link below.

    This code uses the ask-and-tell interface to CMA-ES via the class
    `CMAEvolutionStrategy` to the `options` `dict` is passed.

    To pass additional arguments to the objective function use
    `functools.partial`.

    `not_evaluated` must return `True` if a value indicates (by convention
    of `cma.fitness_models.SurrogatePopulation.EvaluationManager.fvalues`)
    a missing "true" evaluation of the `objective_function`.

    See https://cma-es.github.io/lq-cma for references and details about
    the algorithm.
"""
    if options is None:
        options = {}
    best = ot.BestSolution2()
    restarts = options_parameters.amend_restarts_parameter(restarts)
    for irun in range(1 + restarts['maxrestarts']):
        if irun == 0 or not keep_model:
            surrogate = _fitness_models.SurrogatePopulation(objective_function)
        if irun > 0:  # increase popsize
            options['popsize'] = int(es.sp.popsize * incpopsize + 1/2)
        es = CMAEvolutionStrategy(x0, sigma0, options)
        es.surrogate = surrogate  # may be used in callback
        if irun > 0:  # pass counts from previous state
            surrogate.evaluations = best.count
            es.countevals = best.count
            es.logger.append = True
        while not es.stop():
            X = es.ask()
            F = surrogate(X)
            es.tell(X, F)  # update sample distribution
            es.countevals = surrogate.evaluations  # count only "true" evaluations
            for f, x in zip(surrogate.evals.fvalues, surrogate.evals.X):
                if not_evaluated(f):  # ignore NaN, important for correct count output
                    continue
                best.update(f, x)
            if inject:
                es.inject([surrogate.model.xopt])
            if callback:
                if callable(callback):
                    callback(es)
                else:
                    for c in callback:
                        c(es)
            es.logger.add()  # trigger the logging
            es.disp()  # just checking what's going on
        if es.opts['verb_disp'] > 0:
            es.result_pretty(irun, time.asctime(time.localtime()),
                             best.f)
        if (es.countevals >= restarts['maxfevals'] or
            'ftarget' in es.stop(check=False) or
            'maxfevals' in es.stop(check=False) or
            'callback' in es.stop(check=False)):
            break

    ### assign es.best from best and return
    try:
        i = np.nanargmin(surrogate.evals.fvalues)
    except ValueError as e:
        warnings.warn(str(e) + "\n  Valid assignment of `es.best.last` (best of last/current iteration) failed.")
    else:
        es.best.last.f = surrogate.evals.fvalues[i]
        es.best.last.x = surrogate.evals.X[i]
    es.best.f = best.f
    es.best.x = best.x
    try: es.best.x_geno = es.gp.tf_geno(best.x)
    except: es.best.x_geno = None
    es.best.evals = best.count_saved
    es.best.compared = best.count
    # es.best.evalsall remains as is, not clear what this is though
    es.best_fmin_lq_surr2 = best  # as a reference in case
    return best.x, es

def fmin2(objective_function, x0, sigma0,
         options=None,
         args=(),
         gradf=None,
         restarts=0,
         restart_from_best='False',
         incpopsize=2,
         eval_initial_x=False,
         parallel_objective=None,
         noise_handler=None,
         noise_change_sigma_exponent=1,
         noise_kappa_exponent=0,  # TODO: add max kappa value as parameter
         bipop=False,
         callback=None,
         init_callback=None):
    """wrapper around `cma.fmin` returning the tuple ``(xbest, es)``,

    and with the same in input arguments as `fmin`. Hence a typical
    calling pattern may be::

        x, es = cma.fmin2(...)  # recommended pattern
        es = cma.fmin2(...)[1]  # `es` contains all available information
        x = cma.fmin2(...)[0]   # keep only the best evaluated solution

    `fmin2` is an alias for::

        res = fmin(...)
        return res[0], res[-2]

    `fmin` from `fmin2` is::

        es = fmin2(...)[1]  # fmin2(...)[0] is es.result[0]
        return es.result + (es.stop(), es, es.logger)

    The best found solution is equally available under::

        fmin(...)[0]
        fmin2(...)[0]
        fmin2(...)[1].result[0]
        fmin2(...)[1].result.xbest
        fmin2(...)[1].best.x

    The incumbent, current estimate for the optimum is available under::

        fmin(...)[5]
        fmin2(...)[1].result[5]
        fmin2(...)[1].result.xfavorite

    """
    res = fmin(objective_function, x0, sigma0,
         options,
         args,
         gradf,
         restarts,
         restart_from_best,
         incpopsize,
         eval_initial_x,
         parallel_objective,
         noise_handler,
         noise_change_sigma_exponent,
         noise_kappa_exponent,
         bipop,
         callback,
         init_callback)
    return res[0], res[-2]


all_stoppings = []  # accessable via cma.evolution_strategy.all_stoppings, bound to change
def fmin(objective_function, x0, sigma0,
         options=None,
         args=(),
         gradf=None,
         restarts=0,
         restart_from_best='False',
         incpopsize=2,
         eval_initial_x=False,
         parallel_objective=None,
         noise_handler=None,
         noise_change_sigma_exponent=1,
         noise_kappa_exponent=0,  # TODO: add max kappa value as parameter
         bipop=False,
         callback=None,
         init_callback=None):
    """functional interface to the stochastic optimizer CMA-ES
    for non-convex function minimization.

    `fmin2` provides the cleaner return values.

    Calling Sequences
    =================
    ``fmin(objective_function, x0, sigma0)``
        minimizes ``objective_function`` starting at ``x0`` and with
        standard deviation ``sigma0`` (step-size)
    ``fmin(objective_function, x0, sigma0, options={'ftarget': 1e-5})``
        minimizes ``objective_function`` up to target function value 1e-5,
        which is typically useful for benchmarking.
    ``fmin(objective_function, x0, sigma0, args=('f',))``
        minimizes ``objective_function`` called with an additional
        argument ``'f'``.
    ``fmin(objective_function, x0, sigma0, options={'ftarget':1e-5, 'popsize':40})``
        uses additional options ``ftarget`` and ``popsize``
    ``fmin(objective_function, esobj, None, options={'maxfevals': 1e5})``
        uses the `CMAEvolutionStrategy` object instance ``esobj`` to
        optimize ``objective_function``, similar to ``esobj.optimize()``.

    Arguments
    =========
    ``objective_function``
        called as ``objective_function(x, *args)`` to be minimized.
        ``x`` is a one-dimensional `numpy.ndarray`. See also the
        `parallel_objective` argument.
        ``objective_function`` can return `numpy.NaN`, which is
        interpreted as outright rejection of solution ``x`` and invokes
        an immediate resampling and (re-)evaluation of a new solution
        not counting as function evaluation. The attribute
        ``variable_annotations`` is passed into the
        ``CMADataLogger.persistent_communication_dict``.
    ``x0``
        list or `numpy.ndarray`, initial guess of minimum solution
        before the application of the geno-phenotype transformation
        according to the ``transformation`` option.  It can also be a
        callable that is called (without input argument) before each
        restart to yield the initial guess such that each restart may start
        from a different place. Otherwise, ``x0`` can also be a
        `cma.CMAEvolutionStrategy` object instance, in that case ``sigma0``
        can be ``None``.
    ``sigma0``
        scalar, initial standard deviation in each coordinate.
        ``sigma0`` should be about 1/4th of the search domain width
        (where the optimum is to be expected). The variables in
        ``objective_function`` should be scaled such that they
        presumably have similar sensitivity.
        See also `ScaleCoordinates`.
    ``options``
        a dictionary with additional options passed to the constructor
        of class ``CMAEvolutionStrategy``, see ``cma.CMAOptions`` ()
        for a list of available options.
    ``args=()``
        arguments to be used to call the ``objective_function``
    ``gradf=None``
        gradient of f, where ``len(gradf(x, *args)) == len(x)``.
        ``gradf`` is called once in each iteration if
        ``gradf is not None``.
    ``restarts=0``
        number of restarts with increasing population size, see also
        parameter ``incpopsize``. For the time being (this may change in
        future) ``restarts`` can also be a `dict` where the keys
        ``maxrestarts=9`` and ``maxfevals=np.inf`` are interpreted. An
        empty `dict` is interpreted as ``restarts=0``. An IPOP-CMA-ES
        restart is invoked if ``restarts > 0`` or ``restarts['maxrestarts']
        > 0`` and if ``current_evals < min((restarts['maxfevals'],
        options['maxfevals']))`` and neither the ``'ftarget'`` nor the
        ``termination_callback`` option was triggered;
        ``restarts['maxfevals']`` does not terminate *during* the run or
        restart; to restart from different points (recommended), pass
        ``x0`` as a `callable`; see also parameter ``bipop``.
    ``restart_from_best=False``
        which point to restart from
    ``incpopsize=2``
        multiplier for increasing the population size ``popsize`` before
        each restart
    ``parallel_objective``
        an objective function that accepts a list of `numpy.ndarray` as
        input and returns a `list`, which is mostly used instead of
        `objective_function`, but for the initial (also initial
        elitist) and the final evaluations unless
        ``not callable(objective_function)``. If ``parallel_objective``
        is given, the ``objective_function`` (first argument) may be
        ``None``.
    ``eval_initial_x=None``
        evaluate initial solution, for ``None`` only with elitist option
    ``noise_handler=None``
        must be `True` or a `cma.NoiseHandler` class or instance to invoke
        noise handling. The latter gives control over the specific settings
        for the noise handling, see ``help(cma.NoiseHandler)``.
    ``noise_change_sigma_exponent=1``
        exponent for the sigma increment provided by the noise handler for
        additional noise treatment. 0 means no sigma change.
    ``noise_evaluations_as_kappa=0``
        instead of applying reevaluations, the "number of evaluations"
        is (ab)used as scaling factor kappa (experimental).
    ``bipop=False``
        if ``bool(bipop) is True``, run as BIPOP-CMA-ES; BIPOP is a special
        restart strategy switching between two population sizings - small
        (relative to the large population size and with varying initial
        sigma, the first run is accounted on the "small" budget) and large
        (progressively increased as in IPOP). This makes the algorithm
        potentially solve both, functions with many regularly or
        irregularly arranged local optima (the latter by frequently
        restarting with small populations). Small populations are
        (re-)started as long as the cumulated budget_small is smaller than
        `bipop` x max(1, budget_large). For the `bipop` parameter to
        actually conduct restarts also with the larger population size,
        select a non-zero number of (IPOP) restarts; the recommended
        setting is ``restarts <= 9`` and `x0` passed as a `callable`
        that generates randomized initial solutions. Small-population
        restarts do not count into the total restart count.
    ``callback=None``
        `callable` or list of callables called at the end of each
        iteration with the current `CMAEvolutionStrategy` instance
        as argument.
    ``init_callback=None``
        `callable` or list of callables called at the end of initialization
        of the `CMAEvolutionStrategy` instance with this instance as
        argument (like `callback`). This allows to reassign attributes
        without a corresponding `CMAOption`. For example,
        ``es.integer_centering = lambda *args: None`` disables integer
        centering (which is enabled by default when ``integer_variables``
        are given in the `options`) or ``es.integer_centering =
        cma.integer_centering.IntCentering(es, correct_bias=False)``
        disables its bias correction.

    Optional Arguments
    ==================
    All values in the `options` dictionary are evaluated if they are of
    type `str`, besides `verb_filenameprefix`, see class `CMAOptions` for
    details. The full list is available by calling ``cma.CMAOptions()``.

    >>> import cma
    >>> cma.CMAOptions()  #doctest: +ELLIPSIS
    {...

    Subsets of options can be displayed, for example like
    ``cma.CMAOptions('tol')``, or ``cma.CMAOptions('bound')``,
    see also class `CMAOptions`.

    Return
    ======
    Return the list provided in `CMAEvolutionStrategy.result` appended
    with termination conditions, an `OOOptimizer` and a `BaseDataLogger`::

        res = es.result + (es.stop(), es, logger)

    where
        - ``res[0]`` (``xopt``) -- best evaluated solution
        - ``res[1]`` (``fopt``) -- respective function value
        - ``res[2]`` (``evalsopt``) -- respective number of function evaluations
        - ``res[3]`` (``evals``) -- number of overall conducted objective function evaluations
        - ``res[4]`` (``iterations``) -- number of overall conducted iterations
        - ``res[5]`` (``xmean``) -- mean of the final sample distribution
        - ``res[6]`` (``stds``) -- effective stds of the final sample distribution
        - ``res[-3]`` (``stop``) -- termination condition(s) in a dictionary
        - ``res[-2]`` (``cmaes``) -- class `CMAEvolutionStrategy` instance
        - ``res[-1]`` (``logger``) -- class `CMADataLogger` instance

    Details
    =======
    This function is an interface to the class `CMAEvolutionStrategy`. The
    latter class should be used when full control over the iteration loop
    of the optimizer is desired.

    Examples
    ========
    The following example calls `fmin` optimizing the Rosenbrock function
    in 10-D with initial solution 0.1 and initial step-size 0.5. The
    options are specified for the usage with the `doctest` module.

    >>> import cma
    >>> # cma.CMAOptions()  # returns all possible options
    >>> options = {'CMA_diagonal':100, 'seed':1234, 'verb_time':0}
    >>>
    >>> res = cma.fmin(cma.ff.rosen, [0.1] * 10, 0.3, options)  #doctest: +ELLIPSIS
    (5_w,10)-aCMA-ES (mu_w=3.2,w_1=45%) in dimension 10 (seed=1234...)
       Covariance matrix is diagonal for 100 iterations (1/ccov=26...
    Iterat #Fevals   function value  axis ratio  sigma ...
        1     10 ...
    termination on tolfun=1e-11 ...
    final/bestever f-value = ...
    >>> assert res[1] < 1e-12  # f-value of best found solution
    >>> assert res[2] < 8000  # evaluations

    The above call is pretty much equivalent with the slightly more
    verbose call::

        res = cma.CMAEvolutionStrategy([0.1] * 10, 0.3,
                    options=options).optimize(cma.ff.rosen).result

    where `optimize` returns a `CMAEvolutionStrategy` instance. The
    following example calls `fmin` optimizing the Rastrigin function
    in 3-D with random initial solution in [-2,2], initial step-size 0.5
    and the BIPOP restart strategy (that progressively increases population).
    The options are specified for the usage with the `doctest` module.

    >>> import cma
    >>> # cma.CMAOptions()  # returns all possible options
    >>> options = {'seed':12345, 'verb_time':0, 'ftarget': 1e-8}
    >>>
    >>> res = cma.fmin(cma.ff.rastrigin, lambda : 2. * np.random.rand(3) - 1, 0.5,
    ...                options, restarts=9, bipop=True)  #doctest: +ELLIPSIS
    (3_w,7)-aCMA-ES (mu_w=2.3,w_1=58%) in dimension 3 (seed=12345...

    In either case, the method::

        cma.plot();

    (based on `matplotlib.pyplot`) produces a plot of the run and, if
    necessary::

        cma.s.figshow()

    shows the plot in a window. Finally::

        cma.s.figsave('myfirstrun')  # figsave from matplotlib.pyplot

    will save the figure in a png.

    We can use the gradient like

    >>> import cma
    >>> res = cma.fmin(cma.ff.rosen, np.zeros(10), 0.1,
    ...             options = {'ftarget':1e-8,},
    ...             gradf=cma.ff.grad_rosen,
    ...         )  #doctest: +ELLIPSIS
    (5_w,...
    >>> assert cma.ff.rosen(res[0]) < 1e-8
    >>> assert res[2] < 3600  # 1% are > 3300
    >>> assert res[3] < 3600  # 1% are > 3300

    If solution can only be comparatively ranked, either use
    `CMAEvolutionStrategy` directly or the objective accepts a list
    of solutions as input:

    >>> def parallel_sphere(X): return [cma.ff.sphere(x) for x in X]
    >>> x, es = cma.fmin2(None, 3 * [0], 0.1, {'verbose': -9},
    ...                   parallel_objective=parallel_sphere)
    >>> assert es.result[1] < 1e-9

    :See also: `CMAEvolutionStrategy`, `OOOptimizer.optimize`, `plot`,
        `CMAOptions`, `scipy.optimize.fmin`

    """  # style guides say there should be the above empty line
    if 1 < 3:  # try: # pass on KeyboardInterrupt
        if not objective_function and not parallel_objective:  # cma.fmin(0, 0, 0)
            return CMAOptions()  # these opts are by definition valid

        fmin_options = locals().copy()  # archive original options
        del fmin_options['objective_function']
        del fmin_options['x0']
        del fmin_options['sigma0']
        del fmin_options['options']
        del fmin_options['args']

        if options is None:
            options = cma_default_options
        CMAOptions().check_attributes(options)  # might modify options
        # checked that no options.ftarget =
        opts = CMAOptions(options.copy()).complement()

        callback = _callable_to_list(callback)

        # BIPOP-related variables:
        runs_with_small = 0
        small_i = []
        large_i = []
        popsize0 = None  # to be evaluated after the first iteration
        maxiter0 = None  # to be evaluated after the first iteration
        base_evals = 0

        irun = 0
        best = ot.BestSolution()
        all_stoppings[:] = []
        while True:  # restart loop
            sigma_factor = 1

            # Adjust the population according to BIPOP after a restart.
            if not bipop:
                # BIPOP not in use, simply double the previous population
                # on restart.
                if irun > 0:
                    popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                    opts['popsize'] = popsize0 * popsize_multiplier

            elif irun == 0:
                # Initial run is with "normal" population size; it is
                # the large population before first doubling, but its
                # budget accounting is the same as in case of small
                # population.
                poptype = 'small'

            elif sum(small_i) < bipop * max((1, sum(large_i))):
                # An interweaved run with small population size
                poptype = 'small'
                if 11 < 3:  # not needed when compared to irun - runs_with_small
                    restarts += 1  # A small restart doesn't count in the total
                runs_with_small += 1  # _Before_ it's used in popsize_lastlarge

                sigma_factor = 0.01**np.random.uniform()  # Local search
                popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                opts['popsize'] = np.floor(popsize0 * popsize_multiplier**(np.random.uniform()**2))
                opts['maxiter'] = min(maxiter0, 0.5 * sum(large_i) / opts['popsize'])
                # print('small basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

            else:
                # A run with large population size; the population
                # doubling is implicit with incpopsize.
                poptype = 'large'

                popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                opts['popsize'] = popsize0 * popsize_multiplier
                opts['maxiter'] = maxiter0
                # print('large basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

            if not callable(objective_function) and callable(parallel_objective):
                def objective_function(x, *args):
                    """created from `parallel_objective` argument"""
                    return parallel_objective([x], *args)[0]

            # recover from a CMA object
            if irun == 0 and isinstance(x0, CMAEvolutionStrategy):
                es = x0
                x0 = es.inputargs['x0']  # for the next restarts
                if np.isscalar(sigma0) and np.isfinite(sigma0) and sigma0 > 0:
                    es.sigma = sigma0
                # debatable whether this makes sense:
                sigma0 = es.inputargs['sigma0']  # for the next restarts
                if options is not cma_default_options:
                    if all(str(v) == v for v in options):
                        warnings.warn(
                            'Options must have explicit ("process") values in this \n'
                            'usecase. The passed options are likely to lead to an error \n'
                            'later. Passed options={0}'.format(options))
                    es.opts.set(options)
                # ignore further input args and keep original options
            else:  # default case
                if irun and eval(str(fmin_options['restart_from_best'])):
                    utils.print_warning('CAVE: restart_from_best is often not useful',
                                        verbose=opts['verbose'])
                    es = CMAEvolutionStrategy(best.x, sigma_factor * sigma0, opts)
                else:
                    es = CMAEvolutionStrategy(x0, sigma_factor * sigma0, opts)
                # return opts, es
                if callable(objective_function) and (
                        eval_initial_x
                        or es.opts['CMA_elitist'] == 'initial'
                        or (es.opts['CMA_elitist'] and
                                    eval_initial_x is None)):
                    x = es.gp.pheno(es.mean, copy=True,
                                    into_bounds=es.boundary_handler.repair,
                                    archive=es.sent_solutions)
                    es.f0 = objective_function(x, *args)
                    es.best.update([x], es.sent_solutions,
                                   [es.f0], 1)
                    es.countevals += 1
            es.objective_function = objective_function  # only for the record
            es.parallel_objective = parallel_objective

            opts = es.opts  # processed options, unambiguous
            # a hack:
            fmin_opts = CMAOptions("unchecked", **fmin_options.copy())
            for k in fmin_opts:
                # locals() cannot be modified directly, exec won't work
                # in 3.x, therefore
                fmin_opts.eval(k, loc={'N': es.N,
                                       'popsize': opts['popsize']},
                               correct_key=False)
            fmin_opts['restarts'] = options_parameters.amend_restarts_parameter(
                    fmin_opts['restarts'])

            es.logger.append = opts['verb_append'] or es.countiter > 0 or irun > 0
            # es.logger is "the same" logger, because the "identity"
            # is only determined by the `verb_filenameprefix` option
            logger = es.logger  # shortcut
            try:
                logger.persistent_communication_dict.update(
                    {'variable_annotations':
                    objective_function.variable_annotations})
            except AttributeError:
                pass

            if 11 < 3:
                if es.countiter == 0 and es.opts['verb_log'] > 0 and \
                        not es.opts['verb_append']:
                   logger = CMADataLogger(es.opts['verb_filenameprefix']
                                            ).register(es)
                   logger.add()
                es.writeOutput()  # initial values for sigma etc

            if noise_handler:
                if isinstance(noise_handler, type):
                    noisehandler = noise_handler(es.N)
                elif noise_handler is True:
                    noisehandler = ot.NoiseHandler(es.N)
                else:
                    noisehandler = noise_handler
                noise_handling = True
                if fmin_opts['noise_change_sigma_exponent'] > 0:
                    es.opts['tolfacupx'] = inf
            else:
                noisehandler = ot.NoiseHandler(es.N, 0)  # switched off
                noise_handling = False
            es.noise_handler = noisehandler

            for f in _callable_to_list(init_callback):
                f is None or f(es)

            # the problem: this assumes that good solutions cannot take longer than bad ones:
            # with EvalInParallel(objective_function, 2, is_feasible=opts['is_feasible']) as eval_in_parallel:
            if 1 < 3:
                while not es.stop():  # iteration loop
                    # X, fit = eval_in_parallel(lambda: es.ask(1)[0], es.popsize, args, repetitions=noisehandler.evaluations-1)
                    X, fit = es.ask_and_eval(parallel_objective or objective_function,
                                             args, gradf=gradf,
                                             evaluations=noisehandler.evaluations,
                                             aggregation=np.median,
                                             parallel_mode=parallel_objective)  # treats NaN with resampling if not parallel_mode
                    # TODO: check args and in case use args=(noisehandler.evaluations, )

                    if 11 < 3 and opts['vv']:  # inject a solution
                        # use option check_point = [0]
                        if 0 * np.random.randn() >= 0:
                            X[0] = 0 + opts['vv'] * es.sigma**0 * np.random.randn(es.N)
                            fit[0] = objective_function(X[0], *args)
                            # print fit[0]
                    if es.opts['verbose'] > 4:  # may be undesirable with dynamic fitness (e.g. Augmented Lagrangian)
                        if es.countiter < 2 or min(fit) <= es.best.last.f:
                            degrading_iterations_count = 0  # comes first to avoid code check complaint
                        else:  # min(fit) > es.best.last.f:
                            degrading_iterations_count += 1
                            if degrading_iterations_count > 4:
                                utils.print_message('%d f-degrading iterations (set verbose<=4 to suppress)'
                                                    % degrading_iterations_count,
                                                    iteration=es.countiter)
                    es.tell(X, fit)  # prepare for next iteration
                    if noise_handling:  # it would be better to also use these f-evaluations in tell
                        es.sigma *= noisehandler(X, fit, objective_function, es.ask,
                                                 args=args)**fmin_opts['noise_change_sigma_exponent']

                        es.countevals += noisehandler.evaluations_just_done  # TODO: this is a hack, not important though
                        # es.more_to_write.append(noisehandler.evaluations_just_done)
                        if noisehandler.maxevals > noisehandler.minevals:
                            es.more_to_write.append(noisehandler.evaluations)
                        if 1 < 3:
                            # If sigma was above multiplied by the same
                            #  factor cmean is divided by here, this is
                            #  like only multiplying kappa instead of
                            #  changing cmean and sigma.
                            es.sp.cmean *= np.exp(-noise_kappa_exponent * np.tanh(noisehandler.noiseS))
                            es.sp.cmean[es.sp.cmean > 1] = 1.0  # also works with "scalar arrays" like np.array(1.2)
                    for f in callback:
                        f is None or f(es)
                    es.disp()
                    logger.add(# more_data=[noisehandler.evaluations, 10**noisehandler.noiseS] if noise_handling else [],
                               modulo=1 if es.stop() and logger.modulo else None)
                    if (opts['verb_log'] and opts['verb_plot'] and
                          (es.countiter % max(opts['verb_plot'], opts['verb_log']) == 0 or es.stop())):
                        logger.plot(324)

            # end while not es.stop
            if opts['eval_final_mean'] and callable(objective_function):
                mean_pheno = es.gp.pheno(es.mean, copy=True,
                                         into_bounds=es.boundary_handler.repair,
                                         archive=es.sent_solutions)
                fmean = objective_function(mean_pheno, *args)
                es.countevals += 1
                es.best.update([mean_pheno], es.sent_solutions, [fmean], es.countevals)

            best.update(es.best, es.sent_solutions)  # in restarted case
            # es.best.update(best)

            this_evals = es.countevals - base_evals
            base_evals = es.countevals

            # BIPOP stats update

            if irun == 0:
                popsize0 = opts['popsize']
                maxiter0 = opts['maxiter']
                # XXX: This might be a bug? Reproduced from Matlab
                # small_i.append(this_evals)

            if bipop:
                if poptype == 'small':
                    small_i.append(this_evals)
                else:  # poptype == 'large'
                    large_i.append(this_evals)

            # final message
            if opts['verb_disp']:
                es.result_pretty(irun, time.asctime(time.localtime()),
                                 best.f)

            irun += 1
            # if irun > fmin_opts['restarts'] or 'ftarget' in es.stop() \
            # if irun > restarts or 'ftarget' in es.stop() \
            all_stoppings.append(dict(es.stop(check=False)))  # keeping the order
            if (irun - runs_with_small > fmin_opts['restarts']['maxrestarts']
                    or es.countevals >= fmin_opts['restarts']['maxfevals']
                    or 'ftarget' in es.stop()
                    or 'maxfevals' in es.stop(check=False)
                    or 'callback' in es.stop(check=False)):
                break
            opts['verb_append'] = es.countevals
            opts['popsize'] = fmin_opts['incpopsize'] * es.sp.popsize  # TODO: use rather options?
            try:
                opts['seed'] += 1
            except TypeError:
                pass

        # while irun

        # es.out['best'] = best  # TODO: this is a rather suboptimal type for inspection in the shell
        if irun:
            es.best.update(best)
            # TODO: there should be a better way to communicate the overall best
        return es.result + (es.stop(), es, logger)
        ### 4560
        # TODO refine output, can #args be flexible?
        # is this well usable as it is now?
    else:  # except KeyboardInterrupt:  # Exception as e:
        if eval(options_parameters.safe_str(options['verb_disp'])) > 0:
            print(' in/outcomment ``raise`` in last line of cma.fmin to prevent/restore KeyboardInterrupt exception')
        raise KeyboardInterrupt  # cave: swallowing this exception can silently mess up experiments, if ctrl-C is hit

def no_constraints(x):
    return []

def _al_set_logging(al, kwargs, *more_kwargs):
    """try to figure a good logging value from various verbosity options"""
    def get(d, key):
        v = d[key]
        try: v = v.split('#')[0]
        except (AttributeError, TypeError): pass
        try: v = ast.literal_eval(v)
        except ValueError: pass
        return v
    def extract_logging_value(kwargs):
        if 'logging' in kwargs:
            return kwargs['logging']
        if 'verbose' in kwargs and get(kwargs, 'verbose') <= -3:
            return False
        if 'options' in kwargs:
            ko = kwargs['options']
            if 'verb_log' in ko:
                return get(ko, 'verb_log')
            if 'verbose' in ko and get(ko, 'verbose') <= -3:
                return False
    kwargs = dict(kwargs)
    for m in more_kwargs:
        kwargs.update(m)
    logging = extract_logging_value(kwargs)
    if logging is not None and al is not None:
        al.logging = logging
    return logging

def fmin_con(objective_function, x0, sigma0,
             g=no_constraints, h=no_constraints, post_optimization=False,
             archiving=True, **kwargs):
    """Deprecated: use `cma.ConstrainedFitnessAL` or `cma.fmin_con2` instead.

    Optimize f with constraints g (inequalities) and h (equalities).

    Construct an Augmented Lagrangian instance ``f_aug_lag`` of the type
    `cma.constraints_handler.AugmentedLagrangian` from `objective_function`
    and `g` and `h`.

    Equality constraints should preferably be passed as two inequality
    constraints like ``[h - eps, -h - eps]``, with eps >= 0. When eps > 0,
    also feasible solution tracking can succeed.

    Return a `tuple` ``es.results.xfavorite:numpy.array, es:CMAEvolutionStrategy``,
    where ``es == cma.fmin2(f_aug_lag, x0, sigma0, **kwargs)[1]``.

    Depending on ``kwargs['logging']`` and on the verbosity settings in
    ``kwargs['options']``, the `AugmentedLagrangian` writes (hidden)
    logging files.

    The second return value:`CMAEvolutionStrategy` has an (additional)
    attribute ``best_feasible`` which contains the information about the
    best feasible solution in the ``best_feasible.info`` dictionary, given
    any feasible solution was found. This only works with inequality
    constraints (equality constraints are wrongly interpreted as inequality
    constraints).

    If `post_optimization` is set to True, then the attribute ``best_feasible``
    of the second return value will be updated with the best feasible solution obtained by
    optimizing the sum of the positive constraints squared starting from
    the point ``es.results.xfavorite``. Additionally, the first return value will
    be the best feasible solution obtained in post-optimization.

    In case when equality constraints are present and a "feasible" solution is requested,
    then `post_optimization` must be a strictly positive float indicating the error
    on the inequality constraints.

    The second return value:`CMAEvolutionStrategy` has also a
    `con_archives` attribute which is nonempty if `archiving`. The last
    element of each archive is the best feasible solution if there was any.

    See `cma.fmin` for further parameters ``**kwargs``.

    >>> import cma
    >>> x, es = cma.evolution_strategy.fmin_con(
    ...             cma.ff.sphere, 3 * [0], 1, g=lambda x: [1 - x[0]**2, -(1 - x[0]**2) - 1e-6],
    ...             options={'termination_callback': lambda es: -1e-5 < sum(es.mean**2) - 1 < 1e-5,
    ...                      'verbose':-9})
    >>> assert 'callback' in es.stop()
    >>> assert es.result.evaluations < 1500  # 10%-ish above 1000, 1%-ish above 1300
    >>> assert (sum(es.mean**2) - 1)**2 < 1e-9, es.mean

    >>> x, es = cma.evolution_strategy.fmin_con(
    ...             cma.ff.sphere, 2 * [0], 1, g=lambda x: [1 - x[0]**2],
    ...             options={'termination_callback': lambda es: -1e-8 < sum(es.mean**2) - 1 < 1e-8,
    ...                      'seed':1, 'verbose':-9})
    >>> assert es.best_feasible.f < 1 + 1e-5, es.best_feasible.f
    >>> ".info attribute dictionary keys: {0}".format(sorted(es.best_feasible.info))
    ".info attribute dictionary keys: ['f', 'g', 'g_al', 'x']"

    Details: this is a versatile function subject to changes. It is possible to access
    the `AugmentedLagrangian` instance like

    >>> al = es.augmented_lagrangian
    >>> isinstance(al, cma.constraints_handler.AugmentedLagrangian)
    True
    >>> # al.logger.plot()  # plots the evolution of AL coefficients

    >>> x, es = cma.evolution_strategy.fmin_con(
    ...             cma.ff.sphere, 2 * [0], 1, g=lambda x: [y+1 for y in x],
    ...             post_optimization=True, options={"verbose": -9})
    >>> assert all(y <= -1 for y in x)  # assert feasibility of x

"""
    # TODO: need to rethink equality/inequality interface?

    if 'parallel_objective' in kwargs:
        raise ValueError("`parallel_objective` parameter is not supported by cma.fmin_con")
    if post_optimization and h != no_constraints and (
            not isinstance(post_optimization, float) or post_optimization <= 0):
        raise ValueError("When equality constraints are given, the argument"
                         "``post_optimization`` must be a strictly positive "
                         "float indicating the error on the inequality constraints")
    # prepare callback list
    if callable(kwargs.setdefault('callback', [])):
        kwargs['callback'] = [kwargs['callback']]

    global _al  # for debugging, may be removed at some point
    F = []
    G = []
    _al = AugmentedLagrangian(len(x0))
    _al_set_logging(_al, kwargs)

    # _al.chi_domega = 1.1
    # _al.dgamma = 1.5

    best_feasible_solution = ot.BestSolution2()
    if archiving:
        archives = [
            _constraints_handler.ConstrainedSolutionsArchive(_constraints_handler._g_pos_max),
            _constraints_handler.ConstrainedSolutionsArchive(_constraints_handler._g_pos_sum),
            _constraints_handler.ConstrainedSolutionsArchive(_constraints_handler._g_pos_squared_sum),
        ]
    else:
        archives = []

    def f(x):
        F.append(objective_function(x))
        return F[-1]
    def constraints(x):
        gvals, hvals = g(x), h(x)
        # set m and equality attributes of al
        if _al.lam is None:  # TODO: better abide by an "official" interface?
            _al.set_m(len(gvals) + len(hvals))
            _al._equality = np.asarray(len(gvals) * [False] + len(hvals) * [True],
                                       dtype='bool')
        G.append(list(gvals) + list(hvals))
        return G[-1]
    def auglag(x):
        fval, gvals = f(x), constraints(x)
        alvals = _al(gvals)
        if all([gi <= 0 for gi in gvals]):
            best_feasible_solution.update(fval, x,
                info={'x':x, 'f': fval, 'g':gvals, 'g_al':alvals})
        info = _constraints_handler.constraints_info_dict(
                    _al.count_calls, x, fval, gvals, alvals)
        for a in archives:
            a.update(fval, gvals, info)
        return fval + sum(alvals)
    def set_coefficients(es):
        _al.set_coefficients(F, G)
        F[:], G[:] = [], []
    def update(es):
        x = es.ask(1, sigma_fac=0)[0]
        _al.update(f(x), constraints(x))

    kwargs['callback'].extend([set_coefficients, update])
    # The smallest observed f-values may be below the limit value f(x^*_feas)
    # because f-values depend on the adaptive multipliers. Hence we overwrite
    # the default tolstagnation value:
    kwargs.setdefault('options', {}).setdefault('tolstagnation', 0)
    _, es = fmin2(auglag, x0, sigma0, **kwargs)
    es.objective_function_complements = [_al]  # for historical reasons only
    es.augmented_lagrangian = _al
    es.best_feasible = best_feasible_solution

    if post_optimization:
        def f_post(x):
            return sum(gi ** 2 for gi in g(x) if gi > 0) + sum(
                       hi ** 2 for hi in h(x) if hi ** 2 > post_optimization ** 2)

        kwargs_post = kwargs.copy()
        kwargs_post.setdefault('options', {})['ftarget'] = 0

        _, es_post = fmin2(f_post, es.result.xfavorite, es.sigma,
                           **kwargs_post)
        if es_post.best.f == 0:
            f = objective_function(es_post.best.x)
            es.best_feasible.update(f, x=es_post.best.x, info={
                'x': es_post.best.x,
                'f': f,
                'g': None  # it's a feasible solution, so we don't really care
            })
            return es.best_feasible.x, es
        x_post = es_post.result.xfavorite
        g_x_post, h_x_post = g(x_post), h(x_post)
        if all([gi <= 0 for gi in g_x_post]) and \
                all([hi ** 2 <= post_optimization ** 2 for hi in h_x_post]):
            f_x_post = objective_function(x_post)
            es.best_feasible.update(f_x_post, x=x_post, info={
                'x': x_post,
                'f': f_x_post,
                'g': list(g_x_post) + list(h_x_post)
            })
            return x_post, es
        else:
            utils.print_warning('Post optimization was unsuccessful',
                                verbose=es.opts['verbose'])

    es.con_archives = archives
    return es.result.xfavorite, es  # do not return es.best_feasible.x because it could be quite bad

def fmin_con2(objective_function, x0, sigma0,
             constraints=no_constraints,
             find_feasible_first=False,
             find_feasible_final=False,
             kwargs_confit=None, **kwargs_fmin):
    """optimize f with inequality constraints g.

    `constraints` is a function that returns a list of constraints values,
    where feasibility means <= 0. An equality constraint ``h(x) == 0`` can
    be expressed as two inequality constraints like ``[h(x) - eps, -h(x) -
    eps]`` with ``eps >= 0``.

    `find_feasible_...` arguments toggle to search for a feasible solution
    before and after the constrained problem is optimized. Because this can
    not work with equality constraints, where the feasible domain has zero
    volume, find-feasible arguments are `False` by default.

    `kwargs_confit` are keyword arguments to instantiate
    `constraints_handler.ConstrainedFitnessAL` which is optimized and
    returned as `objective_function` attribute in the second return
    argument (type `CMAEvolutionStrategy`).

    Other and further keyword arguments are passed (in ``**kwargs_fmin``)
    to `cma.fmin2`.

    Consider using `ConstrainedFitnessAL` directly instead of `fmin_con2`.

"""
    if isinstance(find_feasible_first, dict) or isinstance(find_feasible_final, dict):
        raise ValueError("Found an unexected `dict` as argument. Recheck the calling signature."
                         "\nUse the keyword `options={...}` to pass an options argument for `fmin2`."
                         "\nUse the keyword syntax also for any further arguments passed to `fmin2`.")
    if kwargs_confit is None:
        kwargs_confit = {}  # does not change default parameter value
    kwargs_fmin.setdefault('options', {}).setdefault('tolstagnation', 0)

    logging = _al_set_logging(None, kwargs_fmin, kwargs_confit) # the latter overwrites the former
    if logging is not None:
        kwargs_confit['logging'] = logging

    # instantiate unconstrained fitness
    fun = _constraints_handler.ConstrainedFitnessAL(
            objective_function, constraints,
            find_feasible_first=find_feasible_first,
            **kwargs_confit)

    # append fun.update to callback option argument
    if 'callback' not in kwargs_fmin or kwargs_fmin['callback'] is None:
        kwargs_fmin['callback'] = [fun.update]
    else:
        try: kwargs_fmin['callback'] = list(kwargs_fmin['callback']) + [fun.update]
        except: kwargs_fmin['callback'] = [kwargs_fmin['callback']] + [fun.update]

    # optimize fun using fmin2
    _, es = fmin2(fun, x0, sigma0, **kwargs_fmin)
    assert es.objective_function is fun  # we could also just assign it

    # optimize to feasible solution, in case
    if find_feasible_final:
        x = fun.find_feasible(es)  # uses es.optimize
        if kwargs_fmin['options'].get('eval_final_mean', None):
            # this doesn't make sense if xfavorite is returned anyway
            g = constraints(es.result.xfavorite)
            fun._update_best(x, objective_function(x), g, fun.al(g))
            x = fun.best_feas.x
    else:
        x = es.result.xfavorite

    es.best_feasible = fun.best_feas
    es.con_archives = fun.archives
    return x, es  # fun == es.objective_function
