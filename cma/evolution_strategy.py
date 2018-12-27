"""CMA-ES (evolution strategy), the main sub-module of `cma` providing
in particular `CMAOptions`, `CMAEvolutionStrategy`, and `fmin`
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
# from builtins import ...
from .utilities.python3for2 import range  # redefine range in Python 2

import sys
import time  # not really essential
import warnings  # catch numpy warnings
import ast  # for literal_eval
try:
    import collections  # not available in Python 2.5
except ImportError:
    pass
import math
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
from . import transformations
from . import optimization_tools as ot
from . import sampler
from .constraints_handler import BoundNone, BoundPenalty, BoundTransform
from .recombination_weights import RecombinationWeights
from .logger import CMADataLogger  # , disp, plot
from .utilities.utils import BlancClass as _BlancClass
from .utilities.utils import rglen  #, global_verbosity
from .utilities.utils import pprint
# from .utilities.math import Mh
from .sigma_adaptation import *
from . import restricted_gaussian_sampler as _rgs

_where = np.nonzero  # to make pypy work, this is how where is used here anyway
del division, print_function, absolute_import  #, unicode_literals, with_statement

# use_archives uses collections
use_archives = sys.version_info[0] >= 3 or sys.version_info[1] >= 6
# use_archives = False  # on False some unit tests fail
"""speed up for very large population size. `use_archives` prevents the
need for an inverse gp-transformation, relies on collections module,
not sure what happens if set to ``False``. """

class MetaParameters(object):
    """collection of many meta parameters.

    Meta parameters are either annotated constants or refer to
    options from `CMAOptions` or are arguments to `fmin` or to the
    `NoiseHandler` class constructor.

    `MetaParameters` take only effect if the source code is modified by
    a meta parameter weaver module searching for ## meta_parameters....
    and modifying the next line.

    Details
    -------
    This code contains a single class instance `meta_parameters`

    Some interfaces rely on parameters being either `int` or
    `float` only. More sophisticated choices are implemented via
    ``choice_value = {1: 'this', 2: 'or that'}[int_param_value]`` here.

    CAVEAT
    ------
    `meta_parameters` should not be used to determine default
    arguments, because these are assigned only once and for all during
    module import.

    """
    def __init__(self):
        """assign settings to be used"""
        self.sigma0 = None  ## [~0.01, ~10]  # no default available

        # learning rates and back-ward time horizons
        self.CMA_cmean = 1.0  ## [~0.1, ~10]  #
        self.c1_multiplier = 1.0  ## [~1e-4, ~20] l
        self.cmu_multiplier = 2.0  ## [~1e-4, ~30] l  # zero means off
        self.CMA_active = 1.0  ## [~1e-4, ~10] l  # 0 means off, was CMA_activefac
        self.cc_multiplier = 1.0  ## [~0.01, ~20] l
        self.cs_multiplier = 1.0 ## [~0.01, ~10] l  # learning rate for cs
        self.CSA_dampfac = 1.0  ## [~0.01, ~10]
        self.CMA_dampsvec_fac = None  ## [~0.01, ~100]  # def=np.Inf or 0.5, not clear whether this is a log parameter
        self.CMA_dampsvec_fade = 0.1  ## [0, ~2]

        # exponents for learning rates
        self.c1_exponent = 2.0  ## [~1.25, 2]
        self.cmu_exponent = 2.0  ## [~1.25, 2]
        self.cact_exponent = 1.5  ## [~1.25, 2]
        self.cc_exponent = 1.0  ## [~0.25, ~1.25]
        self.cs_exponent = 1.0  ## [~0.25, ~1.75]  # upper bound depends on CSA_clip_length_value

        # selection related parameters
        self.lambda_exponent = 0.0  ## [0, ~2.5]  # usually <= 2, used by adding N**lambda_exponent to popsize-1
        self.CMA_elitist = 0  ## [0, 2] i  # a choice variable
        self.CMA_mirrors = 0.0  ## [0, 0.5)  # values <0.5 are interpreted as fraction, values >1 as numbers (rounded), otherwise about 0.16 is used',

        # sampling strategies
        # self.CMA_sample_on_sphere_surface = 0  ## [0, 1] i  # boolean
        self.mean_shift_line_samples = 0  ## [0, 1] i  # boolean
        self.pc_line_samples = 0  ## [0, 1] i  # boolean

        # step-size adapation related parameters
        self.CSA_damp_mueff_exponent = 0.5  ## [~0.25, ~1.5]  # zero would mean no dependency of damping on mueff, useful with CSA_disregard_length option',
        self.CSA_disregard_length = 0  ## [0, 1] i
        self.CSA_squared = 0  ## [0, 1] i
        self.CSA_clip_length_value = None  ## [0, ~20]  # None reflects inf

        # noise handling
        self.noise_reeval_multiplier = 1.0  ## [0.2, 4]  # usually 2 offspring are reevaluated
        self.noise_choose_reeval = 1  ## [1, 3] i  # which ones to reevaluate
        self.noise_theta = 0.5  ## [~0.05, ~0.9]
        self.noise_alphasigma = 2.0  ## [0, 10]
        self.noise_alphaevals = 2.0  ## [0, 10]
        self.noise_alphaevalsdown_exponent = -0.25  ## [-1.5, 0]
        self.noise_aggregate = None  ## [1, 2] i  # None and 0 == default or user option choice, 1 == median, 2 == mean
        # TODO: more noise handling options (maxreevals...)

        # restarts
        self.restarts = 0  ## [0, ~30]  # but depends on popsize inc
        self.restart_from_best = 0  ## [0, 1] i  # bool
        self.incpopsize = 2.0  ## [~1, ~5]

        # termination conditions (for restarts)
        self.maxiter_multiplier = 1.0  ## [~0.01, ~100] l
        self.mindx = 0.0  ## [1e-17, ~1e-3] l  #v minimal std in any direction, cave interference with tol*',
        self.minstd = 0.0  ## [1e-17, ~1e-3] l  #v minimal std in any coordinate direction, cave interference with tol*',
        self.maxstd = None  ## [~1, ~1e9] l  #v maximal std in any coordinate direction, default is inf',
        self.tolfacupx = 1e3  ## [~10, ~1e9] l  #v termination when step-size increases by tolfacupx (diverges). That is, the initial step-size was chosen far too small and better solutions were found far away from the initial solution x0',
        self.tolupsigma = 1e20  ## [~100, ~1e99] l  #v sigma/sigma0 > tolupsigma * max(sqrt(eivenvals(C))) indicates "creeping behavior" with usually minor improvements',
        self.tolx = 1e-11  ## [1e-17, ~1e-3] l  #v termination criterion: tolerance in x-changes',
        self.tolfun = 1e-11  ## [1e-17, ~1e-3] l  #v termination criterion: tolerance in function value, quite useful',
        self.tolfunhist = 1e-12  ## [1e-17, ~1e-3] l  #v termination criterion: tolerance in function value history',
        self.tolstagnation_multiplier = 1.0  ## [0.01, ~100]  # ': 'int(100 + 100 * N**1.5 / popsize)  #v termination if no improvement over tolstagnation iterations',

        # abandoned:
        # self.noise_change_sigma_exponent = 1.0  ## [0, 2]
        # self.noise_epsilon = 1e-7  ## [0, ~1e-2] l  #
        # self.maxfevals = None  ## [1, ~1e11] l  # is not a performance parameter
        # self.lambda_log_multiplier = 3  ## [0, ~10]
        # self.lambda_multiplier = 0  ## (0, ~10]

meta_parameters = MetaParameters()

def is_feasible(x, f):
    """default to check feasibility of f-values.

    Used for rejection sampling in method `ask_and_eval`.

    :See also: CMAOptions, ``CMAOptions('feas')``.
    """
    return f is not None and not np.isnan(f)


if use_archives:

    from .utilities.utils import SolutionDict
    class _CMASolutionDict(SolutionDict):
        def __init__(self, *args, **kwargs):
            # SolutionDict.__init__(self, *args, **kwargs)
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

else:  # if not use_archives:
    class _CMASolutionDict(dict):
        """a hack to get most code examples running"""
        def insert(self, *args, **kwargs):
            pass
        def get(self, key):
            return None
        def __getitem__(self, key):
            return None
        def __setitem__(self, key, value):
            pass

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

cma_default_options = {
    # the follow string arguments are evaluated if they do not contain "filename"
    'AdaptSigma': 'True  # or False or any CMAAdaptSigmaBase class e.g. CMAAdaptSigmaTPA, CMAAdaptSigmaCSA',
    'CMA_active': 'True  # negative update, conducted after the original update',
#    'CMA_activefac': '1  # learning rate multiplier for active update',
    'CMA_cmean': '1  # learning rate for the mean value',
    'CMA_const_trace': 'False  # normalize trace, 1, True, "arithm", "geom", "aeig", "geig" are valid',
    'CMA_diagonal': '0*100*N/popsize**0.5  # nb of iterations with diagonal covariance matrix, True for always',  # TODO 4/ccov_separable?
    'CMA_eigenmethod': 'np.linalg.eigh  # or cma.utilities.math.eig or pygsl.eigen.eigenvectors',
    'CMA_elitist': 'False  #v or "initial" or True, elitism likely impairs global search performance',
    'CMA_injections_threshold_keep_len': '0  #v keep length if Mahalanobis length is below the given relative threshold',
    'CMA_mirrors': 'popsize < 6  # values <0.5 are interpreted as fraction, values >1 as numbers (rounded), otherwise about 0.16 is used',
    'CMA_mirrormethod': '2  # 0=unconditional, 1=selective, 2=selective with delay',
    'CMA_mu': 'None  # parents selection parameter, default is popsize // 2',
    'CMA_on': '1  # multiplier for all covariance matrix updates',
    # 'CMA_sample_on_sphere_surface': 'False  #v replaced with option randn=cma.utilities.math.randhss, all mutation vectors have the same length, currently (with new_sampling) not in effect',
    'CMA_sampler': 'None  # a class or instance that implements the interface of `cma.interfaces.StatisticalModelSamplerWithZeroMeanBaseClass`',
    'CMA_sampler_options': '{}  # options passed to `CMA_sampler` class init as keyword arguments',
    'CMA_rankmu': '1.0  # multiplier for rank-mu update learning rate of covariance matrix',
    'CMA_rankone': '1.0  # multiplier for rank-one update learning rate of covariance matrix',
    'CMA_recombination_weights': 'None  # a list, see class RecombinationWeights, overwrites CMA_mu and popsize options',
    'CMA_dampsvec_fac': 'np.Inf  # tentative and subject to changes, 0.5 would be a "default" damping for sigma vector update',
    'CMA_dampsvec_fade': '0.1  # tentative fading out parameter for sigma vector update',
    'CMA_teststds': 'None  # factors for non-isotropic initial distr. of C, mainly for test purpose, see CMA_stds for production',
    'CMA_stds': 'None  # multipliers for sigma0 in each coordinate, not represented in C, makes scaling_of_variables obsolete',
    # 'CMA_AII': 'False  # not yet tested',
    'CSA_dampfac': '1  #v positive multiplier for step-size damping, 0.3 is close to optimal on the sphere',
    'CSA_damp_mueff_exponent': '0.5  # zero would mean no dependency of damping on mueff, useful with CSA_disregard_length option',
    'CSA_disregard_length': 'False  #v True is untested, also changes respective parameters',
    'CSA_clip_length_value': 'None  #v poorly tested, [0, 0] means const length N**0.5, [-1, 1] allows a variation of +- N/(N+2), etc.',
    'CSA_squared': 'False  #v use squared length for sigma-adaptation ',
    'BoundaryHandler': 'BoundTransform  # or BoundPenalty, unused when ``bounds in (None, [None, None])``',
    'bounds': '[None, None]  # lower (=bounds[0]) and upper domain boundaries, each a scalar or a list/vector',
     # , eval_parallel2': 'not in use {"processes": None, "timeout": 12, "is_feasible": lambda x: True} # distributes function calls to processes processes'
     # 'callback': 'None  # function or list of functions called as callback(self) at the end of the iteration (end of tell)', # only necessary in fmin and optimize
    'conditioncov_alleviate': '[1e8, 1e12]  # when to alleviate the condition in the coordinates and in main axes',
    'eval_final_mean': 'True  # evaluate the final mean, which is a favorite return candidate',
    'fixed_variables': 'None  # dictionary with index-value pairs like {0:1.1, 2:0.1} that are not optimized',
    'ftarget': '-inf  #v target function value, minimization',
    'integer_variables': '[]  # index list, invokes basic integer handling: prevent std dev to become too small in the given variables',
    'is_feasible': 'is_feasible  #v a function that computes feasibility, by default lambda x, f: f not in (None, np.NaN)',
    'maxfevals': 'inf  #v maximum number of function evaluations',
    'maxiter': '100 + 150 * (N+3)**2 // popsize**0.5  #v maximum number of iterations',
    'mean_shift_line_samples': 'False #v sample two new solutions colinear to previous mean shift',
    'mindx': '0  #v minimal std in any arbitrary direction, cave interference with tol*',
    'minstd': '0  #v minimal std (scalar or vector) in any coordinate direction, cave interference with tol*',
    'maxstd': 'inf  #v maximal std in any coordinate direction',
    'pc_line_samples': 'False #v one line sample along the evolution path pc',
    'popsize': '4+int(3*np.log(N))  # population size, AKA lambda, number of new solution per iteration',
    'randn': 'np.random.randn  #v randn(lam, N) must return an np.array of shape (lam, N), see also cma.utilities.math.randhss',
    'scaling_of_variables': '''None  # depreciated, rather use fitness_transformations.ScaleCoordinates instead (or possibly CMA_stds).
            Scale for each variable in that effective_sigma0 = sigma0*scaling. Internally the variables are divided by scaling_of_variables and sigma is unchanged, default is `np.ones(N)`''',
    'seed': 'time  # random number seed for `numpy.random`; `None` and `0` equate to `time`, `np.nan` means "do nothing", see also option "randn"',
    'signals_filename': 'None  # cma_signals.in  # read versatile options from this file which contains a single options dict, e.g. ``{"timeout": 0}`` to stop, string-values are evaluated, e.g. "np.inf" is valid',
    'termination_callback': 'None  #v a function returning True for termination, called in `stop` with `self` as argument, could be abused for side effects',
    'timeout': 'inf  #v stop if timeout seconds are exceeded, the string "2.5 * 60**2" evaluates to 2 hours and 30 minutes',
    'tolconditioncov': '1e14  #v stop if the condition of the covariance matrix is above `tolconditioncov`',
    'tolfacupx': '1e3  #v termination when step-size increases by tolfacupx (diverges). That is, the initial step-size was chosen far too small and better solutions were found far away from the initial solution x0',
    'tolupsigma': '1e20  #v sigma/sigma0 > tolupsigma * max(eivenvals(C)**0.5) indicates "creeping behavior" with usually minor improvements',
    'tolfun': '1e-11  #v termination criterion: tolerance in function value, quite useful',
    'tolfunhist': '1e-12  #v termination criterion: tolerance in function value history',
    'tolstagnation': 'int(100 + 100 * N**1.5 / popsize)  #v termination if no improvement over tolstagnation iterations',
    'tolx': '1e-11  #v termination criterion: tolerance in x-changes',
    'transformation': '''None  # depreciated, use cma.fitness_transformations.FitnessTransformation instead.
            [t0, t1] are two mappings, t0 transforms solutions from CMA-representation to f-representation (tf_pheno),
            t1 is the (optional) back transformation, see class GenoPheno''',
    'typical_x': 'None  # used with scaling_of_variables',
    'updatecovwait': 'None  #v number of iterations without distribution update, name is subject to future changes',  # TODO: rename: iterwaitupdatedistribution?
    'verbose': '3  #v verbosity e.g. of initial/final message, -1 is very quiet, -9 maximally quiet, may not be fully implemented',
    'verb_append': '0  # initial evaluation counter, if append, do not overwrite output files',
    'verb_disp': '100  #v verbosity: display console output every verb_disp iteration',
    'verb_filenameprefix': CMADataLogger.default_prefix + '  # output path and filenames prefix',
    'verb_log': '1  #v verbosity: write data to files every verb_log iteration, writing can be time critical on fast to evaluate functions',
    'verb_plot': '0  #v in fmin(): plot() is called every verb_plot iteration',
    'verb_time': 'True  #v output timings on console',
    'vv': '{}  #? versatile set or dictionary for hacking purposes, value found in self.opts["vv"]'
}

class CMAOptions(dict):
    """a dictionary with the available options and their default values
    for class `CMAEvolutionStrategy`.

    ``CMAOptions()`` returns a `dict` with all available options and their
    default values with a comment string.

    ``CMAOptions('verb')`` returns a subset of recognized options that
    contain 'verb' in there keyword name or (default) value or
    description.

    ``CMAOptions(opts)`` returns the subset of recognized options in
    ``dict(opts)``.

    Option values can be "written" in a string and, when passed to `fmin`
    or `CMAEvolutionStrategy`, are evaluated using "N" and "popsize" as
    known values for dimension and population size (sample size, number
    of new solutions per iteration). All default option values are given
    as such a string.

    Details
    -------
    `CMAOptions` entries starting with ``tol`` are termination
    "tolerances".

    For `tolstagnation`, the median over the first and the second half
    of at least `tolstagnation` iterations are compared for both, the
    per-iteration best and per-iteration median function value.

    Example
    -------
    ::

        import cma
        cma.CMAOptions('tol')

    is a shortcut for ``cma.CMAOptions().match('tol')`` that returns all
    options that contain 'tol' in their name or description.

    To set an option::

        import cma
        opts = cma.CMAOptions()
        opts.set('tolfun', 1e-12)
        opts['tolx'] = 1e-11

    todo: this class is overly complex and should be re-written, possibly
    with reduced functionality.

    :See also: `fmin` (), `CMAEvolutionStrategy`, `_CMAParameters`

    """

    # @classmethod # self is the class, not the instance
    # @property
    # def default(self):
    #     """returns all options with defaults"""
    #     return fmin([],[])

    @staticmethod
    def defaults():
        """return a dictionary with default option values and description"""
        return dict((str(k), str(v)) for k, v in cma_default_options.items())
        # getting rid of the u of u"name" by str(u"name")
        # return dict(cma_default_options)

    @staticmethod
    def versatile_options():
        """return list of options that can be changed at any time (not
        only be initialized).

        Consider that this list might not be entirely up
        to date.

        The string ' #v ' in the default value indicates a versatile
        option that can be changed any time, however a string will not 
        necessarily be evaluated again.

        """
        return tuple(sorted(i[0] for i in list(CMAOptions.defaults().items()) if i[1].find(' #v ') > 0))
    def check(self, options=None):
        """check for ambiguous keys and move attributes into dict"""
        self.check_values(options)
        self.check_attributes(options)
        self.check_values(options)
        return self
    def check_values(self, options=None):
        corrected_key = CMAOptions().corrected_key  # caveat: infinite recursion
        validated_keys = []
        original_keys = []
        if options is None:
            options = self
        for key in options:
            correct_key = corrected_key(key)
            if correct_key is None:
                raise ValueError("""%s is not a valid option""" % key)
            if correct_key in validated_keys:
                if key == correct_key:
                    key = original_keys[validated_keys.index(key)]
                raise ValueError("%s was not a unique key for %s option"
                    % (key, correct_key))
            validated_keys.append(correct_key)
            original_keys.append(key)
        return options
    def check_attributes(self, opts=None):
        """check for attributes and moves them into the dictionary"""
        if opts is None:
            opts = self
        if 11 < 3:
            if hasattr(opts, '__dict__'):
                for key in opts.__dict__:
                    if key not in self._attributes:
                        raise ValueError("""
                        Assign options with ``opts['%s']``
                        instead of ``opts.%s``
                        """ % (opts.__dict__.keys()[0],
                               opts.__dict__.keys()[0]))
            return self
        else:
        # the problem with merge is that ``opts['ftarget'] = new_value``
        # would be overwritten by the old ``opts.ftarget``.
        # The solution here is to empty opts.__dict__ after the merge
            if hasattr(opts, '__dict__'):
                for key in list(opts.__dict__):
                    if key in self._attributes:
                        continue
                    utils.print_warning(
                        """
        An option attribute has been merged into the dictionary,
        thereby possibly overwriting the dictionary value, and the
        attribute has been removed. Assign options with

            ``opts['%s'] = value``  # dictionary assignment

        or use

            ``opts.set('%s', value)  # here isinstance(opts, CMAOptions)

        instead of

            ``opts.%s = value``  # attribute assignment
                        """ % (key, key, key), 'check', 'CMAOptions')

                    opts[key] = opts.__dict__[key]  # getattr(opts, key)
                    delattr(opts, key)  # is that cosher?
                    # delattr is necessary to prevent that the attribute
                    # overwrites the dict entry later again
            return opts

    def __init__(self, s=None, **kwargs):
        """return an `CMAOptions` instance.

        Return default options if ``s is None and not kwargs``,
        or all options whose name or description contains `s`, if
        `s` is a (search) string (case is disregarded in the match),
        or with entries from dictionary `s` as options,
        or with kwargs as options if ``s is None``,
        in any of the latter cases not complemented with default options
        or settings.

        Returns: see above.

        Details: as several options start with ``'s'``, ``s=value`` is
        not valid as an option setting.

        """
        # if not CMAOptions.defaults:  # this is different from self.defaults!!!
        #     CMAOptions.defaults = fmin([],[])
        if s is None and not kwargs:
            super(CMAOptions, self).__init__(CMAOptions.defaults())  # dict.__init__(self, CMAOptions.defaults()) should be the same
            # self = CMAOptions.defaults()
            s = 'nocheck'
        elif utils.is_str(s) and not s.startswith('unchecked'):
            super(CMAOptions, self).__init__(CMAOptions().match(s))
            # we could return here
            s = 'nocheck'
        elif isinstance(s, dict):
            if kwargs:
                raise ValueError('Dictionary argument must be the only argument')
            super(CMAOptions, self).__init__(s)
        elif kwargs and (s is None or s.startswith('unchecked')):
            super(CMAOptions, self).__init__(kwargs)
        else:
            raise ValueError('The first argument must be a string or a dict or a keyword argument or `None`')
        if not utils.is_str(s) or not s.startswith(('unchecked', 'nocheck')):
            # was main offender
            self.check()  # caveat: infinite recursion
            for key in list(self.keys()):
                correct_key = self.corrected_key(key)
                if correct_key not in CMAOptions.defaults():
                    utils.print_warning('invalid key ``' + str(key) +
                                   '`` removed', '__init__', 'CMAOptions')
                    self.pop(key)
                elif key != correct_key:
                    self[correct_key] = self.pop(key)
        # self.evaluated = False  # would become an option entry
        self._lock_setting = False
        self._attributes = self.__dict__.copy()  # are not valid keys
        self._attributes['_attributes'] = len(self._attributes)

    def init(self, dict_or_str, val=None, warn=True):
        """initialize one or several options.

        Arguments
        ---------
            `dict_or_str`
                a dictionary if ``val is None``, otherwise a key.
                If `val` is provided `dict_or_str` must be a valid key.
            `val`
                value for key

        Details
        -------
        Only known keys are accepted. Known keys are in `CMAOptions.defaults()`

        """
        # dic = dict_or_key if val is None else {dict_or_key:val}
        self.check(dict_or_str)
        dic = dict_or_str
        if val is not None:
            dic = {dict_or_str:val}

        for key, val in dic.items():
            key = self.corrected_key(key)
            if key not in CMAOptions.defaults():
                # TODO: find a better solution?
                if warn:
                    print('Warning in cma.CMAOptions.init(): key ' +
                        str(key) + ' ignored')
            else:
                self[key] = val

        return self

    def set(self, dic, val=None, force=False):
        """assign versatile options.

        Method `CMAOptions.versatile_options` () gives the versatile
        options, use `init()` to set the others.

        Arguments
        ---------
            `dic`
                either a dictionary or a key. In the latter
                case, `val` must be provided
            `val`
                value for `key`, approximate match is sufficient
            `force`
                force setting of non-versatile options, use with caution

        This method will be most probably used with the ``opts`` attribute of
        a `CMAEvolutionStrategy` instance.

        """
        if val is not None:  # dic is a key in this case
            dic = {dic:val}  # compose a dictionary
        for key_original, val in list(dict(dic).items()):
            key = self.corrected_key(key_original)
            if (not self._lock_setting or
                key in CMAOptions.versatile_options() or
                force):
                self[key] = val
            else:
                utils.print_warning('key ' + str(key_original) +
                      ' ignored (not recognized as versatile)',
                               'set', 'CMAOptions')
        return self  # to allow o = CMAOptions(o).set(new)

    def complement(self):
        """add all missing options with their default values"""

        # add meta-parameters, given options have priority
        self.check()
        for key in CMAOptions.defaults():
            if key not in self:
                self[key] = CMAOptions.defaults()[key]
        return self

    @property
    def settable(self):
        """return the subset of those options that are settable at any
        time.

        Settable options are in `versatile_options` (), but the
        list might be incomplete.

        """
        return CMAOptions(dict(i for i in list(self.items())
                                if i[0] in CMAOptions.versatile_options()))

    def __call__(self, key, default=None, loc=None):
        """evaluate and return the value of option `key` on the fly, or
        return those options whose name or description contains `key`,
        case disregarded.

        Details
        -------
        Keys that contain `filename` are not evaluated.
        For ``loc==None``, `self` is used as environment
        but this does not define ``N``.

        :See: `eval()`, `evalall()`

        """
        try:
            val = self[key]
        except:
            return self.match(key)

        if loc is None:
            loc = self  # TODO: this hack is not so useful: popsize could be there, but N is missing
        try:
            if utils.is_str(val):
                val = val.split('#')[0].strip()  # remove comments
                if key.find('filename') < 0:
                        # and key.find('mindx') < 0:
                    val = eval(val, globals(), loc)
            # invoke default
            # TODO: val in ... fails with array type, because it is applied element wise!
            # elif val in (None,(),[],{}) and default is not None:
            elif val is None and default is not None:
                val = eval(str(default), globals(), loc)
        except:
            pass  # slighly optimistic: the previous is bug-free
        return val

    def corrected_key(self, key):
        """return the matching valid key, if ``key.lower()`` is a unique
        starting sequence to identify the valid key, ``else None``

        """
        matching_keys = []
        for allowed_key in CMAOptions.defaults():
            if allowed_key.lower() == key.lower():
                return allowed_key
            if allowed_key.lower().startswith(key.lower()):
                matching_keys.append(allowed_key)
        return matching_keys[0] if len(matching_keys) == 1 else None

    def eval(self, key, default=None, loc=None, correct_key=True):
        """Evaluates and sets the specified option value in
        environment `loc`. Many options need ``N`` to be defined in
        `loc`, some need `popsize`.

        Details
        -------
        Keys that contain 'filename' are not evaluated.
        For `loc` is None, the self-dict is used as environment

        :See: `evalall()`, `__call__`

        """
        # TODO: try: loc['dim'] = loc['N'] etc
        if correct_key:
            # in_key = key  # for debugging only
            key = self.corrected_key(key)
        self[key] = self(key, default, loc)
        return self[key]

    def evalall(self, loc=None, defaults=None):
        """Evaluates all option values in environment `loc`.

        :See: `eval()`

        """
        self.check()
        if defaults is None:
            defaults = cma_default_options
        # TODO: this needs rather the parameter N instead of loc
        if 'N' in loc:  # TODO: __init__ of CMA can be simplified
            popsize = self('popsize', defaults['popsize'], loc)
            for k in list(self.keys()):
                k = self.corrected_key(k)
                self.eval(k, defaults[k],
                          {'N':loc['N'], 'popsize':popsize})
        self._lock_setting = True
        return self

    def match(self, s=''):
        """return all options that match, in the name or the description,
        with string `s`, case is disregarded.

        Example: ``cma.CMAOptions().match('verb')`` returns the verbosity
        options.

        """
        match = s.lower()
        res = {}
        for k in sorted(self):
            s = str(k) + '=\'' + str(self[k]) + '\''
            if match in s.lower():
                res[k] = self[k]
        return CMAOptions(res)

    @property
    def to_namedtuple(self):
        """return options as const attributes of the returned object,
        only useful for inspection. """
        raise NotImplementedError
        # return collections.namedtuple('CMAOptionsNamedTuple',
        #                               self.keys())(**self)

    def from_namedtuple(self, t):
        """update options from a `collections.namedtuple`.
        :See also: `to_namedtuple`
        """
        return self.update(t._asdict())

    def pprint(self, linebreak=80):
        for i in sorted(self.items()):
            s = str(i[0]) + "='" + str(i[1]) + "'"
            a = s.split(' ')

            # print s in chunks
            l = ''  # start entire to the left
            while a:
                while a and len(l) + len(a[0]) < linebreak:
                    l += ' ' + a.pop(0)
                print(l)
                l = '        '  # tab for subsequent lines
try:
    collections.namedtuple
except:
    pass
else:
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

        The best solution of the last completed iteration can be accessed via
        attribute ``pop_sorted[0]`` of `CMAEvolutionStrategy` and the
        respective objective function value via ``fit.fit[0]``.

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

    The best solution of the last completed iteration can be accessed via
    attribute ``pop_sorted[0]`` of `CMAEvolutionStrategy` and the
    respective objective function value via ``fit.fit[0]``.

    Details:

    - This class is of purely declarative nature and for providing this
      docstring. It does not provide any further functionality.
    - ``list(fit.fit).find(0)`` is the index of the first sampled solution
      of the last completed iteration in ``pop_sorted``.

    """
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
            self.gp.pheno(self.mean),
            self.gp.scales * self.sigma * self.sigma_vec.scaling *
                self.dC**0.5))

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
        lie within about `x0` +- ``3*sigma0``. See also options
        `scaling_of_variables`. Often one wants to check for
        solutions close to the initial point. This allows,
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
    >>> assert len(es.result) == 8
    >>> assert es.result[1] < 1e-9

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
    ...         while curr_fit in (None, np.NaN):
    ...             x = es.ask(1)[0]
    ...             curr_fit = cma.ff.somenan(x, cma.ff.elli) # might return np.NaN
    ...         X.append(x)
    ...         fit.append(curr_fit)
    ...     es.tell(X, fit)
    ...     es.logger.add()
    ...     es.disp()  #doctest: +ELLIPSIS
    Itera...
    >>>
    >>> assert es.result[1] < 1e-9
    >>> assert es.result[2] < 9000  # by internal termination
    >>> # es.logger.plot()  # will plot data
    >>> # cma.s.figshow()  # display plot window

    An example with user-defined transformation, in this case to realize
    a lower bound of 2.

    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as warns:
    ...     es = cma.CMAEvolutionStrategy(5 * [3], 0.1,
    ...                 {"transformation": [lambda x: x**2+1.2, None],
    ...                  "verbose": -2,})
    >>> warns[0].message  # doctest:+ELLIPSIS
    UserWarning('in class GenoPheno: user defined transformations have not been tested thoroughly ()'...
    >>> warns[1].message  # doctest:+ELLIPSIS
    UserWarning('computed initial point...
    >>> es.optimize(cma.ff.rosen, verb_disp=0)  #doctest: +ELLIPSIS
    <cma...
    >>> assert cma.ff.rosen(es.result[0]) < 1e-7 + 5.54781521192
    >>> assert es.result[2] < 3300

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
    ...     es = cma.CMAEvolutionStrategy('6 - 8 * np.random.rand(9)',  # 9-D
    ...                                   5,  # initial std sigma0
    ...                                   {'popsize': lam,  # options
    ...                                    'verb_append': bestever.evalsall})
    ...     logger = cma.CMADataLogger().register(es, append=bestever.evalsall)
    ...     while not es.stop():
    ...         X = es.ask()    # get list of new solutions
    ...         fit = [cma.ff.rastrigin(x) for x in X]  # evaluate each solution
    ...         es.tell(X, fit) # besides for termination only the ranking in fit is used
    ...
    ...         # display some output
    ...         logger.add()  # add a "data point" to the log, writing in files
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
    >>> assert es.result[1] < 1e-8

    On the Rastrigin function, usually after five restarts the global
    optimum is located.

    Using the `multiprocessing` module, we can evaluate the function in
    parallel with a simple modification of the example (however
    multiprocessing seems not always reliable):

    >>> from cma.fitness_functions import elli  # cannot be an instance method
    >>> from cma.fitness_transformations import EvalParallel
    >>> es = cma.CMAEvolutionStrategy(22 * [0.0], 1.0, {'maxiter':10})  # doctest:+ELLIPSIS
    (6_w,13)-aCMA-ES (mu_w=...
    >>> with EvalParallel(es.popsize + 1) as eval_all:
    ...     while not es.stop():
    ...         X = es.ask()
    ...         es.tell(X, eval_all(elli, X))
    ...         es.disp()
    ...         # es.logger.add()  # doctest:+ELLIPSIS
    Iterat...

    The final example shows how to resume:

    >>> import pickle
    >>>
    >>> es = cma.CMAEvolutionStrategy(12 * [0.1],  # a new instance, 12-D
    ...                               0.12)         # initial std sigma0
    ...   #doctest: +ELLIPSIS
    (5_w,...
    >>> es.optimize(cma.ff.rosen, iterations=100)  #doctest: +ELLIPSIS
    I...
    >>> pickle.dump(es, open('_saved-cma-object.pkl', 'wb'))
    >>> del es  # let's start fresh
    >>>
    >>> es = pickle.load(open('_saved-cma-object.pkl', 'rb'))
    >>> # resuming
    >>> es.optimize(cma.ff.rosen, verb_disp=200)  #doctest: +ELLIPSIS
      200 ...
    >>> assert es.result[2] < 15000
    >>> assert cma.s.Mh.vequals_approximately(es.result[0], 12 * [1], 1e-5)
    >>> assert len(es.result) == 8

    Details
    =======
    The following two enhancements are implemented, the latter is turned
    on by default for very small population size only.

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
    In selective mirroring, only the worst solutions are  mirrored. With
    the default small number of mirrors, *pairwise selection* (where at
    most one of the two mirrors contribute to the update of the
    distribution mean) is implicitly guarantied under selective
    mirroring and therefore not explicitly implemented.

    References: Brockhoff et al, PPSN 2010, Auger et al, GECCO 2011.

    :See also: `fmin` (), `OOOptimizer`, `CMAOptions`, `plot` (), `ask` (),
        `tell` (), `ask_and_eval` ()

    """
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

    def stop(self, check=True, ignore_list=()):
        """return the termination status as dictionary.

        With ``check==False``, the termination conditions are not checked
        and the status might not reflect the current situation.
        ``stop().clear()`` removes the currently active termination
        conditions.

        As a convenience feature, keywords in `ignore_list` are removed from
        the conditions.

        """
        if (check and self.countiter > 0 and self.opts['termination_callback'] and
                self.opts['termination_callback'] != str(self.opts['termination_callback'])):
            self.callbackstop = self.opts['termination_callback'](self)

        res = self._stopdict(self, check)  # update the stopdict and return a Dict (self)
        if ignore_list:
            for key in ignore_list:
                res.pop(key, None)
        return res

    def __init__(self, x0, sigma0, inopts=None):
        """see class `CMAEvolutionStrategy`

        """
        self.inputargs = dict(locals())  # for the record
        del self.inputargs['self']  # otherwise the instance self has a cyclic reference
        if inopts is None:
            inopts = {}
        self.inopts = inopts
        opts = CMAOptions(inopts).complement()  # CMAOptions() == fmin([],[]) == defaultOptions()
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
        # TODO: no real need here (do rather in fmin)
            self.sigma0 = eval(sigma0)  # like '1./N' or 'np.random.rand(1)[0]+1e-2'
        if np.size(self.sigma0) != 1 or np.shape(self.sigma0):
            raise ValueError('input argument sigma0 must be (or evaluate to) a scalar')
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
        self.randn = opts['randn']
        if not utils.is_nan(opts['seed']):
            if self.randn is np.random.randn:
                if not opts['seed'] or opts['seed'] is time:
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

        self.boundary_handler = opts['BoundaryHandler']
        if isinstance(self.boundary_handler, type):
            self.boundary_handler = self.boundary_handler(opts['bounds'])
        elif opts['bounds'] not in (None, False, [], [None, None]):
            utils.print_warning("""
                Option 'bounds' ignored because a BoundaryHandler *instance* was found.
                Consider to pass only the desired BoundaryHandler class. """,
                                CMAEvolutionStrategy.__init__)
        if not self.boundary_handler.has_bounds():
            self.boundary_handler = BoundNone()  # just a little faster and well defined
        elif not self.boundary_handler.is_in_bounds(self.x0):
            if opts['verbose'] >= 0:
                utils.print_warning("""
            Initial solution is out of the domain boundaries:
                x0   = %s
                ldom = %s
                udom = %s
            THIS MIGHT LEAD TO AN EXCEPTION RAISED LATER ON.
            """ % (str(self.gp.pheno(self.x0)),
                    str(self.boundary_handler.bounds[0]),
                    str(self.boundary_handler.bounds[1])),
                               '__init__', 'CMAEvolutionStrategy')

        # set self.mean to geno(x0)
        tf_geno_backup = self.gp.tf_geno
        if self.gp.tf_pheno and self.gp.tf_geno is None:
            self.gp.tf_geno = lambda x: x  # a hack to avoid an exception
            utils.print_warning(
                "computed initial point may well be wrong, because no\n"
                "inverse for the user provided phenotype transformation "
                "was given")
        self.mean = self.gp.geno(np.array(self.x0, copy=True),
                            from_bounds=self.boundary_handler.inverse,
                            copy=False)
        self.mean0 = array(self.mean, copy=True)  # relevant for initial injection
        self.gp.tf_geno = tf_geno_backup
        # without copy_always interface:
        # self.mean = self.gp.geno(array(self.x0, copy=True), copy_if_changed=False)
        self.N = len(self.mean)
        assert N == self.N
        # self.fmean = np.NaN  # TODO name should change? prints nan in output files (OK with matlab&octave)
        # self.fmean_noise_free = 0.  # for output only

        self.sp = _CMAParameters(N, opts, verbose=opts['verbose'] > 0)
        self.sp0 = self.sp  # looks useless, as it is not a copy

        self.adapt_sigma = opts['AdaptSigma']
        if self.adapt_sigma is None:
            utils.print_warning("""Value `None` for option 'AdaptSigma' is
    ambiguous and hence depreciated. AdaptSigma can be set to `True` or
    `False` or a class or class instance which inherited from
    cma.sigma_adaptation.CMAAdaptSigmaBase""")
            self.adapt_sigma = CMAAdaptSigmaCSA
        elif self.adapt_sigma is True:
            if opts['CMA_diagonal'] is True and N > 299:
                self.adapt_sigma = CMAAdaptSigmaTPA
            else:
                self.adapt_sigma = CMAAdaptSigmaCSA
        elif self.adapt_sigma is False:
            self.adapt_sigma = CMAAdaptSigmaNone()
        if isinstance(self.adapt_sigma, type):  # Is a class?
            # Then we want the instance.
            self.adapt_sigma = self.adapt_sigma(dimension=N, popsize=self.sp.popsize)
        self.mean_shift_samples = True if (isinstance(self.adapt_sigma, CMAAdaptSigmaTPA) or
            opts['mean_shift_line_samples']) else False

        def eval_vector(in_, opts, N, default_value=1.0):
            """return `default_value` as scalar or `in_` after removing
            fixed variables if ``len(in_) == N``
            """
            res = default_value
            if in_ is not None:
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
                else:
                    res = array(in_, dtype=float)
                if np.size(res) not in (1, N):
                    raise ValueError(
                        "CMA_stds option must have dimension %d "
                        "instead of %d" % (N, np.size(res)))
            return res

        opts['minstd'] = eval_vector(opts['minstd'], opts, N, 0)
        opts['maxstd'] = eval_vector(opts['maxstd'], opts, N, np.inf)

        # iiinteger handling, currently very basic:
        # CAVEAT: integer indices may give unexpected results if fixed_variables is used
        if len(opts['integer_variables']) and opts['fixed_variables']:
            utils.print_warning(
                "CAVEAT: fixed_variables change the meaning of "
                "integer_variables indices")
        # 1) prepare minstd to be a vector
        if (len(opts['integer_variables']) and
                np.isscalar(opts['minstd'])):
            opts['minstd'] = N * [opts['minstd']]
        # 2) set minstd to 1 / (2 Nint + 1),
        #    the setting 2 / (2 Nint + 1) already prevents convergence
        for i in opts['integer_variables']:
            if -N <= i < N:
                opts['minstd'][i] = max((opts['minstd'][i],
                    1 / (2 * len(opts['integer_variables']) + 1)))
            else:
                utils.print_warning(
                    """integer index %d not in range of dimension %d""" %
                        (i, N))

        # initialization of state variables
        self.countiter = 0
        self.countevals = max((0, opts['verb_append'])) \
            if not isinstance(opts['verb_append'], bool) else 0
        self.pc = np.zeros(N)
        self.pc_neg = np.zeros(N)
        if 1 < 3:  # new version with class
            self.sigma_vec0 = eval_vector(self.opts['CMA_stds'], opts, N)
            self.sigma_vec = transformations.DiagonalDecoding(self.sigma_vec0)
            if np.isfinite(self.opts['CMA_dampsvec_fac']):
                self.sigma_vec *= np.ones(N)  # make sure to get a vector
        else:
            self.sigma_vec = eval_vector(self.opts['CMA_stds'], opts, N)
            if np.isfinite(self.opts['CMA_dampsvec_fac']):
                self.sigma_vec *= np.ones(N)  # make sure to get a vector
            self.sigma_vec0 = self.sigma_vec if np.isscalar(self.sigma_vec) \
                                            else self.sigma_vec.copy()
        stds = eval_vector(self.opts['CMA_teststds'], opts, N)
        if self.opts['CMA_diagonal']:  # is True or > 0
            # linear time and space complexity
            self.sigma_vec = transformations.DiagonalDecoding(stds * np.ones(N))
            self.sm = sampler.GaussStandardConstant(N, randn=self.opts['randn'])
            self._updateBDfromSM(self.sm)
            if self.opts['CMA_diagonal'] is True:
                self.sp.weights.finalize_negative_weights(N,
                                                      self.sp.c1_sep,
                                                      self.sp.cmu_sep,
                                                      pos_def=False)
            else:  # would ideally be done when switching
                self.sp.weights.finalize_negative_weights(N,
                                                      self.sp.c1,
                                                      self.sp.cmu)
            if self.opts['CMA_diagonal'] is 1:
                raise ValueError("""Option 'CMA_diagonal' == 1 is disallowed.
                Use either `True` or an iteration number > 1 up to which C should be diagonal.
                Only `True` has linear memory demand.""")
        else:
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
        self.dC = self.sm.variances
        self.D = self.dC**0.5  # we assume that the initial C is diagonal
        self.pop_injection_solutions = []
        self.pop_injection_directions = []
        self.number_of_solutions_asked = 0
        self.number_of_injections_delivered = 0  # used/delivered in asked

        # self.gp.pheno adds fixed variables
        relative_stds = ((self.gp.pheno(self.mean + self.sigma * self.sigma_vec * self.D)
                          - self.gp.pheno(self.mean - self.sigma * self.sigma_vec * self.D)) / 2.0
                         / (self.boundary_handler.get_bounds('upper', self.N_pheno)
                            - self.boundary_handler.get_bounds('lower', self.N_pheno)))
        if np.any(relative_stds > 1):
            idx = np.nonzero(relative_stds > 1)[0]
            s = ("Initial standard deviation "
                 "%s larger than the bounded domain size in variable %s.\n"
                 "Consider using option 'CMA_stds', if the bounded "
                 "domain sizes differ significantly. "
                 % (("s (sigma0*stds) are", str(idx))
                    if len(idx) > 1 else (" (sigma0*stds) is",
                                          str(idx[0]))))
            raise ValueError(s)
        self._flgtelldone = True
        self.itereigenupdated = self.countiter
        self.count_eigen = 0
        self.noiseS = 0  # noise "signal"
        self.hsiglist = []

        self.sent_solutions = _CMASolutionDict()
        self.archive = _CMASolutionDict()
        self.best = ot.BestSolution()

        self.const = _BlancClass()
        self.const.chiN = N**0.5 * (1 - 1. / (4.*N) + 1. / (21.*N**2))  # expectation of norm(randn(N,1))

        self.logger = CMADataLogger(opts['verb_filenameprefix'],
                                                     modulo=opts['verb_log']).register(self)

        # attribute for stopping criteria in function stop
        self._stopdict = _CMAStopDict()
        self.callbackstop = 0

        self.fit = _BlancClass()
        self.fit.fit = []  # not really necessary
        self.fit.hist = []  # short history of best
        self.fit.histbest = []  # long history of best
        self.fit.histmedian = []  # long history of median

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

        Input `x0` may be a `callable` or a string (deprecated) or a
        `list` or `numpy.ndarray` of the desired length.

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
                x0 = eval(x0)
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
            raise ValueError('optimization in 1-D is not supported (code was never tested)')
        try:
            self.x0.resize(self.x0.shape[0])  # 1-D array, not really necessary?!
        except NotImplementedError:
            pass
    # ____________________________________________________________
    # ____________________________________________________________
    def ask(self, number=None, xmean=None, sigma_fac=1,
            gradf=None, args=()):
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
                utils.print_warning("""Gradient injection may fail,
    because sampler attributes `B` and `D` are not present""",
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
                        fpenalty = lambda x: self.boundary_handler.__call__(
                            x, SolutionDict({tuple(x): {'geno': x}}), self.gp)
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
                    gradgp = grad_numerical_of_coordinate_map(
                        xmean,
                        lambda x: self.gp.pheno(x, copy=True,
                                into_bounds=boundary_repair))
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
                utils.print_warning("""Gradient injection failed
    presumably due to missing attribute ``self.sm.B or self.sm.D``""")


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
                arinj[1] *= (sum(arinj[0]**2) / sum(arinj[1]**2))**0.5
                if not Mh.vequals_approximately(arinj[0], -arinj[1]):
                    utils.print_warning(
                        "mean_shift_samples, but the first two solutions"
                        " are not mirrors.",
                        "ask_geno", "CMAEvolutionStrategy",
                        self.countiter)
                    arinj[1] /= sum(arinj[0]**2)**0.5 / s1  # revert change
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

        if self.opts['verbose'] > 4 and self.countiter < 3 and len(arinj) and self.adapt_sigma is not CMAAdaptSigmaTPA:
            utils.print_message('   %d pre-injected solutions will be used (popsize=%d)' %
                                (len(arinj), len(ary)))

        pop = xmean + sigma * ary
        self.evaluations_per_f_value = 1
        self.ary = ary
        self.number_of_solutions_asked += len(pop)
        return pop

    def random_rescale_to_mahalanobis(self, x):
        """change `x` like for injection, all on genotypic level"""
        x = x - self.mean  # -= fails if dtypes don't agree
        if any(x):  # let's not divide by zero
            x *= sum(self.randn(1, len(x))[0]**2)**0.5 / self.mahalanobis_norm(x)
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
        return sum(self.randn(1, len(y))[0]**2)**0.5 / self.mahalanobis_norm(y)


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
            # dx *= sum(self.randn(1, self.N)[0]**2)**0.5 / self.mahalanobis_norm(dx)
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
        sampled. By default ``self.is_feasible == cma.feasible == lambda x, f: f not in (None, np.NaN)``.
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
        fit = []  # or np.NaN * np.empty(number)
        X_first = self.ask(popsize, xmean=xmean, gradf=gradf, args=args)
        if xmean is None:
            xmean = self.mean  # might have changed in self.ask
        X = []
        if parallel_mode:
            fit_first = func(X_first, *args)
            # the rest is only book keeping and warnings spitting
            if hasattr(func, 'last_evaluations'):
                self.countevals += func.last_evaluations - self.popsize
            elif hasattr(func, 'evaluations'):
                if self.countevals < func.evaluations:
                    self.countevals = func.evaluations - self.popsize
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
        if any(f is None or np.isnan(f) for f in fit):
            idxs = [i for i in range(len(fit))
                    if fit[i] is None or np.isnan(fit[i])]
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
            if np.alltrue(ary[-1] == 0.0):
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
                        utils.print_warning('''function_values is not a list of scalars,
                        the first element equals %s with non-scalar type %s.
                        Using now ``[v[0] for v in function_values]`` instead (further warnings are suppressed)'''
                                            % (str(function_values[0]), str(type(function_values[0]))))
                    function_values = [val[0] for val in function_values]
                else:
                    raise ValueError('objective function values must be a list of scalars')
            except:
                utils.print_message("function values=%s" % function_values,
                                    method_name='tell', class_name='CMAEvolutionStrategy',
                                    verbose=9, iteration=self.countiter)
                raise
        if any(f is None or np.isnan(f) for f in function_values):
            idx_none = [i for i, f in enumerate(function_values) if f is None]
            idx_nan = [i for i, f in enumerate(function_values) if f is not None and np.isnan(f)]
            m = np.median([f for f in function_values
                           if f is not None and not np.isnan(f)])
            utils.print_warning("function values with index %s/%s are nan/None and will be set to the median value %s"
                                % (str(idx_nan), str(idx_none), str(m)), 'ask',
                                'CMAEvolutionStrategy', self.countiter)
            for i in idx_nan + idx_none:
                function_values[i] = m
        if not np.isfinite(function_values).all():
            idx = [i for i, f in enumerate(function_values)
                   if not np.isfinite(f)]
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
        if lam < sp.weights.mu:  # rather decrease cmean instead of having mu > lambda//2
            raise ValueError('not enough solutions passed to function tell (mu>lambda)')

        self.countiter += 1  # >= 1 now
        self.countevals += sp.popsize * self.evaluations_per_f_value
        self.best.update(solutions, self.sent_solutions, function_values, self.countevals)

        flg_diagonal = self.opts['CMA_diagonal'] is True \
                       or self.countiter <= self.opts['CMA_diagonal']
        if not flg_diagonal and isinstance(self.sm, sampler.GaussStandardConstant):
            self.sm = sampler.GaussFullSampler(N)
            self._updateBDfromSM(self.sm)

        # ## manage fitness
        fit = self.fit  # make short cut

        # CPU for N,lam=20,200: this takes 10s vs 7s
        fit.bndpen = self.boundary_handler.update(function_values, self)(solutions, self.sent_solutions, self.gp)
        # for testing:
        # fit.bndpen = self.boundary_handler.update(function_values, self)([s.unrepaired for s in solutions])
        fit.idx = np.argsort(array(fit.bndpen) + array(function_values))
        fit.fit = array(function_values, copy=False)[fit.idx]

        # update output data TODO: this is obsolete!? However: need communicate current best x-value?
        # old: out['recent_x'] = self.gp.pheno(pop[0])
        # self.out['recent_x'] = array(solutions[fit.idx[0]])  # TODO: change in a data structure(?) and use current as identify
        # self.out['recent_f'] = fit.fit[0]

        # fitness histories
        fit.hist.insert(0, fit.fit[0])
        # if len(self.fit.histbest) < 120+30*N/sp.popsize or  # does not help, as tablet in the beginning is the critical counter-case
        if ((self.countiter % 5) == 0):  # 20 percent of 1e5 gen.
            fit.histbest.insert(0, fit.fit[0])
            fit.histmedian.insert(0, fit.fit[self.popsize // 2] if self.popsize % 2
                                     else np.mean(fit.fit[self.popsize // 2 - 1: self.popsize // 2 + 1]))
        if len(fit.histbest) > 2e4:  # 10 + 30*N/sp.popsize:
            fit.histbest.pop()
            fit.histmedian.pop()
        if len(fit.hist) > 10 + 30 * N / sp.popsize:
            fit.hist.pop()

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
            else:  # to be removed:
                x = self.sent_solutions.pop(s, None)  # 12.7s vs 11.3s with N,lambda=20,200
                if x is not None:
                    pop.append(x['geno'])
                    if x['iteration'] + 1 < self.countiter and check_points not in (False, 0, [], ()):
                        self.repair_genotyp(pop[-1])
                    # TODO: keep additional infos or don't pop s from sent_solutions in the first place
                else:
                    # this case is expected for injected solutions
                    pop.append(self.gp.geno(s, self.boundary_handler.inverse, copy=copy))  # cannot recover the original genotype with boundary handling
                    if check_points not in (False, 0, [], ()):
                        self.repair_genotype(pop[-1])  # necessary if pop[-1] was changed or injected by the user.
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
        if check_points not in (None, False, 0, [], ()):  # useful in case of injected solutions and/or adaptive encoding, however is automatic with use_sent_solutions
            try:
                if len(check_points):
                    idx = check_points
            except:
                idx = range(sp.popsize)

            for k in idx:
                self.repair_genotype(pop[k])

        # sort pop for practicability, now pop != self.pop, which is unsorted
        pop = np.asarray(pop)[fit.idx]

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
                pop = array([x_elit] + list(pop), copy=False)
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
        # compute new mean
        self.mean = mold + self.sp.cmean * \
                    (np.sum(np.asarray(sp.weights.positive_weights) * pop[0:sp.weights.mu].T, 1) - mold)


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
        # adjust missing variance due to hsig, in 4-D with damps=1e99 and sig0 small
        #       hsig leads to premature convergence of C otherwise
        # hsiga = (1-hsig**2) * c1 * cc * (2-cc)  # to be removed in future
        c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))  # adjust for variance loss

        if 11 < 3:  # diagnostic data
            # self.out['hsigcount'] += 1 - hsig
            if not hsig:
                self.hsiglist.append(self.countiter)
        if 11 < 3:  # diagnostic message
            if not hsig:
                print(str(self.countiter) + ': hsig-stall')
        if 11 < 3:  # for testing purpose
            hsig = 1  # TODO:
            #       put correction term, but how?
            if self.countiter == 1:
                print('hsig=1')

        self.pc = (1 - cc) * self.pc + hsig * (
                    (cc * (2 - cc) * self.sp.weights.mueff)**0.5 / self.sigma
                        / cmean) * (self.mean - mold) / self.sigma_vec.scaling

        # covariance matrix adaptation/udpate
        pop_zero = pop - mold
        if c1a + cmu > 0:
            # TODO: make sure cc is 1 / N**0.5 rather than 1 / N
            # TODO: simplify code: split the c1 and cmu update and call self.sm.update twice
            #       caveat: for this the decay factor ``c1_times_delta_hsigma - sum(weights)`` should be zero in the second update
            sampler_weights = [c1a] + [cmu * w for w in sp.weights]
            if len(pop_zero) > len(sp.weights):
                sampler_weights = (
                        sampler_weights[:1+sp.weights.mu] +
                        (len(pop_zero) - len(sp.weights)) * [0] +
                        sampler_weights[1+sp.weights.mu:])
            if 'inc_cmu_pos' in self.opts['vv']:
                sampler_weights = np.asarray(sampler_weights)
                sampler_weights[sampler_weights > 0] *= 1 + self.opts['vv']['inc_cmu_pos']
            # logger = logging.getLogger(__name__)  # "global" level needs to be DEBUG
            # logger.debug("w[0,1]=%f,%f", sampler_weights[0],
            #               sampler_weights[1]) if self.countiter < 2 else None
            if flg_diagonal:
                self.sigma_vec.update(
                    [self.sm.transform_inverse(self.pc)] +
                    list(self.sm.transform_inverse(pop_zero /
                                        (self.sigma * self.sigma_vec.scaling))),
                    array(sampler_weights) / 2)  # TODO: put the 1/2 into update function!?
            else:
                self.sm.update([(c1 / (c1a + 1e-23))**0.5 * self.pc] +  # c1a * pc**2 gets c1 * pc**2
                              list(pop_zero / (self.sigma * self.sigma_vec.scaling)),
                              sampler_weights)
            if any(np.asarray(self.sm.variances) < 0):
                raise RuntimeError("""A sampler variance has become
    negative after update, this must be considered as a bug.
    Variances `self.sm.variances`=%s""" % str(self.sm.variances))
        self._updateBDfromSM(self.sm)

        # step-size adaptation, adapt sigma
        # in case of TPA, function_values[0] and [1] must reflect samples colinear to xmean - xmean_old
        try:
            self.sigma *= self.adapt_sigma.update2(self,
                                        function_values=function_values)
        except (NotImplementedError, AttributeError):
            self.adapt_sigma.update(self, function_values=function_values)

        if 11 < 3 and self.opts['vv']:
            if self.countiter < 2:
                print('constant sigma applied')
                print(self.opts['vv'])  # N=10,lam=10: 0.8 is optimal
            self.sigma = self.opts['vv'] * self.sp.weights.mueff * sum(self.mean**2)**0.5 / N

        if any(self.sigma * self.sigma_vec.scaling * self.dC**0.5 <
                       np.asarray(self.opts['minstd'])):
            self.sigma = max(np.asarray(self.opts['minstd']) /
                                (self.sigma_vec * self.dC**0.5))
            assert all(self.sigma * self.sigma_vec * self.dC**0.5 >=
                       (1-1e-9) * np.asarray(self.opts['minstd']))
        elif any(self.sigma * self.sigma_vec.scaling * self.dC**0.5 >
                       np.asarray(self.opts['maxstd'])):
            self.sigma = min(np.asarray(self.opts['maxstd']) /
                             self.sigma_vec * self.dC**0.5)
        # g = self.countiter
        # N = self.N
        # mindx = eval(self.opts['mindx'])
        #  if utils.is_str(self.opts['mindx']) else self.opts['mindx']
        if self.sigma * min(self.D) < self.opts['mindx']:  # TODO: sigma_vec is missing here
            self.sigma = self.opts['mindx'] / min(self.D)

        if self.sigma > 1e9 * self.sigma0:
            alpha = self.sigma / max(self.sm.variances)**0.5
            if alpha > 1:
                self.sigma /= alpha**0.5  # adjust only half
                self.opts['tolupsigma'] /= alpha**0.5  # to be compared with sigma
                self.sm *= alpha
                self._updateBDfromSM()

        # TODO increase sigma in case of a plateau?

        # Uncertainty noise measurement is done on an upper level

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
            if self.opts['verbose'] > 4 and self.countiter < 3 and type(self.adapt_sigma) is not CMAAdaptSigmaTPA and len(self.pop_injection_directions):
                utils.print_message('   %d directions prepared for injection %s' %
                                    (len(self.pop_injection_directions),
                                     "(no more messages will be shown)" if
                                     self.countiter == 2 else ""))
            self.number_of_injections_delivered = 0
        self.pop = []  # remove this in case pop is still needed
        # self.pop_sorted = []
        self._flgtelldone = True
        try:  # shouldn't fail, but let's be nice to code abuse
            self.timer.pause()
        except AttributeError:
            utils.print_warning("""
    "timer" attribute not found, probably because `ask` was never called.
     Timing is likely to work only until `tell` is called (again), because
     `tic` will never be called again afterwards.
     """,
                                'tell', 'CMAEvolutionStrategy',
                                self.countiter)
            self.timer = utils.ElapsedWCTime()

        self.more_to_write.check()
    # end tell()

    def inject(self, solutions, force=None):
        """inject list of one or several genotypic solution(s).

        Unless `force is True`, the solutions are used as direction
        relative to the distribution mean to compute a new candidate
        solution returned in method `ask_geno` which in turn is used in
        method `ask`. `inject` is to be called before `ask` or after
        `tell` and can be called repeatedly.

        >>> import cma
        >>> es = cma.CMAEvolutionStrategy(4 * [1], 2)  #doctest: +ELLIPSIS
        (4_w,...
        >>> while not es.stop():
        ...     es.inject([4 * [0.0]])
        ...     X = es.ask()
        ...     if es.countiter == 0:
        ...         assert X[0][0] == X[0][1]  # injected sol. is on the diagonal
        ...     es.tell(X, [cma.ff.sphere(x) for x in X])

        """
        for solution in solutions:
            if solution is None:
                continue
            if len(solution) != self.N:
                raise ValueError('method `inject` needs a list or array'
                    + (' each el with dimension (`len`) %d' % self.N))
            solution = array(solution, copy=False, dtype=float)
            if force:
                self.pop_injection_solutions.append(solution)
            else:
                self.pop_injection_directions.append(solution - self.mean)

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
        res = self.best.get() + (  # (x, f, evals) triple
            self.countevals,
            self.countiter,
            self.gp.pheno(self.mean),
            self.gp.scales * self.sigma * self.sigma_vec.scaling *
                self.dC**0.5,
            self.stop())
        try:
            return CMAEvolutionStrategyResult(*res)
        except NameError:
            return res

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

        print('final/bestever f-value = %e %e' % (self.best.last.f,
                                                  fbestever))
        if self.N < 9:
            print('incumbent solution: ' + str(list(self.gp.pheno(self.mean, into_bounds=self.boundary_handler.repair))))
            print('std deviation: ' + str(list(self.sigma * self.sigma_vec.scaling * np.sqrt(self.dC) * self.gp.scales)))
        else:
            print('incumbent solution: %s ...]' % (str(self.gp.pheno(self.mean, into_bounds=self.boundary_handler.repair)[:8])[:-1]))
            print('std deviations: %s ...]' % (str((self.sigma * self.sigma_vec.scaling * np.sqrt(self.dC) * self.gp.scales)[:8])[:-1]))
        return self.result

    def repair_genotype(self, x, copy_if_changed=False):
        """make sure that solutions fit to the sample distribution.

        This interface is versatile and likely to change.

        In particular the frequency of ``x - self.mean`` being long in
        Mahalanobis distance is limited, currently clipping at
        ``N**0.5 + 2 * N / (N + 2)`` is implemented.
        """
        x = array(x, copy=False)
        mold = array(self.mean, copy=False)
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
        else:
            if 'checktail' not in self.__dict__:  # hasattr(self, 'checktail')
                raise NotImplementedError
                # from check_tail_smooth import CheckTail  # for the time being
                # self.checktail = CheckTail()
                # print('untested feature checktail is on')
            fac = self.checktail.addchin(self.mahalanobis_norm(x - mold))

            if fac < 1:
                x = fac * (x - mold) + mold

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
        >>> es = cma.CMAEvolutionStrategy(4 * [0], 1)  #doctest: +ELLIPSIS
        (4_w,...
        >>> while not es.stop():
        ...     X = es.ask()
        ...     es.tell(X, f(X))
        ...     es.logger.add()
        ...     es.manage_plateaus()
        >>> assert es.sigma > 1.5**5

        """
        if not self._flgtelldone:
            utils.print_warning("Inbetween `ask` and `tell` plateaus cannot" +
            " be managed, because `sigma` should not change.",
                           "manage_plateaus", "CMAEvolutionStrategy",
                                self.countiter)
            return
        idx = Mh.sround(sample_fraction * (self.popsize - 1))
        if self.fit.fit[0] == self.fit.fit[idx]:
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
        if condition and np.isfinite(condition) and max(self.dC) / min(self.dC) > condition:
            # allows for much larger condition numbers, if axis-parallel
            if hasattr(self, 'sm') and isinstance(self.sm, sampler.GaussFullSampler):
                old_coordinate_condition = max(self.dC) / min(self.dC)
                old_condition = self.sm.condition_number
                factors = self.sm.to_correlation_matrix()
                self.sigma_vec *= factors
                self.pc /= factors
                self._updateBDfromSM(self.sm)
                utils.print_message('\ncondition in coordinate system exceeded'
                                    ' %.1e, rescaled to %.1e, '
                                    '\ncondition changed from %.1e to %.1e'
                                      % (old_coordinate_condition, max(self.dC) / min(self.dC),
                                         old_condition, self.sm.condition_number),
                                    iteration=self.countiter)

    def alleviate_conditioning(self, condition=1e12):
        """pass conditioning of `C` to linear transformation in `self.gp`.

        Argument `condition` defines the limit condition number above
        which the action is taken.

        Details: the action applies only if `self.gp.isidentity`. Then,
        the covariance matrix `C` is set (back) to identity and a
        respective linear transformation is "added" to `self.gp`.
        """
        # new interface: if sm.condition_number > condition ...
        if not self.gp.isidentity or not condition or self.condition_number < condition:
            return
        try:
            old_condition_number = self.condition_number
            tf_inv = self.sm.to_linear_transformation_inverse()
            tf = self.sm.to_linear_transformation(reset=True)
            self.pc = np.dot(tf_inv, self.pc)
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
        self.sigma_vec = transformations.DiagonalDecoding(1)

        # TODO: refactor old_scales * old_sigma_vec into sigma_vec0 to prevent tolfacupx stopping

        self.gp.tf_pheno = lambda x: np.dot(self.gp._tf_matrix, x)
        self.gp.tf_geno = lambda x: np.dot(self.gp._tf_matrix_inv, x)  # not really necessary
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
        for i in rglen((X) / popsize):
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
        z = self.sm.transform_inverse((self.mean - self.mean_old) /
                                      self.sigma_vec.scaling)
        # TODO:
        # works unless a re-parametrisation has been done, and otherwise?
        # assert Mh.vequals_approximately(z, np.dot(es.B, (1. / es.D) *
        #         np.dot(es.B.T, (es.mean - es.mean_old) / es.sigma_vec)))
        z /= self.sigma * self.sp.cmean
        z *= self.sp.weights.mueff**0.5
        return z

    def disp_annotation(self):
        """print annotation line for `disp` ()"""
        print('Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]')
        sys.stdout.flush()

    def disp(self, modulo=None):
        """print current state variables in a single-line.

        Prints only if ``iteration_counter % modulo == 0``.

        :See also: `disp_annotation`.
        """
        if modulo is None:
            modulo = self.opts['verb_disp']

        # console display

        if modulo:
            if (self.countiter - 1) % (10 * modulo) < 1:
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
                                stime)))
                # if self.countiter < 4:
                sys.stdout.flush()
        return self
    def plot(self):
        """plot current state variables using `matplotlib`.

        Details: calls `self.logger.plot`.
        """
        try:
            self.logger.plot()
        except AttributeError:
            utils.print_warning('plotting failed, no logger attribute found')
        except:
            utils.print_warning(('plotting failed with:', sys.exc_info()[0]),
                           'plot', 'CMAEvolutionStrategy')
        return self

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
        self._stoplist = []  # to keep multiple entries
        self.lastiter = 0  # probably not necessary
        try:
            self._stoplist = d._stoplist  # multiple entries
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

        self.clear()  # compute conditions from scratch

        N = es.N
        opts = es.opts
        self.opts = opts  # a hack to get _addstop going

        # check user signals
        try:
            # adds about 40% time in 5-D, 15% if file is not present
            # simple resolution: set signals_filename to None or ''
            if 1 < 3 and self.opts['signals_filename']:
                with open(self.opts['signals_filename'], 'r') as f:
                    s = f.read()
                d = dict(ast.literal_eval(s.strip()))
                for key in list(d):
                    if key not in opts.versatile_options():
                        utils.print_warning(
        """\n        unkown or non-versatile option '%s' found in file %s.
        Check out the #v annotation in ``cma.CMAOptions()``.
        """ % (key, self.opts['signals_filename']))
                        d.pop(key)
                opts.update(d)
                for key in d:
                    opts.eval(key, {'N': N, 'dim': N})
        except IOError:
            pass  # no warning, as signals file doesn't need to be present

        # fitness: generic criterion, user defined w/o default
        self._addstop('ftarget',
                      es.best.f < opts['ftarget'])
        # maxiter, maxfevals: generic criteria
        self._addstop('maxfevals',
                      es.countevals - 1 >= opts['maxfevals'])
        self._addstop('maxiter',
                      ## meta_parameters.maxiter_multiplier == 1.0
                      es.countiter >= 1.0 * opts['maxiter'])
        # tolx, tolfacupx: generic criteria
        # tolfun, tolfunhist (CEC:tolfun includes hist)
        self._addstop('tolx',
                      all([es.sigma * xi < opts['tolx'] for xi in es.sigma_vec * es.pc]) and
                      all([es.sigma * xi < opts['tolx'] for xi in es.sigma_vec * np.sqrt(es.dC)]))
        self._addstop('tolfacupx',
                      any(es.sigma * es.sigma_vec.scaling * es.dC**0.5 >
                          es.sigma0 * es.sigma_vec0 * opts['tolfacupx']))
        self._addstop('tolfun',
                      es.fit.fit[-1] - es.fit.fit[0] < opts['tolfun'] and
                      max(es.fit.hist) - min(es.fit.hist) < opts['tolfun'])
        self._addstop('tolfunhist',
                      len(es.fit.hist) > 9 and
                      max(es.fit.hist) - min(es.fit.hist) < opts['tolfunhist'])

        # worst seen false positive: table N=80,lam=80, getting worse for fevals=35e3 \approx 50 * N**1.5
        # but the median is not so much getting worse
        # / 5 reflects the sparsity of histbest/median
        # / 2 reflects the left and right part to be compared
        ## meta_parameters.tolstagnation_multiplier == 1.0
        l = int(max(( 1.0 * opts['tolstagnation'] / 5. / 2, len(es.fit.histbest) / 10)))
        # TODO: why max(..., len(histbest)/10) ???
        # TODO: the problem in the beginning is only with best ==> ???
        if 11 < 3:  # print for debugging
            print(es.countiter, (opts['tolstagnation'], es.countiter > N * (5 + 100 / es.popsize),
                  len(es.fit.histbest) > 100,
                  np.median(es.fit.histmedian[:l]) >= np.median(es.fit.histmedian[l:2 * l]),
                  np.median(es.fit.histbest[:l]) >= np.median(es.fit.histbest[l:2 * l])))
        # equality should handle flat fitness
        self._addstop('tolstagnation',  # leads sometimes early stop on ftablet, fcigtab, N>=50?
                      1 < 3 and opts['tolstagnation'] and es.countiter > N * (5 + 100 / es.popsize) and
                      len(es.fit.histbest) > 100 and 2 * l < len(es.fit.histbest) and
                      np.median(es.fit.histmedian[:l]) >= np.median(es.fit.histmedian[l:2 * l]) and
                      np.median(es.fit.histbest[:l]) >= np.median(es.fit.histbest[l:2 * l]))
        # iiinteger: stagnation termination can prevent to find the optimum

        self._addstop('tolupsigma', opts['tolupsigma'] and
                      es.sigma / np.max(es.D) > es.sigma0 * opts['tolupsigma'])
        try:
            self._addstop('timeout',
                          es.timer.elapsed > opts['timeout'])
        except AttributeError:
            if es.countiter <= 0: 
                pass 
            # else: raise

        if 11 < 3 and 2 * l < len(es.fit.histbest):  # TODO: this might go wrong, because the nb of written columns changes
            tmp = np.array((-np.median(es.fit.histmedian[:l]) + np.median(es.fit.histmedian[l:2 * l]),
                        - np.median(es.fit.histbest[:l]) + np.median(es.fit.histbest[l:2 * l])))
            es.more_to_write += [(10**t if t < 0 else t + 1) for t in tmp]  # the latter to get monotonicy

        if 1 < 3:
            # non-user defined, method specific
            # noeffectaxis (CEC: 0.1sigma), noeffectcoord (CEC:0.2sigma), conditioncov
            idx = np.nonzero(es.mean == es.mean + 0.2 * es.sigma *
                             es.sigma_vec.scaling * es.dC**0.5)[0]
            self._addstop('noeffectcoord', any(idx), list(idx))
#                         any([es.mean[i] == es.mean[i] + 0.2 * es.sigma *
#                                                         (es.sigma_vec if np.isscalar(es.sigma_vec) else es.sigma_vec[i]) *
#                                                         sqrt(es.dC[i])
#                              for i in range(N)])
#                )
            if opts['CMA_diagonal'] is not True and es.countiter > opts['CMA_diagonal']:
                i = es.countiter % N
                try:
                    self._addstop('noeffectaxis',
                                 sum(es.mean == es.mean + 0.1 * es.sigma *
                                     es.sm.D[i] * es.sigma_vec.scaling *
                                     (es.sm.B[:, i] if len(es.sm.B.shape) > 1 else es.sm.B[0])) == N)
                except AttributeError:
                    pass
            self._addstop('tolconditioncov',
                          opts['tolconditioncov'] and
                          es.D[-1] > opts['tolconditioncov']**0.5 * es.D[0], opts['tolconditioncov'])

            self._addstop('callback', es.callbackstop)  # termination_callback

        if 1 < 3 or len(self): # only if another termination criterion is satisfied
            if 1 < 3:  # warn, in case
                if es.fit.fit[0] == es.fit.fit[-1] == es.best.last.f:
                    utils.print_warning(
                    """flat fitness (f=%f, sigma=%.2e).
                    For small sigma, this could indicate numerical convergence.
                    Otherwise, please (re)consider how to compute the fitness more elaborately.""" %
                    (es.fit.fit[0], es.sigma), iteration=es.countiter)
            if 1 < 3:  # add stop condition, in case
                self._addstop('flat fitness',  # message via stopdict
                         len(es.fit.hist) > 9 and
                         max(es.fit.hist) == min(es.fit.hist) and
                              es.fit.fit[0] == es.fit.fit[-2],
                         "please (re)consider how to compute the fitness more elaborately if sigma=%.2e is large" % es.sigma)
        if 11 < 3 and opts['vv'] == 321:
            self._addstop('||xmean||^2<ftarget', sum(es.mean**2) <= opts['ftarget'])

        return self

    def _addstop(self, key, cond, val=None):
        if cond:
            self.stoplist.append(key)  # can have the same key twice
            self[key] = val if val is not None \
                            else self.opts.get(key, None)

    def clear(self):
        """empty the stopdict"""
        for k in list(self):
            self.pop(k)
        self.stoplist = []

class _CMAParameters(object):
    """strategy parameters like population size and learning rates.

    Note:
        contrary to `CMAOptions`, `_CMAParameters` is not (yet) part of the
        "user-interface" and subject to future changes (it might become
        a `collections.namedtuple`)

    Example
    -------
    >>> import cma
    >>> es = cma.CMAEvolutionStrategy(20 * [0.1], 1)  #doctest: +ELLIPSIS
    (6_w,12)-aCMA-ES (mu_w=3.7,w_1=40%) in dimension 20 (seed=...)
    >>>
    >>> type(es.sp)  # sp contains the strategy parameters
    <class 'cma.evolution_strategy._CMAParameters'>
    >>> es.sp.disp()  #doctest: +ELLIPSIS
    {'CMA_on': True,
     'N': 20,
     'c1': 0.00437235...,
     'c1_sep': 0.0343279...,
     'cc': 0.171767...,
     'cc_sep': 0.252594...,
     'cmean': array(1...,
     'cmu': 0.00921656...,
     'cmu_sep': 0.0565385...,
     'lam_mirr': 0,
     'mu': 6,
     'popsize': 12,
     'weights': [0.4024029428...,
                 0.2533890840...,
                 0.1662215645...,
                 0.1043752252...,
                 0.05640347757...,
                 0.01720770576...,
                 -0.05018713636...,
                 -0.1406167894...,
                 -0.2203813963...,
                 -0.2917332686...,
                 -0.3562788884...,
                 -0.4152044225...]}
    >>>

    :See: `CMAOptions`, `CMAEvolutionStrategy`

    """
    def __init__(self, N, opts, ccovfac=1, verbose=True):
        """Compute strategy parameters, mainly depending on
        dimension and population size, by calling `set`

        """
        self.N = N
        if ccovfac == 1:
            ccovfac = opts['CMA_on']  # that's a hack
        self.popsize = None  # declaring the attribute, not necessary though
        self.set(opts, ccovfac=ccovfac, verbose=verbose)

    def set(self, opts, popsize=None, ccovfac=1, verbose=True):
        """Compute strategy parameters as a function
        of dimension and population size """

        limit_fac_cc = 4.0  # in future: 10**(1 - N**-0.33)?

        def conedf(df, mu, N):
            """used for computing separable learning rate"""
            return 1. / (df + 2. * np.sqrt(df) + float(mu) / N)

        def cmudf(df, mu, alphamu):
            """used for computing separable learning rate"""
            return (alphamu + mu + 1. / mu - 2) / (df + 4 * np.sqrt(df) + mu / 2.)

        sp = self  # mainly for historical reasons
        N = sp.N
        if popsize:
            opts.evalall({'N':N, 'popsize':popsize})
        else:
            popsize = opts.evalall({'N':N})['popsize']  # the default popsize is computed in CMAOptions()
        ## meta_parameters.lambda_exponent == 0.0
        popsize = int(popsize + N** 0.0 - 1)

        # set weights
        sp.weights = RecombinationWeights(popsize)
        if opts['CMA_mu']:
            sp.weights = RecombinationWeights(2 * opts['CMA_mu'])
            while len(sp.weights) < popsize:
                sp.weights.insert(sp.weights.mu, 0.0)
        if utils.is_(opts['CMA_recombination_weights']):
            sp.weights[:] = opts['CMA_recombination_weights']
            sp.weights.set_attributes_from_weights()
            popsize = len(sp.weights)
        # weights.finalize_negative_weights will be called below
        sp.popsize = popsize
        sp.mu = sp.weights.mu  # not used anymore but for the record

        if opts['CMA_mirrors'] < 0.5:
            sp.lam_mirr = int(0.5 + opts['CMA_mirrors'] * popsize)
        elif opts['CMA_mirrors'] > 1:
            sp.lam_mirr = int(0.5 + opts['CMA_mirrors'])
        else:
            sp.lam_mirr = int(0.5 + 0.16 * min((popsize, 2 * N + 2)) + 0.29)  # 0.158650... * popsize is optimal
            # lam = arange(2,22)
            # mirr = 0.16 + 0.29/lam
            # print(lam); print([int(0.5 + l) for l in mirr*lam])
            # [ 2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21]
            # [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4]
        # in principle we have mu_opt = popsize/2 + lam_mirr/2,
        # which means in particular weights should only be negative for q > 0.5+mirr_frac/2
        if sp.popsize // 2 > sp.popsize - 2 * sp.lam_mirr + 1:
            utils.print_warning("pairwise selection is not implemented, therefore " +
                  " mu = %d > %d = %d - 2*%d + 1 = popsize - 2*mirr + 1 can produce a bias" % (
                    sp.popsize // 2, sp.popsize - 2 * sp.lam_mirr + 1, sp.popsize, sp.lam_mirr))
        if sp.lam_mirr > sp.popsize // 2:
            raise ValueError("fraction of mirrors in the population as read from option CMA_mirrors cannot be larger 0.5, " +
                         "theoretically optimal is 0.159")

        mueff = sp.weights.mueff

        # line 3415
        ## meta_parameters.cc_exponent == 1.0
        b = 1.0
        ## meta_parameters.cc_multiplier == 1.0
        sp.cc = 1.0 * (limit_fac_cc + mueff / N)**b / \
                (N**b + (limit_fac_cc + 2 * mueff / N)**b)
        sp.cc_sep = (1 + 1 / N + mueff / N) / \
                    (N**0.5 + 1 / N + 2 * mueff / N)
        if hasattr(opts['vv'], '__getitem__'):
            if 'sweep_ccov1' in opts['vv']:
                sp.cc = 1.0 * (4 + mueff / N)**0.5 / ((N + 4)**0.5 +
                                                    (2 * mueff / N)**0.5)
            if 'sweep_cc' in opts['vv']:
                sp.cc = opts['vv']['sweep_cc']
                sp.cc_sep = sp.cc
                print('cc is %f' % sp.cc)

        ## meta_parameters.c1_multiplier == 1.0
        sp.c1 = (1.0 * opts['CMA_rankone'] * ccovfac * min(1, sp.popsize / 6) *
                 ## meta_parameters.c1_exponent == 2.0
                 2 / ((N + 1.3)** 2.0 + mueff))
                 # 2 / ((N + 1.3)** 1.5 + mueff))  # TODO
                 # 2 / ((N + 1.3)** 1.75 + mueff))  # TODO
        # 1/0
        sp.c1_sep = opts['CMA_rankone'] * ccovfac * conedf(N, mueff, N)
        if 11 < 3:
            sp.c1 = 0.
            print('c1 is zero')
        if utils.is_(opts['CMA_rankmu']):  # also empty
            ## meta_parameters.cmu_multiplier == 2.0
            alphacov = 2.0
            ## meta_parameters.rankmu_offset == 0.25
            rankmu_offset = 0.25
            # the influence of rankmu_offset in [0, 1] on performance is
            # barely visible
            if hasattr(opts['vv'], '__getitem__') and 'sweep_rankmu_offset' in opts['vv']:
                rankmu_offset = opts['vv']['sweep_rankmu_offset']
                print("rankmu_offset = %.2f" % rankmu_offset)
            mu = mueff
            sp.cmu = min(1 - sp.c1,
                         opts['CMA_rankmu'] * ccovfac * alphacov *
                         # simpler nominator would be: (mu - 0.75)
                         (rankmu_offset + mu + 1 / mu - 2) /
                         ## meta_parameters.cmu_exponent == 2.0
                         ((N + 2)** 2.0 + alphacov * mu / 2))
                         # ((N + 2)** 1.5 + alphacov * mu / 2))  # TODO
                         # ((N + 2)** 1.75 + alphacov * mu / 2))  # TODO
                         # cmu -> 1 for mu -> N**2 * (2 / alphacov)
            if hasattr(opts['vv'], '__getitem__') and 'sweep_ccov' in opts['vv']:
                sp.cmu = opts['vv']['sweep_ccov']
            sp.cmu_sep = min(1 - sp.c1_sep, ccovfac * cmudf(N, mueff, rankmu_offset))
        else:
            sp.cmu = sp.cmu_sep = 0
        if hasattr(opts['vv'], '__getitem__') and 'sweep_ccov1' in opts['vv']:
            sp.c1 = opts['vv']['sweep_ccov1']

        if any(w < 0 for w in sp.weights):
            if opts['CMA_active'] and opts['CMA_on'] and opts['CMA_rankmu']:
                sp.weights.finalize_negative_weights(N, sp.c1, sp.cmu)
                # this is re-done using self.sm.parameters()['c1']...
            else:
                sp.weights.zero_negative_weights()

        # line 3834
        sp.CMA_on = sp.c1 + sp.cmu > 0
        # print(sp.c1_sep / sp.cc_sep)

        if not opts['CMA_on'] and opts['CMA_on'] not in (None, [], (), ''):
            sp.CMA_on = False
            # sp.c1 = sp.cmu = sp.c1_sep = sp.cmu_sep = 0
        # line 3480
        if 11 < 3:
            # this is worse than damps = 1 + sp.cs for the (1,10000)-ES on 40D parabolic ridge
            sp.damps = 0.3 + 2 * max([mueff / sp.popsize, ((mueff - 1) / (N + 1))**0.5 - 1]) + sp.cs
        if 11 < 3:
            # this does not work for lambda = 4*N^2 on the parabolic ridge
            sp.damps = opts['CSA_dampfac'] * (2 - 0 * sp.lam_mirr / sp.popsize) * mueff / sp.popsize + 0.3 + sp.cs
            # nicer future setting
            print('damps =', sp.damps)
        if 11 < 3:
            sp.damps = 10 * sp.damps  # 1e99 # (1 + 2*max(0,sqrt((mueff-1)/(N+1))-1)) + sp.cs;
            # sp.damps = 20 # 1. + 20 * sp.cs**-1  # 1e99 # (1 + 2*max(0,sqrt((mueff-1)/(N+1))-1)) + sp.cs;
            print('damps is %f' % (sp.damps))

        sp.cmean = np.asarray(opts['CMA_cmean'], dtype=float)
        # sp.kappa = 1  # 4-D, lam=16, rank1, kappa < 4 does not influence convergence rate
                        # in larger dim it does, 15-D with defaults, kappa=8 factor 2
        if 11 < 3 and np.any(sp.cmean != 1):
            print('  cmean = ' + str(sp.cmean))

        if verbose:
            if not sp.CMA_on:
                print('covariance matrix adaptation turned off')
            if opts['CMA_mu'] != None:
                print('mu = %d' % (sp.weights.mu))

        # return self  # the constructor returns itself

    def disp(self):
        pprint(self.__dict__)


def fmin2(*args, **kwargs):
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
    res = fmin(*args, **kwargs)
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
         callback=None):
    """functional interface to the stochastic optimizer CMA-ES
    for non-convex function minimization.

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
        according to the ``transformation`` option.  It can also be
        a string holding a Python expression that is evaluated
        to yield the initial guess - this is important in case
        restarts are performed so that they start from different
        places.  Otherwise ``x0`` can also be a `cma.CMAEvolutionStrategy`
        object instance, in that case ``sigma0`` can be ``None``.
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
        parameter ``incpopsize``, implementing the IPOP-CMA-ES restart
        strategy, see also parameter ``bipop``; to restart from
        different points (recommended), pass ``x0`` as a string.
    ``restart_from_best=False``
        which point to restart from
    ``incpopsize=2``
        multiplier for increasing the population size ``popsize`` before
        each restart
    ``parallel_objective``
        an objective function that accepts a list of `numpy.ndarray` as
        input and returns a `list`, which is mostly used instead of
        `objective_function`, but for the initial (also initial
        elitist) and the final evaluations. If ``parallel_objective``
        is given, the ``objective_function`` (first argument) may be
        ``None``.
    ``eval_initial_x=None``
        evaluate initial solution, for ``None`` only with elitist option
    ``noise_handler=None``
        a ``NoiseHandler`` class or instance or ``None``. Example:
        ``cma.fmin(f, 6 * [1], 1, noise_handler=cma.NoiseHandler(6))``
        see ``help(cma.NoiseHandler)``.
    ``noise_change_sigma_exponent=1``
        exponent for the sigma increment provided by the noise handler for
        additional noise treatment. 0 means no sigma change.
    ``noise_evaluations_as_kappa=0``
        instead of applying reevaluations, the "number of evaluations"
        is (ab)used as scaling factor kappa (experimental).
    ``bipop=False``
        if `True`, run as BIPOP-CMA-ES; BIPOP is a special restart
        strategy switching between two population sizings - small
        (like the default CMA, but with more focused search) and
        large (progressively increased as in IPOP). This makes the
        algorithm perform well both on functions with many regularly
        or irregularly arranged local optima (the latter by frequently
        restarting with small populations).  For the `bipop` parameter
        to actually take effect, also select non-zero number of
        (IPOP) restarts; the recommended setting is ``restarts<=9``
        and `x0` passed as a string using `numpy.rand` to generate
        initial solutions. Note that small-population restarts
        do not count into the total restart count.
    ``callback=None``
        `callable` or list of callables called at the end of each
        iteration with the current `CMAEvolutionStrategy` instance
        as argument.

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
    >>> res = cma.fmin(cma.ff.rastrigin, '2. * np.random.rand(3) - 1', 0.5,
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

        if callback is None:
            callback = []
        elif callable(callback):
            callback = [callback]

        # BIPOP-related variables:
        runs_with_small = 0
        small_i = []
        large_i = []
        popsize0 = None  # to be evaluated after the first iteration
        maxiter0 = None  # to be evaluated after the first iteration
        base_evals = 0

        irun = 0
        best = ot.BestSolution()
        all_stoppings = []
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

            elif sum(small_i) < sum(large_i):
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

            # recover from a CMA object
            if irun == 0 and isinstance(x0, CMAEvolutionStrategy):
                es = x0
                x0 = es.inputargs['x0']  # for the next restarts
                if np.isscalar(sigma0) and np.isfinite(sigma0) and sigma0 > 0:
                    es.sigma = sigma0
                # debatable whether this makes sense:
                sigma0 = es.inputargs['sigma0']  # for the next restarts
                if options is not None:
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
                    x = es.gp.pheno(es.mean,
                                    into_bounds=es.boundary_handler.repair,
                                    archive=es.sent_solutions)
                    es.f0 = objective_function(x, *args)
                    es.best.update([x], es.sent_solutions,
                                   [es.f0], 1)
                    es.countevals += 1
            es.objective_function = objective_function  # only for the record

            opts = es.opts  # processed options, unambiguous
            # a hack:
            fmin_opts = CMAOptions("unchecked", **fmin_options.copy())
            for k in fmin_opts:
                # locals() cannot be modified directly, exec won't work
                # in 3.x, therefore
                fmin_opts.eval(k, loc={'N': es.N,
                                       'popsize': opts['popsize']},
                               correct_key=False)

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
                else:
                    noisehandler = noise_handler
                noise_handling = True
                if fmin_opts['noise_change_sigma_exponent'] > 0:
                    es.opts['tolfacupx'] = inf
            else:
                noisehandler = ot.NoiseHandler(es.N, 0)  # switched off
                noise_handling = False
            es.noise_handler = noisehandler

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
                    if es.opts['verbose'] > 4:
                        if es.countiter > 1 and min(fit) > es.best.last.f:
                            unsuccessful_iterations_count += 1
                            if unsuccessful_iterations_count > 4:
                                utils.print_message('%d unsuccessful iterations'
                                                    % unsuccessful_iterations_count,
                                                    iteration=es.countiter)
                        else:
                            unsuccessful_iterations_count = 0
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
                mean_pheno = es.gp.pheno(es.mean,
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
            if irun - runs_with_small > fmin_opts['restarts'] or 'ftarget' in es.stop() \
                    or 'maxfevals' in es.stop(check=False) or 'callback' in es.stop(check=False):
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
        if eval(str(options['verb_disp'])) > 0:
            print(' in/outcomment ``raise`` in last line of cma.fmin to prevent/restore KeyboardInterrupt exception')
        raise KeyboardInterrupt  # cave: swallowing this exception can silently mess up experiments, if ctrl-C is hit


# BEGIN cmaplt.py

class old_CMADataLogger(interfaces.BaseDataLogger):
    """data logger for class `CMAEvolutionStrategy`.

    The logger is identified by its name prefix and (over-)writes or
    reads according data files. Therefore, the logger must be
    considered as *global* variable with unpredictable side effects,
    if two loggers with the same name and on the same working folder
    are used at the same time.

    Examples
    ========
    ::

        import cma
        es = cma.CMAEvolutionStrategy(...)
        logger = cma.CMADataLogger().register(es)
        while not es.stop():
            ...
            logger.add()  # add can also take an argument

        logger.plot() # or a short cut can be used:
        cma.plot()  # plot data from logger with default name

        logger2 = cma.CMADataLogger('just_another_filename_prefix').load()
        logger2.plot()
        logger2.disp()

        import cma
        from matplotlib.pylab import *
        res = cma.fmin(cma.ff.sphere, rand(10), 1e-0)
        logger = res[-1]  # the CMADataLogger
        logger.load()  # by "default" data are on disk
        semilogy(logger.f[:,0], logger.f[:,5])  # plot f versus iteration, see file header
        cma.s.figshow()

    Details
    =======
    After loading data, the logger has the attributes `xmean`, `xrecent`,
    `std`, `f`, `D` and `corrspec` corresponding to ``xmean``,
    ``xrecentbest``, ``stddev``, ``fit``, ``axlen`` and ``axlencorr``
    filename trails.

    :See: `disp` (), `plot` ()
    """
    default_prefix = 'outcmaes'
    # names = ('axlen','fit','stddev','xmean','xrecentbest')
    # key_names_with_annotation = ('std', 'xmean', 'xrecent')

    def __init__(self, name_prefix=default_prefix, modulo=1, append=False):
        """initialize logging of data from a `CMAEvolutionStrategy`
        instance, default ``modulo=1`` means logging with each call

        """
        # super(CMAData, self).__init__({'iter':[], 'stds':[], 'D':[],
        #        'sig':[], 'fit':[], 'xm':[]})
        # class properties:
        self.name_prefix = name_prefix if name_prefix \
            else old_CMADataLogger.default_prefix
        if isinstance(self.name_prefix, CMAEvolutionStrategy):
            self.name_prefix = self.name_prefix.opts.eval(
                'verb_filenameprefix')
        self.file_names = ('axlen', 'axlencorr', 'fit', 'stddev', 'xmean',
                'xrecentbest')
        """used in load, however hard-coded in add"""
        self.key_names = ('D', 'corrspec', 'f', 'std', 'xmean', 'xrecent')
        """used in load, however hard-coded in plot"""
        self._key_names_with_annotation = ('std', 'xmean', 'xrecent')
        """used in load to add one data row to be modified in plot"""
        self.modulo = modulo
        """how often to record data, allows calling `add` without args"""
        self.append = append
        """append to previous data"""
        self.counter = 0
        """number of calls to `add`"""
        self.last_iteration = 0
        self.registered = False
        self.last_correlation_spectrum = None
        self._eigen_counter = 1  # reduce costs
        self.persistent_communication_dict = utils.DictFromTagsInString()
    @property
    def data(self):
        """return dictionary with data.

        If data entries are None or incomplete, consider calling
        ``.load().data`` to (re-)load the data from files first.

        """
        d = {}
        for name in self.key_names:
            d[name] = self.__dict__.get(name, None)
        return d
    def register(self, es, append=None, modulo=None):
        """register a `CMAEvolutionStrategy` instance for logging,
        ``append=True`` appends to previous data logged under the same name,
        by default previous data are overwritten.

        """
        if not isinstance(es, CMAEvolutionStrategy):
            utils.print_warning("""only class CMAEvolutionStrategy should
    be registered for logging. The used "%s" class may not to work
    properly. This warning may also occur after using `reload`. Then,
    restarting Python should solve the issue.""" %
                                str(type(es)))
        self.es = es
        if append is not None:
            self.append = append
        if modulo is not None:
            self.modulo = modulo
        self.registered = True
        return self

    def initialize(self, modulo=None):
        """reset logger, overwrite original files, `modulo`: log only every modulo call"""
        if modulo is not None:
            self.modulo = modulo
        try:
            es = self.es  # must have been registered
        except AttributeError:
            pass  # TODO: revise usage of es... that this can pass
            raise AttributeError('call register() before initialize()')

        self.counter = 0  # number of calls of add
        self.last_iteration = 0  # some lines are only written if iteration>last_iteration
        if self.modulo <= 0:
            return self

        # write headers for output
        fn = self.name_prefix + 'fit.dat'
        strseedtime = 'seed=%s, %s' % (str(es.opts['seed']), time.asctime())

        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, sigma, axis ratio, ' +
                        'bestever, best, median, worst objective function value, ' +
                        'further objective values of best", ' +
                        strseedtime +
                        ', ' + self.persistent_communication_dict.as_python_tag +
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'axlen.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, sigma, ' +
                        'max axis length, ' +
                        ' min axis length, all principle axes lengths ' +
                        ' (sorted square roots of eigenvalues of C)", ' +
                        strseedtime +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)
        fn = self.name_prefix + 'axlencorr.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, min max(neg(.)) min(pos(.))' +
                        ' max correlation, correlation matrix principle axes lengths ' +
                        ' (sorted square roots of eigenvalues of correlation matrix)", ' +
                        strseedtime +
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)
        fn = self.name_prefix + 'stddev.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, sigma, void, void, ' +
                        ' stds==sigma*sqrt(diag(C))", ' +
                        strseedtime +
                        ', ' + self.persistent_communication_dict.as_python_tag +
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'xmean.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, void, void, void, xmean", ' +
                        strseedtime +
                        ', ' + self.persistent_communication_dict.as_python_tag
                        )
                f.write(' # scaling_of_variables: ')  # todo: put as python tag
                if np.size(es.gp.scales) > 1:
                    f.write(' '.join(map(str, es.gp.scales)))
                else:
                    f.write(str(es.gp.scales))
                f.write(', typical_x: ')
                if np.size(es.gp.typical_x) > 1:
                    f.write(' '.join(map(str, es.gp.typical_x)))
                else:
                    f.write(str(es.gp.typical_x))
                f.write('\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        fn = self.name_prefix + 'xrecentbest.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iter, evals, sigma, 0, fitness, xbest" ' +
                        strseedtime +
                        ', ' + self.persistent_communication_dict.as_python_tag +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        return self
    # end def __init__

    def load(self, filenameprefix=None):
        """load (or reload) data from output files, `load` is called in
        `plot` and `disp`.

        Argument `filenameprefix` is the filename prefix of data to be
        loaded (six files), by default ``'outcmaes'``.

        Return self with (added) attributes `xrecent`, `xmean`,
        `f`, `D`, `std`, 'corrspec'

        """
        if not filenameprefix:
            filenameprefix = self.name_prefix
        assert len(self.file_names) == len(self.key_names)
        for i in range(len(self.file_names)):
            fn = filenameprefix + self.file_names[i] + '.dat'
            try:
                # list of rows to append another row latter
                with warnings.catch_warnings():
                    if self.file_names[i] == 'axlencorr':
                        warnings.simplefilter("ignore")
                    try:
                        self.__dict__[self.key_names[i]] = list(
                                np.loadtxt(fn, comments=['%', '#']))
                    except:
                        self.__dict__[self.key_names[i]] = list(
                                np.loadtxt(fn, comments='%'))
                # read dict from <python> tag in first line
                with open(fn) as file:
                    self.persistent_communication_dict.update(
                                string_=file.readline())
            except IOError:
                utils.print_warning('reading from file "' + fn + '" failed',
                               'load', 'CMADataLogger')
            try:
                # duplicate last row to later fill in annotation
                # positions for display
                if self.key_names[i] in self._key_names_with_annotation:
                    self.__dict__[self.key_names[i]].append(
                        self.__dict__[self.key_names[i]][-1])
                self.__dict__[self.key_names[i]] = \
                    np.asarray(self.__dict__[self.key_names[i]])
            except:
                utils.print_warning('no data for %s' % fn, 'load',
                               'CMADataLogger')
        # convert single line to matrix of shape (1, len)
        for key in self.key_names:
            try:
                d = getattr(self, key)
            except AttributeError:
                utils.print_warning("attribute %s missing" % key, 'load',
                                    'CMADataLogger')
                continue
            if len(d.shape) == 1:  # one line has shape (8, )
                setattr(self, key, d.reshape((1, len(d))))

        return self

    def add(self, es=None, more_data=(), modulo=None):
        """append some logging data from `CMAEvolutionStrategy` class instance `es`,
        if ``number_of_times_called % modulo`` equals to zero, never if ``modulo==0``.

        ``more_data`` is a list of additional data to be recorded where each
        data entry must have the same length.

        When used for a different optimizer class, this function can be
        (easily?) adapted by changing the assignments under INTERFACE
        in the implemention.

        """
        mod = modulo if modulo is not None else self.modulo
        self.counter += 1
        if mod == 0 or (self.counter > 3 and (self.counter - 1) % mod):
            return
        if es is None:
            try:
                es = self.es  # must have been registered
            except AttributeError :
                raise AttributeError('call `add` with argument `es` or ``register(es)`` before ``add()``')
        elif not self.registered:
            self.register(es)

        if self.counter == 1 and not self.append and self.modulo != 0:
            self.initialize()  # write file headers
            self.counter = 1

        # --- INTERFACE, can be changed if necessary ---
        if not isinstance(es, CMAEvolutionStrategy):  # not necessary
            utils.print_warning('type CMAEvolutionStrategy expected, found '
                                + str(type(es)), 'add', 'CMADataLogger')
        evals = es.countevals
        iteration = es.countiter
        eigen_decompositions = es.count_eigen
        sigma = es.sigma
        if es.opts['CMA_diagonal'] is True or es.countiter <= es.opts['CMA_diagonal']:
            stds = es.sigma_vec.scaling * es.sm.variances**0.5
            axratio = max(stds) / min(stds)
        else:
            axratio = es.D.max() / es.D.min()
        xmean = es.mean  # TODO: should be optionally phenotype?
        fmean_noise_free = 0  # es.fmean_noise_free  # meaningless as
        fmean = 0  # es.fmean                        # only inialized
        # TODO: find a different way to communicate current x and f?
        try:
            besteverf = es.best.f
            bestf = es.fit.fit[0]
            worstf = es.fit.fit[-1]
            medianf = es.fit.fit[es.sp.popsize // 2]
        except:
            if iteration > 0:  # first call without f-values is OK
                raise
        try:
            xrecent = es.best.last.x
        except:
            xrecent = None
        diagC = es.sigma * es.sigma_vec.scaling * es.sm.variances**0.5
        if es.opts['CMA_diagonal'] is True or es.countiter <= es.opts['CMA_diagonal']:
            maxD = max(es.sigma_vec * es.sm.variances**0.5)  # dC should be 1 though
            minD = min(es.sigma_vec * es.sm.variances**0.5)
            diagD = [1] if es.opts['CMA_diagonal'] is True else diagC
        elif isinstance(es.sm, _rgs.GaussVkDSampler):
            diagD = list(1e2 * es.sm.D) + list(1e-2 * (es.sm.S + 1)**0.5)
            axratio = ((max(es.sm.S) + 1) / (min(es.sm.S) + 1))**0.5
            maxD = (max(es.sm.S) + 1)**0.5
            minD = (min(es.sm.S) + 1)**0.5
            sigma = es.sm.sigma
        elif isinstance(es.sm, _rgs.GaussVDSampler):
            # this may not be reflective of the shown annotations
            diagD = list(1e2 * es.sm.dvec) + [1e-2 * es.sm.norm_v]
            maxD = minD = 1
            axratio = 1  # es.sm.condition_number**0.5
            # sigma = es.sm.sigma
        else:
            try:
                diagD = es.sm.D
            except:
                diagD = [1]
            maxD = max(diagD)
            minD = min(diagD)
        more_to_write = es.more_to_write
        es.more_to_write = utils.MoreToWrite()
        # --- end interface ---

        try:
            # fit
            if iteration > self.last_iteration:
                fn = self.name_prefix + 'fit.dat'
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(sigma) + ' '
                            + str(axratio) + ' '
                            + str(besteverf) + ' '
                            + '%.16e' % bestf + ' '
                            + str(medianf) + ' '
                            + str(worstf) + ' '
                            # + str(es.sp.popsize) + ' '
                            # + str(10**es.noiseS) + ' '
                            # + str(es.sp.cmean) + ' '
                            + ' '.join(str(i) for i in more_to_write) + ' '
                            + ' '.join(str(i) for i in more_data) + ' '
                            + '\n')
            # axlen
            fn = self.name_prefix + 'axlen.dat'
            if 1 < 3:
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(sigma) + ' '
                            + str(maxD) + ' '
                            + str(minD) + ' '
                            + ' '.join(map(str, diagD))
                            + '\n')
            # correlation matrix eigenvalues
            if 1 < 3:
                fn = self.name_prefix + 'axlencorr.dat'
                try:
                    c = es.sm.correlation_matrix
                except (AttributeError, NotImplemented, NotImplementedError):
                    c = None
                if c is not None:
                    # accept at most 50% internal loss
                    if 11 < 3 or self._eigen_counter < eigen_decompositions / 2:
                        self.last_correlation_spectrum = \
                            sorted(es.opts['CMA_eigenmethod'](c)[0]**0.5)
                        self._eigen_counter += 1
                    if self.last_correlation_spectrum is None:
                        self.last_correlation_spectrum = len(diagD) * [1]
                    c = c[c < 1 - 1e-14]  # remove diagonal elements
                    c[c > 1 - 1e-14] = 1 - 1e-14
                    c[c < -1 + 1e-14] = -1 + 1e-14
                    c_min = np.min(c)
                    c_max = np.max(c)
                    if np.min(abs(c)) == 0:
                        c_medminus = 0  # thereby zero "is negative"
                        c_medplus = 0  # thereby zero "is positive"
                    else:
                        c_medminus = c[np.argmin(1/c)]  # c is flat
                        c_medplus = c[np.argmax(1/c)]  # c is flat

                    with open(fn, 'a') as f:
                        f.write(str(iteration) + ' '
                                + str(evals) + ' '
                                + str(c_min) + ' '
                                + str(c_medminus) + ' ' # the one closest to 0
                                + str(c_medplus) + ' ' # the one closest to 0
                                + str(c_max) + ' '
                                + ' '.join(map(str,
                                        self.last_correlation_spectrum))
                                + '\n')

            # stddev
            fn = self.name_prefix + 'stddev.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(sigma) + ' '
                        + '0 0 '
                        + ' '.join(map(str, diagC))
                        + '\n')
            # xmean
            fn = self.name_prefix + 'xmean.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        # + str(sigma) + ' '
                        + '0 '
                        + str(fmean_noise_free) + ' '
                        + str(fmean) + ' '  # TODO: this does not make sense
                        # TODO should be optional the phenotyp?
                        + ' '.join(map(str, xmean))
                        + '\n')
            # xrecent
            fn = self.name_prefix + 'xrecentbest.dat'
            if iteration > 0 and xrecent is not None:
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(sigma) + ' '
                            + '0 '
                            + str(bestf) + ' '
                            + ' '.join(map(str, xrecent))
                            + '\n')
        except (IOError, OSError):
            if iteration <= 1:
                utils.print_warning(('could not open/write file %s: ' % fn,
                                     sys.exc_info()))
        self.last_iteration = iteration

    def figclose(self):
        from matplotlib.pyplot import close
        close(self.fighandle)

    def save(self, name=None):
        """data are saved to disk the moment they are added"""

    def save_to(self, nameprefix, switch=False):
        """saves logger data to a different set of files, for
        ``switch=True`` also the loggers name prefix is switched to
        the new value

        """
        if not nameprefix or not utils.is_str(nameprefix):
            raise ValueError('filename prefix must be a non-empty string')

        if nameprefix == self.default_prefix:
            raise ValueError('cannot save to default name "' + nameprefix + '...", chose another name')

        if nameprefix == self.name_prefix:
            return

        for name in self.file_names:
            open(nameprefix + name + '.dat', 'w').write(open(self.name_prefix + name + '.dat').read())

        if switch:
            self.name_prefix = nameprefix
    def select_data(self, iteration_indices):
        """keep only data of `iteration_indices`"""
        dat = self
        iteridx = iteration_indices
        dat.f = dat.f[_where([x in iteridx for x in dat.f[:, 0]])[0], :]
        dat.D = dat.D[_where([x in iteridx for x in dat.D[:, 0]])[0], :]
        try:
            iteridx = list(iteridx)
            iteridx.append(iteridx[-1])  # last entry is artificial
        except:
            pass
        dat.std = dat.std[_where([x in iteridx
                                    for x in dat.std[:, 0]])[0], :]
        dat.xmean = dat.xmean[_where([x in iteridx
                                        for x in dat.xmean[:, 0]])[0], :]
        try:
            dat.xrecent = dat.x[_where([x in iteridx for x in
                                          dat.xrecent[:, 0]])[0], :]
        except AttributeError:
            pass
        try:
            dat.corrspec = dat.x[_where([x in iteridx for x in
                                           dat.corrspec[:, 0]])[0], :]
        except AttributeError:
            pass
    def plot(self, fig=None, iabscissa=1, iteridx=None,
             plot_mean=False, # was: plot_mean=True
             foffset=1e-19, x_opt=None, fontsize=7,
             downsample_to=1e7):
        """plot data from a `CMADataLogger` (using the files written
        by the logger).

        Arguments
        ---------
        `fig`
            figure number, by default 325
        `iabscissa`
            ``0==plot`` versus iteration count,
            ``1==plot`` versus function evaluation number
        `iteridx`
            iteration indices to plot

        Return `CMADataLogger` itself.

        Examples
        --------
        ::

            import cma
            logger = cma.CMADataLogger()  # with default name
            # try to plot the "default logging" data (e.g.
            #   from previous fmin calls, which is essentially what
            #   also cma.plot() does)
            logger.plot()
            cma.s.figsave('fig325.png')  # save current figure
            logger.figclose()

        Dependencies: matlabplotlib.pyplot

        """
        try:
            from matplotlib import pyplot
            from matplotlib.pyplot import figure, subplot, gcf
        except ImportError:
            ImportError('could not find matplotlib.pyplot module, function plot() is not available')
            return

        if fig is None:
            fig = 325
        if iabscissa not in (0, 1):
            iabscissa = 1

        self.load()  # better load only conditionally?
        if self.f.shape[0] > downsample_to:
            self.downsampling(1 + self.f.shape[0] // downsample_to)
            self.load()

        dat = self
        dat.x = dat.xmean  # this is the genotyp
        if not plot_mean:
            if len(dat.x) < 2:
                print('not enough data to plot recent x')
            else:
                dat.x = dat.xrecent

        # index out some data
        if iteridx is not None:
            self.select_data(iteridx)

        if len(dat.f) <= 1:
            print('nothing to plot')
            return

        # not in use anymore, see formatter above
        # xticklocs = np.arange(5) * np.round(minxend/4., -int(np.log10(minxend/4.)))

        # dfit(dfit<1e-98) = NaN;

        # TODO: if abscissa==0 plot in chunks, ie loop over subsets where
        # dat.f[:,0]==countiter is monotonous

        figure(fig)
        finalize = self._finalize_plotting
        self._finalize_plotting = lambda : None
        self._enter_plotting(fontsize)
        self.fighandle = gcf()  # fighandle.number
        self.fighandle.clear()

        subplot(2, 2, 1)
        self.plot_divers(iabscissa, foffset)
        pyplot.xlabel('')

        # Scaling
        subplot(2, 2, 3)
        self.plot_axes_scaling(iabscissa)

        # spectrum of correlation matrix
        if 11 < 3 and hasattr(dat, 'corrspec'):
            figure(fig+10000)
            pyplot.gcf().clear()  # == clf(), replaces hold(False)
            self.plot_correlations(iabscissa)
        figure(fig)

        subplot(2, 2, 2)
        if plot_mean:
            self.plot_mean(iabscissa, x_opt)
        else:
            self.plot_xrecent(iabscissa, x_opt)
        pyplot.xlabel('')
        # pyplot.xticks(xticklocs)

        # standard deviations
        subplot(2, 2, 4)
        self.plot_stds(iabscissa)

        self._finalize_plotting = finalize
        self._finalize_plotting()
        return self

    def plot_all(self, fig=None, iabscissa=1, iteridx=None,
             foffset=1e-19, x_opt=None, fontsize=7):
        """
        plot data from a `CMADataLogger` (using the files written by the logger).

        Arguments
        ---------
        `fig`
            figure number, by default 425
        `iabscissa`
            ``0==plot`` versus iteration count,
            ``1==plot`` versus function evaluation number
        `iteridx`
            iteration indices to plot

        Return `CMADataLogger` itself.

        Examples
        --------
        ::

            import cma
            logger = cma.CMADataLogger()  # with default name
            # try to plot the "default logging" data (e.g.
            #   from previous fmin calls, which is essentially what
            #   also cma.plot() does)
            logger.plot_all()
            cma.s.figsave('fig425.png')  # save current figure
            logger.s.figclose()

        Dependencies: matlabplotlib/pyplot.

        """
        try:
            # pyplot: prodedural interface for matplotlib
            from matplotlib import pyplot
            from matplotlib.pyplot import figure, subplot, gcf
        except ImportError:
            ImportError('could not find matplotlib.pyplot module, function plot() is not available')
            return

        if fig is None:
            fig = 426
        if iabscissa not in (0, 1):
            iabscissa = 1

        self.load()
        dat = self

        # index out some data
        if iteridx is not None:
            self.select_data(iteridx)

        if len(dat.f) == 0:
            print('nothing to plot')
            return

        # not in use anymore, see formatter above
        # xticklocs = np.arange(5) * np.round(minxend/4., -int(np.log10(minxend/4.)))

        # dfit(dfit<1e-98) = NaN;

        # TODO: if abscissa==0 plot in chunks, ie loop over subsets where
        #       dat.f[:,0]==countiter is monotonous

        figure(fig)
        self._enter_plotting(fontsize)
        self.fighandle = gcf()  # fighandle.number
        self.fighandle.clear()

        if 11 < 3:
            subplot(3, 2, 1)
            self.plot_divers(iabscissa, foffset)
            pyplot.xlabel('')

            # Scaling
            subplot(3, 2, 3)
            self.plot_axes_scaling(iabscissa)
            pyplot.xlabel('')

            # spectrum of correlation matrix
            subplot(3, 2, 5)
            self.plot_correlations(iabscissa)

            # x-vectors
            subplot(3, 2, 2)
            self.plot_xrecent(iabscissa, x_opt)
            pyplot.xlabel('')
            subplot(3, 2, 4)
            self.plot_mean(iabscissa, x_opt)
            pyplot.xlabel('')

            # standard deviations
            subplot(3, 2, 6)
            self.plot_stds(iabscissa)
        else:
            subplot(2, 3, 1)
            self.plot_divers(iabscissa, foffset)
            pyplot.xlabel('')

            # standard deviations
            subplot(2, 3, 4)
            self.plot_stds(iabscissa)

            # Scaling
            subplot(2, 3, 2)
            self.plot_axes_scaling(iabscissa)
            pyplot.xlabel('')

            # spectrum of correlation matrix
            subplot(2, 3, 5)
            self.plot_correlations(iabscissa)

            # x-vectors
            subplot(2, 3, 3)
            self.plot_xrecent(iabscissa, x_opt)
            pyplot.xlabel('')

            subplot(2, 3, 6)
            self.plot_mean(iabscissa, x_opt)

        self._finalize_plotting()
        return self
    def plot_axes_scaling(self, iabscissa=1):
        from matplotlib import pyplot
        if not hasattr(self, 'D'):
            self.load()
        dat = self
        if np.max(dat.D[:, 5:]) == np.min(dat.D[:, 5:]):
            pyplot.text(0, dat.D[-1, 5],
                        'all axes scaling values equal to %s'
                        % str(dat.D[-1, 5]),
                        verticalalignment='center')
            return self  # nothing interesting to plot
        self._enter_plotting()
        pyplot.semilogy(dat.D[:, iabscissa], dat.D[:, 5:], '-b')
        # pyplot.hold(True)
        pyplot.grid(True)
        ax = array(pyplot.axis())
        # ax[1] = max(minxend, ax[1])
        pyplot.axis(ax)
        pyplot.title('Principle Axes Lengths')
        # pyplot.xticks(xticklocs)
        self._xlabel(iabscissa)
        self._finalize_plotting()
        return self
    def plot_stds(self, iabscissa=1):
        from matplotlib import pyplot
        if not hasattr(self, 'std'):
            self.load()
        dat = self
        self._enter_plotting()
        # remove sigma from stds (graphs become much better readible)
        dat.std[:, 5:] = np.transpose(dat.std[:, 5:].T / dat.std[:, 2].T)
        # ax = array(pyplot.axis())
        # ax[1] = max(minxend, ax[1])
        # axis(ax)
        if 1 < 2 and dat.std.shape[1] < 100:
            # use fake last entry in x and std for line extension-annotation
            minxend = int(1.06 * dat.std[-2, iabscissa])
            # minxend = int(1.06 * dat.x[-2, iabscissa])
            dat.std[-1, iabscissa] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat.std[-2, 5:])
            # idx2 = np.argsort(idx)
            dat.std[-1, 5 + idx] = np.logspace(np.log10(np.min(dat.std[:, 5:])),
                            np.log10(np.max(dat.std[:, 5:])), dat.std.shape[1] - 5)

            dat.std[-1, iabscissa] = minxend  # TODO: should be ax[1]
            pyplot.semilogy(dat.std[:, iabscissa], dat.std[:, 5:], '-')
            # pyplot.hold(True)
            ax = array(pyplot.axis())

            # yy = np.logspace(np.log10(ax[2]), np.log10(ax[3]), dat.std.shape[1] - 5)
            # yyl = np.sort(dat.std[-1,5:])
            idx = np.argsort(dat.std[-1, 5:])
            # idx2 = np.argsort(idx)
            # plot(np.dot(dat.std[-2, iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-') # vertical separator
            # vertical separator
            pyplot.plot(np.dot(dat.std[-2, iabscissa], [1, 1]),
                        array([ax[2] * (1 + 1e-6), ax[3] / (1 + 1e-6)]),
                        # array([np.min(dat.std[:, 5:]), np.max(dat.std[:, 5:])]),
                        'k-')
            # pyplot.hold(True)
            # plot([dat.std[-1, iabscissa], ax[1]], [dat.std[-1,5:], yy[idx2]], 'k-') # line from last data point
            annotations = self.persistent_communication_dict.get('variable_annotations')
            if annotations is None:
                annotations = range(len(idx))
            for i, s in enumerate(annotations):
                # text(ax[1], yy[i], ' '+str(idx[i]))
                pyplot.text(dat.std[-1, iabscissa], dat.std[-1, 5 + i],
                            ' ' + str(s))
        else:
            pyplot.semilogy(dat.std[:, iabscissa], dat.std[:, 5:], '-')
        # pyplot.hold(True)
        pyplot.grid(True)
        pyplot.title(r'Standard Deviations $\times$ $\sigma^{-1}$ in All Coordinates')
        # pyplot.xticks(xticklocs)
        self._xlabel(iabscissa)
        self._finalize_plotting()
        return self
    def plot_mean(self, iabscissa=1, x_opt=None, annotations=None):
        if not hasattr(self, 'xmean'):
            self.load()
        self.x = self.xmean
        self._plot_x(iabscissa, x_opt, 'mean', annotations=annotations)
        self._xlabel(iabscissa)
        return self
    def plot_xrecent(self, iabscissa=1, x_opt=None, annotations=None):
        if not hasattr(self, 'xrecent'):
            self.load()
        self.x = self.xrecent
        self._plot_x(iabscissa, x_opt, 'curr best', annotations=annotations)
        self._xlabel(iabscissa)
        return self
    def plot_correlations(self, iabscissa=1):
        """spectrum of correlation matrix and largest correlation"""
        if not hasattr(self, 'corrspec'):
            self.load()
        if len(self.corrspec) < 2:
            return self
        x = self.corrspec[:, iabscissa]
        y = self.corrspec[:, 6:]  # principle axes
        ys = self.corrspec[:, :6]  # "special" values

        from matplotlib.pyplot import semilogy, text, grid, axis, title
        self._enter_plotting()
        semilogy(x, y, '-c')
        # hold(True)
        semilogy(x[:], np.max(y, 1) / np.min(y, 1), '-r')
        text(x[-1], np.max(y[-1, :]) / np.min(y[-1, :]), 'axis ratio')
        if ys is not None:
            semilogy(x, 1 + ys[:, 2], '-b')
            text(x[-1], 1 + ys[-1, 2], '1 + min(corr)')
            semilogy(x, 1 - ys[:, 5], '-b')
            text(x[-1], 1 - ys[-1, 5], '1 - max(corr)')
            semilogy(x[:], 1 + ys[:, 3], '-k')
            text(x[-1], 1 + ys[-1, 3], '1 + max(neg corr)')
            semilogy(x[:], 1 - ys[:, 4], '-k')
            text(x[-1], 1 - ys[-1, 4], '1 - min(pos corr)')
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        title('Spectrum (roots) of correlation matrix')
        # pyplot.xticks(xticklocs)
        self._xlabel(iabscissa)
        self._finalize_plotting()
        return self
    def plot_divers(self, iabscissa=1, foffset=1e-19):
        """plot fitness, sigma, axis ratio...

        :param iabscissa: 0 means vs evaluations, 1 means vs iterations
        :param foffset: added to f-value

        :See: `plot`

        """
        from matplotlib import pyplot
        from matplotlib.pyplot import semilogy, grid, \
            axis, title, text
        fontsize = pyplot.rcParams['font.size']

        if not hasattr(self, 'f'):
            self.load()
        dat = self

        # correct values which are rather not reasonable
        if not np.isfinite(dat.f[0, 5]):
            dat.f[0, 5:] = dat.f[1, 5:]  # best, median and worst f-value
        for i, val in enumerate(dat.f[0, :]): # hack to prevent warnings
            if np.isnan(val):
                dat.f[0, i] = dat.f[1, i]
        minfit = np.nanmin(dat.f[:, 5])
        dfit = dat.f[:, 5] - minfit  # why not using idx?
        dfit[dfit < 1e-98] = np.NaN

        self._enter_plotting()
        if dat.f.shape[1] > 7:
            # semilogy(dat.f[:, iabscissa], abs(dat.f[:,[6, 7, 10, 12]])+foffset,'-k')
            semilogy(dat.f[:, iabscissa], abs(dat.f[:, [6, 7]]) + foffset, '-k')
            # hold(True)

        # (larger indices): additional fitness data, for example constraints values
        if dat.f.shape[1] > 8:
            # dd = abs(dat.f[:,7:]) + 10*foffset
            # dd = _where(dat.f[:,7:]==0, np.NaN, dd) # cannot be
            semilogy(dat.f[:, iabscissa], np.abs(dat.f[:, 8:]) + 10 * foffset, 'y')
            # hold(True)

        idx = _where(dat.f[:, 5] > 1e-98)[0]  # positive values
        semilogy(dat.f[idx, iabscissa], dat.f[idx, 5] + foffset, '.b')
        # hold(True)
        grid(True)


        semilogy(dat.f[:, iabscissa], abs(dat.f[:, 5]) + foffset, '-b')
        text(dat.f[-1, iabscissa], abs(dat.f[-1, 5]) + foffset,
             r'$|f_\mathsf{best}|$', fontsize=fontsize + 2)

        # negative f-values, dots
        sgn = np.sign(dat.f[:, 5])
        sgn[np.abs(dat.f[:, 5]) < 1e-98] = 0
        idx = _where(sgn < 0)[0]
        semilogy(dat.f[idx, iabscissa], abs(dat.f[idx, 5]) + foffset,
                 '.m')  # , markersize=5

        # lines between negative f-values
        dsgn = np.diff(sgn)
        start_idx = 1 + _where((dsgn < 0) * (sgn[1:] < 0))[0]
        stop_idx = 1 + _where(dsgn > 0)[0]
        if sgn[0] < 0:
            start_idx = np.concatenate(([0], start_idx))
        for istart in start_idx:
            istop = stop_idx[stop_idx > istart]
            istop = istop[0] if len(istop) else 0
            idx = range(istart, istop if istop else dat.f.shape[0])
            if len(idx) > 1:
                semilogy(dat.f[idx, iabscissa], abs(dat.f[idx, 5]) + foffset,
                        'm')  # , markersize=5
            # lines between positive and negative f-values
            # TODO: the following might plot values very close to zero
            if istart > 0:  # line to the left of istart
                semilogy(dat.f[istart-1:istart+1, iabscissa],
                         abs(dat.f[istart-1:istart+1, 5]) +
                         foffset, '--m')
            if istop:  # line to the left of istop
                semilogy(dat.f[istop-1:istop+1, iabscissa],
                         abs(dat.f[istop-1:istop+1, 5]) +
                         foffset, '--m')
                # mark the respective first positive values
                semilogy(dat.f[istop, iabscissa], abs(dat.f[istop, 5]) +
                         foffset, '.b', markersize=7)
            # mark the respective first negative values
            semilogy(dat.f[istart, iabscissa], abs(dat.f[istart, 5]) +
                     foffset, '.r', markersize=7)

        # standard deviations std
        semilogy(dat.std[:-1, iabscissa],
                 np.vstack([list(map(max, dat.std[:-1, 5:])),
                            list(map(min, dat.std[:-1, 5:]))]).T,
                     '-m', linewidth=2)
        text(dat.std[-2, iabscissa], max(dat.std[-2, 5:]), 'max std',
             fontsize=fontsize)
        text(dat.std[-2, iabscissa], min(dat.std[-2, 5:]), 'min std',
             fontsize=fontsize)

        # delta-fitness in cyan
        idx = np.isfinite(dfit)
        if any(idx):
            idx_nan = _where(~idx)[0]  # gaps
            if not len(idx_nan):  # should never happen
                semilogy(dat.f[:, iabscissa][idx], dfit[idx], '-c')
            else:
                i_start = 0
                for i_end in idx_nan:
                    if i_end > i_start:
                        semilogy(dat.f[:, iabscissa][i_start:i_end],
                                                dfit[i_start:i_end], '-c')
                    i_start = i_end + 1
                if len(dfit) > idx_nan[-1] + 1:
                    semilogy(dat.f[:, iabscissa][idx_nan[-1]+1:],
                                            dfit[idx_nan[-1]+1:], '-c')
            text(dat.f[idx, iabscissa][-1], dfit[idx][-1],
                 r'$f_\mathsf{best} - \min(f)$', fontsize=fontsize + 2)

        elif 11 < 3 and any(idx):
            semilogy(dat.f[:, iabscissa][idx], dfit[idx], '-c')
            text(dat.f[idx, iabscissa][-1], dfit[idx][-1],
                 r'$f_\mathsf{best} - \min(f)$', fontsize=fontsize + 2)

        if 11 < 3:  # delta-fitness as points
            dfit = dat.f[1:, 5] - dat.f[:-1, 5]  # should be negative usually
            semilogy(dat.f[1:, iabscissa],  # abs(fit(g) - fit(g-1))
                np.abs(dfit) + foffset, '.c')
            i = dfit > 0
            # print(np.sum(i) / float(len(dat.f[1:,iabscissa])))
            semilogy(dat.f[1:, iabscissa][i],  # abs(fit(g) - fit(g-1))
                np.abs(dfit[i]) + foffset, '.r')

        # overall minimum
        i = np.argmin(dat.f[:, 5])
        semilogy(dat.f[i, iabscissa], np.abs(dat.f[i, 5]), 'ro',
                 markersize=9)
        if any(idx):
            semilogy(dat.f[i, iabscissa], dfit[idx][np.argmin(dfit[idx])]
                 + 1e-98, 'ro', markersize=9)
        # semilogy(dat.f[-1, iabscissa]*np.ones(2), dat.f[-1,4]*np.ones(2), 'rd')

        # AR and sigma
        semilogy(dat.f[:, iabscissa], dat.f[:, 3], '-r')  # AR
        semilogy(dat.f[:, iabscissa], dat.f[:, 2], '-g')  # sigma
        text(dat.f[-1, iabscissa], dat.f[-1, 3], r'axis ratio',
             fontsize=fontsize)
        text(dat.f[-1, iabscissa], dat.f[-1, 2] / 1.5, r'$\sigma$',
             fontsize=fontsize+3)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        text(ax[0] + 0.01, ax[2],  # 10**(log10(ax[2])+0.05*(log10(ax[3])-log10(ax[2]))),
             '.min($f$)=' + repr(minfit))
             #'.f_recent=' + repr(dat.f[-1, 5]))

        # title('abs(f) (blue), f-min(f) (cyan), Sigma (green), Axis Ratio (red)')
        # title(r'blue:$\mathrm{abs}(f)$, cyan:$f - \min(f)$, green:$\sigma$, red:axis ratio',
        #       fontsize=fontsize - 0.0)
        title(r'$|f_{\mathrm{best},\mathrm{med},\mathrm{worst}}|$, $f - \min(f)$, $\sigma$, axis ratio')

        # if __name__ != 'cma':  # should be handled by the caller
        self._xlabel(iabscissa)
        self._finalize_plotting()
        return self
    def _enter_plotting(self, fontsize=7):
        """assumes that a figure is open """
        from matplotlib import pyplot
        # interactive_status = matplotlib.is_interactive()
        self.original_fontsize = pyplot.rcParams['font.size']
        # if font size deviates from default, we assume this is on purpose and hence leave it alone
        if pyplot.rcParams['font.size'] == pyplot.rcParamsDefault['font.size']:
            pyplot.rcParams['font.size'] = fontsize
        # was: pyplot.hold(False)
        # pyplot.gcf().clear()  # opens a figure window, if non exists
        pyplot.ioff()
    def _finalize_plotting(self):
        from matplotlib import pyplot
        pyplot.tight_layout(rect=(0, 0, 0.96, 1))
        pyplot.draw()  # update "screen"
        pyplot.ion()  # prevents that the execution stops after plotting
        pyplot.show()
        pyplot.rcParams['font.size'] = self.original_fontsize
    def _xlabel(self, iabscissa=1):
        from matplotlib import pyplot
        pyplot.xlabel('iterations' if iabscissa == 0
                      else 'function evaluations')
    def _plot_x(self, iabscissa=1, x_opt=None, remark=None,
                annotations=None):
        """If ``x_opt is not None`` the difference to x_opt is plotted
        in log scale

        """
        if not hasattr(self, 'x'):
            utils.print_warning('no x-attributed found, use methods ' +
                           'plot_xrecent or plot_mean', 'plot_x',
                           'CMADataLogger')
            return
        if annotations is None:
            annotations = self.persistent_communication_dict.get('variable_annotations')
        from matplotlib.pyplot import plot, semilogy, text, grid, axis, title
        dat = self  # for convenience and historical reasons
        # modify fake last entry in x for line extension-annotation
        if dat.x.shape[1] < 100:
            minxend = int(1.06 * dat.x[-2, iabscissa])
            # write y-values for individual annotation into dat.x
            dat.x[-1, iabscissa] = minxend  # TODO: should be ax[1]
            if x_opt is None:
                idx = np.argsort(dat.x[-2, 5:])
                # idx2 = np.argsort(idx)
                dat.x[-1, 5 + idx] = np.linspace(np.min(dat.x[:, 5:]),
                            np.max(dat.x[:, 5:]), dat.x.shape[1] - 5)
            else: # y-axis is in log
                xdat = np.abs(dat.x[:, 5:] - np.array(x_opt, copy=False))
                idx = np.argsort(xdat[-2, :])
                # idx2 = np.argsort(idx)
                xdat[-1, idx] = np.logspace(np.log10(np.min(abs(xdat[xdat!=0]))),
                            np.log10(np.max(np.abs(xdat))),
                            dat.x.shape[1] - 5)
        else:
            minxend = 0
        self._enter_plotting()
        if x_opt is not None:  # TODO: differentate neg and pos?
            semilogy(dat.x[:, iabscissa], abs(xdat), '-')
        else:
            plot(dat.x[:, iabscissa], dat.x[:, 5:], '-')
        # hold(True)
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        ax[1] -= 1e-6  # to prevent last x-tick annotation, probably superfluous
        if dat.x.shape[1] < 100:
            # yy = np.linspace(ax[2] + 1e-6, ax[3] - 1e-6, dat.x.shape[1] - 5)
            # yyl = np.sort(dat.x[-1,5:])
            if x_opt is not None:
                # semilogy([dat.x[-1, iabscissa], ax[1]], [abs(dat.x[-1, 5:]), yy[idx2]], 'k-')  # line from last data point
                semilogy(np.dot(dat.x[-2, iabscissa], [1, 1]),
                         array([ax[2] * (1+1e-6), ax[3] / (1+1e-6)]), 'k-')
            else:
                # plot([dat.x[-1, iabscissa], ax[1]], [dat.x[-1,5:], yy[idx2]], 'k-') # line from last data point
                plot(np.dot(dat.x[-2, iabscissa], [1, 1]),
                     array([ax[2] + 1e-6, ax[3] - 1e-6]), 'k-')
            # plot(array([dat.x[-1, iabscissa], ax[1]]),
            #      reshape(array([dat.x[-1,5:], yy[idx2]]).flatten(), (2,4)), '-k')
            for i in range(len(idx)):
                # TODOqqq: annotate phenotypic value!?
                # text(ax[1], yy[i], 'x(' + str(idx[i]) + ')=' + str(dat.x[-2,5+idx[i]]))

                text(dat.x[-1, iabscissa], dat.x[-1, 5 + i]
                            if x_opt is None else np.abs(xdat[-1, i]),
                     ('x[' + str(i) + ']=' if annotations is None
                        else str(i) + ':' + annotations[i] + "=")
                     + str(dat.x[-2, 5 + i]))
        i = 2  # find smallest i where iteration count differs (in case the same row appears twice)
        while i < len(dat.f) and dat.f[-i][0] == dat.f[-1][0]:
            i += 1
        title('Object Variables (' +
                (remark + ', ' if remark is not None else '') +
                str(dat.x.shape[1] - 5) + '-D, popsize~' +
                (str(int((dat.f[-1][1] - dat.f[-i][1]) / (dat.f[-1][0] - dat.f[-i][0])))
                    if len(dat.f.T[0]) > 1 and dat.f[-1][0] > dat.f[-i][0] else 'NA')
                + ')')
        self._finalize_plotting()
    def downsampling(self, factor=10, first=3, switch=True, verbose=True):
        """
        rude downsampling of a `CMADataLogger` data file by `factor`,
        keeping also the first `first` entries. This function is a
        stump and subject to future changes. Return self.

        Arguments
        ---------
           - `factor` -- downsampling factor
           - `first` -- keep first `first` entries
           - `switch` -- switch the new logger to the downsampled logger
                original_name+'down'

        Details
        -------
        ``self.name_prefix+'down'`` files are written

        Example
        -------
        ::

            import cma
            cma.downsampling()  # takes outcmaes* files
            cma.plot('outcmaesdown')

        """
        newprefix = self.name_prefix + 'down'
        for name in self.file_names:
            with open(newprefix + name + '.dat', 'wt') as f:
                iline = 0
                cwritten = 0
                for line in open(self.name_prefix + name + '.dat'):
                    if iline < first or iline % factor < 1:
                        f.write(line)
                        cwritten += 1
                    iline += 1
            if verbose and iline > first:
                print('%d' % (cwritten) + ' lines written in ' + newprefix + name + '.dat')
        if switch:
            self.name_prefix += 'down'
        return self

    # ____________________________________________________________
    # ____________________________________________________________
    #
    def disp(self, idx=100):  # r_[0:5,1e2:1e9:1e2,-10:0]):
        """displays selected data from (files written by) the class
        `CMADataLogger`.

        Arguments
        ---------
           `idx`
               indices corresponding to rows in the data file;
               if idx is a scalar (int), the first two, then every idx-th,
               and the last three rows are displayed. Too large index
               values are removed. If ``idx=='header'``, the header
               line is printed.

        Example
        -------
        >>> import cma, numpy as np
        >>> res = cma.fmin(cma.ff.elli, 7 * [0.1], 1, {'verb_disp':1e9})  # generate data
        ...  #doctest: +ELLIPSIS
        (4...
        >>> assert res[1] < 1e-9
        >>> assert res[2] < 4400
        >>> l = cma.evolution_strategy.CMADataLogger()  # == res[-1], logger with default name, "points to" above data
        >>> l.disp([0,-1])  # first and last
        ...  #doctest: +ELLIPSIS
        Iterat Nfevals  function value    axis ratio maxstd  minstd...
        >>> l.disp(20)  # some first/last and every 20-th line
        ...  #doctest: +ELLIPSIS
        Iterat Nfevals  function value    axis ratio maxstd  minstd...
        >>> l.disp(np.r_[0:999999:100, -1]) # every 100-th and last
        ...  #doctest: +ELLIPSIS
        Iterat Nfevals  function value    axis ratio maxstd  minstd...
        >>> l.disp(np.r_[0, -10:0]) # first and ten last
        ...  #doctest: +ELLIPSIS
        Iterat Nfevals  function value    axis ratio maxstd  minstd...
        >>> cma.disp(l.name_prefix, np.r_[0:9999999:100, -10:])  # the same as l.disp(...)
        ...  #doctest: +ELLIPSIS
        Iterat Nfevals  function value    axis ratio maxstd  minstd...

        Details
        -------
        The data line with the best f-value is displayed as last line.

        Use `CMADataLogger.disp` if the logger does not have the default
        name.

        :See: `CMADataLogger.disp`, `CMADataLogger.disp`

        """
        if utils.is_str(idx):
            if idx == 'header':
                self.disp_header()
                return

        filenameprefix = self.name_prefix

        def printdatarow(dat, iteration):
            """print data of iteration i"""
            i = _where(dat.f[:, 0] == iteration)[0][0]
            j = _where(dat.std[:, 0] == iteration)[0][0]
            print('%5d' % (int(dat.f[i, 0])) + ' %6d' % (int(dat.f[i, 1])) + ' %.14e' % (dat.f[i, 5]) +
                  ' %5.1e' % (dat.f[i, 3]) +
                  ' %6.2e' % (max(dat.std[j, 5:])) + ' %6.2e' % min(dat.std[j, 5:]))

        dat = old_CMADataLogger(filenameprefix).load()
        ndata = dat.f.shape[0]

        # map index to iteration number, is difficult if not all iteration numbers exist
        # idx = idx[_where(map(lambda x: x in dat.f[:,0], idx))[0]] # TODO: takes pretty long
        # otherwise:
        if idx is None:
            idx = 100
        if np.isscalar(idx):
            # idx = np.arange(0, ndata, idx)
            if idx:
                idx = np.r_[0, 1, idx:ndata - 3:idx, -3:0]
            else:
                idx = np.r_[0, 1, -3:0]

        idx = array(idx)
        idx = idx[idx < ndata]
        idx = idx[-idx <= ndata]
        iters = dat.f[idx, 0]
        idxbest = np.argmin(dat.f[:, 5])
        iterbest = dat.f[idxbest, 0]

        if len(iters) == 1:
            printdatarow(dat, iters[0])
        else:
            self.disp_header()
            for i in iters:
                printdatarow(dat, i)
            self.disp_header()
            printdatarow(dat, iterbest)
        sys.stdout.flush()
    def disp_header(self):
        heading = 'Iterat Nfevals  function value    axis ratio maxstd  minstd'
        print(heading)

    # end class CMADataLogger

_old_last_figure_number = 324
def _old_plot(name=None, fig=None, abscissa=1, iteridx=None,
         plot_mean=False,
         foffset=1e-19, x_opt=None, fontsize=7, downsample_to=3e3):
    """
    plot data from files written by a `CMADataLogger`,
    the call ``cma.plot(name, **argsdict)`` is a shortcut for
    ``cma.CMADataLogger(name).plot(**argsdict)``

    Arguments
    ---------
    `name`
        name of the logger, filename prefix, None evaluates to
        the default 'outcmaes'
    `fig`
        filename or figure number, or both as a tuple (any order)
    `abscissa`
        0==plot versus iteration count,
        1==plot versus function evaluation number
    `iteridx`
        iteration indices to plot

    Return `None`

    Examples
    --------
    ::

       cma.plot()  # the optimization might be still
                   # running in a different shell
       cma.s.figsave('fig325.png')
       cma.s.figclose()

       cdl = cma.CMADataLogger().downsampling().plot()
       # in case the file sizes are large

    Details
    -------
    Data from codes in other languages (C, Java, Matlab, Scilab) have the same
    format and can be plotted just the same.

    :See also: `CMADataLogger`, `CMADataLogger.plot`

    """
    global _old_last_figure_number
    if not fig:
        _old_last_figure_number += 1
        fig = _old_last_figure_number
    if isinstance(fig, (int, float)):
        _old_last_figure_number = fig
    return old_CMADataLogger(name).plot(fig, abscissa, iteridx, plot_mean, foffset,
                             x_opt, fontsize, downsample_to)

def _old_disp(name=None, idx=None):
    """displays selected data from (files written by) the class
    `CMADataLogger`.

    The call ``cma.disp(name, idx)`` is a shortcut for
    ``cma.CMADataLogger(name).disp(idx)``.

    Arguments
    ---------
    `name`
        name of the logger, filename prefix, `None` evaluates to
        the default ``'outcmaes'``
    `idx`
        indices corresponding to rows in the data file; by
        default the first five, then every 100-th, and the last
        10 rows. Too large index values are removed.

    The best ever observed iteration is also printed by default.

    Examples
    --------
    ::

       import cma
       from numpy import r_
       # assume some data are available from previous runs
       cma.disp(None, r_[0, -1])  # first and last
       cma.disp(None, r_[0:int(1e9):100, -1]) # every 100-th and last
       cma.disp(idx=r_[0, -10:0]) # first and ten last
       cma.disp(idx=r_[0:int(1e9):1000, -10:0])

    :See also: `CMADataLogger.disp`

    """
    return old_CMADataLogger(name if name else old_CMADataLogger.default_prefix
                         ).disp(idx)

# END cmaplt.py

