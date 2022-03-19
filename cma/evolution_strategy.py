# -*- coding: utf-8 -*-
"""CMA-ES (evolution strategy), the main sub-module of `cma` providing
in particular `CMAOptions`, `CMAEvolutionStrategy`, and `fmin2`
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
# from builtins import ...
from .utilities.python3for2 import range  # redefine range in Python 2

import sys
import os
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
from .utilities import utils as _utils
from . import constraints_handler as _constraints_handler
from .constraints_handler import BoundNone, BoundPenalty, BoundTransform, AugmentedLagrangian
from .recombination_weights import RecombinationWeights
from .logger import CMADataLogger  # , disp, plot
from .utilities.utils import BlancClass as _BlancClass
from .utilities.utils import rglen  #, global_verbosity
from .utilities.utils import pprint
from .utilities.utils import seval as eval
from .utilities.utils import SolutionDict as _SolutionDict
from .utilities.math import Mh
from .sigma_adaptation import *
from . import restricted_gaussian_sampler as _rgs

_where = np.nonzero  # to make pypy work, this is how where is used here anyway
del division, print_function, absolute_import  #, unicode_literals, with_statement

class InjectionWarning(UserWarning):
    """Injected solutions are not passed to tell as expected"""

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
        self.tolfunrel = 0  ## [1e-17, ~1e-2] l  #v termination criterion: relative tolerance in function value',
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
    return f is not None and not utils.is_nan(f)


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

def cma_default_options_(  # to get keyword completion back
    # the follow string arguments are evaluated if they do not contain "filename"
    AdaptSigma='True  # or False or any CMAAdaptSigmaBase class e.g. CMAAdaptSigmaTPA, CMAAdaptSigmaCSA',
    CMA_active='True  # negative update, conducted after the original update',
#    CMA_activefac='1  # learning rate multiplier for active update',
    CMA_active_injected='0  #v weight multiplier for negative weights of injected solutions',
    CMA_cmean='1  # learning rate for the mean value',
    CMA_const_trace='False  # normalize trace, 1, True, "arithm", "geom", "aeig", "geig" are valid',
    CMA_diagonal='0*100*N/popsize**0.5  # nb of iterations with diagonal covariance matrix,'\
                                        ' True for always',  # TODO 4/ccov_separable?
    CMA_eigenmethod='np.linalg.eigh  # or cma.utilities.math.eig or pygsl.eigen.eigenvectors',
    CMA_elitist='False  #v or "initial" or True, elitism likely impairs global search performance',
    CMA_injections_threshold_keep_len='1  #v keep length if Mahalanobis length is below the given relative threshold',
    CMA_mirrors='popsize < 6  # values <0.5 are interpreted as fraction, values >1 as numbers (rounded),'\
                              ' for `True` about 0.16 is used',
    CMA_mirrormethod='2  # 0=unconditional, 1=selective, 2=selective with delay',
    CMA_mu='None  # parents selection parameter, default is popsize // 2',
    CMA_on='1  # multiplier for all covariance matrix updates',
    # CMA_sample_on_sphere_surface='False  #v replaced with option randn=cma.utilities.math.randhss, all mutation vectors have the same length, currently (with new_sampling) not in effect',
    CMA_sampler='None  # a class or instance that implements the interface of'\
                       ' `cma.interfaces.StatisticalModelSamplerWithZeroMeanBaseClass`',
    CMA_sampler_options='{}  # options passed to `CMA_sampler` class init as keyword arguments',
    CMA_rankmu='1.0  # multiplier for rank-mu update learning rate of covariance matrix',
    CMA_rankone='1.0  # multiplier for rank-one update learning rate of covariance matrix',
    CMA_recombination_weights='None  # a list, see class RecombinationWeights, overwrites CMA_mu and popsize options',
    CMA_dampsvec_fac='np.Inf  # tentative and subject to changes, 0.5 would be a "default" damping for sigma vector update',
    CMA_dampsvec_fade='0.1  # tentative fading out parameter for sigma vector update',
    CMA_teststds='None  # factors for non-isotropic initial distr. of C, mainly for test purpose, see CMA_stds for production',
    CMA_stds='None  # multipliers for sigma0 in each coordinate, not represented in C, better use `cma.ScaleCoordinates` instead',
    # CMA_AII='False  # not yet tested',
    CSA_dampfac='1  #v positive multiplier for step-size damping, 0.3 is close to optimal on the sphere',
    CSA_damp_mueff_exponent='0.5  # zero would mean no dependency of damping on mueff, useful with CSA_disregard_length option',
    CSA_disregard_length='False  #v True is untested, also changes respective parameters',
    CSA_clip_length_value='None  #v poorly tested, [0, 0] means const length N**0.5, [-1, 1] allows a variation of +- N/(N+2), etc.',
    CSA_squared='False  #v use squared length for sigma-adaptation ',
    BoundaryHandler='BoundTransform  # or BoundPenalty, unused when ``bounds in (None, [None, None])``',
    bounds='[None, None]  # lower (=bounds[0]) and upper domain boundaries, each a scalar or a list/vector',
     # , eval_parallel2='not in use {"processes": None, "timeout": 12, "is_feasible": lambda x: True} # distributes function calls to processes processes'
     # 'callback='None  # function or list of functions called as callback(self) at the end of the iteration (end of tell)', # only necessary in fmin and optimize
    conditioncov_alleviate='[1e8, 1e12]  # when to alleviate the condition in the coordinates and in main axes',
    eval_final_mean='True  # evaluate the final mean, which is a favorite return candidate',
    fixed_variables='None  # dictionary with index-value pairs like {0:1.1, 2:0.1} that are not optimized',
    ftarget='-inf  #v target function value, minimization',
    integer_variables='[]  # index list, invokes basic integer handling: prevent std dev to become too small in the given variables',
    is_feasible='is_feasible  #v a function that computes feasibility, by default lambda x, f: f not in (None, np.NaN)',
    maxfevals='inf  #v maximum number of function evaluations',
    maxiter='100 + 150 * (N+3)**2 // popsize**0.5  #v maximum number of iterations',
    mean_shift_line_samples='False #v sample two new solutions colinear to previous mean shift',
    mindx='0  #v minimal std in any arbitrary direction, cave interference with tol*',
    minstd='0  #v minimal std (scalar or vector) in any coordinate direction, cave interference with tol*',
    maxstd='inf  #v maximal std in any coordinate direction',
    pc_line_samples='False #v one line sample along the evolution path pc',
    popsize='4 + 3 * np.log(N)  # population size, AKA lambda, int(popsize) is the number of new solution per iteration',
    popsize_factor='1  # multiplier for popsize, convenience option to increase default popsize',
    randn='np.random.randn  #v randn(lam, N) must return an np.array of shape (lam, N), see also cma.utilities.math.randhss',
    scaling_of_variables='None  # deprecated, rather use fitness_transformations.ScaleCoordinates instead (or possibly CMA_stds). Scale for each variable in that effective_sigma0 = sigma0*scaling. Internally the variables are divided by scaling_of_variables and sigma is unchanged, default is `np.ones(N)`',
    seed='time  # random number seed for `numpy.random`; `None` and `0` equate to `time`,'\
                ' `np.nan` means "do nothing", see also option "randn"',
    signals_filename='cma_signals.in  # read versatile options from this file (use `None` or `""` for no file)'\
                                      ' which contains a single options dict, e.g. ``{"timeout": 0}`` to stop,'\
                                      ' string-values are evaluated, e.g. "np.inf" is valid',
    termination_callback='[]  #v a function or list of functions returning True for termination, called in'\
                              ' `stop` with `self` as argument, could be abused for side effects',
    timeout='inf  #v stop if timeout seconds are exceeded, the string "2.5 * 60**2" evaluates to 2 hours and 30 minutes',
    tolconditioncov='1e14  #v stop if the condition of the covariance matrix is above `tolconditioncov`',
    tolfacupx='1e3  #v termination when step-size increases by tolfacupx (diverges). That is, the initial'\
                     ' step-size was chosen far too small and better solutions were found far away from the initial solution x0',
    tolupsigma='1e20  #v sigma/sigma0 > tolupsigma * max(eivenvals(C)**0.5) indicates "creeping behavior" with usually'\
                       ' minor improvements',
    tolflatfitness='1  #v iterations tolerated with flat fitness before termination',
    tolfun='1e-11  #v termination criterion: tolerance in function value, quite useful',
    tolfunhist='1e-12  #v termination criterion: tolerance in function value history',
    tolfunrel='0  #v termination criterion: relative tolerance in function value:'\
                   ' Delta f current < tolfunrel * (median0 - median_min)',
    tolstagnation='int(100 + 100 * N**1.5 / popsize)  #v termination if no improvement over tolstagnation iterations',
    tolx='1e-11  #v termination criterion: tolerance in x-changes',
    transformation='''None  # depreciated, use cma.fitness_transformations.FitnessTransformation instead.
            [t0, t1] are two mappings, t0 transforms solutions from CMA-representation to f-representation (tf_pheno),
            t1 is the (optional) back transformation, see class GenoPheno''',
    typical_x='None  # used with scaling_of_variables',
    updatecovwait='None  #v number of iterations without distribution update, name is subject to future changes',  # TODO: rename: iterwaitupdatedistribution?
    verbose='3  #v verbosity e.g. of initial/final message, -1 is very quiet, -9 maximally quiet, may not be fully implemented',
    verb_append='0  # initial evaluation counter, if append, do not overwrite output files',
    verb_disp='100  #v verbosity: display console output every verb_disp iteration',
    verb_filenameprefix=CMADataLogger.default_prefix + '  # output path (folder) and filenames prefix',
    verb_log='1  #v verbosity: write data to files every verb_log iteration, writing can be'\
                  ' time critical on fast to evaluate functions',
    verb_log_expensive='N * (N <= 50)  # allow to execute eigendecomposition for logging every verb_log_expensive iteration,'\
                                       ' 0 or False for never',
    verb_plot='0  #v in fmin2(): plot() is called every verb_plot iteration',
    verb_time='True  #v output timings on console',
    vv='{}  #? versatile set or dictionary for hacking purposes, value found in self.opts["vv"]'
    ):
    """use this function to get keyword completion for `CMAOptions`.

    ``cma.CMAOptions('substr')`` provides even substring search.

    returns default options as a `dict` (not a `cma.CMAOptions` `dict`).
    """
    return dict(locals())  # is defined before and used by CMAOptions, so it can't return CMAOptions

cma_default_options = cma_default_options_()  # will later be reassigned as CMAOptions(dict)
cma_versatile_options = tuple(sorted(k for (k, v) in cma_default_options.items()
                                     if v.find(' #v ') > 0))
cma_allowed_options_keys = dict([s.lower(), s] for s in cma_default_options)

def safe_str(s):
    """return a string safe to `eval` or raise an exception.

    Selected words and chars are considered safe such that all default
    string-type option values from `CMAOptions()` pass. This function is
    implemented for convenience, to keep the default option format
    backwards compatible, and to be able to pass, for example, `3 * N`.
    Function or class names other than those from the default values cannot
    be passed as strings (any more) but only as the function or class
    themselves.
    """
    from . import purecma
    return purecma.safe_str(s.split('#')[0],
                            dict([k, k] for k in
                                 ['True', 'False', 'None',
                                 'N', 'dim', 'popsize', 'int', 'np.Inf', 'inf',
                                 'np.log', 'np.random.randn', 'time',
                                 # 'cma_signals.in', 'outcmaes/',
                                 'BoundTransform', 'is_feasible', 'np.linalg.eigh',
                                 '{}', '/'])
                            ).replace('N one', 'None'  # if purecma.safe_str could avoid substring substitution, this would not be necessary
                                      ).replace('/  /', '//')

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

    Option values can be "written" in a string and, when passed to `fmin2`
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

    :See also: `fmin2` (), `CMAEvolutionStrategy`, `_CMAParameters`

    """

    # @classmethod # self is the class, not the instance
    # @property
    # def default(self):
    #     """returns all options with defaults"""
    #     return fmin([],[])

    @staticmethod
    def defaults():
        """return a dictionary with default option values and description"""
        return cma_default_options
        # return dict((str(k), str(v)) for k, v in cma_default_options_().items())
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
        return cma_versatile_options
        # return tuple(sorted(i[0] for i in list(CMAOptions.defaults().items()) if i[1].find(' #v ') > 0))
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
                raise ValueError('%s is not a valid option.\n'
                                 'Similar valid options are %s\n'
                                 'Valid options are %s' %
                                (key, str(list(cma_default_options(key))),
                                 str(list(cma_default_options))))
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
                    val = eval(safe_str(val), globals(), loc)
            # invoke default
            # TODO: val in ... fails with array type, because it is applied element wise!
            # elif val in (None,(),[],{}) and default is not None:
            elif val is None and default is not None:
                val = eval(safe_str(default), globals(), loc)
        except:
            pass  # slighly optimistic: the previous is bug-free
        return val

    def corrected_key(self, key):
        """return the matching valid key, if ``key.lower()`` is a unique
        starting sequence to identify the valid key, ``else None``

        """
        matching_keys = []
        key = key.lower()  # this was somewhat slow, so it is speed optimized now
        if key in cma_allowed_options_keys:
            return cma_allowed_options_keys[key]
        for allowed_key in cma_allowed_options_keys:
            if allowed_key.startswith(key):
                if len(matching_keys) > 0:
                    return None
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
            defaults = cma_default_options_()
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

cma_default_options = CMAOptions(cma_default_options_())

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
            self.gp.pheno(self.mean, into_bounds=self.boundary_handler.repair),
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
    >>> assert es.result[1] < 1e-8

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
    >>> assert es.result[2] < 15000
    >>> assert cma.s.Mh.vequals_approximately(es.result[0], 12 * [1], 1e-5)
    >>> assert len(es.result) == 8

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

    def __init__(self, x0, sigma0, inopts=None):
        """see class `CMAEvolutionStrategy`

        """
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
            warnings.warn("""
                Option 'bounds' ignored because a BoundaryHandler *instance* was found.
                Consider to pass only the desired BoundaryHandler class. """)
        if not self.boundary_handler.has_bounds():
            self.boundary_handler = BoundNone()  # just a little faster and well defined
        elif not self.boundary_handler.is_in_bounds(self.x0):
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

        # set self.mean to geno(x0)
        tf_geno_backup = self.gp.tf_geno
        if self.gp.tf_pheno and self.gp.tf_geno is None:
            self.gp.tf_geno = lambda x: x  # a hack to avoid an exception
            warnings.warn(
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
        self.callbackstop = ()
        "    return values of callbacks, used like ``if any(callbackstop)``"
        self.fit = _BlancClass()
        self.fit.fit = []  # not really necessary
        self.fit.hist = []  # short history of best
        self.fit.histbest = []  # long history of best
        self.fit.histmedian = []  # long history of median
        self.fit.median = None
        self.fit.median0 = None
        self.fit.median_min = np.inf
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
            raise ValueError('Could not digest initial solution argument x0=%s.\n'
                             'Optimization in 1-D is not supported (code was never tested)'
                             % str(self.x0))
        try:
            self.x0.resize(self.x0.shape[0])  # 1-D array, not really necessary?!
        except NotImplementedError:
            pass
    
    def _copy_light(self, sigma=None, inopts=None):
        """tentative copy of self, versatile (interface and functionalities may change).
        
        `sigma` overwrites the original initial `sigma`.
        `inopts` allows to overwrite any of the original options.

        This copy may not work as expected depending on the used sampler.
        
        Copy mean and sample distribution parameters and input options. Do
        not copy evolution paths, termination status or other state variables.

        >>> import cma
        >>> es = cma.CMAEvolutionStrategy(3 * [1], 0.1,
        ...          {'verbose':-9}).optimize(cma.ff.elli, iterations=10)
        >>> es2 = es._copy_light()
        >>> assert es2.sigma == es.sigma
        >>> assert sum((es.sm.C - es2.sm.C).flat < 1e-12)
        >>> es3 = es._copy_light(sigma=10)
        >>> assert es3.sigma == es3.sigma0 == 10
        >>> es4 = es._copy_light(inopts={'CMA_on': False})
        >>> assert es4.sp.c1 == es4.sp.cmu == 0

        """
        if sigma is None:
            sigma = self.sigma
        opts = dict(self.inopts)
        if inopts is not None:
            opts.update(inopts)
        es = type(self)(self.mean[:], sigma, opts)
        es.sigma_vec = transformations.DiagonalDecoding(self.sigma_vec.scaling)
        try: es.sm.C = self.sm.C.copy()
        except: warnings.warn("self.sm.C.copy failed")
        es.sm.update_now(-1)  # make B and D consistent with C
        es._updateBDfromSM()
        return es    
    
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
                        fpenalty = lambda x: self.boundary_handler.__call__(
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
                    utils.print_warning(
                        "mean_shift_samples, but the first two solutions"
                        " are not mirrors.",
                        "ask_geno", "CMAEvolutionStrategy",
                        self.countiter)
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
        return np.sum(self.randn(1, len(y))[0]**2)**0.5 / self.mahalanobis_norm(y)


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
        sampled. By default
        ``self.is_feasible == cma.feasible == lambda x, f: f not in (None, np.NaN)``.
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
        if lam < sp.weights.mu:  # rather decrease cmean instead of having mu > lambda//2
            raise ValueError('not enough solutions passed to function tell (mu>lambda)')

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
        fit.idx = np.argsort(array(fit.bndpen) + array(function_values))
        fit.fit = array(function_values, copy=False)[fit.idx]

        # update output data TODO: this is obsolete!? However: need communicate current best x-value?
        # old: out['recent_x'] = self.gp.pheno(pop[0])
        # self.out['recent_x'] = array(solutions[fit.idx[0]])  # TODO: change in a data structure(?) and use current as identify
        # self.out['recent_f'] = fit.fit[0]

        # fitness histories
        fit.hist.insert(0, fit.fit[0])  # caveat: this may neither be the best nor the best in-bound fitness, TODO
        fit.median = (fit.fit[self.popsize // 2] if self.popsize % 2
                      else np.mean(fit.fit[self.popsize // 2 - 1: self.popsize // 2 + 1]))
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
            # print(' injected solutions', tuple(self._injected_solutions_archive.values()))
            for i, x in enumerate(pop):
                try:
                    self._injected_solutions_archive.pop(x)
                    # self.gp.repaired_solutions.pop(x)
                except KeyError:
                    pass  # print(i)
                else:
                    # print(i + 1, '-th weight set to zero')
                    # sampler_weights[i + 1] = 0  # weight index 0 is for pc
                    sampler_weights[i + 1] *= self.opts['CMA_active_injected']  # weight index 0 is for pc
            for s in list(self._injected_solutions_archive):
                if self._injected_solutions_archive[s]['iteration'] < self.countiter - 2:
                    warnings.warn("""orphanated injected solution %s
                        This could be a bug in the calling order/logics or due to
                        a too small popsize used in `ask()` or when only using
                        `ask(1)` repeatedly. Please check carefully.
                        In case this is desired, the warning can be surpressed with
                        ``warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)``
                        """ % str(self._injected_solutions_archive.pop(s)),
                        InjectionWarning)
            assert len(sampler_weights) == len(pop_zero) + 1
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
                raise RuntimeError("A sampler variance has become negative "
                                   "after the update, this must be considered as a bug.\n"
                                   "Variances `self.sm.variances`=%s" % str(self.sm.variances))
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
        r0 = _utils.ranks(vals + list(self.fit.fit))
        r1 = _utils.ranks(vals + list(function_values))
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
        x, f, evals = self.best.get()
        return CMAEvolutionStrategyResult(
            x,
            f,
            evals,
            self.countevals,
            self.countiter,
            self.gp.pheno(self.mean, into_bounds=self.boundary_handler.repair),
            self.gp.scales * self.sigma * self.sigma_vec.scaling *
                self.dC**0.5,
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

        print('final/bestever f-value = %e %e' % (self.best.last.f,
                                                  fbestever))
        if self.N < 9:
            print('incumbent solution: ' + str(list(self.gp.pheno(self.mean, into_bounds=self.boundary_handler.repair))))
            print('std deviation: ' + str(list(self.sigma * self.sigma_vec.scaling * np.sqrt(self.dC) * self.gp.scales)))
        else:
            print('incumbent solution: %s ...]' % (str(self.gp.pheno(self.mean, into_bounds=self.boundary_handler.repair)[:8])[:-1]))
            print('std deviations: %s ...]' % (str((self.sigma * self.sigma_vec.scaling * np.sqrt(self.dC) * self.gp.scales)[:8])[:-1]))
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
        if condition and np.isfinite(condition) and np.max(self.dC) / np.min(self.dC) > condition:
            # allows for much larger condition numbers, if axis-parallel
            if hasattr(self, 'sm') and isinstance(self.sm, sampler.GaussFullSampler):
                old_coordinate_condition = np.max(self.dC) / np.min(self.dC)
                old_condition = self.sm.condition_number
                factors = self.sm.to_correlation_matrix()
                self.sigma_vec *= factors
                self.pc /= factors
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
    def plot(self, *args, **kwargs):
        """plot current state variables using `matplotlib`.

        Details: calls `self.logger.plot`.
        """
        try:
            self.logger.plot(*args, **kwargs)
        except AttributeError:
            utils.print_warning('plotting failed, no logger attribute found')
        except:
            utils.print_warning('plotting failed with: {}'.format(sys.exc_info()),
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
                      es.best.f < opts['ftarget'])
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
                        len(es.fit.histbest) > 100 and 2 * l < len(es.fit.histbest) and
                        np.median(es.fit.histmedian[:l]) >= np.median(es.fit.histmedian[l:2 * l]) and
                        np.median(es.fit.histbest[:l]) >= np.median(es.fit.histbest[l:2 * l]))
            # iiinteger: stagnation termination can prevent to find the optimum

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
                if es.fit.fit[0] < es.fit.fit[int(0.75 * es.popsize)]:
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
        self.popsize = None  # type: int
        """number of candidation solutions per iteration, AKA population size"""
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
            popsize *= opts['popsize_factor']
        ## meta_parameters.lambda_exponent == 0.0
        popsize = int(popsize + N** 0.0 - 1)

        # set weights
        if utils.is_(opts['CMA_recombination_weights']):
            sp.weights = RecombinationWeights(opts['CMA_recombination_weights'])
            popsize = len(sp.weights)
        elif opts['CMA_mu']:
            sp.weights = RecombinationWeights(2 * opts['CMA_mu'])
            while len(sp.weights) < popsize:
                sp.weights.insert(sp.weights.mu, 0.0)  # doesn't change mu or mueff
        else:  # default
            sp.weights = RecombinationWeights(popsize)
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
         callback=None):
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
         callback)
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
        elitist) and the final evaluations unless
        ``not callable(objective_function)``. If ``parallel_objective``
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
        if eval(safe_str(options['verb_disp'])) > 0:
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
    ...             options={'termination_callback': lambda es: -1e-5 < sum(es.mean**2) - 1 < 1e-5,
    ...                      'seed':1, 'verbose':-9})
    >>> es.best_feasible.f < 1 + 1e-5
    True
    >>> ".info attribute dictionary keys: {}".format(sorted(es.best_feasible.info))
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
    volume, find-feasible are off by default.

    `kwargs_confit` are keyword arguments to instantiate
    `constraints_handler.ConstrainedFitnessAL` which is optimized and
    returned as `objective_function` attribute in the second return
    argument (type `CMAEvolutionStrategy`).

    Other and further keyword arguments are passed (in ``**kwargs_fmin``)
    to `cma.fmin2`.

    Consider using `ConstrainedFitnessAL` directly instead of `fmin_con2`.

"""
    if isinstance(find_feasible_first, dict) or isinstance(find_feasible_last, dict):
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
