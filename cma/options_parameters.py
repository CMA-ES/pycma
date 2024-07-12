# -*- coding: utf-8 -*-
"""Parameters and Options for CMA-ES.
"""
from math import inf  # used to eval options
import warnings as _warnings
import numpy as np
from . import constraints_handler
from . import utilities
from .utilities import utils
from .logger import CMADataLogger
from .recombination_weights import RecombinationWeights

integer_std_lower_bound_factor1 = 1
'''factor used in `integer_std_lower_bound` as multiplier to ``mueff/N``'''
integer_std_lower_bound_factor2 = 1
'''factor used in `integer_std_lower_bound` as multiplier to
   `integer_std_lower_bound_limit_when_mu_is_large`'''
integer_std_lower_bound_limit = 0.2

integer_active_limit_std = inf
'''limit coordinate stds of solutions in C update, by default off, may go away'''
integer_active_limit_recombination_weight_condition = None
'''None or True or a function float->bool, None -> limit only negative updates'''

default_restart_number_if_not_zero = 9

def cma_default_options_(  # to get keyword completion back
    # the follow string arguments are evaluated if they do not contain "filename"
    AdaptSigma='True  # or False or any CMAAdaptSigmaBase class e.g. CMAAdaptSigmaTPA, CMAAdaptSigmaCSA',
    CMA_active='True  # negative update, conducted after the original update',
    #  CMA_activefac='1  # learning rate multiplier for active update',
    CMA_active_injected='0  #v weight multiplier for negative weights of injected solutions',
    CMA_cmean='1  # learning rate for the mean value',
    CMA_const_trace='False  # normalize trace, 1, True, "arithm", "geom", "aeig", "geig" are valid',
    CMA_diagonal='0*100*N/popsize**0.5  # nb of iterations with diagonal covariance matrix,'\
                                        ' True for always',  # TODO 4/ccov_separable?
    CMA_diagonal_decoding='0  # learning rate multiplier for additional diagonal update',
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
    CMA_dampsvec_fac='np.inf  # tentative and subject to changes, 0.5 would be a "default" damping for sigma vector update',
    CMA_dampsvec_fade='0.1  # tentative fading out parameter for sigma vector update',
    CMA_teststds='None  # factors for non-isotropic initial distr. of C, mainly for test purpose, see CMA_stds for production',
    CMA_stds='None  # multipliers for sigma0 in each coordinate (not represented in C), or use `cma.ScaleCoordinates` instead',
    # CMA_AII='False  # not yet tested',
    CSA_dampfac='1  #v positive multiplier for step-size damping, 0.3 is close to optimal on the sphere',
    CSA_damp_mueff_exponent='None  # exponent for mueff/N, by default 0.5 and 1 if CSA_squared,'
        ' zero means no dependency of damping on mueff, useful with CSA_disregard_length option',
    CSA_disregard_length='False  #v True is untested, also changes respective parameters',
    CSA_clip_length_value='None  #v poorly tested, [0, 0] means const length N**0.5, [-1, 1] allows a variation of +- N/(N+2), etc.',
    CSA_squared='False  #v use squared length for sigma-adaptation ',
    CSA_invariant_path='False  #v pc is invariant and ps (default) is unbiased',
    stall_sigma_change_on_divergence_iterations='False  #v number of iterations of median'
        ' worsenings threshold at which the sigma change is stalled; the default may become 2',
    BoundaryHandler='BoundTransform  # or BoundPenalty, unused when ``bounds in (None, [None, None])``',
    bounds='[None, None]  # lower (=bounds[0]) and upper domain boundaries, each a scalar or a list/vector',
     # , eval_parallel2='not in use {"processes": None, "timeout": 12, "is_feasible": lambda x: True} # distributes function calls to processes processes'
     # 'callback='None  # function or list of functions called as callback(self) at the end of the iteration (end of tell)', # only necessary in fmin and optimize
    conditioncov_alleviate='[1e8, 1e12]  # when to alleviate the condition in the coordinates and in main axes',
    eval_final_mean='True  # evaluate the final mean, which is a favorite return candidate',
    fixed_variables='None  # dictionary with index-value pairs like {0:1.1, 2:0.1} that are not optimized',
    ftarget='-inf  #v target function value, minimization',
    integer_variables='[]  # index list, invokes basic integer handling by setting minstd of integer variables if it was not given and by integer centering',
    is_feasible='is_feasible  #v a function that computes feasibility, by default lambda x, f: f not in (None, np.nan)',
    maxfevals='inf  #v maximum number of function evaluations',
    maxiter='100 + 150 * (N+3)**2 // popsize**0.5  #v maximum number of iterations',
    mean_shift_line_samples='False #v sample two new solutions colinear to previous mean shift',
    mindx='0  #v minimal std in any arbitrary direction, cave interference with tol*',
    minstd='0  #v minimal std (scalar or vector) in any coordinate direction, cave interference with tol*',
    maxstd='None  #v maximal std (scalar or vector) in any coordinate direction',
    maxstd_boundrange='1/3  # maximal std relative to bound_range per coordinate, overruled by maxstd',
    pc_line_samples='False #v one line sample along the evolution path pc',
    popsize='4 + 3 * np.log(N)  # population size, AKA lambda, int(popsize) is the number of new solution per iteration',
    popsize_factor='1  # multiplier for popsize, convenience option to increase default popsize',
    randn='np.random.randn  #v randn(lam, N) must return an np.array of shape (lam, N), see also cma.utilities.math.randhss',
    scaling_of_variables='None  # deprecated, rather use fitness_transformations.ScaleCoordinates instead (or CMA_stds). WAS: Scale for each variable in that effective_sigma0 = sigma0*scaling. Internally the variables are divided by scaling_of_variables and sigma is unchanged, default is `np.ones(N)`',
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
    tolxstagnation='[1e-9, 20, 0.1]  #v termination thresholds for Delta of [mean, iterations, iterations fraction], the latter two are summed; '
                   'trigger termination if Dmean stays below the threshold over Diter iterations, '
                   'pass `False` or a negative value to turn off tolxstagnation',
    tolx='1e-11  #v termination criterion: tolerance in x-changes',
    transformation='None  # deprecated, use a wrapper like those in cma.fitness_transformations instead.',
    # WAS:
    # '''   t0, t1] are two mappings, t0 transforms solutions from CMA-representation to f-representation (tf_pheno),
    #       t1 is the (optional) back transformation, see class GenoPheno''',
    typical_x='None  # deprecated, use `cma.fitness_transformations.Shifted` instead',
    updatecovwait='None  #v number of iterations without distribution update, name is subject to future changes',  # TODO: rename: iterwaitupdatedistribution?
    verbose='3  #v verbosity e.g. of initial/final message, -1 is very quiet, -9 maximally quiet, may not be fully implemented',
    verb_append='0  # initial evaluation counter, if append, do not overwrite output files',
    verb_disp='100  #v verbosity: display console output every verb_disp iteration',
    verb_disp_overwrite='inf  #v start overwriting after given iteration',
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

def integer_std_lower_bound(N, mueff, N_int=None, binary=False):
    """can be reassigned/overwritten like a global "parameter setting"

    This function returns the minimum of three bounds, an absolute bound
    (default 0.2), mueff / N, and a bound computed from the normal quantile
    (when the mean is assumed to be in the domain middle) which takes
    ``ptarget=integer_lower_bound_target_probability(N, N_int)`` (which
    should be roughly 1/N) as input. This bound becomes < 0.2 only when
    ptarget < 1/152 (==> < -2.5-sigma).
    """
    ptarget = integer_lower_bound_target_probability(N, N_int if N_int else N)
    ppf = utilities.math.normal_ppf(ptarget / (1 if binary else 2))  # AKA sigma(ptail)
    return min((integer_std_lower_bound_limit,
                integer_std_lower_bound_factor1 * mueff / N,  # TODO: this is too small for 100D-leadingones, add (1+tanh(N_int / N)) / 2 ?
                integer_std_lower_bound_factor2 * 0.5 / -ppf  # p=1/152.85 -> sigma=0.2
               ))

def integer_lower_bound_target_probability(N, N_int):
    """target probability for an integer mutation assuming a centered mean

    and no boundaries (two-tailed).

    ``2 / (N + N_int)`` should keep at least 37% of the solutions unaffected.

    Details: from ``ptarget`` we can compute ``sigma = 1 / PPF(ptarget/2) /
    2`` where ``PPF`` is the quantile function of the standard normal
    distribution and the first ``2`` accounts for two-sided sampling of the
    tails and the second ``2`` is needed because the distance to the value
    domain bound is 1/2.
    """
    if N_int > N:
        raise ValueError("{0}=N_int > N = {1} is not a valid cases".format(N_int, N))
    return 2 / (2 + N + N_int)  # ad hoc setting, not validated

def amend_restarts_parameter(restarts):
    """return a `dict` with ``'maxrestarts'`` and ``'maxfevals'`` as keys.

    `restarts` is a parameter to ``cma.fmin*``, see `cma.fmin`.
    """
    if restarts is True:
        restarts = {'maxrestarts': default_restart_number_if_not_zero}
    elif not restarts:
        restarts = {'maxrestarts': 0}
    if not isinstance(restarts, dict):  # kinda assume that restart is an int
        restarts = {'maxrestarts': restarts}
    restarts.setdefault('maxrestarts', default_restart_number_if_not_zero)
    restarts.setdefault('maxfevals', np.inf)
    return restarts

def is_feasible(x, f):
    """default to check feasibility of f-values.

    Used for rejection sampling in method `ask_and_eval`.

    :See also: CMAOptions, ``CMAOptions('feas')``.
    """
    return f is not None and not utils.is_nan(f)

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
                                 'N', 'dim', 'popsize', 'int', 'np.inf', 'inf',
                                 'np.log', 'np.random.randn', 'time',
                                 # 'cma_signals.in', 'outcmaes/',
                                 'BoundTransform', 'is_feasible', 'np.linalg.eigh',
                                 '{}', '/'])
                            ).replace('N one', 'None'  # if purecma.safe_str could avoid substring substitution, this would not be necessary
                                      ).replace('/  /', '//')

options_environment = {
    name: getattr(constraints_handler, name) for name in
       ['BoundNone', 'BoundPenalty', 'BoundTransform', 'AugmentedLagrangian']}

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

    :See also: `fmin2` (), `CMAEvolutionStrategy`, `CMAParameters`

    """
    _ps_for_pc = False
    _hsig = True  # False == never toggle hsig
    _stationary_sphere = False  # True or callable like lambda x: cma.ff.elli(x)**0.5
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
        global_env = dict(globals())
        global_env.update(options_environment)

        try:
            val = self[key]
        except Exception:
            return self.match(key)

        if loc is None:
            loc = self  # TODO: this hack is not so useful: popsize could be there, but N is missing
        try:
            if utils.is_str(val):
                val = val.split('#')[0].strip()  # remove comments
                if key.find('filename') < 0 and not (key == 'seed' and val.startswith('time')):
                        # and key.find('mindx') < 0:
                    val = eval(safe_str(val), global_env, loc)
            # invoke default
            # TODO: val in ... fails with array type, because it is applied element wise!
            # elif val in (None,(),[],{}) and default is not None:
            elif val is None and default is not None:
                val = eval(safe_str(default), global_env, loc)
        except Exception as e:
            if not str(e).startswith('"initial"'):
                _warnings.warn(str(e))
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

    def amend_integer_options(self, dimension, inopts):
        """amend options when integer variables are indicated
        """
        try:
            self['integer_variables'] = list(self['integer_variables'])
        except Exception:
            if self['integer_variables']:
                raise
            self['integer_variables'] = []

        if self['integer_variables']:
            self.amend_integer_variables(dimension)
        if not self['integer_variables']:  # may have changed
            return

        if len(self['integer_variables']) > dimension:
            raise ValueError("{0} = dimension < len(options['integer_variables']) = {1}"
                             " is not a valid setting"
                             .format(dimension, len(self['integer_variables'])))

        if self['conditioncov_alleviate']:
            if len(self['conditioncov_alleviate']) == 1:
                self['conditioncov_alleviate'] = [self['conditioncov_alleviate'][0], 0]
            self['conditioncov_alleviate'][-1] = 0
        if inopts.get('popsize', None) in (None, cma_default_options['popsize']):
            self['popsize'] = 6 + 3 * (np.log(dimension) +  # for the time being, why not sqrt(N)?
                                       np.log(len(self['integer_variables']) + 0/2))

        # number of early tol-triggers before success on the 2D sphere with one int-variable:
        # code: es = cma.CMA([2, 0.1], 0.22, {'integer_variables': [0],...
        #
        # activated tol criterion: percentage (number of triggers observed)
        # -----------------------
        #   tolfunhist:  12% (96)
        #   tolfun:       9% (72)
        # tolflatfit=1:   9% (93)
        # tolflatfit=3:   9.5% (57)
        # tolflatfit=10:  4.7% (28)
        # tolflatfit=30:  0.5% (3)

        if inopts.get('tolflatfitness', None) in (
                    None, cma_default_options['tolflatfitness']):
            self['tolflatfitness'] = 3 + 30 * (
                len(self['integer_variables']) / dimension)
        if len(self['integer_variables']) == dimension:
            if inopts.get('tolfun', None) in (None, cma_default_options['tolfun']):
                self['tolfun'] = 0
            if inopts.get('tolfunhist', None) in (None, cma_default_options['tolfunhist']):
                self['tolfunhist'] = 0

    def amend_integer_variables(self, dimension):
        """removed fixed variables from the integer variable index values"""
        if not self['fixed_variables']:
            return
        # CAVEAT: this has not be thoroughly tested
        # transform integer indices to genotype
        popped = []  # just for the record
        for i in reversed(range(dimension)):
            if i in self['fixed_variables']:
                self['integer_variables'].remove(i)
                if 1 < 3:  # just for catching errors
                    popped.append(i)
                    if i in self['integer_variables']:
                        raise ValueError("index {0} appeared more than once in `'integer_variables'` option".format(i))
                # reduce integer variable indices > i by one
                for j, idx in enumerate(self['integer_variables']):
                    if idx > i:
                        self['integer_variables'][j] -= 1
        if self['verbose'] >= 0:
            _warnings.warn("Handling integer variables when some variables are fixed."
                        "\n  This code is poorly tested and may fail for negative indices."
                        "\n  Variables {0} are fixed integer variables but are"
                        " now dropped and discarded for integer handling."
                        .format(popped))

    def set_integer_min_std(self, N, mueff):
        """set lower std bounds for integer variables.

        Uses the above defined `integer_std_lower_bound` function which can
        be reassigned.
        """
        if not self['integer_variables']:
            return
        # 1) prepare minstd to be a vector
        if np.isscalar(self['minstd']) and self['minstd'] == 0:
            self['minstd'] = self['minstd'] * np.ones(N)
        # 2) set minstd to 0.7 mueff / N, was: 1 / (2 Nint + 1)
        #    the setting 2 / (2 Nint + 1) already prevents convergence
        if not np.isscalar(self['minstd']):
            for i in self['integer_variables']:
                if -N <= i < N:  # when i < 0, the index computes to N + i
                    if self['minstd'][i] == 0:  # don't change negative values too
                        # self['minstd'][i] = 1 / (2 * len(self['integer_variables']) + 1)
                        self['minstd'][i] = integer_std_lower_bound(
                                N, mueff, len(self['integer_variables']))
                else:
                    utils.print_warning(
                        "dropping integer index %d as it is not in range of dimension %d"
                            % (i, N))
                    self['integer_variables'].pop(self['integer_variables'].index(i))

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
            line = ''  # start entire to the left
            while a:
                while a and len(line) + len(a[0]) < linebreak:
                    line += ' ' + a.pop(0)
                print(line)
                line = '        '  # tab for subsequent lines

cma_default_options = CMAOptions(cma_default_options_())

class CMAParameters(object):
    """strategy parameters like population size and learning rates.

    Note:
        contrary to `CMAOptions`, `CMAParameters` is not (yet) part of the
        "user-interface" and subject to future changes (it might become
        a `collections.namedtuple`)

    Example
    -------
    >>> import cma
    >>> es = cma.CMAEvolutionStrategy(20 * [0.1], 1)  #doctest: +ELLIPSIS
    (6_w,12)-aCMA-ES (mu_w=3.7,w_1=40%) in dimension 20 (seed=...)
    >>>
    >>> type(es.sp)  # sp contains the strategy parameters
    <class 'cma.options_parameters.CMAParameters'>
    >>> es.sp.disp()  #doctest: +ELLIPSIS
    {'CMA_on': True,
     'N': 20,
     'c1': 0.00437235...,
     'c1_sep': ...0.0343279...,
     'cc': 0.171767...,
     'cc_sep': 0.252594...,
     'cmean': array(1...,
     'cmu': 0.00921656...,
     'cmu_sep': ...0.0565385...,
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
            if 'sweep_cc' in opts['vv']:  # caveat: cc2 and cc_sep
                sp.cc = opts['vv']['sweep_cc']
                sp.cc_sep = sp.cc
                print('cc is %f' % sp.cc)

        ## meta_parameters.c1_multiplier == 1.0
        sp.c1 = (1.0 * opts['CMA_rankone'] * ccovfac * min(1, sp.popsize / 6) *
                 ## meta_parameters.c1_exponent == 2.0
                 2 / ((N + 1.3)** 2.0 + mueff))
                 # 2 / ((N + 1.3)** 1.5 + mueff))  # TODO
                 # 2 / ((N + 1.3)** 1.75 + mueff))  # TODO
        # caveat: sp.c1 is NOT used in the update but for computing cmu
        # c1 given by interfaces.StatisticalModelSampler...parameters() equals to
        #    min((1, lam / 6)) * 2 / ((N + 1.3)**2 + mueff)
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
            sp.cmu = min(1 - sp.c1,  # TODO: this is a bug if sp.c1 is smaller than
                                     # interface...parameters()['c1']
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
            if opts['CMA_mu'] is not None:
                print('mu = %d' % (sp.weights.mu))

        # return self  # the constructor returns itself

    def disp(self):
        utils.pprint(self.__dict__)


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
        self.CMA_dampsvec_fac = None  ## [~0.01, ~100]  # def=np.inf or 0.5, not clear whether this is a log parameter
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
