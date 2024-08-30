# -*- coding: utf-8 -*-
"""logger class mainly to be used with `CMAEvolutionStrategy`

"""

from __future__ import absolute_import, division, print_function, unicode_literals
__author__ = "Nikolaus Hansen"

import os  # path
import sys  # flush
import warnings
import time
# from collections import defaultdict as _defaultdict
import numpy as np
from . import interfaces
from .utilities import utils
from .utilities import math as _mathutils
from . import restricted_gaussian_sampler as _rgs

_where = np.nonzero  # to make pypy work, this is how where is used here anyway
array = np.array

def _fix_lower_xlim_and_clipping():
    """minimize space wasted below x=0"""
    from matplotlib.pyplot import gca
    a = gca()
    for line in a.get_lines():
        line.set_clip_on(False)
    for key in a.spines:  # avoid border lines hiding data
        a.spines[key].set_zorder(0)  # 1.01 is still below the grid
    a.set_ymargin(0)
    if a.get_xlim()[0] < 0:  # first value is probably zero or one
       a.set_xlim(0 * -0.006 * a.get_xlim()[1], a.get_xlim()[1])
        # a.spines['left'].set_visible(False)
        # a.spines['left'].set_clip_on(False)
    else:
        a.set_xmargin(0)

def _monotone_abscissa(x, iabscissa=0):
    """make x monotone if ``not iabscissa``.

    Useful to plot restarted runs where the iteration is reset after
    each restart.

    TODO: handle same iteration number better, but how?
    Currently we assume it was written "on purpose" twice.
    """
    if iabscissa:
        return x  # mainly for efficiency
    x = np.asarray(x)
    d = np.diff(x)
    idxs = d < 0  # indices as boolean
    if not np.any(idxs):
        return x
    for i in np.nonzero(idxs)[0]:
        x[i+1:] -= d[i]  # d[i] is smaller than zero
    return x

class CMADataLogger(interfaces.BaseDataLogger):
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

        es = cma.CMAEvolutionStrategy(12 * [3], 4)
        es.optimize(cma.ff.elli, callback=es.logger.plot)

    plots into the current `matplotlib` figure (or opens one if none
    exists) via a generic callback. ``es.optimize`` already adds by default
    ``es.logger.add`` to its callback list to add data to the logger. This
    call::

        x, es = cma.fmin2(cma.ff.elli, 12 * [3], 4, {'verb_plot': 1})

    is very similar, but plots hard-coded into figure number 324::

        es = cma.CMAEvolutionStrategy(...)
        logger = cma.CMADataLogger().register(es)
        while not es.stop():
            ...
            logger.add()  # add can also take an argument

        logger.plot() # or a short cut can be used:
        cma.plot()  # plot data from logger with default name into new figure

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
    default_prefix = 'outcmaes' + os.sep
    # default_prefix = 'outcmaes'
    # names = ('axlen','fit','stddev','xmean','xrecentbest')
    # key_names_with_annotation = ('std', 'xmean', 'xrecent')

    def __init__(self, name_prefix=default_prefix, modulo=1, append=False, expensive_modulo=1):
        """initialize logging of data from a `CMAEvolutionStrategy` instance.

        Default ``modulo=1`` means logging with each call of `add`. If
        ``append is True`` data is appended to already existing data of a
        logger with the same name. Additional eigendecompositions are
        allowed only every `expensive_modulo`-th logged iteration.
        """
        # super(CMAData, self).__init__({'iter':[], 'stds':[], 'D':[],
        #        'sig':[], 'fit':[], 'xm':[]})
        # class properties:
        #        if isinstance(name_prefix, CMAEvolutionStrategy):
        #            name_prefix = name_prefix.opts.eval('verb_filenameprefix')
        if name_prefix is None:
            name_prefix = CMADataLogger.default_prefix
        if os.path.isfile(name_prefix) and name_prefix.endswith('.tar.gz'):
            raise NotImplementedError("sorry, you need to unzip the file manually"
                                      " and pass the folder name")
        self.name_prefix = os.path.abspath(os.path.join(*os.path.split(name_prefix)))
        if name_prefix is not None and name_prefix.endswith((os.sep, '/')):
            self.name_prefix = self.name_prefix + os.sep
        self.file_names = ('axlen', 'axlencorr', 'axlenprec', 'fit', 'stddev', 'sigvec',
                           'xmean', 'xrecentbest')
        """used in load, however hard-coded in add, because data must agree with name"""
        self.key_names = ('D', 'corrspec', 'precspec', 'f', 'std', 'sigvec', 'xmean', 'xrecent')
        """used in load, however hard-coded in plot"""
        self._key_names_with_annotation = ('std', 'sigvec', 'xmean', 'xrecent')
        """used in load to add one data row to be modified in plot"""
        self.modulo = modulo
        """how often to record data, allows calling `add` without args"""
        self.append = append
        """append to previous data"""
        self.expensive_modulo = expensive_modulo
        """log also values that need an eigendecomposition to be generated every `expensive` iteration"""
        self.counter = 0
        """number of calls to `add`"""
        self.last_iteration = 0
        self.last_skipped_iteration = 0  # skipped during plotting
        self.registered = False
        self.last_correlation_spectrum = {}
        self._eigen_counter = 1  # reduce costs
        self.skip_finalize_plotting = False  # flag to temporarily turn off finalization
        self.persistent_communication_dict = utils.DictFromTagsInString()
        self.relative_allowed_time_for_plotting = 25. / 100
        self.timer_plot = utils.ElapsedWCTime().pause()
        self.timer_all = utils.ElapsedWCTime()
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
        #        if not isinstance(es, CMAEvolutionStrategy):
        #            utils.print_warning("""only class CMAEvolutionStrategy should
        #    be registered for logging. The used "%s" class may not to work
        #    properly. This warning may also occur after using `reload`. Then,
        #    restarting Python should solve the issue.""" %
        #                                str(type(es)))
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

        # create path if necessary
        if os.path.dirname(self.name_prefix):
            try:
                os.makedirs(os.path.dirname(self.name_prefix))
            except OSError:
                pass  # folder exists

        # write x0
        fn = self.name_prefix + 'x0.dat'
        try:
            with open(fn, 'w') as f:
                f.write(repr(list(es.x0)))
        except (IOError, OSError):
            print('could not open file ' + fn)
        except Exception as e:
            warnings.warn("could not save `X0` due to this exception:\n    {0}".format(repr(e)))

        # write headers for output
        fn = self.name_prefix + 'fit.dat'
        strseedtime = 'seed=%s, %s' % (str(es.opts['seed']), time.asctime())

        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, sigma, axis ratio, ' +
                        'bestever, best, median, worst objective function value, interquartile range, ' +
                        'further/more values", ' +
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
                        ' min axis length, all principal axes lengths ' +
                        ' (sorted square roots of eigenvalues of C)", ' +
                        strseedtime +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)
        for name in ['axlencorr.dat', 'axlenprec.dat']:
            fn = self.name_prefix + name
            try:
                with open(fn, 'w') as f:
                    f.write('% # columns="iteration, evaluation, min max(neg(.)) min(pos(.))' +
                            ' max correlation, correlation matrix principal axes lengths ' +
                            ' (sorted square roots of eigenvalues of correlation matrix)", ' +
                            strseedtime +
                            '\n')
            except (IOError, OSError):
                print('could not open file ' + fn)
        fn = self.name_prefix + 'stddev.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, sigma, void, void, ' +
                        ' stds==sigma*sigma_vec.scaling*sqrt(diag(C))", ' +
                        strseedtime +
                        ', ' + self.persistent_communication_dict.as_python_tag +
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)
        # sigvec scaling factors from diagonal decoding
        fn = self.name_prefix + 'sigvec.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, sigma, beta, void, ' +
                        ' sigvec==sigma_vec.scaling factors from diagonal decoding", ' +
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
        loaded (5-8 files), by default ``'outcmaes/'`` (where / stands
        for the OS-specific path seperator).

        Return self with (added) attributes `xrecent`, `xmean`,
        `f`, `D`, `std`, `corrspec`, `sigvec`.

        """
        if not filenameprefix:
            filenameprefix = self.name_prefix
        assert len(self.file_names) == len(self.key_names)
        for i in range(len(self.file_names)):
            fn = filenameprefix + self.file_names[i] + '.dat'
            try:
                # list of rows to append another row latter
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    if self.file_names[i] in ['axlencorr', 'axlenprec']:
                        warnings.simplefilter("ignore")
                    try:
                        self.__dict__[self.key_names[i]] = list(
                                np.loadtxt(fn, comments=['%', '#'], ndmin=2))
                    except:
                        try:
                            self.__dict__[self.key_names[i]] = list(
                                np.loadtxt(fn, comments='%'))
                        except ValueError:  # fit.dat may have different number of columns
                            with open(fn, 'r') as f_:
                                lines = f_.readlines()
                            import ast, collections
                            data = [[ast.literal_eval(s) for s in line.split()]
                                    for line in lines if len(line.strip()) and not line.lstrip().startswith(('#', '%'))]
                            lengths = collections.Counter(len(row) for row in data)
                            if len(lengths) > 1:
                                print("rows in {0} have different lengths {1}, filling in `np.nan`"
                                      .format(fn, lengths))  # warnings are filtered out
                                l = max(lengths)
                                for row in data:
                                    while len(row) < l:
                                        row.append(np.nan)
                            self.__dict__[self.key_names[i]] = data
                # read dict from <python> tag in first line
                with open(fn) as file:
                    self.persistent_communication_dict.update(
                                string_=file.readline())
            except IOError:
                m = utils.format_warning('reading from file "' + fn + '" failed',
                        'load', 'CMADataLogger'); m and warnings.warn(m)
                self.__dict__[self.key_names[i]] = []
            try:
                # duplicate last row to later fill in annotation
                # positions for display
                if self.key_names[i] in self._key_names_with_annotation:
                    self.__dict__[self.key_names[i]].append(
                        self.__dict__[self.key_names[i]][-1])
                self.__dict__[self.key_names[i]] = \
                    np.asarray(self.__dict__[self.key_names[i]])
            except:
                if self.file_names[i] != 'sigvec':
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
            if len(d) and len(d.shape) == 1:  # one line has shape (8, )
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
        #        if not isinstance(es, CMAEvolutionStrategy):  # not necessary
        #            utils.print_warning('type CMAEvolutionStrategy expected, found '
        #                                + str(type(es)), 'add', 'CMADataLogger')
        evals = es.countevals
        iteration = es.countiter
        try: eigen_decompositions = es.sm.count_eigen
        except: eigen_decompositions = 0  # no correlations will be plotted
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
            medianf = es.fit.fit[len(es.fit.fit) // 2]
            iqrangef = np.diff(_mathutils.Mh.prctile(
                es.fit.fit, [25, 75], sorted_=True))[0]
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
                diagD = es.sm.eigenspectrum**0.5
            except:
                diagD = [1]
            maxD = max(diagD)
            minD = min(diagD)
        diagonal_scaling = es.sigma_vec.scaling
        try: diagonal_scaling_beta = es.sm._beta_diagonal_acceleration  # is for free
        except: diagonal_scaling_beta = 1
        correlation_matrix = None
        if not hasattr(self, 'last_precision_matrix'):
            self.last_precision_matrix = None
        if self.expensive_modulo:
            try:
                correlation_matrix = es.sm.correlation_matrix
                if correlation_matrix is not None and (
                        self.last_precision_matrix is None or eigen_decompositions % self.expensive_modulo == 0):
                    try:
                        self.last_precision_matrix = np.linalg.inv(correlation_matrix)
                        self.last_precision_matrix = _mathutils.to_correlation_matrix(self.last_precision_matrix)
                    except:  # diagonal case
                        warnings.warn("CMADataLogger failed to compute precision matrix")
            except (AttributeError, NotImplementedError):
                pass
            # TODO: write also np.linalg.eigvalsh(sigma_vec.transform_covariance_matrix(es.C))
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
                            + str(float(besteverf)) + ' '  # float converts Fraction
                            + '%.16e' % bestf + ' '
                            + str(float(medianf)) + ' '
                            + str(float(worstf)) + ' '
                            + str(float(iqrangef)) + ' '
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
            if self.expensive_modulo:
                for name, matrix in [['axlencorr.dat', correlation_matrix],
                                     ['axlenprec.dat', self.last_precision_matrix]]:
                    fn = self.name_prefix + name
                    if (matrix is not None
                        and not np.isscalar(matrix)
                        and len(matrix) > 1):
                        if (name not in self.last_correlation_spectrum
                            or eigen_decompositions % self.expensive_modulo == 0):
                            self.last_correlation_spectrum[name] = \
                                sorted(es.opts['CMA_eigenmethod'](matrix)[0]**0.5)
                            self._eigen_counter += matrix is self.last_precision_matrix  # hack to add only once per for loop
                        c = np.asarray(matrix)
                        c = c[np.triu_indices(c.shape[0], 1)]
                        if 11 < 3:  # old version
                            c_min = np.min(c)
                            c_max = np.max(c)
                            if np.min(abs(c)) == 0:
                                c_medminus = 0  # thereby zero "is negative"
                                c_medplus = 0  # thereby zero "is positive"
                            else:
                                cinv = 1 / c
                                c_medminus = 1 / np.min(cinv)  # negative close to zero
                                c_medplus = 1 / np.max(cinv)  # positive close to zero
                            if c_max <= 0:  # no positive values
                                c_max = c_medplus = 0  # set both "positive" values to zero
                            elif c_min >=0:  # no negative values
                                c_min = c_medminus = 0
                        c_min, c_medminus, c_medplus, c_max = _mathutils.Mh.prctile(c, [0, 25, 75, 100])
                        if 11 < 3:  # log correlations instead of eigenvalues, messes up KL display
                            _KL = 1e-3 - 0.5 * np.mean(np.log(self.last_correlation_spectrum[name]))  # doesn't work as expected
                            cs = np.asarray(sorted(c))
                            self.last_correlation_spectrum[name] = (1 + cs) / (1 - cs)
                            # c_min, c_medminus, c_medplus, c_max = 4 * [(KL - 1) / (KL + 1)]  # something is wrong
                            # c_min = (KL - 1) / (KL + 1)
                        with open(fn, 'a') as f:
                            f.write(str(iteration) + ' '
                                    + str(evals) + ' '
                                    + str(c_min) + ' '
                                    + str(c_medminus) + ' ' # the one closest to 0
                                    + str(c_medplus) + ' ' # the one closest to 0
                                    + str(c_max) + ' '
                                    + ' '.join(map(str,
                                            self.last_correlation_spectrum[name]))
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
            # sigvec scaling factors from diagonal decoding
            if np.size(diagonal_scaling) > 1:
                fn = self.name_prefix + 'sigvec.dat'
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(sigma) + ' '
                            + str(diagonal_scaling_beta) + ' '
                            + '0 '
                            + ' '.join(map(str, diagonal_scaling))
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
                            + str(float(bestf)) + ' '  # float converts Fraction
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

    def zip(self, filename=None, unique=True):
        """write logger data to file ``filename [+ ...] + '.tar.gz'``.

        `filename` defaults to ``self.name_prefix``.

        When `unique` is true, `filename` is furnished with a unique time
        stamp.

        Return the path (relative or absolute) of the output file.

        This function does in essence just ``tar -czf filename.tar.gz folder``
        where ``folder`` defaults to the current plot data folder and
        ``filename`` is created to be unique.

        TODO: we may want to be able to add a time stamp to the folder name
        too, to prevent conficts later?
        """
        # extract: zipfile.ZipFile(filename, "r").extractall(dirname) ?
        #          tarfile.open(filename, "r").extractall()
        import tarfile

        def unique_time_stamp(decimals=3):
            "a unique timestamp up to 1/10^decimals seconds"
            s = '{0:.' + str(decimals) + 'f}' + '.' * (decimals == 0)
            return time.strftime("%Y-%m-%dd%Hh%Mm%Ss") + (
                    s.format(time.time()).split('.')[1])

        if filename is None:
            filename = self.name_prefix  # name_prefix is an absolute path
        filename = filename.rstrip('/').rstrip(os.path.sep)
        if unique:
            filename += '-' + unique_time_stamp(3 if unique is True else unique)
        filename += '.tar.gz'

        with tarfile.open(filename, 'w:gz') as tar:
            tar.add(os.path.relpath(self.name_prefix))

        return filename

    def _unzip(self, filename):
        """to unzip use "tar -xf filename" in a system shell

        and afterwards::

            logger = cma.CMADataLogger(extacted_folder_name).plot()

        in a Python shell.
        """
        raise NotImplementedError(self._unzip.__doc__)

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
            if len(dat.corrspec):
                dat.corrspec = dat.x[_where([x in iteridx for x in
                                             dat.corrspec[:, 0]])[0], :]
        except AttributeError:
            pass
        try:
            if len(dat.precspec):
                dat.precspec = dat.x[_where([x in iteridx for x in
                                             dat.precspec[:, 0]])[0], :]
        except AttributeError:
            pass
        return self
    def plot(self, fig=None, iabscissa=0, iteridx=None,
             plot_mean=False, # was: plot_mean=True
             foffset=1e-19, x_opt=None, fontsize=7,
             downsample_to=1e7,
             xsemilog=False,
             xnormalize=False,
             fshift=0,
             addcols=None,
             load=True,
             message=None):
        """plot data from a `CMADataLogger` (using the files written
        by the logger).

        Arguments
        ---------
        `fig`
            figure number, by default starting from 325
        `iabscissa`
            ``0==plot`` versus iteration count,
            ``1==plot`` versus function evaluation number
        `iteridx`
            iteration indices to plot, e.g. ``range(100)`` for the first 100 evaluations.
        `x_opt`
            If `isscalar(x_opt)` it is interpreted as iteration number and the
            difference of ``x`` to the respective iteration is plotted. If it is
            a negative scalar the respective index rather than the iteration is used.
            Namely in particular, ``x_opt=0`` subtracts the initial solution ``X0``
            and ``x_opt=-1`` subtracts the final solution of the data.
            If ``len(x_opt) == dimension``, the difference to `x_opt` is
            plotted, otherwise the first row of ``x_opt`` are the indices of
            the variables to be plotted and the second row, if present, is used
            to take the difference.
        `fshift`
            is added to all displayed fitness values

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
            return self
        if hasattr(self, 'es') and self.es is not None:
            if fig is self.es:      # in case of usage in a callback
                fig = gcf().number  # plot in current figure
            if message is None:
                message = ''
            if not 'stop()' in message:
                if not message.endswith('\n'):
                    message += '\n'
                message += "stop()={0}".format(self.es.stop())
            # check whether self.es may be running and we want to negotiate timings
            if not self.es.stop() and self.es.countiter > self.last_skipped_iteration:
                # print(self.timer_plot.toc, self.relative_allowed_time_for_plotting, self.timer_all.toc)
                # check whether plotting is cheap enough
                if self.es.countiter < 3 or self.timer_all.elapsed < 0.15 or (  # avoid warning when too few data are available
                    self.timer_plot.toc > self.relative_allowed_time_for_plotting * self.timer_all.toc
                   ):
                    self.timer_plot.pause()  # just in case
                    self.last_skipped_iteration = self.es.countiter
                    return self
        self.timer_all.tic
        self.timer_plot.tic

        if fig is None:
            global last_figure_number  # we need this because the variable is not mutable
            last_figure_number += 1
            fig = last_figure_number
        if iabscissa not in (0, 1):
            iabscissa = 1

        load and self.load()  # load only conditionally
        if addcols is None:
            addcols = 1 if np.size(self.sigvec) else 0
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
            return self

        # not in use anymore, see formatter above
        # xticklocs = np.arange(5) * np.round(minxend/4., -int(np.log10(minxend/4.)))

        # dfit(dfit<1e-98) = NaN;

        # TODO: if abscissa==0 plot in chunks, ie loop over subsets where
        # dat.f[:,0]==countiter is monotonous

        figure(fig)
        self.skip_finalize_plotting = True  # disable finalize until end of this plot function
        self._enter_plotting(fontsize)
        self.fighandle = gcf()  # fighandle.number
        self.fighandle.clear()

        subplot(2, 2 + addcols, 1)
        if (self.fighandle.get_size_inches()[0] / self.fighandle.get_size_inches()[1]
            < 1.4 + 0.5 * addcols):
            self.fighandle.set_figwidth((1 + 0.45 * addcols) * self.fighandle.get_figwidth())

        self.plot_divers(iabscissa, foffset, fshift=fshift, message=message)
        pyplot.xlabel('')

        # Scaling
        subplot(2, 2 + addcols, 3 + addcols)
        self.plot_axes_scaling(iabscissa)

        # spectrum of correlation matrix
        if 1 < 3 and addcols and hasattr(dat, 'corrspec'):
            # figure(fig+10000)
            # pyplot.gcf().clear()  # == clf(), replaces hold(False)
            subplot(2, 2 + addcols, 3)
            self.plot_correlations(iabscissa)
            pyplot.xlabel('')
            subplot(2, 2 + addcols, 2 + addcols + 3)  # 3rd column in second row
            self.plot_sigvec(iabscissa)
            if addcols > 1:
                subplot(2, 2 + addcols, 4)
                self.plot_correlations(iabscissa, name='precspec')

        subplot(2, 2 + addcols, 2)
        if plot_mean:
            self.plot_mean(iabscissa, x_opt, xsemilog=False if plot_mean=='linear' else xsemilog, xnormalize=xnormalize)
        else:
            self.plot_xrecent(iabscissa, x_opt, xsemilog=xsemilog, xnormalize=xnormalize)
        pyplot.xlabel('')
        # pyplot.xticks(xticklocs)

        # standard deviations
        subplot(2, 2 + addcols, 4 + addcols)
        self.plot_stds(iabscissa, idx=x_opt)

        self.skip_finalize_plotting = False
        self._finalize_plotting()
        self.timer_plot.pause()
        return self

    def plot_all(self, fig=None, iabscissa=0, iteridx=None,
             foffset=1e-19, x_opt=None, fontsize=7):
        """
        plot data from a `CMADataLogger` (using the files written by the logger).

        Superseded by `plot`?

        Arguments
        ---------
        `fig`
            figure number, by default 425
        `iabscissa`
            ``0==plot`` versus iteration count,
            ``1==plot`` versus function evaluation number
        `iteridx`
            iteration indices to plot
        `x_opt`
            If `isscalar(x_opt)` it is interpreted as iteration number and the
            difference of ``x`` to the respective iteration is plotted. If it is
            a negative scalar the respective index rather than the iteration is used.
            Namely in particular, ``x_opt=0`` subtracts the initial solution ``X0``
            and ``x_opt=-1`` subtracts the final solution of the data.
            If ``len(x_opt) == dimension``, the difference to `x_opt` is
            plotted, otherwise the first row of ``x_opt`` are the indices of
            the variables to be plotted and the second row, if present, is used
            to take the difference.

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
            self.plot_stds(iabscissa, idx=x_opt)
        else:
            subplot(2, 3, 1)
            self.plot_divers(iabscissa, foffset)
            pyplot.xlabel('')

            # standard deviations
            subplot(2, 3, 4)
            self.plot_stds(iabscissa, idx=x_opt)

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
    def plot_axes_scaling(self, iabscissa=0):
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
        color = iter(pyplot.get_cmap('plasma_r')(
                    np.linspace(0.35, 1, dat.D.shape[1] - 5)))
        _x = _monotone_abscissa(dat.D[:, iabscissa], iabscissa)
        for i in range(5, dat.D.shape[1]):
            pyplot.semilogy(_x, dat.D[:, i],
                            '-', color=next(color))
        # pyplot.hold(True)
        smartlogygrid()
        ax = array(pyplot.axis())
        # ax[1] = max(minxend, ax[1])
        pyplot.axis(ax)
        pyplot.title('Principal Axes Lengths')
        # pyplot.xticks(xticklocs)
        self._xlabel(iabscissa)
        self._finalize_plotting()
        return self
    def plot_stds(self, iabscissa=0, idx=None):
        """``iabscissa=1`` means vs function evaluations, `idx` picks variables to plot"""
        from matplotlib import pyplot
        if not hasattr(self, 'std'):
            self.load()
        # quick fix of not cp issue without changing much code
        class _tmp: pass
        dat = _tmp()
        dat.std = np.array(self.std, copy=True)
        self._enter_plotting()
        try:
            if len(np.shape(idx)) > 1:
                idx = idx[0]  # take only first row
            if len(idx) < dat.std.shape[1] - 5:  # idx reduces the displayed variables
                dat.std = dat.std[:, list(range(5)) + [5 + i for i in idx]]
        except TypeError: pass  # idx has no len
        # remove sigma from stds (graphs become much better readible)
        dat.std[:, 5:] = np.transpose(dat.std[:, 5:].T / dat.std[:, 2].T)
        # ax = array(pyplot.axis())
        # ax[1] = max(minxend, ax[1])
        # axis(ax)
        _x = _monotone_abscissa(dat.std[:, iabscissa], iabscissa)
        if 1 < 2 and dat.std.shape[1] < 100:
            # use fake last entry in x and std for line extension-annotation
            minxend = int(1.06 * _x[-2])
            # minxend = int(1.06 * dat.x[-2, iabscissa])
            _x[-1] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat.std[-2, 5:])
            # idx2 = np.argsort(idx)
            dat.std[-1, 5 + idx] = np.logspace(np.log10(np.min(dat.std[:, 5:])),
                            np.log10(np.max(dat.std[:, 5:])), dat.std.shape[1] - 5)

            _x[-1] = minxend  # TODO: should be ax[1]
            pyplot.semilogy(_x, dat.std[:, 5:], '-')
            # pyplot.hold(True)
            ax = array(pyplot.axis())

            # yy = np.logspace(np.log10(ax[2]), np.log10(ax[3]), dat.std.shape[1] - 5)
            # yyl = np.sort(dat.std[-1,5:])
            idx = np.argsort(dat.std[-1, 5:])
            # idx2 = np.argsort(idx)
            # plot(np.dot(dat.std[-2, iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-') # vertical separator
            # vertical separator
            pyplot.plot(np.dot(_x[-2], [1, 1]),
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
                pyplot.text(_x[-1], dat.std[-1, 5 + i],
                            ' ' + str(s))
        else:
            pyplot.semilogy(_x, dat.std[:, 5:], '-')
        # pyplot.hold(True)
        smartlogygrid()
        pyplot.title(r'Standard Deviations $\times$ $\sigma^{-1}$ in All Coordinates')
        # pyplot.xticks(xticklocs)
        self._xlabel(iabscissa)
        self._finalize_plotting()
        return self
    def plot_sigvec(self, iabscissa=0, idx=None):
        """plot (outer) scaling from diagonal decoding.

        ``iabscissa=1`` plots vs function evaluations

        `idx` picks variables to plot if len(idx) < N, otherwise it picks
        iteration indices (in case, after downsampling).
        """
        from matplotlib import pyplot as plt
        if not hasattr(self, 'sigvec'):
            self.load()
        if not np.size(self.sigvec):  # nothing to plot
            plt.text(0, 0, 'nothing to plot')
            return self
        dat = np.array(self.sigvec, copy=True)  # we change the data in the process
        self._enter_plotting()
        if idx is not None:
            try: idx = list(idx)
            except TypeError: pass  # idx has no len
            else:
                if len(np.shape(idx)) > 1:
                    idx = idx[0]  # take only first row
                if len(idx) < dat.shape[1] - 5:  # idx reduces the displayed variables
                    dat = dat[:, list(range(5)) + [5 + i for i in idx]]
                else:
                    dat = dat[idx, :]
        # remove sigma from stds (graphs become much better readible)
        # dat[:, 5:] = np.transpose(dat[:, 5:].T / dat[:, 2].T)
        _x = _monotone_abscissa(dat[:, iabscissa], iabscissa)
        if 1 < 2 and dat.shape[1] < 100:
            # use fake last entry in x and std for line extension-annotation
            minxend = int(1.06 * _x[-2])
            # minxend = int(1.06 * dat.x[-2, iabscissa])
            _x[-1] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat[-2, 5:])
            # idx2 = np.argsort(idx)
            dat[-1, 5 + idx] = np.logspace(np.log10(np.min(dat[:, 5:])),
                            np.log10(np.max(dat[:, 5:])), dat.shape[1] - 5)

            _x[-1] = minxend  # TODO: should be ax[1]
            plt.semilogy(_x, dat[:, 5:], '-')
            # plt.hold(True)
            ax = array(plt.axis())

            # vertical separator
            idx = np.argsort(dat[-1, 5:])
            plt.plot(np.dot(_x[-2], [1, 1]),
                        array([ax[2] * (1 + 1e-6), ax[3] / (1 + 1e-6)]),
                        # array([np.min(dat[:, 5:]), np.max(dat[:, 5:])]),
                        'k-')
            annotations = self.persistent_communication_dict.get('variable_annotations')
            if annotations is None:
                annotations = range(len(idx))
            for i, s in enumerate(annotations):
                # text(ax[1], yy[i], ' '+str(idx[i]))
                plt.text(_x[-1], dat[-1, 5 + i],
                            ' ' + str(s))
        else:
            plt.semilogy(_x, dat[:, 5:], '-')
        plt.plot(_x[:-1], 1 / dat[:-1, 3], 'k',
                 label=r'$\beta=\max(2, \sqrt{cond(CORR)}) - 1$')
        plt.text(_x[-2], 1 / dat[-2, 3], '$1/\\beta$')
        plt.legend(framealpha=0.3)
        smartlogygrid()
        plt.title(r'Diagonal Decoding Scaling Factors')
        self._xlabel(iabscissa)
        self._finalize_plotting()
        return self
    def plot_mean(self, iabscissa=0, x_opt=None, annotations=None, xsemilog=None, xnormalize=None):
        if not hasattr(self, 'xmean'):
            self.load()
        self.x = self.xmean
        if xsemilog is None and x_opt is not None:
            xsemilog = True
        self._plot_x(iabscissa, x_opt, 'mean', annotations=annotations,
                     xsemilog=xsemilog, xnormalize=xnormalize)
        self._xlabel(iabscissa)
        return self
    def plot_xrecent(self, iabscissa=0, x_opt=None, annotations=None,
                     xsemilog=None, xnormalize=None):
        if not hasattr(self, 'xrecent'):
            self.load()
        self.x = self.xrecent
        self._plot_x(iabscissa, x_opt, 'curr best', annotations=annotations,
                     xsemilog=xsemilog, xnormalize=xnormalize)
        self._xlabel(iabscissa)
        return self
    def plot_correlations(self, iabscissa=0, name='corrspec'):
        """spectrum of correlation or precision matrix and percentiles of off-diagonal entries"""
        if not hasattr(self, name):
            self.load()
        if len(getattr(self, name)) < 2:
            return self
        from matplotlib import pyplot
        x = _monotone_abscissa(getattr(self, name)[:, iabscissa], iabscissa)
        y = getattr(self, name)[:, 6:]  # principal axes
        ys = getattr(self, name)[:, :6]  # "special" values

        from matplotlib.pyplot import semilogy, text, axis, title, gca
        self._enter_plotting()
        if 11 < 3:  # to be removed
            semilogy(x[:], np.max(y, 1) / np.min(y, 1), '-r')
            # text(x[-1], np.max(y[-1, :]) / np.min(y[-1, :]), 'axis ratio')
            labels = ['axis ratio']
        else:
            semilogy(x[:], 1e-3 - 0.5 * np.mean(np.log(y), axis=1), '-r')
            labels = [r'$10^{-3}$' + ' + KL(K || I) / D']  # mutual information / dimension
        if ys is not None:
            if 11 < 3:  # to be removed
                semilogy(x, 1 + ys[:, 2], '-b')
                text(x[-1], 1 + ys[-1, 2], '1 + min(corr)')
                semilogy(x, 1 - ys[:, 5], '-b')
                text(x[-1], 1 - ys[-1, 5], '1 - max(corr)')
                semilogy(x[:], 1 + ys[:, 3], '-k')
                text(x[-1], 1 + ys[-1, 3], '1 + max(neg corr)')
                semilogy(x[:], 1 - ys[:, 4], '-k')
                text(x[-1], 1 - ys[-1, 4], '1 - min(pos corr)')
            else:
                minmaxcorrs = ys[:, 2:6]  # 0, 25, 75, and 100 percentile correlation
                semilogy(x, (1 + minmaxcorrs) / (1 - minmaxcorrs), 'c',
                         linewidth=0.5)
                labels += ['(1 + c) / (1 - c) of (0,25,75,100)-prctile']
        pyplot.legend(labels, framealpha=0.3)
        # semilogy(x, y, '-c')
        color = iter(pyplot.get_cmap('plasma_r')(np.linspace(0.35, 1,
                                                    y.shape[1])))
        for i in range(y.shape[1]):
            semilogy(x, y[:, i], '-', color=next(color), zorder=1)
        smartlogygrid()
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        gca().yaxis.set_label_position("right")
        gca().yaxis.tick_right()
        title('Spectrum (roots) of %s matrix' % ('white precision' if name.startswith('prec') else 'correlation'))
        # pyplot.xticks(xticklocs)
        self._xlabel(iabscissa)
        self._finalize_plotting()
        return self
    def plot_divers(self, iabscissa=0, foffset=1e-19, fshift=0, message=None):
        """plot fitness, sigma, axis ratio...

        :param iabscissa: 0 means vs evaluations, 1 means vs iterations
        :param foffset: added to f-value after abs(f) is taken

        :See: `plot`

        """
        from matplotlib import pyplot
        from matplotlib.pyplot import semilogy, \
            axis, title, text
        fontsize = pyplot.rcParams['font.size']

        if not hasattr(self, 'f'):
            self.load()
        self.f[:, 5:8] += fshift

        dat = self

        # correct values which are rather not reasonable
        if not np.isfinite(dat.f[0, 5]):
            dat.f[0, 5:] = dat.f[1, 5:]  # best, median and worst f-value
        for i, val in enumerate(dat.f[0, :]): # hack to prevent warnings
            if np.isnan(val):
                dat.f[0, i] = dat.f[1, i]
        minfit = np.nanmin(dat.f[:, 5])
        dfit1 = dat.f[:, 5] - minfit  # why not using idx?
        dfit1[dfit1 < 1e-98] = np.nan
        dfit2 = dat.f[:, 5] - dat.f[-1, 5]
        dfit2[dfit2 < 1e-28] = np.nan

        self._enter_plotting()
        _x = _monotone_abscissa(dat.f[:, iabscissa], iabscissa)
        if dat.f.shape[1] > 7:
            # semilogy(dat.f[:, iabscissa], abs(dat.f[:,[6, 7, 10, 12]])+foffset,'-k')
            semilogy(_x, abs(dat.f[:, [6, 7]]) + foffset, '-k')
            # hold(True)
        if dat.f.shape[1] > 8:  # interquartile f-range
            # semilogy(dat.f[:, iabscissa], abs(dat.f[:,[6, 7, 10, 12]])+foffset,'-k')
            semilogy(_x, abs(dat.f[:, [8]]) + foffset, 'grey', linewidth=0.7)  # darkorange is nice

        # (larger indices): additional fitness data, for example constraints values
        if dat.f.shape[1] > 9:
            # dd = abs(dat.f[:,7:]) + 10*foffset
            # dd = _where(dat.f[:,7:]==0, np.nan, dd) # cannot be
            semilogy(_x, np.abs(dat.f[:, 8:]) + 10 * foffset, 'y')
            # hold(True)

        idx = _where(dat.f[:, 5] > 1e-98)[0]  # positive values
        semilogy(_x[idx], dat.f[idx, 5] + foffset, '.b')
        # hold(True)
        smartlogygrid()

        semilogy(_x, abs(dat.f[:, 5]) + foffset, '-b')
        text(_x[-1], abs(dat.f[-1, 5]) + foffset,
             r'$|f_\mathsf{best}|$', fontsize=fontsize + 2)

        # negative f-values, dots
        sgn = np.sign(dat.f[:, 5])
        sgn[np.abs(dat.f[:, 5]) < 1e-98] = 0
        idx = _where(sgn < 0)[0]
        semilogy(_x[idx], abs(dat.f[idx, 5]) + foffset,
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
                semilogy(_x[idx], abs(dat.f[idx, 5]) + foffset,
                        'm')  # , markersize=5
            # lines between positive and negative f-values
            # TODO: the following might plot values very close to zero
            if istart > 0:  # line to the left of istart
                semilogy(_x[istart-1:istart+1],
                         abs(dat.f[istart-1:istart+1, 5]) +
                         foffset, '--m')
            if istop:  # line to the left of istop
                semilogy(_x[istop-1:istop+1],
                         abs(dat.f[istop-1:istop+1, 5]) +
                         foffset, '--m')
                # mark the respective first positive values
                semilogy(_x[istop], abs(dat.f[istop, 5]) +
                         foffset, '.b', markersize=7)
            # mark the respective first negative values
            semilogy(_x[istart], abs(dat.f[istart, 5]) +
                     foffset, '.r', markersize=7)

        # standard deviations std
        _x_std = _monotone_abscissa(dat.std[:, iabscissa])
        semilogy(_x_std[:-1],
                 np.vstack([list(map(max, dat.std[:-1, 5:])),
                            list(map(min, dat.std[:-1, 5:]))]).T,
                     '-m', linewidth=2)
        text(_x_std[-2], max(dat.std[-2, 5:]), 'max std',
             fontsize=fontsize)
        text(_x_std[-2], min(dat.std[-2, 5:]), 'min std',
             fontsize=fontsize)

        # delta-fitness in cyan
        for dfit, label in [
            [dfit2, r'$f_\mathsf{best} - f_\mathsf{last}$'],
            [dfit1, r'$f_\mathsf{best} - f_\mathsf{min}$']]:
            idx = np.isfinite(dfit)
            if any(idx):
                idx_nan = _where(np.logical_not(idx))[0]  # gaps
                if not len(idx_nan):  # should never happen
                    semilogy(_x[idx], dfit[idx], '-c')
                else:
                    i_start = 0
                    for i_end in idx_nan:
                        if i_end > i_start:
                            semilogy(_x[i_start:i_end],
                                                    dfit[i_start:i_end], '-c')
                        i_start = i_end + 1
                    if len(dfit) > idx_nan[-1] + 1:
                        semilogy(_x[idx_nan[-1]+1:],
                                                dfit[idx_nan[-1]+1:], '-c')
                text(_x[-1], dfit[idx][-1],
                     label, fontsize=fontsize + 2)

            elif 11 < 3 and any(idx):
                semilogy(_x[idx], dfit[idx], '-c')
                text(_x[-1], dfit[idx][-1],
                     r'$f_\mathsf{best} - \min(f)$', fontsize=fontsize + 2)

            if 11 < 3:  # delta-fitness as points
                dfit = dat.f[1:, 5] - dat.f[:-1, 5]  # should be negative usually
                semilogy(_x[1:],  # abs(fit(g) - fit(g-1))
                    np.abs(dfit) + foffset, '.c')
                i = dfit > 0
                # print(np.sum(i) / float(len(dat.f[1:,iabscissa])))
                semilogy(_x[1:][i],  # abs(fit(g) - fit(g-1))
                    np.abs(dfit[i]) + foffset, '.r')
            # postcondition: dfit, idx = dfit1, ...

        # fat red dot for overall minimum
        i = np.argmin(dat.f[:, 5])
        semilogy(_x[i], np.abs(dat.f[i, 5]), 'ro',
                 markersize=9)
        if any(idx):  # another fat red dot
            semilogy(_x[i], dfit[idx][np.argmin(dfit[idx])]
                 + 1e-98, 'ro', markersize=9)
        # semilogy(dat.f[-1, iabscissa]*np.ones(2), dat.f[-1,4]*np.ones(2), 'rd')

        # AR and sigma
        semilogy(_x, dat.f[:, 3], '-r')  # AR
        semilogy(_x, dat.f[:, 2], '-g')  # sigma
        text(_x[-1], dat.f[-1, 3], r'axis ratio',
             fontsize=fontsize)
        text(_x[-1], dat.f[-1, 2] / 1.5, r'$\sigma$',
             fontsize=fontsize+3)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        text(ax[0] + 0.003 * (ax[1] - ax[0]), ax[2] * (ax[3]/ax[2])**0.002,
                    # 10**(log10(ax[2])+0.05*(log10(ax[3])-log10(ax[2]))),
             (message + '\n' if message else '') +
             'evals/iter={0}/{1} min($f$)={2}'.format(
                 int(dat.f[-1, 1]), int(dat.f[-1, 0]), repr(minfit))
            )
             #'.f_recent=' + repr(dat.f[-1, 5]))

        self.f[:, 5:8] -= fshift

        # AR and damping of diagonal decoding
        if np.size(dat.sigvec) > 1:  # try to figure out whether we have data
            _x = _monotone_abscissa(dat.sigvec[:, iabscissa], iabscissa)
            semilogy(_x, 1 / dat.sigvec[:, 3], 'k', label='$\\beta=\\sqrt{cond(CORR)} - 1$')
            text(_x[-1], 1 / dat.sigvec[-1, 3], 'dd-damp$\\approx1/\\sqrt{cond(CORR)}$')
            semilogy(_x, np.max(dat.sigvec[:, 5:], axis=1) / np.min(dat.sigvec[:, 5:], axis=1),
                    'darkred', label='axis ratio of diagonal decoding')
            text(_x[-1], np.max(dat.sigvec[-1, 5:]) / np.min(dat.sigvec[-1, 5:]), 'dd-AR')
        if np.size(dat.corrspec) > 1:
            def c_odds(c):
                cc = (c + 1) / (c - 1)
                cc[cc < 0] = -1 / cc[cc < 0]
                return cc
            _x = _monotone_abscissa(dat.corrspec[:, iabscissa], iabscissa)
            semilogy(_x, c_odds(dat.corrspec[:, 2]), 'c', label=r'$\min (c + 1) / (c - 1)$')
            semilogy(_x, c_odds(dat.corrspec[:, 5]), 'c', label=r'$\max (c + 1) / (c - 1)$')
            text(_x[-1], c_odds(np.asarray([dat.corrspec[-1, 2]])), r'$\max (c + 1) / (c - 1)$')
            text(_x[-1], c_odds(np.asarray([dat.corrspec[-1, 5]])), r'$-{\min}^{-1} (c + 1)\dots$')


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
        ## was: pyplot.hold(False)
        ## pyplot.gcf().clear()  # opens a figure window, if non exists
        pyplot.ioff()  # I assume this should save some time?
    def _finalize_plotting(self):
        if self.skip_finalize_plotting:
            return
        from matplotlib import pyplot
        pyplot.subplots_adjust(left=0.05, top=0.96, bottom=0.07, right=0.95)
        # pyplot.tight_layout(rect=(0, 0, 0.96, 1))
        pyplot.gcf().canvas.draw()  # update figure immediately
        pyplot.ion()  # prevents that the execution blocks after plotting
        # pyplot.show()  # in non-interactive mode: block until the figures have been closed
        # https://github.com/efiring/matplotlib/commit/94c5e161d1f3306d90092c986694d3f611cc5609
        # https://stackoverflow.com/questions/6130341/exact-semantics-of-matplotlibs-interactive-mode-ion-ioff
        pyplot.rcParams['font.size'] = self.original_fontsize  # changes font size in current figure which defeats the original purpose
    def _xlabel(self, iabscissa=0):
        from matplotlib import pyplot
        pyplot.xlabel('iterations' if iabscissa == 0
                      else 'function evaluations')
        _fix_lower_xlim_and_clipping()
    def _plot_x(self, iabscissa=0, x_opt=None, remark=None,
                annotations=None, xsemilog=None, xnormalize=False):
        """see def plot(...
        """
        if not hasattr(self, 'x'):
            utils.print_warning('no x-attributed found, use methods ' +
                           'plot_xrecent or plot_mean', 'plot_x',
                           'CMADataLogger')
            return
        if annotations is None:
            annotations = self.persistent_communication_dict.get('variable_annotations')
        import matplotlib
        from matplotlib.pyplot import plot, yscale, text, grid, axis, title
        dat = self  # for convenience and historical reasons
        # interpret x_opt
        dat_x = dat.x
        if np.isscalar(x_opt):  # interpret as iteration number or negative index
            if x_opt == 0:  # subtract X0
                import ast
                fn = self.name_prefix + 'x0.dat'
                try:
                    with open(fn, 'r') as f:
                        x_opt = ast.literal_eval(f.read())
                except (IOError, OSError):
                    warnings.warn('could not open file {0} to subtract ``X0``'.format(fn))
                except Exception as e:
                    warnings.warn("could not read `X0` due to this exception:\n    {0}".format(repr(e)))
            elif x_opt < 0:  # interpret x_opt as index in data
                x_opt = self.x[x_opt, 5:]
            else:  # interpret x_opt as iteration number
                _i = np.searchsorted(self.x[:, 0], x_opt)
                _m = np.shape(self.x)[0]
                if _i == _m or (  # index is above range
                    _i > 0 and x_opt - self.x[_i-1, 0] < self.x[_i, 0] - x_opt):
                    _i -= 1  # left iteration number is closer to x_opt
                if self.x[_i, 0] != x_opt:
                    warnings.warn("\n  Using iteration {0} (index={1}/{2}) instead"
                                  " of iteration {3} from `x_opt` argument"
                                  .format(self.x[_i, 0], _i, _m, x_opt))
                x_opt = self.x[_i, 5:]
            dat_x = dat.x[:,:]
            dat_x[:, 5:] -= x_opt
        elif x_opt is not None:  # x_opt is a vector or an index array to reduce dimension or both
            dat_x = dat.x[:,:]
            try:
                dat_x[:, 5:] -= x_opt
            except ValueError:  # interpret x_opt as index
                def apply_xopt(dat_x, x_opt_idx):
                    """first row of `x_opt_idx` are indices, second (optional) row are values"""
                    x_opt_vals = None
                    if len(np.shape(x_opt_idx)) > 1:
                        x_opt_vals = x_opt_idx[1]
                        x_opt_idx = np.asarray(x_opt_idx[0])
                    dat_x = dat_x[:, list(range(5)) + [5 + i for i in x_opt_idx]]
                    if x_opt_vals is not None:
                        dat_x[:, 5:] -= x_opt_vals
                    return dat_x
                dat_x = apply_xopt(dat_x, x_opt)
        if xnormalize:
            dat_x[:, 5:] /= dat.std[:, 5:]
            if xsemilog is None:
                xsemilog = True  # normalization assumes that zero is meaningful

        # modify fake last entry in x for line extension-annotation
        if dat_x.shape[1] < 100:
            minxend = int(1.06 * dat_x[-2, iabscissa])
            # write y-values for individual annotation into dat_x
            dat_x[-1, iabscissa] = minxend  # TODO: should be ax[1]
            if xsemilog or (xsemilog is None and remark and remark.startswith('mean')):
                dat_x[-1, 5:] = dat_x[-2, 5:]  # set horizontal
                idx = np.arange(dat_x.shape[1] - 5)
            else:
                idx = np.argsort(dat_x[-2, 5:])
                # idx2 = np.argsort(idx)
                dat_x[-1, 5 + idx] = np.linspace(np.min(dat_x[:, 5:]),
                            np.max(dat_x[:, 5:]), dat_x.shape[1] - 5)
        else:
            minxend = 0
        self._enter_plotting()
        _x = _monotone_abscissa(dat_x[:, iabscissa], iabscissa)
        plot(_x, dat_x[:, 5:], '-')
        if xsemilog or (xsemilog is None and remark and remark.startswith('mean')):
            _d = dat_x[:, 5:]
            _d_pos = np.abs(_d[_d != 0])
            if len(_d_pos):
                if matplotlib.__version__[:3] < '3.3':
                    # a terrible interface change that swallows the new/old parameter and breaks code
                    yscale('symlog', linthreshy=np.min(_d_pos))  # see matplotlib.scale.SymmetricalLogScale
                else:
                    yscale('symlog', linthresh=np.min(_d_pos))
            smartlogygrid(linthresh=np.min(_d_pos))
        if dat_x.shape[1] < 100:  # annotations
            ax = array(axis())
            axis(ax)
            # yy = np.linspace(ax[2] + 1e-6, ax[3] - 1e-6, dat_x.shape[1] - 5)
            # yyl = np.sort(dat_x[-1,5:])
            # plot([dat_x[-1, iabscissa], ax[1]], [dat_x[-1,5:], yy[idx2]], 'k-') # line from last data point
            plot(np.dot(_x[-2], [1, 1]),
                array([ax[2] + 1e-6, ax[3] - 1e-6]), 'k-')
            # plot(array([dat_x[-1, iabscissa], ax[1]]),
            #      reshape(array([dat_x[-1,5:], yy[idx2]]).flatten(), (2,4)), '-k')
            for i in range(len(idx)):
                # TODOqqq: annotate phenotypic value!?
                # text(ax[1], yy[i], 'x(' + str(idx[i]) + ')=' + str(dat_x[-2,5+idx[i]]))
                text(_x[-1], dat_x[-1, 5 + i],
                    ('' + str(i) + ': ' if annotations is None
                        else str(i) + ':' + annotations[i] + "=")
                    + utils.num2str(dat_x[-2, 5 + i],
                                    significant_digits=2,
                                    desired_length=4))
        smartlogygrid()
        i = 2  # find smallest i where iteration count differs (in case the same row appears twice)
        while i < len(dat.f) and dat.f[-i][0] == dat.f[-1][0]:
            i += 1
        popsi = ((dat.f[-1][1] - dat.f[-i][1]) / (dat.f[-1][0] - dat.f[-i][0])
                 if len(dat.f.T[0]) > 1 and dat.f[-1][0] > dat.f[-i][0] else None)
        mpopsi = ((dat.f[-1][1] - dat.f[0][1]) / (dat.f[-1][0] - dat.f[0][0])
                  if dat.f[-1][0] > dat.f[0][0] else None)
        title('Object Variables (' +
                (remark + ', ' if remark is not None else '') +
                str(dat_x.shape[1] - 5) + '-D, popsize=' +
                ((str(int(popsi)) if popsi is not None else 'NA') +
                  ('|' + str(np.round(mpopsi, 1)) if mpopsi != popsi else '')
                )
                + ')')
        self._finalize_plotting()
    def downsampling(self, factor=10, first=3, switch=True, verbose=False):
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
            cma.downsampling()  # takes outcma/cma* files
            cma.plot('outcma/cmadown')

        """
        newprefix = self.name_prefix + 'down'
        for name in self.file_names:
            if not os.path.isfile(self.name_prefix + name + '.dat'):
                warnings.warn('CMADataLogger.downsampling: file {0} not found'
                              .format(self.name_prefix + name + '.dat'))
                continue
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
        >>> l = cma.CMADataLogger()  # == res[-1], logger with default name, "points to" above data
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

        dat = CMADataLogger(filenameprefix).load()
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

last_figure_number = 324
def plot(name=None, fig=None, abscissa=0, iteridx=None,
         plot_mean=False,
         foffset=1e-19, x_opt=None, fontsize=7, downsample_to=3e3,
         xsemilog=None, xnormalize=None, fshift=0, addcols=None, **kwargs):
    """
    plot data from files written by a `CMADataLogger`,
    the call ``cma.plot(name, **argsdict)`` is a shortcut for
    ``cma.CMADataLogger(name).plot(**argsdict)``

    Arguments
    ---------
    `name`
        name of the logger, filename prefix, None evaluates to
        the default 'outcmaes/'
    `fig`
        filename or figure number, or both as a tuple (any order)
    `abscissa`
        0==plot versus iteration count,
        1==plot versus function evaluation number
    `iteridx`
        iteration indices to plot
    `x_opt`
        If `isscalar(x_opt)` it is interpreted as iteration number and the
        difference of ``x`` to the respective iteration is plotted. If it is
        a negative scalar the respective index rather than the iteration is used.
        Namely in particular, ``x_opt=0`` subtracts the initial solution ``X0``
        and ``x_opt=-1`` subtracts the final solution of the data.
        If ``len(x_opt) == dimension``, the difference to `x_opt` is
        plotted, otherwise the first row of ``x_opt`` are the indices of
        the variables to be plotted and the second row, if present, is used
        to take the difference.
    `xsemilog`
        customized semilog plot for x-values
    `fshift`
        is added to plotted and shown f-values

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
    global last_figure_number
    if not fig:
        last_figure_number += 1
        fig = last_figure_number
    if isinstance(fig, (int, float)):
        last_figure_number = fig
    return CMADataLogger(name).plot(fig, abscissa, iteridx, plot_mean, foffset,
                             x_opt, fontsize, downsample_to, xsemilog, xnormalize,
                             fshift, addcols, **kwargs)

def plot_zip(name=None, filename=None, unique=True):
    """create tar file ``filename [+ ...] + '.tar.gz'`` of folder `name`.

    The resulting tar file serves to recreate the current `cma.plot`
    elsewhere.

    `name` defaults to the folder plotted with `cma.plot` by default.
    `filename` defaults to `name` while adding a unique time stamp if
    `unique` is true.

    Return the path of the created file (may be absolute or relative).

    Details
    =======

    This is a convenience replacement for executing ``tar -czf
    filename.tar.gz name`` in a system shell where ``name`` defaults to the
    default plot data folder and ``filename`` is created to be unique.

    This function calls `CMADataLogger.zip` to do the real work.
    """
    return CMADataLogger(name if name else CMADataLogger.default_prefix
                         ).zip(filename=filename, unique=unique)

def disp(name=None, idx=None):
    """displays selected data from (files written by) the class
    `CMADataLogger`.

    The call ``cma.disp(name, idx)`` is a shortcut for
    ``cma.CMADataLogger(name).disp(idx)``.

    Arguments
    ---------
    `name`
        name of the logger, filename prefix, `None` evaluates to
        the default ``'outcma/cma'``
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
    return CMADataLogger(name if name else CMADataLogger.default_prefix
                         ).disp(idx)

# END cmaplt.py

class LoggerDummy(object):
    """use to fake a `Logger` in non-verbose setting"""
    def __init__(self, *args, **kwargs):
        self.count = 0
        self.name = None
        self.attributes = []
        self.callables = []
        self.labels = []
    def __call__(self, *args, **kwargs):
        return self.push()
    def add(self, *args, **kwargs):
        return self
    def push(self, *args, **kwargs):
        return self
    def push_header(self, *args, **kwargs):
        pass
    def load(self, *args, **kwargs):
        self.data = []
        return self
    def delete(self):
        self.count = 0
    @property
    def filename(self):
        return ""
    def plot(self, *args, **kwargs):
        warnings.warn("loggers is in dummy (silent) mode,"
                      " there is nothing to plot")

class Logger(object):
    r"""log an arbitrary number of data (a data row) per "timestep".

    The `add` method can be called several times per timestep, the `push`
    method must be called once per timestep. Callables are called in the
    `push` method and their output is logged. `push` finally also dumps the
    current data row to disk. `load` reads all logged data in and, like
    `plot`, will only work if the same number of data was pushed each and
    every time.

    To-be-logged "values" can be scalars or iterables (like lists
    or nparrays).

    The current data is saved to a file and cleared after each timestep.
    The `name` and `filename` attributes are based on either the name or
    the logged instance class as given as first argument. Only if a name
    was given, `push` overwrites the derived file if it exists.

    To append data, set `self.counter` > 0 or call `load` before to call
    `push` the first time (make sure that the `_name` attribute has the
    desired value either way). ``len(self.load().data)`` is the number of
    current data.

    A minimal practical example logging some nicely transformed attribute
    values of an object (looping over `Logger` and `LoggerDummy` for
    testing purpose only):

    >>> import numpy as np
    >>> import cma
    >>> for Logger in [cma.logger.Logger, cma.logger.LoggerDummy]:
    ...     es = cma.CMAEvolutionStrategy(3 * [1], 2, dict(maxiter=9, verbose=-9))
    ...     lg = Logger(es,  # es-instance serves as argument to callables and for attribute access
    ...                 callables=[lambda s: s.best.f,
    ...                            lambda s: np.log10(np.abs(s.best.f)),
    ...                            lambda s: np.log10(s.sigma),
    ...                           ],
    ...                 labels=['best f', 'lg(best f)', r'lg($\sigma$)'])
    ...     _ = es.optimize(cma.ff.sphere, callback=lg.push)
    ...     # lg.plot()  # caveat: requires matplotlib and clears current figure like gcf().clear()
    ...     lg2 = Logger(lg.name).load()  # same logger without callables assigned
    ...     lg3 = Logger(lg.filename).load()  # ditto
    ...     assert len(lg.load().data) == lg.count == 9 or isinstance(lg, cma.logger.LoggerDummy)
    ...     assert np.all(lg.data == lg2.data) and np.all(lg.data == lg3.data)
    ...     assert lg.labels == lg2.labels
    ...     lg.delete()  # delete data file, logger can still be (re-)used for new data

    """
    extension = ".logdata"
    fields_read = ['attributes', 'labels']
    "  names of attributes written to and read from file"

    def __init__(self, obj_or_name, attributes=None, callables=None,
                 path='outcmaes/', name=None, labels=None,
                 delete=False, plot_transformations=None):
        """`obj_or_name` is the instance that we want to observe,

        or a name, or an absolute path to a file.

        `attributes` are attribute names[:`str`] of `obj_or_name`, however

        `callables` are more general in their usage and hence recommended.
        They allow attribute access and transformations, for example like
        ``lambda es: np.log10(sum(es.mean**2)**0.5 / es.sigma)``.

        When a `callable` accepts an argument, it is called with
        `obj_or_name` as argument. The returned value of `callables` and
        the current values of `attributes` are logged each time when
        `push` is called.

        Details: `path` is not used if `obj_or_name` is an absolute path
        name, e.g. the `filename` attribute from another logger. If
        ``delete is True``, the data file is deleted when the instance is
        destructed. This can also be controlled or prevented by setting the
        boolean `_delete` attribute.
        """
        self.format = "%.19e"
        if obj_or_name == str(obj_or_name) and attributes is not None:
            raise ValueError('string obj %s has no attributes %s' % (
                str(obj_or_name), str(attributes)))
        self.obj = obj_or_name
        self.name = name
        # handle output location, TODO: streamline
        self.path = path
        self._delete = delete
        self.plot_transformations = plot_transformations
        self._autoname(obj_or_name)  # set _name attribute which is the output filename
        if self._name != os.path.abspath(self._name):
            self._name = self._create_path(path) + self._name
        if obj_or_name != str(obj_or_name):
            id = self._unique_name_addition(self._name)  # needs full path
            self._name = self._compose_name(self._name, id)
            self.name = self._compose_name(self.name, id)
        # self.taken_names.append(self._name)
        if 11 < 3 and os.path.isfile(self._name):
            utils.print_message('Logger uses existing file "%s" '
                                'which may be overwritten' % self._name)
        # print(self._name)
        self.attributes = attributes or []
        self.callables = callables or []
        self.labels = labels or []
        self.count = 0
        self.current_data = []
        # print('Logger:', self.name, self._name)

    @property
    def filename(self):
        """full filename as absolute path (stored in attribute `_name`)"""
        return self._name

    def delete(self):
        """delete current data file and reset count to zero"""
        os.remove(self._name)
        self.count = 0

    def __del__(self):
        if self._delete is True:
            self.delete()

    def _create_path(self, name_prefix=None):
        """return absolute path or '' if not `name_prefix`"""
        if not name_prefix:
            return ''
        path = os.path.abspath(os.path.join(*[a for a in os.path.split(name_prefix) if a]))
        if name_prefix.endswith((os.sep, '/')):
            path = path + os.sep
        # create path if necessary
        if os.path.dirname(path):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError:
                pass  # folder exists
        return path

    def _autoname(self, obj):
        """set `name` and `_name` attributes.

        Loggers based on the same class are separted by calling
        `_unique_name_addition` afterwards.
        """
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
        if not os.path.isfile(self._name):  # we want to be able to load an existing logger
            if '.' not in self._name:
                self._name = self._name + self.extension
            if not self._name.startswith(('._', '_')):
                self._name = '._' + self._name

    def _compose_name(self, name, unique_id):
        """add unique_id to name before ".logdata" """
        i = name.find(self.extension)
        return name[:i] + unique_id + name[i:] if i >= 0 else name + unique_id

    def _unique_name_addition(self, name=None):
        """return id:`str` that makes ``name or self._name`` unique"""
        if name is None:
            name = self._name
        if not os.path.isfile(name):
            return ''
        i = 2
        while os.path.isfile(self._compose_name(name, str(i))):
            i += 1
        if i % 99 == 0:
            utils.print_message('%d Logger data files like %s found. \n'
                                'Consider removing old data files and/or using '
                                'the delete parameter to delete on destruction and/or\n'
                                'using the `delete` method to delete the current log.'
                                % (i, name))
        return str(i)

    def _stack(self, data):
        """stack data into current row managing the different access...

        ...and type formats.
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
        """add data from registered attributes and callables, called by `push`"""
        for name in self.attributes:
            data = getattr(self.obj, name)
            self._stack(data)
        for callable in self.callables:
            try:
                self._stack(callable(self.obj))
            except TypeError:
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
        return self

    def push_header(self):
        mode = 'at' if self.count else 'wt'
        with open(self._name, mode) as file_:
            for name in self.fields_read:
                if getattr(self, name, None):
                    file_.write("# {'%s': %s}\n" % (name, repr(getattr(self, name))))

    def load(self):
        import ast
        self.data = np.loadtxt(self._name)
        with open(self._name, 'rt') as file_:  # read meta data/labels
            line = file_.readline()
            while line.startswith('#'):
                res = ast.literal_eval((line[1:].lstrip()))
                if isinstance(res, dict):
                    for name in self.fields_read:
                        if name in res:
                            setattr(self, name, res[name])
                else:  # backward compatible, to be removed (TODO)
                    self.labels = res
                line = file_.readline()
        if self.count == 0:  # prevent overwriting of data
            self.count = len(self.data)
        return self

    def plot(self, plot=None, clear=True, transformations=None):
        """plot logged data using the `plot` function.

        If `clear`, this calls `matplotlib.pyplot.gca().clear()` before
        to plot in the current figure. The default value of `clear` may
        change in future.

        If ``transformations[i]`` is a `callable` it is used to transform the i-th
        data column like ``i_th_column = transformations[i](data[:,i])``.
        """
        try:
            from matplotlib import pyplot as plt
        except ImportError: pass
        if not callable(plot):  # this may allow to use this method as callback
            from matplotlib.pyplot import plot
        self.load()
        n = len(self.data)  # number of data rows
        try:
            m = len(self.data[0])  # number of "variables"
        except TypeError:
            m = 0
        if clear:
            plt.gca().clear()
        if transformations is None:
            transformations = self.plot_transformations
        if m < 2:  # data cannot be indexed like data[:,0]
            try: data = transformations[0](self.data)
            except (IndexError, TypeError): data = self.data
            plot(range(1, n + 1), data,
                 label=self.labels[0] if self.labels else None)
        else:
            color = iter(plt.get_cmap('plasma')(np.linspace(0.01, 0.9, m)))  # plasma was: winter_r
            idx_labels = [int(i * m / len(self.labels)) for i in range(len(self.labels))]
            if len(idx_labels) > 1:
                idx_labels[-1] = m - 1  # last label goes to the line m - 1
            labels = iter(self.labels)
            for i in range(m):
                column = self.data[:, i]
                try: column = transformations[i](column)
                except (IndexError, TypeError): pass
                plot(range(1, n + 1), column,
                    color=next(color),
                    label=next(labels) if i in idx_labels else None,
                    linewidth=1 - 0.7 * m / (m + 10))
                # plt.gca().get_lines()[0].set_color(next(color))
        if self.labels:
            plt.legend(framealpha=0.3)  # more opaque than not
        plt.gcf().canvas.draw()  # allows online use
        return self

def smartlogygrid(**kwargs):
    """turn on grid and also minor grid depending on y-limits"""
    try:
        from matplotlib import pyplot as plt
    except ImportError: return
    plt.grid(True)
    lims = plt.ylim()
    scale = plt.gca().get_yscale()
    if scale.startswith('lin'):
        return  # minor grid has no effect
    if scale.startswith('symlog'):
        try: s = max(np.abs(lims)) / kwargs['linthresh']
        except KeyError: pass
        else:
            if s < 1e3:
                plt.grid(True, which='minor')  # minor grid isn't yet supported in symlog
        return
    if lims[1] / lims[0] < 1e5:
        plt.grid(True, which='minor')
    _fix_lower_xlim_and_clipping()

def custom_default_matplotlib():
    """reduce wasted margin area from 19.0% to 4.0% and make both grids default and

    show legend when labels are available.
    """
    import matplotlib.pyplot as plt
    plt.grid(True)
    plt.grid(True, which='minor', linewidth=0.5)
    plt.margins(x=.01, y=.01)  # mpl.rcParams['axes.xmargin'] = 0.01, default is 0.05

    labels = [line.get_label() for line in plt.gca().get_lines()]
    if any([label and not label.startswith('_') for label in labels]):
        plt.legend()
