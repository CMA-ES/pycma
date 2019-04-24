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
from .optimization_tools import semilogy_signed
from . import restricted_gaussian_sampler as _rgs

_where = np.nonzero  # to make pypy work, this is how where is used here anyway
array = np.array

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
    default_prefix = 'outcmaes' + os.sep
    # default_prefix = 'outcmaes'
    # names = ('axlen','fit','stddev','xmean','xrecentbest')
    # key_names_with_annotation = ('std', 'xmean', 'xrecent')

    def __init__(self, name_prefix=default_prefix, modulo=1, append=False):
        """initialize logging of data from a `CMAEvolutionStrategy`
        instance, default ``modulo=1`` means logging with each call

        """
        # super(CMAData, self).__init__({'iter':[], 'stds':[], 'D':[],
        #        'sig':[], 'fit':[], 'xm':[]})
        # class properties:
#        if isinstance(name_prefix, CMAEvolutionStrategy):
#            name_prefix = name_prefix.opts.eval('verb_filenameprefix')
        if name_prefix is None:
            name_prefix = CMADataLogger.default_prefix
        self.name_prefix = os.path.abspath(os.path.join(*os.path.split(name_prefix)))
        if name_prefix is not None and name_prefix.endswith((os.sep, '/')):
            self.name_prefix = self.name_prefix + os.sep
        self.file_names = ('axlen', 'axlencorr', 'fit', 'stddev', 'xmean',
                'xrecentbest')
        """used in load, however hard-coded in add, because data must agree with name"""
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
        loaded (six files), by default ``'outcma/cma'``.

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
#        if not isinstance(es, CMAEvolutionStrategy):  # not necessary
#            utils.print_warning('type CMAEvolutionStrategy expected, found '
#                                + str(type(es)), 'add', 'CMADataLogger')
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
        try:
            correlation_matrix = es.sm.correlation_matrix
        except (AttributeError, NotImplemented, NotImplementedError):
            correlation_matrix = None
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
                if (correlation_matrix is not None
                    and not np.isscalar(correlation_matrix)
                    and len(correlation_matrix) > 1):
                    # accept at most 50% internal loss
                    if 11 < 3 or self._eigen_counter < eigen_decompositions / 2:
                        self.last_correlation_spectrum = \
                            sorted(es.opts['CMA_eigenmethod'](c)[0]**0.5)
                        self._eigen_counter += 1
                    if self.last_correlation_spectrum is None:
                        self.last_correlation_spectrum = len(diagD) * [1]
                    c = np.asarray(correlation_matrix)
                    c = c[np.triu_indices(c.shape[0], 1)]
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
             downsample_to=1e7,
             xsemilog=False):
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
            iteration indices to plot, e.g. ``range(100)`` for the first 100 evaluations.

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
            self.plot_mean(iabscissa, x_opt, xsemilog=xsemilog)
        else:
            self.plot_xrecent(iabscissa, x_opt, xsemilog=xsemilog)
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
        color = iter(pyplot.cm.plasma_r(np.linspace(0.35, 1,
                                                    dat.D.shape[1] - 5)))
        for i in range(5, dat.D.shape[1]):
            pyplot.semilogy(dat.D[:, iabscissa], dat.D[:, i],
                            '-', color=next(color))
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
        # quick fix of not cp issue without changing much code
        class _tmp: pass
        dat = _tmp()
        dat.std = np.asarray(self.std).copy()
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
    def plot_mean(self, iabscissa=1, x_opt=None, annotations=None, xsemilog=None):
        if not hasattr(self, 'xmean'):
            self.load()
        self.x = self.xmean
        if xsemilog is None and x_opt is not None:
            xsemilog = True
        self._plot_x(iabscissa, x_opt, 'mean', annotations=annotations, xsemilog=xsemilog)
        self._xlabel(iabscissa)
        return self
    def plot_xrecent(self, iabscissa=1, x_opt=None, annotations=None, xsemilog=None):
        if not hasattr(self, 'xrecent'):
            self.load()
        self.x = self.xrecent
        self._plot_x(iabscissa, x_opt, 'curr best', annotations=annotations, xsemilog=xsemilog)
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
        if ys is not None:
            semilogy(x, 1 + ys[:, 2], '-b')
            text(x[-1], 1 + ys[-1, 2], '1 + min(corr)')
            semilogy(x, 1 - ys[:, 5], '-b')
            text(x[-1], 1 - ys[-1, 5], '1 - max(corr)')
            semilogy(x[:], 1 + ys[:, 3], '-k')
            text(x[-1], 1 + ys[-1, 3], '1 + max(neg corr)')
            semilogy(x[:], 1 - ys[:, 4], '-k')
            text(x[-1], 1 - ys[-1, 4], '1 - min(pos corr)')
        semilogy(x, y, '-c')
        semilogy(x[:], np.max(y, 1) / np.min(y, 1), '-r')
        text(x[-1], np.max(y[-1, :]) / np.min(y[-1, :]), 'axis ratio')
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
        dfit1 = dat.f[:, 5] - minfit  # why not using idx?
        dfit1[dfit1 < 1e-98] = np.NaN
        dfit2 = dat.f[:, 5] - dat.f[-1, 5]
        dfit2[dfit2 < 1e-28] = np.NaN

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
        for dfit, label in [
            [dfit2, r'$f_\mathsf{best} - f_\mathsf{last}$'],
            [dfit1, r'$f_\mathsf{best} - f_\mathsf{min}$']]:
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
                     label, fontsize=fontsize + 2)

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
            # postcondition: dfit, idx = dfit1, ...

        # fat red dot for overall minimum
        i = np.argmin(dat.f[:, 5])
        semilogy(dat.f[i, iabscissa], np.abs(dat.f[i, 5]), 'ro',
                 markersize=9)
        if any(idx):  # another fat red dot
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
        pyplot.subplots_adjust(left=0.05, top=0.96, bottom=0.07, right=0.95)
        # pyplot.tight_layout(rect=(0, 0, 0.96, 1))
        pyplot.draw()  # update "screen"
        pyplot.ion()  # prevents that the execution stops after plotting
        pyplot.show()
        pyplot.rcParams['font.size'] = self.original_fontsize
    def _xlabel(self, iabscissa=1):
        from matplotlib import pyplot
        pyplot.xlabel('iterations' if iabscissa == 0
                      else 'function evaluations')
    def _plot_x(self, iabscissa=1, x_opt=None, remark=None,
                annotations=None, xsemilog=None):
        """If ``x_opt is not None`` the difference to x_opt is plotted
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
        if x_opt in (None, 0):
            dat_x = dat.x
        else:
            dat_x = dat.x[:,:]
            dat_x[:, 5:] -= x_opt

        # modify fake last entry in x for line extension-annotation
        if dat_x.shape[1] < 100:
            minxend = int(1.06 * dat_x[-2, iabscissa])
            # write y-values for individual annotation into dat_x
            dat_x[-1, iabscissa] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat_x[-2, 5:])
            # idx2 = np.argsort(idx)
            dat_x[-1, 5 + idx] = np.linspace(np.min(dat_x[:, 5:]),
                        np.max(dat_x[:, 5:]), dat_x.shape[1] - 5)
        else:
            minxend = 0
        self._enter_plotting()
        if xsemilog or (xsemilog is None and remark and remark.startswith('mean')):
            semilogy_signed(dat_x[:, iabscissa], dat_x[:, 5:])
        else:
            plot(dat_x[:, iabscissa], dat_x[:, 5:], '-')
        # hold(True)
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        ax[1] -= 1e-6  # to prevent last x-tick annotation, probably superfluous
        if dat_x.shape[1] < 100:
            # yy = np.linspace(ax[2] + 1e-6, ax[3] - 1e-6, dat_x.shape[1] - 5)
            # yyl = np.sort(dat_x[-1,5:])
            # plot([dat_x[-1, iabscissa], ax[1]], [dat_x[-1,5:], yy[idx2]], 'k-') # line from last data point
            plot(np.dot(dat_x[-2, iabscissa], [1, 1]),
                 array([ax[2] + 1e-6, ax[3] - 1e-6]), 'k-')
            # plot(array([dat_x[-1, iabscissa], ax[1]]),
            #      reshape(array([dat_x[-1,5:], yy[idx2]]).flatten(), (2,4)), '-k')
            for i in range(len(idx)):
                # TODOqqq: annotate phenotypic value!?
                # text(ax[1], yy[i], 'x(' + str(idx[i]) + ')=' + str(dat_x[-2,5+idx[i]]))
                text(dat_x[-1, iabscissa], dat_x[-1, 5 + i],
                     ('' + str(i) + ': ' if annotations is None
                        else str(i) + ':' + annotations[i] + "=")
                     + utils.num2str(dat_x[-2, 5 + i],
                                     significant_digits=2,
                                     desired_length=4))
        i = 2  # find smallest i where iteration count differs (in case the same row appears twice)
        while i < len(dat.f) and dat.f[-i][0] == dat.f[-1][0]:
            i += 1
        title('Object Variables (' +
                (remark + ', ' if remark is not None else '') +
                str(dat_x.shape[1] - 5) + '-D, popsize~' +
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
            cma.downsampling()  # takes outcma/cma* files
            cma.plot('outcma/cmadown')

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
def plot(name=None, fig=None, abscissa=1, iteridx=None,
         plot_mean=False,
         foffset=1e-19, x_opt=None, fontsize=7, downsample_to=3e3,
         xsemilog=None):
    """
    plot data from files written by a `CMADataLogger`,
    the call ``cma.plot(name, **argsdict)`` is a shortcut for
    ``cma.CMADataLogger(name).plot(**argsdict)``

    Arguments
    ---------
    `name`
        name of the logger, filename prefix, None evaluates to
        the default 'outcma/cma'
    `fig`
        filename or figure number, or both as a tuple (any order)
    `abscissa`
        0==plot versus iteration count,
        1==plot versus function evaluation number
    `iteridx`
        iteration indices to plot
    `x_opt`
        negative offset for plotting x-values
    `xsemilog`
        customized semilog plot for x-values

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
                             x_opt, fontsize, downsample_to, xsemilog)

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

class Logger(object):
    """log an arbitrary number of data (a data row) per "timestep".

    The `add` method can be called several times per timestep, the
    `push` method must be called once per timestep. `load` and `plot`
    will only work if each time the same number of data was pushed.

    For the time being, the data is saved to a file after each timestep.

    To append data, set `self.counter` > 0 before to call `push` the first
    time. ``len(self.load().data)`` is the number of current data.

    Useless example::

        >> es = cma.CMAEvolutionStrategy
        >> lg = Logger(es, ['countiter'])  # prepare to log the countiter attribute of es
        >> lg.push()  # execute logging

    """
    def __init__(self, obj_or_name, attributes=None, callables=None, path='outcmaes/', name=None, labels=None):
        """obj can also be a name"""
        self.format = "%.19e"
        if obj_or_name == str(obj_or_name) and attributes is not None:
            raise ValueError('string obj %s has no attributes %s' % (
                str(obj_or_name), str(attributes)))
        self.obj = obj_or_name
        self.name = name
        # handle output location, TODO: streamline
        self.path = path
        self._autoname(obj_or_name)  # set _name attribute
        self._name = self._create_path(path) + self._name
        # print(self._name)
        self.attributes = attributes or []
        self.callables = callables or []
        self.labels = labels or []
        self.count = 0
        self.current_data = []
        # print('Logger:', self.name, self._name)

    def _create_path(self, name_prefix=None):
        """return absolute path or '' if not `name_prefix`"""
        if not name_prefix:
            return ''
        path = os.path.abspath(os.path.join(*os.path.split(name_prefix)))
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

        TODO: how to handle two loggers in the same class??
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
            if self.labels:
                file_.write('# %s\n' % repr(self.labels))
            if self.attributes:
                file_.write('# %s\n' % repr(self.attributes))

    def load(self):
        import ast
        self.data = np.loadtxt(self._name)
        with open(self._name, 'rt') as file_:
            first_line = file_.readline()
        if first_line.startswith('#'):
            self.labels = ast.literal_eval((first_line[1:].lstrip()))
        return self

    def plot(self, plot=None):
        try:
            from matplotlib import pyplot as plt
        except ImportError: pass
        if plot is None:
            from matplotlib.pyplot import plot
        self.load()
        n = len(self.data)  # number of data rows
        try:
            m = len(self.data[0])  # number of "variables"
        except TypeError:
            m = 0
        plt.gca().clear()
        if not m or len(self.labels) == 1:  # data cannot be indexed like data[:,0]
            plot(range(1, n + 1), self.data,
                 label=self.labels[0] if self.labels else None)
            return
        color=iter(plt.cm.winter_r(np.linspace(0.15, 1, m)))
        for i in range(m):
            plot(range(1, n + 1), self.data[:, i],
                 label=self.labels[i] if i < len(self.labels) else None)
            plt.gca().get_lines()[0].set_color(next(color))
        plt.legend(framealpha=0.3)  # more opaque than not
        return self

