from __future__ import absolute_import  # use from . import
from __future__ import division  # use // for integer division
from __future__ import print_function  # use print() instead of print
from __future__ import unicode_literals  # all the strings are unicode
"""Iterative Algorithm Interface
This module provides the framework for iterative algorithms. It consists of the following classes
    ABCIterativeAlgorithm:
        The interface of iterative algorithms equipped with logging and plotting functionality.
    OptionBaseClass:
        The container of optional parameters that provides the function to parse the string of an expression that
        depends on the other parameters contained.
    IterativeAlgorithmOption: Derived from OptionBaseClass
        The default option for ABCIterativeAlgorithm. Once a class inheriting ABCIterativeAlgorithm is defined, a class
         derived from IterativeAlgorithmOption should be defined for the optional parameters for the algorithm class.
    Summarizer:
        A class to post-process the results of the same algorithm with the same optional parameters.
    Comparator:
        A class to post-process the results of different algorithms and/or different parameters.
The following functions are used with Comparator
    get_multi_setting_info
    dict_cartesian
    piv
To demonstrate the usage of the framework, it also provides an example algorithm:
    RandomSearch, RandomSearchOption
        A very simple random search
Example
-------
The interactive demo model will be started by
>>> python iterative_algorithm_interface.py
Dependencies
------------
numpy, matplotlib (Plotting), pandas (Summarizer and Comparator)
All of these packages are included in Anaconda python distribution.
Last Update
-----------
2017/06/29: First Public Version
"""
__author__ = 'youhei akimoto'

import sys
import os
from abc import ABCMeta, abstractmethod, abstractproperty
import time
import pprint
import math

try:
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    print('Requirement: Numpy, Pandas, Matplotlib')

mpl.rc('lines', linewidth=2, markersize=8)
mpl.rc('font', size=12)
mpl.rc('grid', color='0.75', linestyle=':')
mpl.rc('ps', useafm=True)  # Force to use
mpl.rc('pdf', use14corefonts=True)  # only Type 1 fonts
mpl.rc('text', usetex=True)  # for a paper submision

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

NITER = 0
MEASURE = 1
ELAPSED = 2
CRITERION = 3
PROFILE = 'profile'
EXT = '.dat'
DELIM = '_'
PATTERNS = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*", ""]

if sys.version_info[0] >= 3:
    # For Python Version >= 3.0
    basestring = str
else:
    # For Python Version < 3.0    
    range = xrange


def my_formatter(x, pos):
    """Float Number Format for Axes"""
    float_str = "{0:2.1e}".format(x)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0}e{1}".format(base, int(exponent))
    else:
        return r"" + float_str + ""


def _myeval(s, g, l):
    if isinstance(s, basestring):
        return eval(s, g, l)
    else:
        return s


class OptionBaseClass(object):
    def __init__(self, **kwargs):
        """OptionBaseClass
        
        It is a container of variables, whose values can depends on the other variables.
        The method `parse` automatically determines the right order to parse the variables
        and returns a new instance consisting of the parsed values.
        Parameters
        ----------
        The pair (key, value) in the keyward arguments is stored in the instance as 
            
            instance.key = value
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.is_parsed = False

    def __str__(self):
        return type(self).__name__ + str(self.__dict__)

    def setattr_from_local(self, locals_):
        for key in locals_:
            if key != 'self' and key != 'kwargs':
                setattr(self, key, locals_[key])

    def disp(self):
        """Display all the options"""
        print(type(self).__name__)
        pp = pprint.PrettyPrinter()
        pp.pprint(self.__dict__)

    def to_latex(self):
        """Parse the option to LaTeX
        Return
        ------
        LaTeX string. 
        """
        res = ''
        res += '\\begin{longtable}{ll}\n'
        res += '\\caption{The parameters of ' + type(self).__name__ + '.}\\\\\n'
        res += '\\hline\n'
        res += 'Key & Value \\\\\n'
        res += '\\hline\n'
        for key in sorted(list(self.__dict__)):
            if key != 'is_parsed':
                res += '\\detokenize{' + str(key) + '} & \\detokenize{' + str(getattr(self, key)) + '}\\\\\n'
        res += '\\hline\n'
        res += '\\end{longtable}\n'
        return res

    def parse(self, env=None, flg_print=False):
        """Parse the member variables that are string expressions
        Parameters
        ----------
        env : dict, default None
            The dictionary of a namespace. A typical choice is globals().
            The values in the instance are evaluated on the environment `env`
             updated with globals() called inside the method.
        flg_print : bool, default False
            Print the parse process.
        Returns
        -------
        parsed : an instance of the same class as `self`
            all the member variables are parsed.
        Example
        -------
        Case 1.
        >>> opts = OptionBaseClass(N='10', M='int(N)', L='int(N)')
        >>> parsed = opts.parse()
        >>> parsed.disp()
        OptionBaseClass
        {'L': 10, 'M': 10, 'N': 10, 'is_parsed': True}
        Case 2.
        repr(N) is evaluated before it is passed to the function
        >>> N = 100
        >>> opts = OptionBaseClass(N='10', L=repr(N))
        >>> parsed = opts.parse()
        >>> parsed.disp()
        OptionBaseClass
        {'L': 100, 'N': 10, 'is_parsed': True}
        Case 3.
        >>> N = 100
        >>> opts = OptionBaseClass(M='N', L=repr(N))
        >>> parsed = opts.parse(globals())
        >>> parsed.disp()
        OptionBaseClass
        {'L': 100, 'M': 100, 'is_parsed': True}
        In the following cases, one may obtain unexpected outputs.
        Case A.
            opts = OptionBaseClass(N='10', M='N')
        The output of `parse` method is undefined.
        If `M` is evaluated before `N`, the result will be M = '10' (string).
        To prevent this unexpected behavior, consider to cast the variable like
            opts = OptionBaseClass(N='10', M='int(N)')
        Case B.
            opts = OptionBaseClass(N='M', M='N')
        Obviously, the variables can not be evaluated if there is a pair of
        variables that depend on each other.
        Case C.
            N = 100
            mypow = pow
            opts = OptionBaseClass(M='mypow(N, L)', L='2')
            parsed = opts.parse(globals())
        To refer to variables and objects defined in the caller,
        call `parse` with env=globals()
        Case D.
        Call `parse` with env=globals() if some modules are required
        to evaluate some variables
            import numpy as np
            opts = OptionBaseClass(N='np.arange(5)', M='np.sqrt(N)')
            parsed = opts.parse(globals())
        """
        if self.is_parsed:
            print("Already parsed. Returns itself.")
            return self

        parsed_dict = dict()
        failure_count = 0
        key_list = list(self.__dict__)
        key_list.remove('is_parsed')

        if env is None:
            env = dict(globals())
        else:
            env = dict(env)
        env.update(globals())
        env.update(self.__dict__)

        while key_list:
            if failure_count >= len(key_list):
                print("Some options couldn't be parsed: " + str(key_list))
                print("Their values are as follows.")
                for key in key_list:
                    print(key + ': ' + getattr(self, key))

                print("\n" + "To find out the reason, see the document of " +
                      "`OptionBaseClass.parse`, and\n" +
                      "A. type-cast the option variables (see Case A);\n" +
                      "B. remove a cycle of variables (see Case B);\n" +
                      "C. call `parse` with env=globals() " +
                      "to use global variables (see Case C);\n" +
                      "D. import modules and functions such as " +
                      "`numpy`, `exp`, `sqrt` (see Case D).\n")

                print("Here are the parsed values:")
                pp = pprint.PrettyPrinter()
                pp.pprint(parsed_dict)

                print("Try 'OptionBaseClass.parse' with 'flg_print' option.")
                raise ValueError()

            key = key_list.pop()
            try:
                val = _myeval(getattr(self, key), env, parsed_dict)
                parsed_dict[key] = val
                if flg_print:
                    print(key + ': ' + repr(val))
                failure_count = 0
            except:
                key_list.insert(0, key)
                failure_count += 1

        parsed = self.create()
        key_list = list(self.__dict__)
        key_list.remove('is_parsed')
        for key in key_list:
            setattr(parsed, key, parsed_dict[key])
        parsed.is_parsed = True
        return parsed

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    
class ABCIterativeAlgorithm(object):
    """Iterative Algorithm Interface
    Main functionalities are: onestep, check, log, run, plotme, and plot
    onestep : perform one iteration of the algorithm, 
              call _onestep
    check   : check if a termination condition is satisfied, 
              call _check
    log     : write and display the internal states, 
              call _log_preprocess before taking a log
    run     : perform onestep-check-log loop untile 
              a termination condition is satisfied
    plotme  : plot the log data, shortcut of `plot`
    plot    : plot the log data
    Derived Classes need to implement the following methods and properties
    
    _onestep : one iteration of the algorithm
    _check   : termination checker 
    _log_preprocess : optional, preprocessing for log output.
                      mainly for monkeypatch.
    measure   : measure (int) of the run-length of the algorithm
    criterion : measure (float) of the progress of the algorithm
    recommendation : current recommendation (ndarray)
    See
    ---
    RandomSearch, RandomSearchOption
    """
    __metaclass__ = ABCMeta

    def __init__(self, opts):
        """
        Parameters
        ----------
        opts : IterativeAlgorithmOption or its derived class instance
            It can be either parsed or not parsed.
        """
        self._niter = 0
        self._time_step = 0.0
        self._time_check = 0.0
        self._time_log = 0.0
        assert isinstance(opts, IterativeAlgorithmOption)
        if opts.is_parsed:
            self.opts = opts
        else:
            self.opts = opts.parse()
            self.opts_original = opts
        # Logger dictionary
        self.logger = dict()
        if self.opts.log_prefix and self.opts.log_span > 0:
            self.profile_logger = self.opts.log_prefix + DELIM + PROFILE + EXT
            with open(self.profile_logger, 'w') as f:
                f.write('#' + type(self).__name__ + "\n")
        for key in self.opts.log_variable_list:
            self.logger[key] = self.opts.log_prefix + DELIM + key + EXT
            with open(self.logger[key], 'w') as f:
                f.write('#' + type(self).__name__ + "\n")

    def onestep(self):
        t0 = time.clock()
        self._onestep()
        self._time_step += time.clock() - t0
        self._niter += 1

    def check(self):
        t0 = time.clock()
        is_satisfied, condition = self._check()
        self._time_check += time.clock() - t0
        return is_satisfied, condition

    def log(self, flgforce=False, condition=''):
        """Take a log
        Parameters
        ----------
        flgforce : bool, optional
            force to take a log.
        condition : str, optional
            termination condition.
        """

        elapsed_time = self.time_step + self.time_check
        # Check if the log should be taken
        if ((flgforce and self.opts.log_span > 0) or
            (self.opts.log_span >= 1 and self.niter % self.opts.log_span == 0)
                or (1.0 > self.opts.log_span > 0.0 and
                    elapsed_time * self.opts.log_span > self.time_log)):

            t0 = time.clock()
            # Preprocess
            self._log_preprocess()
            # Display
            displayline = "{0} {1} {2} {3}".format(
                repr(self.niter),
                repr(self.measure), repr(elapsed_time), str(self.criterion))
            if self.opts.log_display:
                print(displayline)
                if condition:
                    print('# End with condition = ' + condition)
            with open(self.profile_logger, 'a') as f:
                f.write(displayline + "\n")
                if condition:
                    f.write('# End with condition = ' + condition)
            for key, log in self.logger.items():
                var = self.__dict__[key]
                if isinstance(var, np.ndarray) and len(var.shape) > 1:
                    var = var.flatten()
                varlist = np.hstack(
                    (self.niter, self.measure, elapsed_time, var))
                with open(log, 'a') as f:
                    f.write(' '.join(map(repr, varlist)) + "\n")
            self._time_log += time.clock() - t0

    def _log_preprocess(self):
        """Preprocess for logging
        This function is called at the top of `log` function.
        By default, it does nothing. This method will be implemented
        when one wants to record some values that are not the attribute
        of the `iterative_algorithm`.
        Two possible usage of this functions are:
            1. to implement this method in a derived class
            2. to monkeypatch this method outside the class definition
        An example of the second usage is as follows
        Example
        -------
        Assume that we want to record the product of `a` and `b`
        defined in the `iterative_algorithm`.
        >>> # Define a function to be monkeypatched
        >>> def monkeypatch(self):
        >>>    a = self.iterative_algorithm.a
        >>>    b = self.iterative_algorithm.b
        >>>    # set a variable to `self.iterative_algorithm`
        >>>    self.iterative_algorithm.a_times_b = a * b
        >>> # Replace `_log_preprocess`
        >>> ABCIterativeAlgorithm._log_preprocess = monkeypatch
        """
        pass

    def run(self):
        """Run the iterative algorithm
        If you want to monitor the behavior of the algorithm during the run,
        it is possible to do it by either using the multithreading or run a different interpreter
        Example 1
        ---------
        from threading import Thread
        th = Thread(target=algo.run, name='algo.run') # algo is an instance of this class
        th.start()
        algo.plotme()
        Example 2
        ---------
        algo.run()
        # Open a separate interpreter
        ABCIterativeAlgorithmOption.plot(...)
        """
        is_satisfied = False
        while not is_satisfied:
            if self.opts.check_exception:
                try:
                    self.onestep()
                    is_satisfied, condition = self.check()
                    self.log(flgforce=is_satisfied, condition=condition)
                except Exception as e:
                    is_satisfied = True
                    self.log(flgforce=True, condition='exception')
            else:
                self.onestep()
                is_satisfied, condition = self.check()
                self.log(flgforce=is_satisfied, condition=condition)

    def plotme(self, xaxis=MEASURE, **kwargs):
        """Plot the results using `plot` function
        Parameters
        ----------
        xaxis : int
            see xaxis in `plot`
        kwargs : optional parameters
            optional arguments to `plot`
        """
        return ABCIterativeAlgorithm.plot(opts=self.opts, xaxis=xaxis, **kwargs)

    @staticmethod
    def plot(prefix='',
             xaxis=MEASURE,
             variable_list=None,
             opts=None,
             ncols=None,
             figsize=None,
             cmap_='winter'):
        """Plot the result
        This allows to plot previously generated or post-processed data that
        has the same format as the log file generated by ``log``.
        Parameters
        ----------
        prefix : str
            the prefix of the log file
        xaxis : int
            NITER == 0. vs iter
            MEASURE == 1. vs measure
            ELAPSED == 2. vs cpu time in sec.
        variable_list : list of str
            names of variables
        opts : OptionBaseClass or its derived class
            If this is given, the other parameters are not needed
        Returns
        -------
        fig : figure object.
            figure object
        axdict : dictionary of axes
            the keys are the names of variables given in `variable_list`
        The log files must be located at ``prefix`` + PROFILE + EXT,
        and ``prefix`` + ``variable_list[i]`` + EXT.
        """
        if opts:
            prefix = opts.log_prefix
            variable_list = opts.log_variable_list
        if variable_list is None:
            variable_list = []

        # Default settings
        nfigs = 1 + len(variable_list)
        if ncols is None:
            ncols = int(np.ceil(np.sqrt(nfigs)))
        nrows = int(np.ceil(nfigs / ncols))
        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)
        axdict = dict()
        # Figure
        fig = plt.figure(figsize=figsize)
        # The first figure
        x = np.loadtxt(prefix + DELIM + PROFILE + EXT)
        x = x[~np.isnan(x[:, xaxis]), :]  # remove columns where xaxis is nan
        # Axis
        ax = plt.subplot(nrows, ncols, 1)
        ax.set_title('criterion')
        ax.grid(True)
        ax.grid(which='major', linewidth=0.50)
        ax.grid(which='minor', linewidth=0.25)
        plt.plot(x[:, xaxis], x[:, 3:])
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
        axdict['criterion'] = ax

        # The other figures
        idx = 1
        for key in variable_list:
            idx += 1
            x = np.loadtxt(prefix + DELIM + key + EXT)
            x = x[~np.isnan(
                x[:, xaxis]), :]  # remove columns where xaxis is nan
            ax = plt.subplot(nrows, ncols, idx)
            ax.set_title(r'\detokenize{' + key + '}')
            ax.grid(True)
            ax.grid(which='major', linewidth=0.50)
            ax.grid(which='minor', linewidth=0.25)
            if False:  # Before 2017/02/11
                plt.plot(x[:, xaxis], x[:, 3:])
            else:  # New
                cmap = plt.get_cmap(cmap_)
                cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 1)
                scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
                for i in range(x.shape[1] - 3):
                    plt.plot(
                        x[:, xaxis], x[:, 3 + i], color=scalarMap.to_rgba(i))
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(my_formatter))
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(my_formatter))
            axdict[key] = ax

        plt.tight_layout() # NOTE: not sure if it works fine
        return fig, axdict

    @property
    def niter(self):
        return self._niter

    @property
    def time_step(self):
        return self._time_step

    @property
    def time_check(self):
        return self._time_check

    @property
    def time_log(self):
        return self._time_log

    @abstractmethod
    def _onestep(self):
        """Do one step (one iteration) of the algorithm"""
        pass

    @abstractmethod
    def _check(self):
        """Check the termination criteria
        Returns
        -------
        flag : bool
            True if one of the termination condition is satisfied, False otherwise
        condition : str
            String of condition, empty string '' if no condition is satisfied
        """
        if self.criterion < self.opts.check_target:
            return True, 'target'
        elif self.niter >= self.opts.check_maxiter:
            return True, 'maxiter'
        elif self.time_step + self.time_check >= self.opts.check_maxsec:
            return True, 'maxsec'
        elif self.measure >= self.opts.check_maxruntime:
            return True, 'maxruntime'
        else:
            return False, ''

    @abstractproperty
    def measure(self):
        """Measure (int) of the run-length of the algorithm"""

    @abstractproperty
    def criterion(self):
        """Measure (float) of the progress of the algorithm"""

    @abstractproperty
    def recommendation(self):
        """The current recommendation (ndarray)"""


class IterativeAlgorithmOption(OptionBaseClass):
    """Option Container for Iterative Algorithm Class"""
    def __init__(self,
                 log_prefix=repr('.'),
                 log_span=repr(0.1),
                 log_display=repr(True),
                 log_variable_list=repr([]),
                 check_target="-np.inf",
                 check_maxiter="np.inf",
                 check_maxsec="np.inf",
                 check_maxruntime="np.inf",
                 check_exception=repr(False),
                 **kwargs):

        OptionBaseClass.__init__(self, **kwargs)
        self.setattr_from_local(locals())


class Summarizer:
    """Repeat the experiments with the same setting (parameters)
    For the usage, see `demo`
    """

    def __init__(self, target_value_array=None):
        """Summarizer
        Parameters
        ----------
        target_value_array : numpy 1D array
            vector of the target criteria values
        """
        if target_value_array is None:
            target_value_array = np.array([np.NaN])

        self.frame_target = pd.Series(
            np.sort(target_value_array)[::-1], dtype=float)
        self.frame_target.index.name = 'target_id'
        self.frame_target.name = 'target_val'
        self.frame_niter = pd.DataFrame(index=self.frame_target.index)
        self.frame_measure = pd.DataFrame(index=self.frame_target.index)
        self.frame_elapsed = pd.DataFrame(index=self.frame_target.index)
        self.dict_rawdata = dict()

    @staticmethod
    def _get_xaxis_name(number):
        if number == NITER:
            return 'No. iterations'
        elif number == MEASURE:
            return 'Measure'
        elif number == ELAPSED:
            return 'CPU time (sec)'

    def add(self, log_prefix):
        """Add a new result
        Parameteres
        -----------
        log_prefix : str
            path to the log files.
        """
        # Load the profile data
        profile_filename = log_prefix + DELIM + PROFILE + EXT
        dat = np.loadtxt(profile_filename)
        self.dict_rawdata[profile_filename] = dat
        # Compute target information
        s_niter, s_measure, s_elapsed = self._compute_target_info(dat)
        self.frame_niter[profile_filename] = s_niter
        self.frame_measure[profile_filename] = s_measure
        self.frame_elapsed[profile_filename] = s_elapsed

    def change_target(self, target_value_array):
        """Change the target array
        Parameters
        ----------
        target_value_array : ndarray (1D)
            the array of target values
        """
        # Set target frame
        self.frame_target = pd.Series(
            np.sort(target_value_array)[::-1], dtype=float)
        self.frame_target.index.name = 'target_id'
        self.frame_target.name = 'target_val'
        self.frame_niter = pd.DataFrame(index=self.frame_target.index)
        self.frame_measure = pd.DataFrame(index=self.frame_target.index)
        self.frame_elapsed = pd.DataFrame(index=self.frame_target.index)
        # Update target frame
        for profile_name in self.dict_rawdata:
            dat = self.dict_rawdata[profile_name]
            # Compute target information
            s_niter, s_measure, s_elapsed = self._compute_target_info(dat)
            self.frame_niter[profile_name] = s_niter
            self.frame_measure[profile_name] = s_measure
            self.frame_elapsed[profile_name] = s_elapsed

    def _compute_target_info(self, dat):
        """(Re-)Compute the runtime with respect to given target
        Parameters
        ----------
        dat : ndarray (2D)
            the array that is loaded from profile.dat by np.loadtxt
        """
        # Series
        series_niter = pd.Series(index=self.frame_target.index)
        series_measure = pd.Series(index=self.frame_target.index)
        series_elapsed = pd.Series(index=self.frame_target.index)
        # Set the profile dict
        for itarget in self.frame_target.index:
            target = self.frame_target[itarget]
            if dat.ndim == 1:
                if dat[CRITERION] > target:
                    break
                series_niter[itarget] = dat[NITER]
                series_measure[itarget] = dat[MEASURE]
                series_elapsed[itarget] = dat[ELAPSED]
            elif dat.ndim == 2:
                dat = dat[dat[:, CRITERION] <= target]
                if dat.size == 0:
                    break
                series_niter[itarget] = dat[0, NITER]
                series_measure[itarget] = dat[0, MEASURE]
                series_elapsed[itarget] = dat[0, ELAPSED]
            else:
                raise RuntimeError(
                    'dat file is empty or not correctly loaded.')

        return series_niter, series_measure, series_elapsed

    def save_as_csv(self, xaxis_name=MEASURE, float_format='{:,.2e}'.format):
        if xaxis_name == NITER:
            dat = self.frame_niter
        elif xaxis_name == MEASURE:
            dat = self.frame_measure
        elif xaxis_name == ELAPSED:
            dat = self.frame_elapsed
        else:
            raise NotImplementedError()
        #TODO: save as csv
        raise NotImplementedError()

    def get_summary(self, xaxis_name=MEASURE, float_format='{:,.2e}'.format):
        """Summary of the results w.r.t. the given xaxis
        Parameters
        ----------
        xaxis_name : int, default is 1.
            0 == Summarizer.NITER : number of iteration
            1 == Summarizer.MEASURE : value of measure
            2 == Summarizer.ELAPSED : cpu time in second
        float_format : format, default is '{:,.2e}'.format
            floating point number format for each cell
        """

        if xaxis_name == NITER:
            summary = self.frame_niter.T.describe().T
        elif xaxis_name == MEASURE:
            summary = self.frame_measure.T.describe().T
        elif xaxis_name == ELAPSED:
            summary = self.frame_elapsed.T.describe().T
        else:
            raise NotImplementedError()
        summary[self.frame_target.name] = self.frame_target
        # The floating point numbers are formated by the given formatter
        for col in summary.columns:
            if col != 'count':
                summary[col] = summary[col].map(float_format)
        return summary

    def get_data_profile(self, xaxis_name=MEASURE):
        """Data Profile
        Parameters
        ----------
        xaxis_name : int, default is 1.
            0 == Summarizer.NITER : number of iteration
            1 == Summarizer.MEASURE : value of measure
            2 == Summarizer.ELAPSED : cpu time in second
        fig : figure object, optional
            if it is given, the figure is drawn on the current one
        Returns
        -------
        xaxis : numpy.ndarray (1D) of NITER, MEASURE, or ELAPSED
        yaxis : numpy.ndarray (1D) of the proportion of the reached targets
        """
        if xaxis_name == NITER:
            xaxis = np.r_[[0], np.sort(self.frame_niter.values.flatten())]
        elif xaxis_name == MEASURE:
            xaxis = np.r_[[0], np.sort(self.frame_measure.values.flatten())]
        elif xaxis_name == ELAPSED:
            xaxis = np.r_[[0], np.sort(self.frame_elapsed.values.flatten())]
        else:
            raise NotImplementedError()
        yaxis = np.arange(xaxis.shape[0]) / float(xaxis.shape[0] - 1)
        return xaxis, yaxis

    def plot_proportion_of_reached_targets(self,
                                           xaxis_name=MEASURE,
                                           fig=None,
                                           **plot_option):
        """Data Profile
        Parameters
        ----------
        xaxis_name : int, default is 1.
            0 == Summarizer.NITER : number of iteration
            1 == Summarizer.MEASURE : value of measure
            2 == Summarizer.ELAPSED : cpu time in second
        fig : figure object, optional
            if it is given, the figure is drawn on the current one
        Returns
        -------
        fig : figure
        """

        plot_dict = {'ls': '-', 'drawstyle': 'steps-post'}
        plot_dict.update(plot_option)
        xaxis, yaxis = self.get_data_profile(xaxis_name=xaxis_name)
        # Plot
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()
        ax.plot(xaxis, yaxis, **plot_dict)
        xmin, xmax = ax.get_xlim()
        ax.set_xlabel(self._get_xaxis_name(xaxis_name))
        ax.set_ylabel('prop. of reached targets')
        ax.set_xlim(left=0, right=max(xmax, xaxis.max()))
        ax.grid(True)
        ax.grid(which='major', linewidth=0.50)
        ax.grid(which='minor', linewidth=0.25)
        return fig

    def plot_proportion_for_each_trial(self, xaxis_name=MEASURE, fig=None):
        """Data Profile for each run, similar to plot_rawdata
        Parameters
        ----------
        xaxis_name : int, default is 1.
            0 == Summarizer.NITER : number of iteration
            1 == Summarizer.MEASURE : value of measure
            2 == Summarizer.ELAPSED : cpu time in second
        fig : figure object, optional
            if it is given, the figure is drawn on the current one
        Returns
        -------
        fig : figure
        """
        plot_dict = {'ls': '-', 'drawstyle': 'steps-post'}
        if xaxis_name == NITER:
            xaxis = self.frame_niter.values
        elif xaxis_name == MEASURE:
            xaxis = self.frame_measure.values
        elif xaxis_name == ELAPSED:
            xaxis = self.frame_elapsed.values
        else:
            raise NotImplementedError()
        yaxis = np.arange(xaxis.shape[0]) / float(xaxis.shape[0] - 1)
        # Plot
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()
        for i in range(xaxis.shape[1]):
            ax.plot(xaxis[:, i], yaxis, **plot_dict)
        ax.set_xlabel(self._get_xaxis_name(xaxis_name))
        ax.set_ylabel('prop. of reached targets')
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1])
        ax.grid(True)
        ax.grid(which='major', linewidth=0.50)
        ax.grid(which='minor', linewidth=0.25)
        return fig

    def plot_raw_data(self, xaxis_name=MEASURE, fig=None):
        """Plot raw data
        Parameters
        ----------
        xaxis_name : int, default is 1.
            0 == Summarizer.NITER : number of iteration
            1 == Summarizer.MEASURE : value of measure
            2 == Summarizer.ELAPSED : cpu time in second
        fig : figure object, optional
            if it is given, the figure is drawn on the current one
        """

        plot_dict = {'ls': '-', 'drawstyle': 'steps-post'}
        # Plot
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()
        for key in self.dict_rawdata:
            xaxis = self.dict_rawdata[key][:, xaxis_name]
            yaxis = self.dict_rawdata[key][:, CRITERION]
            ax.plot(xaxis, yaxis, **plot_dict)
        ax.set_xlabel(self._get_xaxis_name(xaxis_name))
        ax.set_ylabel('criterion')
        ax.set_xlim(left=0)
        ax.grid(True)
        ax.grid(which='major', linewidth=0.50)
        ax.grid(which='minor', linewidth=0.25)
        return fig

    def plot_bandwidth(self,
                       xaxis_name=MEASURE,
                       quantile=[25, 50, 75],
                       fig=None):
        """Data Profile with bandwidth
        Parameters
        ----------
        xaxis_name : int, default is 1.
            0 == Summarizer.NITER : number of iteration
            1 == Summarizer.MEASURE : value of measure
            2 == Summarizer.ELAPSED : cpu time in second
        quantile : array of 3 numbers between 1 and 100
            quantile = [lower, center, upper]
            The lower and upper determine the bandwidth, while the center is the representative line.
        fig : figure object, optional
            if it is given, the figure is drawn on the current one
        """
        # Note: pandas.describe neglects NaN, while numpy.percentile does not.
        if xaxis_name == NITER:
            x = self.frame_niter
        elif xaxis_name == MEASURE:
            x = self.frame_measure
        elif xaxis_name == ELAPSED:
            x = self.frame_elapsed
        else:
            raise NotImplementedError()
        lower, median, upper = np.percentile(x, q=quantile, axis=1)
        y = np.arange(0, self.frame_target.index.shape[0] +
                      1) / float(self.frame_target.index.shape[0])
        # Plot
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()
        ax.fill_betweenx(
            y, np.r_[[0], upper], x2=np.r_[[0], lower], color='b', alpha=0.2)
        ax.plot(np.r_[[0], median], y, color='b')
        ax.set_xlabel(self._get_xaxis_name(xaxis_name))
        ax.set_ylabel('prop. of reached targets')
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1])
        ax.grid(True)
        ax.grid(which='major', linewidth=0.50)
        ax.grid(which='minor', linewidth=0.25)
        return fig

    def plot_probability(self, fig=None):
        """Data Profile for each run, similar to plot_rawdata
        Parameters
        ----------
        fig : figure object, optional
            if it is given, the figure is drawn on the current one
        Returns
        -------
        fig : figure
        """

        plot_dict = {'ls': '-', 'drawstyle': 'steps-post'}
        y = np.mean(~pd.isnull(self.frame_niter), axis=1)
        x = np.arange(1, self.frame_target.index.shape[0] +
                      1) / float(self.frame_target.index.shape[0])
        # Plot
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()
        ax.plot(np.r_[[0], x], np.r_[[1], y], color='g', **plot_dict)
        ax.set_xlabel('prop. of reached targets')
        ax.set_ylabel('success probability')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.01])
        ax.grid(True)
        ax.grid(which='major', linewidth=0.50)
        ax.grid(which='minor', linewidth=0.25)
        return fig

    def saveaslatex(self, prefix, opts=None, xaxis=MEASURE):
        """Output the summary figures and tables
        Parameters
        ----------
        prefix : str
            All the results will be produced in `prefix` + DELIM + 'summary/'
        opts : OptionBaseClass, optional
            If it is given, the parameter settings will be included in the resulting tex file
        xaxis : int, optional
            Either NITER, MEASURE, or ELAPSED.
        """
        directory = prefix + DELIM + 'summary/'
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Directory ' + directory + ' has been created.')
        print('All the result will be produced in ' + directory + '.')

        with open(directory + 'summary.tex', 'w') as f:
            f.write('\\documentclass[10pt]{article}\n')
            f.write(
                '\\usepackage[textwidth=0.9\\paperwidth,textheight=0.9\\paperheight]{geometry}\n'
            )
            f.write('\\usepackage[T1]{fontenc}\n')
            f.write('\\usepackage{booktabs}\n')
            f.write('\\usepackage{longtable}\n')
            f.write('\\usepackage{subcaption}\n')
            f.write('\\usepackage{graphicx}\n')
            f.write('\\begin{document}\n\n')
            # opts
            if opts:
                f.write(opts.to_latex())
                f.write('\\clearpage\n\n')
            # get_summary
            df = self.get_summary(xaxis_name=xaxis)
            f.write('\\begin{table}[t]\n')
            f.write('\\centering\n')
            f.write('\\caption{Statistics over ' + str(len(self.dict_rawdata))
                    + ' independent runs.}\n')
            f.write(df.reset_index().to_latex(index=False))
            f.write('\\end{table}\n')
            f.write('\\clearpage\n\n')
            # plot_proportion_of_reached_targets
            self.plot_proportion_of_reached_targets()
            epsname = 'proportion_of_reached_targets.eps'
            plt.savefig(directory + epsname)
            f.write('\\begin{figure}[t]\n')
            f.write('\\centering\n')
            f.write('\\includegraphics[]{' + epsname + '}\n')
            f.write(
                '\\caption{Proportion of reached targets over all runs.}\n')
            f.write('\\end{figure}\n')
            f.write('\\clearpage\n\n')
            # plot_proportion_of_reached_targets
            self.plot_proportion_for_each_trial()
            epsname = 'proportion_for_each_trial.eps'
            plt.savefig(directory + epsname)
            f.write('\\begin{figure}[h]\n')
            f.write('\\centering\n')
            f.write('\\includegraphics[]{' + epsname + '}\n')
            f.write('\\caption{Proportion of reached targets for each run.}\n')
            f.write('\\end{figure}\n')
            f.write('\\clearpage\n\n')
            # plot_probability
            self.plot_probability()
            epsname = 'probability.eps'
            plt.savefig(directory + epsname)
            f.write('\\begin{figure}[h]\n')
            f.write('\\centering\n')
            f.write('\\includegraphics[]{' + epsname + '}\n')
            f.write('\\caption{Probability to reach each target.}\n')
            f.write('\\end{figure}\n')
            f.write('\\clearpage\n\n')
            # end
            f.write('\\end{document}')


class Comparator(object):
    
    def __init__(self, target_value_array=[np.NaN]):
        self.s_target = pd.Series(target_value_array, dtype=float)
        self.s_target.index.name = 'target_id'
        self.s_target.name = 'target_val'
        self.dict_summarizer = dict()

    def change_target(self, target_value_array):
        self.s_target = pd.Series(target_value_array, dtype=float)
        self.s_target.index.name = 'target_id'
        self.s_target.name = 'target_val'
        for key in self.dict_summarizer:
            self.dict_summarizer[key].change_target(self.s_target.values)

    def add(self, summarizer, name):
        assert isinstance(summarizer, Summarizer), \
            "`summarizer` must be an instance of `Summarizer` class."
        self.dict_summarizer[name] = summarizer
        self.dict_summarizer[name].change_target(self.s_target.values)

    def get_summary(self, xaxis_name=MEASURE, float_format='{:,.2e}'.format):
        """
        """
        # Useful commands: pivot, xs
        isthefirst = True
        for key in self.dict_summarizer:
            summary = self.dict_summarizer[key].get_summary(
                xaxis_name=xaxis_name, float_format=float_format).T
            if isthefirst:
                frame = pd.DataFrame(summary.unstack(), columns=[key])
                frame.columns.name = 'log_prefix'
                isthefirst = False
            else:
                frame[key] = summary.unstack()
        return frame

    def get_summary_with_multi_param_index(self,
                                           frame_param,
                                           series_prefix,
                                           xaxis_name=MEASURE,
                                           float_format='{:,.2e}'.format):

        frame = self.get_summary(xaxis_name, float_format=float_format)
        index_frame = frame_param.copy()
        index_frame['log_prefix'] = series_prefix

        #print(list(map(tuple, index_frame.values)))
        multiindex = pd.MultiIndex.from_tuples(
            list(map(tuple, index_frame.values)), names=index_frame.columns)
        frame = frame[series_prefix.values].T
        frame.index = multiindex
        return frame

    def plot_proportion_of_reached_targets(self,
                                          xaxis_name=MEASURE,
                                          fig=None,
                                          **plot_option):

        xmax = []
        ymax = []
        keylist = []

        for key in self.dict_summarizer:
            xaxis, yaxis = self.dict_summarizer[key].get_data_profile(
                xaxis_name=xaxis_name)
            keylist.append(key)
            imax = xaxis.shape[0] - 1 - np.sum(np.isnan(xaxis))
            if imax < 0:
                ymax.append(np.nan)
                xmax.append(np.nan)
            else:
                ymax.append(yaxis[imax])
                xmax.append(xaxis[imax])
        xmax = np.array(xmax)
        ymax = np.array(ymax)

        if 11 < 3:
            # keep for backward compatibility check
            isort_by_xmax = xmax.argsort()[::-1]
            isort = isort_by_xmax[ymax[isort_by_xmax].argsort(
                kind='mergesort')]  # Stable sort
            isort = isort[::-1]
        else:
            isort = np.lexsort((xmax, -ymax))

        if fig == None:
            fig = plt.figure()
        for i in isort:
            key = keylist[i]
            label = r'\texttt{' + key.replace('_',
                                              '\_') + '}'  # Escape underscores
            self.dict_summarizer[key].plot_proportion_of_reached_targets(
                xaxis_name=xaxis_name, fig=fig, label=label, **plot_option)
        ax = fig.gca()
        ax.legend(loc='best', fancybox=True, shadow=True, fontsize=8)
        return fig

    # TODO: summary w.r.t. budget

    def saveaslatex(self, prefix, opts=None, xaxis=MEASURE):
        """Output the summary figure and tables
        Parameters
        ----------
        prefix : str
            All the results will be produced in `prefix` + DELIM + 'summary/'
        opts : OptionBaseClass, optional
            If it is given, the parameter settings will be included in the resulting tex file
        xaxis : int, optional
            Either NITER, MEASURE, or ELAPSED.
        """
        directory = prefix + DELIM + 'summary/'
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Directory ' + directory + ' has been created.')
        print('All the result will be produced in ' + directory + '.')

        with open(directory + 'summary.tex', 'w') as f:
            f.write('\\documentclass[10pt]{article}\n')
            f.write(
                '\\usepackage[textwidth=0.9\\paperwidth,textheight=0.9\\paperheight]{geometry}\n'
            )
            f.write('\\usepackage[T1]{fontenc}\n')
            f.write('\\usepackage{booktabs}\n')
            f.write('\\usepackage{rotating}\n')
            f.write('\\usepackage{longtable}\n')
            f.write('\\usepackage{subcaption}\n')
            f.write('\\usepackage{graphicx}\n')
            f.write('\\begin{document}\n\n')
            # opts
            if opts:
                f.write(opts.to_latex())
                f.write('\\clearpage\n\n')
            # get_summary
            df = self.get_summary(xaxis_name=xaxis)
            for i in range(self.s_target.shape[0]):
                f.write('\\begin{sidewaystable}[t]\n')
                f.write('\\centering\n')
                f.write('\\caption{Target(' + str(i) + ') = ' + str(
                    self.s_target[i]) + '.}\n')
                f.write(df.T[i].to_latex())
                f.write('\\end{sidewaystable}\n')
                f.write('\\clearpage\n\n')

            # plot_proportion_of_reached_targets
            self.plot_proportion_of_reached_targets()
            epsname = 'proportion_of_reached_targets.eps'
            plt.savefig(directory + epsname)
            f.write('\\begin{figure}[t]\n')
            f.write('\\centering\n')
            f.write('\\includegraphics[]{' + epsname + '}\n')
            f.write(
                '\\caption{Proportion of reached targets over all runs.}\n')
            f.write('\\end{figure}\n')
            f.write('\\clearpage\n\n')
            # end
            f.write('\\end{document}')


def get_multi_setting_info(dict_modified_param):
    """
    a DataFrame of all parameters tuples and
    a Series of string to identify the parameter settings
    Parameters
    ----------
    `dict_modified_param` : dict
        a dict of the parameters to be calibrated
        key : the name of a parameter
        value : the list of parameter values
    Returns
    -------
    `frame_param` : DataFrame
        consisting of all the combinations of the parameters
        index : int starting from 0
        column name : the name of a parameter
        values : value of each parameter
    `series_id` : Series
        consisting of strings that identify the settings
    Example
    -------
    >>> diff = dict()
    >>> diff['lam'] = ['10', '20', '40']
    >>> diff['cm'] = ['1', '0.1', '0.01']
    >>> diff['cmu'] = ['1', '0.1', '0.01']
    >>> frame_param, series_id = get_multi_setting_info(diff)
    See Also
    --------
    `Summarizer`, `Comparator`.
    """
    diff_values, diff_keys = dict_cartesian(dict_modified_param)
    frame_param = pd.DataFrame(data=diff_values, columns=diff_keys)
    series_id = pd.Series(index=frame_param.index)
    for idx in frame_param.index:
        series_id[idx] = DELIM.join([
            '='.join([key, str(frame_param.ix[idx, key])])
            for key in frame_param.columns
        ])
    return frame_param, series_id


def dict_cartesian(dict_):
    """Cartesian product of a dictionary
    Parameters
    ----------
    dict_ : dict
        key : string
        value : list of string
    Returns
    -------
    * ndarray (2D) consisting of the Cartesian product of the list in dict_
    * list of keys of dict_
    """

    list_ = list(dict_.values())
    n_list = [len(item) for item in list_]
    out = [[None] * np.prod(n_list) for i in range(len(n_list))]

    for i in range(len(n_list)):
        array_ = list_[i]
        n = 1
        if i < len(n_list) - 1:
            n = np.prod(n_list[i + 1:])
        for j in range(len(out[i])):
            out[i][j] = array_[(j % (n_list[i] * n)) // n]
    return np.array(out).T, dict_.keys()


def piv(path,
        df,
        diff,
        val_list,
        row,
        col,
        is_row_numbers=False,
        is_col_numbers=False):
    """
    Parameter
    ---------
    path     : str
        Output tex file name
    df       : pandas DataFrame
        The date frame to be parsed to latex table. The index must be reset.
        The typical example is as follows: Provided that `comp` is an instance of the comparator,
            df_param, s_param = get_multi_setting_info(diff)
                                            # get the parameter information
            comp.change_target([1e-8])      # set the only target
            df_ = comp.get_summary_with_multi_param_index(df_param, s_param)
                                            # get the summary
            df = df[0].reset_index()        # select the only target, reset the index
    diff     : dict
        dictionary of the parameter, typically the input of `get_multi_setting_info`
    val_list : list of str
        array of the output values, e.g., ['count', 'mean', 'std']
        For the other possible output values, see the output of the get_summary
    row      : str
        label of a column of `df`
    col      : str
        label of a column of `df`
    is_row_numbers : bool, default = False
        the resulting rows of the data frame will be sorted as numbers
    is_col_numbers : bool, default = False
        the resulting columns of the data frame will be sorted as numbers
    """
    diff_clone = dict(diff)
    _ = diff_clone.pop(row)
    _ = diff_clone.pop(col)
    if not diff_clone:  # diff_clone is empty
        _piv_2dim(path, df, val_list, row, col)
        return

    df_param, s_param = get_multi_setting_info(diff_clone)
    with open(path, 'w') as f:
        # LaTeX Preamble
        f.write('\\documentclass[10pt]{article}\n')
        f.write(
            '\\usepackage[textwidth=0.9\\paperwidth,textheight=0.9\\paperheight]{geometry}\n'
        )
        f.write('\\usepackage[T1]{fontenc}\n')
        f.write('\\usepackage{booktabs}\n')
        f.write('\\usepackage{longtable}\n')
        f.write('\\usepackage{subcaption}\n')
        f.write('\\usepackage{graphicx}\n')
        f.write('')
        f.write('\\begin{document}\n')

        for idx in s_param.index:

            # Each Table
            f.write('\\begin{table}[t]\n')
            f.write('\\centering\n')
            f.write('\\caption{\\detokenize{' + s_param[idx] + '}}\n')

            # Select the rows
            selected = df
            for key, value in df_param.ix[idx].iteritems():
                selected = selected[selected[key] == value]
            #print(selected)

            for val in val_list:

                # Pivot the Table
                # pandas pivot function tends to break the order
                # when the indeces or columns are `string of numbers`
                # The following code is a hack for it.
                pivoted = selected.reset_index().pivot(index=row, columns=col)
                _col = pivoted[val].columns
                _idx = pivoted[val].index
                if is_col_numbers:
                    icol = np.argsort(np.array(_col, dtype=float))
                else:
                    icol = np.arange(len(_col))
                if is_row_numbers:
                    iidx = np.argsort(np.array(_idx, dtype=float))
                else:
                    iidx = np.arange(len(_idx))
                to_print = pivoted[val].ix[_idx[iidx], _col[icol]]

                code = to_print.to_latex()

                # Each Sub-Table
                f.write('\\begin{subtable}[h]{\\textwidth}\n')
                f.write('\\centering\n')
                f.write('\\caption{\\detokenize{' + val + '}}\n')
                f.write(code)
                f.write('\\end{subtable}\n')

            f.write('\\end{table}\n')
        f.write('\\end{document}')


def _piv_2dim(path, df, val_list, row, col):

    with open(path, 'w') as f:
        # LaTeX Preamble
        f.write('\\documentclass[10pt]{article}\n')
        f.write(
            '\\usepackage[textwidth=0.9\\paperwidth,textheight=0.9\\paperheight]{geometry}\n'
        )
        f.write('\\usepackage[T1]{fontenc}\n')
        f.write('\\usepackage{booktabs}\n')
        f.write('\\usepackage{longtable}\n')
        f.write('\\usepackage{subcaption}\n')
        f.write('\\usepackage{graphicx}\n')
        f.write('')
        f.write('\\begin{document}\n')

        # Each Table
        f.write('\\begin{table}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{}\n')

        for val in val_list:

            # Pivot the Table
            pivoted = df.reset_index().pivot(index=row, columns=col)
            code = pivoted[val].to_latex()

            # Each Sub-Table
            f.write('\\begin{subtable}[h]{\\textwidth}\n')
            f.write('\\centering\n')
            f.write('\\caption{\\detokenize{' + val + '}}\n')
            f.write(code)
            f.write('\\end{subtable}\n')

        f.write('\\end{table}\n')
        f.write('\\end{document}')


class RandomSearch(ABCIterativeAlgorithm):
    """Uniform Random Search
    It is a very simple example usage of `ABCIterativeAlgorithm`.
    Example
    -------
    import numpy as np
    import matplotlib.pylab as plt
    from util import iterative_algorithm_interface as iai
    # func
    def func(x):
        return (x * x).sum()
    # Option
    opts = iai.RandomSearchOption(N='10',
                                 lbound='-1.0',
                                 ubound='1.0',
                                 log_span='0.01',
                                 log_variable_list="['xbest']",
                                 check_maxsec='3.')
    opts.disp()  # check the default setting
    # Run
    seed = 100
    np.random.seed(seed)
    para = opts.parse()
    para.log_prefix = para.log_prefix + iai.DELIM + 'seed=' + str(seed)
    algo = iai.RandomSearch(para, func)
    algo.run()
    # Plot
    algo.plotme()
    plt.savefig(para.log_prefix + '.eps')
    """

    def __init__(self, opts, func):

        super(RandomSearch, self).__init__(opts)

        self.func = func
        self.neval = 1

        self.N = self.opts.N
        self.lbound = self.opts.lbound
        self.ubound = self.opts.ubound
        self.xbest = self.lbound + (self.ubound - self.lbound) * np.random.rand(self.N)
        self.fbest = self.func(self.xbest)

    def _onestep(self):

        xnew = self.lbound + (self.ubound - self.lbound) * np.random.rand(self.N)
        fnew = self.func(xnew)
        if fnew <= self.fbest:
            self.xbest = xnew
            self.fbest = fnew
        self.neval += 1

    def _check(self):
        return super(RandomSearch, self)._check()

    def _log_preprocess(self):
        pass

    @property
    def measure(self):
        return self.neval

    @property
    def criterion(self):
        return self.fbest

    @property
    def recommendation(self):
        return self.xbest


class RandomSearchOption(IterativeAlgorithmOption):

    def __init__(self,
                 N='0',
                 lbound='0 # either float or 1d numpy array',
                 ubound='0 # either float or 1d numpy array',
                 **kwargs):

        super(RandomSearchOption, self).__init__(**kwargs)
        self.setattr_from_local(locals())

if __name__ == '__main__':

    print("Interactive Demo Mode")
    print("1: Single Run")
    print("2: Multiple run with Summarizer")
    print("3: Multiple settings with Comparator")
    demo_mode = int(input("Select the mode number (int): "))
    print("")
    print("Flag for experiment")
    print("1: the experiment is (re)run.")
    print("0: the existing .dat file is loaded and post-processed.")
    flg_experiment = bool(int(input("1 or 0 (int): ")))

    
    def func(x):
        y = np.asarray(x)
        return np.dot(y, y)

    if demo_mode == 1:
        demo_directory = 'demo1'        
        if not os.path.exists(demo_directory):
            os.makedirs(demo_directory)    

        # Option
        opts = RandomSearchOption(
            N=5,
            lbound=-5.,
            ubound=5.,
            check_target=1e-4,
            check_maxruntime=1e4,
            log_prefix=repr(os.path.join(demo_directory, 'RandomSearchDemo')),
            log_variable_list=["xbest"]).parse()

        if flg_experiment:
            # Algorithm
            algo = RandomSearch(opts, func)
            algo.run()

            # Plot
            # One can call ABCIterativeAlgorithm.plot function instead of plotme method
            fig, axdict = algo.plotme()
            axdict['criterion'].set_yscale('log')
            plt.savefig(algo.opts.log_prefix + '.eps')
            plt.savefig(algo.opts.log_prefix + '.pdf')
            plt.savefig(algo.opts.log_prefix + '.png')                        
        else:
            # Plot
            fig, axdict = ABCIterativeAlgorithm.plot(opts=opts)
            axdict['criterion'].set_yscale('log')
            plt.savefig(opts.log_prefix + '.eps')
            plt.savefig(opts.log_prefix + '.pdf')
            plt.savefig(opts.log_prefix + '.png')                        

    elif demo_mode == 2:
        demo_directory = 'demo2'        
        if not os.path.exists(demo_directory):
            os.makedirs(demo_directory)    
            
        # Seed
        seedarray = [100, 200, 300, 400, 500]

        # Option
        opts = RandomSearchOption(
            N='2',
            lbound='-1.0',
            ubound='1.0',
            log_span='0.01',
            log_variable_list=repr([]),
            log_prefix=repr(os.path.join(demo_directory, 'SummarizerDemo')),
            check_maxsec='0.5',
            check_target=1e-4)

        # Run
        if flg_experiment:
            for seed in seedarray:
                np.random.seed(
                    seed
                )  # Set the seed for the pseudo random number generator `np.random`
                para = opts.parse(
                )  # Parse the options. All the options are evaluated.
                para.log_prefix = para.log_prefix + DELIM + 'seed=' + str(
                    seed)  # Set the path to the output data
                algo = RandomSearch(para,
                                    func)  # Create a RandomSearch instance
                algo.run()  # Run

        # Post-process
        target_array = np.logspace(3.0, -4.0, num=50, endpoint=True, base=10.0)
        prefix = opts.parse().log_prefix
        summarizer = Summarizer(target_array)
        for seed in seedarray:
            summarizer.add(prefix + DELIM + 'seed=' + str(seed))
        summarizer.saveaslatex(prefix, opts=opts)

    elif demo_mode == 3:
        demo_directory = 'demo3'        
        if not os.path.exists(demo_directory):
            os.makedirs(demo_directory)    
        
        prefix = os.path.join(demo_directory, 'ComparatorDemo')

        # Parameter Setting
        diff = dict()
        diff['N'] = ["2", "3", "5", "10"]
        diff['lbound'] = ['-4.', '-2.', '0.']
        diff['ubound'] = ['5.', '3.', '1.']
        df, s = get_multi_setting_info(diff)

        # Seed array
        seedarray = [100, 200, 300, 400, 500]

        # Run
        if flg_experiment:
            for i in range(s.shape[0]):
                print('#=================================================#')
                print('Run for ' + s[i])
                opts = RandomSearchOption(
                    check_target=repr(1e-4),
                    check_maxruntime=repr(int(4e3)),
                    log_variable_list=repr([]),
                    log_span='10',
                    log_prefix=repr(prefix),
                    **df.ix[i])
                for seed in seedarray:
                    print('#-----------------------------#')
                    print('Seed = ' + repr(seed))
                    np.random.seed(seed)
                    parsed = opts.parse()
                    parsed.log_prefix += DELIM + s[i] + DELIM + 'seed=' + repr(
                        seed)
                    algo = RandomSearch(parsed, func)
                    algo.run()

        # Post-process
        target_array = np.logspace(3.0, -8.0, num=50, endpoint=True, base=10.0)
        comp = Comparator(target_array)
        for i in range(s.shape[0]):
            summ = Summarizer()
            for seed in seedarray:
                summ.add(prefix + DELIM + s[i] + DELIM + 'seed=' + repr(seed))
            comp.add(summ, s[i])

        # Summary
        #comp.saveaslatex(prefix)

        # Figure
        fig = comp.plot_proportion_of_reached_targets()
        ax = fig.gca()
        ax.set_xlim(xmin=1)
        ax.set_xscale('log')
        plt.savefig(prefix + '.eps')

        # Pivot LaTeX
        target_array = np.array([1e-8])
        comp.change_target(target_array)
        df_summary = comp.get_summary_with_multi_param_index(df, s)
        df_summary_flatten = df_summary[target_array.shape[0] -
                                        1].reset_index()
        piv(prefix + '.tex',
            df_summary_flatten,
            diff, ['count', 'mean', 'std'],
            row='lbound',
            col='ubound')

    else:
        print('`demo_mode` must be either 1, 2, or 3. ')
