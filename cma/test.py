#!/usr/bin/env python
"""test module of `cma` package.

Usage::

    python -m cma.test -h    # print this docstring
    python -m cma.test       # doctest all (listed) files
    python -m cma.test list  # list files to be doctested
    python -m cma.test interfaces.py [file2 [file3 [...]]] # doctest only these

or possibly by executing this file as a script::

    python cma/test.py  # same options as above work
    cma/test.py         # the same

or equivalently by passing Python code::

    python -c "import cma.test; cma.test.main()"  # doctest all (listed) files
    python -c "import cma.test; cma.test.main('list')"  # show files in doctest list
    python -c "import cma.test; cma.test.main('interfaces.py [file2 [file3 [...]]]')"
    python -c "import cma.test; help(cma.test)"  # print this docstring

File(name)s are interpreted within the package. Without a filename
argument, all files from attribute `files_for_doc_test` are tested.
"""

# (note to self) for testing:
#   pyflakes cma.py   # finds bugs by static analysis
#   pychecker --limit 60 cma.py  # also executes, all 60 warnings checked
#   or python ~/Downloads/pychecker-0.8.19/pychecker/checker.py cma.py
#   python -3 -m cma  2> out2to3warnings.txt # produces no warnings from here

from __future__ import (absolute_import, division, print_function,
                        )  # unicode_literals)
import os, sys
import doctest
del absolute_import, division, print_function  #, unicode_literals

files_for_doctest = ['bbobbenchmarks.py',
                     'constraints_handler.py',
                     'evolution_strategy.py',
                     'fitness_functions.py',
                     'fitness_models.py',
                     'fitness_transformations.py',
                     'interfaces.py',
                     'optimization_tools.py',
                     'purecma.py',
                     'recombination_weights.py',
                     'restricted_gaussian_sampler.py',
                     'sampler.py',
                     'sigma_adaptation.py',
                     'test.py',
                     'transformations.py',
                     os.path.join('utilities', 'math.py'),
                     os.path.join('utilities', 'utils.py'),
    ]
_files_written = ['_saved-cma-object.pkl',
                  'outcmaesaxlen.dat',
                  'outcmaesaxlencorr.dat',
                  'outcmaesfit.dat',
                  'outcmaesstddev.dat',
                  'outcmaesxmean.dat',
                  'outcmaesxrecentbest.dat',
    ]
"""files written by the doc tests and hence, in case, to be deleted"""

PY2 = sys.version_info[0] == 2
def _clean_up(folder, start_matches, protected):
    """(permanently) remove entries in ``folder`` which begin with any of
    ``start_matches``, where ``""`` matches any string, and which are not
    in ``protected``.

    CAVEAT: use with care, as with ``"", ""`` as second and third
    arguments this could delete all files in ``folder``.
    """
    if not os.path.isdir(folder):
        return
    if not protected and "" in start_matches:
        raise ValueError(
            '''_clean_up(folder, [..., "", ...], []) is not permitted as it
               resembles "rm *"''')
    protected = protected + ["/"]
    for file_ in os.listdir(folder):
        if any(file_.startswith(s) for s in start_matches) \
                and not any(file_.startswith(p) for p in protected):
            os.remove(os.path.join(folder, file_))
def is_str(var):  # copy from utils to avoid relative import
    """`bytes` (in Python 3) also fit the bill"""
    if PY2:
        types_ = (str, unicode)
    else:
        types_ = (str, bytes)
    return any(isinstance(var, type_) for type_ in types_)

def various_doctests():
    """various doc tests.

    This function describes test cases and might in future become
    helpful as an experimental tutorial as well. The main testing feature
    at the moment is by doctest with ``cma.test.main()`` in a Python shell
    or by ``python -m cma.test`` in a system shell.

    A simple first overall test:

        >>> import cma
        >>> res = cma.fmin(cma.ff.elli, 3*[1], 1,
        ...                {'CMA_diagonal':2, 'seed':1, 'verb_time':0})
        ...                # doctest: +ELLIPSIS
        (3_w,7)-aCMA-ES (mu_w=2.3,w_1=58%) in dimension 3 (seed=1,...)
           Covariance matrix is diagonal for 2 iterations (1/ccov=6...
        Iterat #Fevals   function value  axis ratio  sigma ...
        >>> assert res[1] < 1e-6
        >>> assert res[2] < 2000

    Testing output file consistency with diagonal option:

        >>> import cma
        >>> for val in (0, True, 2, 3):
        ...     _ = cma.fmin(cma.ff.sphere, 3 * [1], 1,
        ...                  {'verb_disp':0, 'CMA_diagonal':val, 'maxiter':5})
        ...     _ = cma.CMADataLogger().load()

    Test on the Rosenbrock function with 3 restarts. The first trial only
    finds the local optimum, which happens in about 20% of the cases.

        >>> import cma
        >>> res = cma.fmin(cma.ff.rosen, 4 * [-1], 0.01,
        ...                options={'ftarget':1e-6,
        ...                     'verb_time':0, 'verb_disp':500,
        ...                     'seed':3},
        ...                restarts=3)
        ...                # doctest: +ELLIPSIS
        (4_w,8)-aCMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=3,...)
        Iterat #Fevals ...
        >>> assert res[1] <= 1e-6

    Notice the different termination conditions. Termination on the target
    function value ftarget prevents further restarts.

    Test of scaling_of_variables option

        >>> import cma
        >>> opts = cma.CMAOptions()
        >>> opts['seed'] = 4567
        >>> opts['verb_disp'] = 0
        >>> opts['CMA_const_trace'] = True
        >>> # rescaling of third variable: for searching in  roughly
        >>> #   x0 plus/minus 1e3*sigma0 (instead of plus/minus sigma0)
        >>> opts['scaling_of_variables'] = [1, 1, 1e3, 1]
        >>> res = cma.fmin(cma.ff.rosen, 4 * [0.1], 0.1, opts)
        >>> assert res[1] < 1e-9
        >>> es = res[-2]
        >>> es.result_pretty()  # doctest: +ELLIPSIS
        termination on tolfun=1e-11
        final/bestever f-value = ...

    The printed std deviations reflect the actual value in the
    parameters of the function (not the one in the internal
    representation which can be different).

    Test of CMA_stds scaling option.

        >>> import cma
        >>> opts = cma.CMAOptions()
        >>> s = 5 * [1]
        >>> s[0] = 1e3
        >>> opts.set('CMA_stds', s)  #doctest: +ELLIPSIS
        {'...
        >>> opts.set('verb_disp', 0)  #doctest: +ELLIPSIS
        {'...
        >>> res = cma.fmin(cma.ff.cigar, 5 * [0.1], 0.1, opts)
        >>> assert res[1] < 1800

    Testing combination of ``fixed_variables`` and ``CMA_stds`` options.

        >>> import cma
        >>> options = {
        ...     'fixed_variables':{1:2.345},
        ...     'CMA_stds': 4 * [1],
        ...     'minstd': 3 * [1]}
        >>> es = cma.CMAEvolutionStrategy(4 * [1], 1, options) #doctest: +ELLIPSIS
        (3_w,7)-aCMA-ES (mu_w=2.3,w_1=58%) in dimension 3 (seed=...

    Test of elitism:

        >>> import cma
        >>> res = cma.fmin(cma.ff.rastrigin, 10 * [0.1], 2,
        ...       {'CMA_elitist':'initial', 'ftarget':1e-3, 'verbose':-9})
        >>> assert 'ftarget' in res[7]

    Test CMA_on option and similar:

        >>> import cma
        >>> res = cma.fmin(cma.ff.sphere, 4 * [1], 2,
        ...      {'CMA_on':False, 'ftarget':1e-8, 'verbose':-9})
        >>> assert 'ftarget' in res[7] and res[2] < 1e3
        >>> res = cma.fmin(cma.ff.sphere, 3 * [1], 2,
        ...      {'CMA_rankone':0, 'CMA_rankmu':0, 'ftarget':1e-8,
        ...       'verbose':-9})
        >>> assert 'ftarget' in res[7] and res[2] < 1e3
        >>> res = cma.fmin(cma.ff.sphere, 2 * [1], 2,
        ...      {'CMA_rankone':0, 'ftarget':1e-8, 'verbose':-9})
        >>> assert 'ftarget' in res[7] and res[2] < 1e3
        >>> res = cma.fmin(cma.ff.sphere, 2 * [1], 2,
        ...      {'CMA_rankmu':0, 'ftarget':1e-8, 'verbose':-9})
        >>> assert 'ftarget' in res[7] and res[2] < 1e3

    Check rotational invariance:

        >>> import cma
        >>> felli = cma.s.ft.Shifted(cma.ff.elli)
        >>> frot = cma.s.ft.Rotated(felli)
        >>> res_elli = cma.CMAEvolutionStrategy(3 * [1], 1,
        ...           {'ftarget': 1e-8}).optimize(felli).result
        ...                  #doctest: +ELLIPSIS
        (3_w,7)-...
        >>> res_rot = cma.CMAEvolutionStrategy(3 * [1], 1,
        ...         {'ftarget': 1e-8}).optimize(frot).result
        ...                  #doctest: +ELLIPSIS
        (3_w,7)-...
        >>> assert res_rot[3] < 2 * res_elli[3]

    Both condition alleviation transformations are applied during this
    test, first in iteration 62, second in iteration 257:

        >>> import cma
        >>> ftabletrot = cma.fitness_transformations.Rotated(cma.ff.tablet, seed=10)
        >>> es = cma.CMAEvolutionStrategy(4 * [1], 1, {
        ...                                   'tolconditioncov':False,
        ...                                   'seed': 8,
        ...                                   'CMA_mirrors': 0,
        ...                                   'ftarget': 1e-9,
        ...                                })  # doctest:+ELLIPSIS
        (4_w...
        >>> while not es.stop() and es.countiter < 82:
        ...    X = es.ask()
        ...    es.tell(X, [cma.ff.elli(x, cond=1e22) for x in X])  # doctest:+ELLIPSIS
        NOTE ...iteration=81...
        >>> while not es.stop():
        ...    X = es.ask()
        ...    es.tell(X, [ftabletrot(x, cond=1e32) for x in X])  # doctest:+ELLIPSIS
        >>> assert es.countiter <= 344 and 'ftarget' in es.stop(), (
        ...             "transformation bug in alleviate_condition?",
        ...             es.countiter, es.stop())

    Integer handling:

    >>> import warnings
    >>> idx = [0, 1, 5, -1]
    >>> f = cma.s.ft.IntegerMixedFunction(cma.ff.elli, idx)
    >>> with warnings.catch_warnings(record=True) as warns:
    ...     es = cma.CMAEvolutionStrategy(4 * [5], 10, dict(
    ...                   ftarget=1e-9, seed=5,
    ...                   integer_variables=idx
    ...                ))  # doctest:+ELLIPSIS
    (4_w,8)-...
    >>> warns[0].message  # doctest:+ELLIPSIS
    UserWarning('integer index 5 not in range of dimension 4 ()'...
    >>> es.optimize(f)  # doctest:+ELLIPSIS
    Iterat #Fevals   function value ...
    >>> assert 'ftarget' in es.stop() and es.result[3] < 1800

    Parallel objective:

    >>> def parallel_sphere(X): return [cma.ff.sphere(x) for x in X]
    >>> x, es = cma.fmin2(cma.ff.sphere, 3 * [0], 0.1, {
    ...     'verbose': -9, 'eval_final_mean': True, 'CMA_elitist': 'initial'},
    ...                   parallel_objective=parallel_sphere)
    >>> assert es.result[1] < 1e-9
    >>> x, es = cma.fmin2(None, 3 * [0], 0.1, {
    ...     'verbose': -9, 'eval_final_mean': True, 'CMA_elitist': 'initial'},
    ...                   parallel_objective=parallel_sphere)
    >>> assert es.result[1] < 1e-9

    Some sort of interactive control via an options file:

    >>> es = cma.CMAEvolutionStrategy(4 * [2], 1, dict(
    ...                      signals_filename='cma_signals.in',
    ...                      verbose=-9))
    >>> s = es.stop()
    >>> es = es.optimize(cma.ff.sphere)

    Test of huge lambda:

    >>> es = cma.CMAEvolutionStrategy(3 * [0.91], 1, {
    ...     'verbose': -9,
    ...     'popsize': 200,
    ...     'ftarget': 1e-8 })
    >>> es = es.optimize(cma.ff.tablet)
    >>> assert es.result.evaluations < 5000

    For VD- and VkD-CMA, see `cma.restricted_gaussian_sampler`.

    """

def doctest_files(file_list=files_for_doctest, **kwargs):
    """doctest all (listed) files of the `cma` package.

    Details: accepts ``verbose`` and all other keyword arguments that
    `doctest.testfile` would accept, while negative ``verbose`` values
    are passed as 0.
    """
    # print("__name__ is", __name__, sys.modules[__name__])
    # print(__package__)
    if not isinstance(file_list, list) and is_str(file_list):
        file_list = [file_list]
    verbosity_here = kwargs.get('verbose', 0)
    if verbosity_here < 0:
        kwargs['verbose'] = 0
    failures = 0
    for file_ in file_list:
        file_ = file_.strip().strip(os.path.sep)
        if file_.startswith('cma' + os.path.sep):
            file_ = file_[4:]
        if verbosity_here >= 0:
            print('doctesting %s ...' % file_,
                  ' ' * (max(len(_file) for _file in file_list) -
                         len(file_)),
                  end="")  # does not work in Python 2.5
            sys.stdout.flush()
        protected_files = os.listdir('.')
        report = doctest.testfile(file_,
                                  package=__package__,  # 'cma', # sys.modules[__name__],
                                  **kwargs)
        _clean_up('.', _files_written, protected_files)
        failures += report[0]
        if verbosity_here >= 0:
            print(report)
    return failures

def get_version():
    try:
        with open(__file__[:-7] + '__init__.py', 'r') as f:
            for line in f.readlines():
                if line.startswith('__version__'):
                    return line[15:].split()[0]
    except:
        return ""
        print(__file__)
        raise

def main(*args, **kwargs):
    """test the `cma` package.

    The first argument can be '-h' or '--help' or 'list' to list all
    files to be tested. Otherwise, arguments can be file(name)s to be
    tested, where names are interpreted relative to the package root
    and a leading 'cma' + path separator is ignored.

    By default all files are tested.

    :See also: ``python -c "import cma.test; help(cma.test)"``
    """
    if len(args) > 0:
        if args[0].startswith(('-h', '--h')):
            print(__doc__)
            exit(0)
        elif args[0].startswith('list'):
            for file_ in files_for_doctest:
                print(file_)
            exit(0)
    else:
        v = get_version()
        print("doctesting `cma` package%s by calling `doctest_files`:"
              % ((" (v%s)" % v) if v else ""))
    return doctest_files(args if args else files_for_doctest, **kwargs)

if __name__ == "__main__":
    exit(main(*sys.argv[1:]) > 0)  # 0 if failures == 0 else 1