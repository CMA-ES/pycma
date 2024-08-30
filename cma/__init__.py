# -*- coding: utf-8 -*-
"""Package `cma` implements the CMA-ES (Covariance Matrix Adaptation
Evolution Strategy).

CMA-ES is a stochastic optimizer for robust non-linear non-convex
derivative- and function-value-free numerical optimization.

This implementation can be used with Python versions >= 2.7, namely,
it was tested with 2.7, 3.5, 3.6, 3.7, 3.8.

CMA-ES searches for a minimizer (a solution x in :math:`R^n`) of an
objective function f (cost function), such that f(x) is minimal.
Regarding f, only a passably reliable ranking of the candidate
solutions in each iteration is necessary. Neither the function values
itself, nor the gradient of f need to be available or do matter (like
in the downhill simplex Nelder-Mead algorithm). Some termination
criteria however depend on actual f-values.

The `cma` module provides two independent implementations of the
CMA-ES algorithm in the classes `cma.CMAEvolutionStrategy` and
`cma.purecma.CMAES`.

In each implementation two interfaces are provided:

- functions `fmin2` and `purecma.fmin`:
    run a complete minimization of the passed objective function with
    CMA-ES. `fmin` also provides optional restarts and noise handling.

- class `CMAEvolutionStrategy` and `purecma.CMAES`:
    allow for minimization such that the control of the iteration
    loop remains with the user.

The `cma` package root provides shortcuts to these and other classes and
functions.

Used external packages are `numpy` (only `purecma` does not depend on
`numpy`) and `matplotlib.pyplot` (for `plot` etc., optional but highly
recommended).

Install
=======

To install the package from `PyPI`_ with `pip`_, type::

    pip install cma

from the command line.

To install the package from a ``cma`` folder::

    pip install -e cma

To upgrade the currently installed version use additionally the ``-U``
option.

To use the package `from source`_, only the source folder ``cma`` needs to
be visible in the python path, e.g. in the current working directory.


.. _PyPI: https://pypi.org/project/cma
.. _pip: https://pip.pypa.io
.. _from source: https://github.com/CMA-ES/pycma

Testing
=======
From the system shell::

    python -m cma.test -h
    python -m cma.test
    python -c "import cma.test; cma.test.main()"  # the same

or from any (i)python shell::

    import cma.test
    cma.test.main()

should run without complaints in about between 20 and 100 seconds.

Example
=======
From a python shell::

    import cma
    help(cma)  # "this" help message, use cma? in ipython
    help(cma.fmin)
    help(cma.CMAEvolutionStrategy)
    help(cma.CMAOptions)
    cma.CMAOptions('tol')  # display 'tolerance' termination options
    cma.CMAOptions('verb') # display verbosity options
    x, es = cma.fmin2(cma.ff.tablet, 15 * [1], 1)
    es = cma.CMAEvolutionStrategy(15 * [1], 1).optimize(cma.ff.tablet)
    help(es.result)
    x, es.result[0]  # best evaluated solution
    es.result[5]  # mean solution, presumably better with noise

:See also: `fmin` (), `CMAOptions`, `CMAEvolutionStrategy`

:Author: Nikolaus Hansen, 2008-
:Author: Petr Baudis, 2014
:Author: Youhei Akimoto, 2017-

:License: BSD 3-Clause, see LICENSE file.

"""

# How to create a html documentation file:
#    pydoctor --docformat=restructuredtext --make-html cma
# old:
#    pydoc -w cma  # edit the header (remove local pointers)
#    epydoc cma.py  # comes close to javadoc but does not find the
#                   # links of function references etc
#    doxygen needs @package cma as first line in the module docstring
#       some things like class attributes are not interpreted correctly
#    sphinx: doc style of doc.python.org, could not make it work (yet)
# __docformat__ = "reStructuredText"  # this hides some comments entirely?

from __future__ import absolute_import as _ab # now local imports must use .
from __future__ import division as _di
from __future__ import print_function as _pr
del _ab, _di, _pr


___author__ = "Nikolaus Hansen and Petr Baudis and Youhei Akimoto"
__license__ = "BSD 3-clause"

import warnings as _warnings

# __package__ = 'cma'
from . import purecma
try:
    import numpy as _np
    del _np
except ImportError:
    _warnings.warn('Only `cma.purecma` has been imported. Install `numpy` ("pip'
          ' install numpy") if you want to import the entire `cma`'
          ' package.')
else:
    from . import (constraints_handler, evolution_strategy, fitness_functions,
                    fitness_transformations, interfaces, optimization_tools,
                    sampler, sigma_adaptation, transformations, utilities,
                    )
    # from . import test  # gives a warning with python -m cma.test (since Python 3.5.3?)
    test = 'type "import cma.test" to access the `test` module of `cma`'
    from . import s
    from .fitness_functions import ff
    from .fitness_transformations import GlueArguments, ScaleCoordinates
    from .evolution_strategy import fmin, fmin2, fmin_con, fmin_con2, fmin_lq_surr, fmin_lq_surr2
    from .evolution_strategy import CMAEvolutionStrategy
    from .options_parameters import CMAOptions, cma_default_options_
    CMA = CMAEvolutionStrategy  # shortcut for typing without completion
    from .logger import disp, plot, plot_zip, CMADataLogger
    from .optimization_tools import NoiseHandler
    from .constraints_handler import BoundPenalty, BoundTransform, ConstrainedFitnessAL

# fcts = ff  # historical reasons only, replace cma.fcts with cma.ff first

__version__ = "4.0.0"
# $Source$  # according to PEP 8 style guides, but what is it good for?
# $Id: __init__.py 4432 2020-05-28 18:39:09Z hansen $
# bash $: svn propset svn:keywords 'Date Revision Id' __init__.py
