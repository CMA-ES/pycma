# -*- coding: utf-8 -*-
"""Package `cma` implements the CMA-ES (Covariance Matrix Adaptation
Evolution Strategy).

CMA-ES is a stochastic optimizer for robust non-linear non-convex
derivative- and function-value-free numerical optimization.

This implementation can be used with Python versions >= 2.6, namely
2.6, 2.7, 3.3, 3.4, 3.5, 3.6.

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

- functions `fmin` and `purecma.fmin`:
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
To use the module, the folder ``cma`` only needs to be visible in the
python path, e.g. in the current working directory.

To install the module from pipy, type::

    pip install cma

from the command line.

To install the module from a ``cma`` folder::

    pip install -e cma

To upgrade the currently installed version use additionally the ``-U``
option.

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
    res = cma.fmin(cma.ff.tablet, 15 * [1], 1)
    es = cma.CMAEvolutionStrategy(15 * [1], 1).optimize(cma.ff.tablet)
    help(es.result)
    res[0], es.result[0]  # best evaluated solution
    res[5], es.result[5]  # mean solution, presumably better with noise

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

from __future__ import absolute_import  # now local imports must use .
# big difference between PY2 and PY3:
from __future__ import division
from __future__ import print_function
# only necessary for python 2.5 (not supported) and not in heavy use
from __future__ import with_statement
___author__ = "Nikolaus Hansen and Petr Baudis and Youhei Akimoto"
__license__ = "BSD 3-clause"

# __package__ = 'cma'
from . import purecma
try:
    from . import (constraints_handler, evolution_strategy, fitness_functions,
                   fitness_transformations, interfaces, optimization_tools,
                   sampler, sigma_adaptation, transformations, utilities,
                   )
except ImportError:
    print('Only `cma.purecma` has been imported. Install `numpy` ("pip'
          ' install numpy") if you want to import the entire `cma`'
          ' package.')
else:
    # from . import test  # gives a warning with python -m cma.test (since Python 3.5.3?)
    test = 'type "import cma.test" to access the `test` module of `cma`'
    from . import s
    from .fitness_functions import ff
    from .fitness_transformations import GlueArguments, ScaleCoordinates
    from .evolution_strategy import fmin, fmin2, CMAEvolutionStrategy, CMAOptions
    from .logger import disp, plot, CMADataLogger
    from .optimization_tools import NoiseHandler
    from .constraints_handler import BoundPenalty, BoundTransform

del division, print_function, absolute_import, with_statement  #, unicode_literals

# fcts = ff  # historical reasons only, replace cma.fcts with cma.ff first

__author__ = 'Nikolaus Hansen'
__version__ = "2.7.0  $Revision: 4426 $ $Date: 2019-04-24 18:03:09 +0200 (Wed, 24 Apr 2019) $"
# $Source$  # according to PEP 8 style guides, but what is it good for?
# $Id: __init__.py 4426 2019-04-24 16:03:09Z hansen $
# bash $: svn propset svn:keywords 'Date Revision Id' __init__.py
