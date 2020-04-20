# pycma &nbsp; &nbsp; &nbsp; &nbsp; 
[![CircleCI](https://circleci.com/gh/CMA-ES/pycma/tree/master.svg?style=shield)](https://circleci.com/gh/CMA-ES/pycma/tree/master)
[![Build status](https://ci.appveyor.com/api/projects/status/1rge11pwyt55b26k?svg=true)](https://ci.appveyor.com/project/nikohansen/pycma)
[![DOI](https://zenodo.org/badge/68926339.svg)](https://doi.org/10.5281/zenodo.2559634)
[[BibTeX](http://cma.gforge.inria.fr/pycmabibtex.bib)] cite as:
> Nikolaus Hansen, Youhei Akimoto, and Petr Baudis. CMA-ES/pycma on Github. Zenodo, [DOI:10.5281/zenodo.2559634](https://doi.org/10.5281/zenodo.2559634), February 2019. 
---

<!--- 

[![Build status](https://ci.appveyor.com/api/projects/status/1rge11pwyt55b26k/branch/master?svg=true)](https://ci.appveyor.com/project/nikohansen/pycma/branch/master)

Zenodo: 34 points to the latest, this is 35: https://zenodo.org/badge/latestdoi/68926339 

--->
  
``pycma`` is a Python implementation of [CMA-ES](http://cma.gforge.inria.fr/) and a few related numerical optimization tools.

The [Covariance Matrix Adaptation Evolution Strategy](https://en.wikipedia.org/wiki/CMA-ES) 
([CMA-ES](http://cma.gforge.inria.fr/)) is a stochastic derivative-free numerical optimization
algorithm for difficult (non-convex, ill-conditioned, multi-modal, rugged, noisy) optimization
problems in continuous search spaces.

Useful links:

* [A quick start guide with a few usage examples](https://pypi.python.org/pypi/cma)

* [The API Documentation](http://cma.gforge.inria.fr/apidocs-pycma)

* [Hints for how to use this (kind of) optimization module in practice](http://cma.gforge.inria.fr/cmaes_sourcecode_page.html#practical)

## Installation of the [(almost) latest release](https://pypi.python.org/pypi/cma)

Type

    python -m pip install cma

in a system shell to install the [latest _release_](https://pypi.python.org/pypi/cma)
from the [Python Package Index (PyPI)](https://pypi.python.org/pypi) (which may be 
behind the lastest release tag on Github). The release link also provides more installation hints and a quick start guide.

    conda install --channel cma-es cma

installs from the conda cloud channel `cma-es`.

## Installation of the current master branch

The quick way (requires git to be installed):

    pip install git+https://github.com/CMA-ES/pycma.git@master

The long version: download and unzip the code (see green button above) or
``git clone https://github.com/CMA-ES/pycma.git`` and

- either, copy (or move) the ``cma`` source code folder into a folder visible to Python, 
  namely a folder which is in the Python path (e.g. the current folder). Then, 
  ``import cma`` works without any further installation.

- or, install the ``cma`` package by typing within the folder, where the ``cma`` source 
  code folder is visible,

      pip install -e cma

  Moving the ``cma`` folder away from its location would invalidate this
  installation.

It may be necessary to replace ``pip`` with ``python -m pip`` and/or prefixing
either of these with ``sudo``.

## Version History

* [Release ``3.0.2``](https://github.com/CMA-ES/pycma/releases/tag/r3.0.2) provides parallelization with ``OOOptimizer.optimize(..., n_jobs=...)`` and improved `pickle` support.

* [Release ``3.0.0``](https://github.com/CMA-ES/pycma/releases/tag/r3.0.0) provides non-linear constraints handling, improved plotting and termination options and better resilience to injecting bad solutions, and further various fixes.

* Version ``2.7.1`` allows for a list of termination callbacks and a light copy of `CMAEvolutionStrategy` instances.

* [Release ``2.7.0``](https://github.com/CMA-ES/pycma/releases/tag/r2.7.0) logger now writes into a folder, new fitness model module, various fixes.

* [Release ``2.6.1``](https://github.com/CMA-ES/pycma/releases/tag/r2.6.1) allow possibly much larger condition numbers, fix corner case with growing more-to-write list.

* [Release ``2.6.0``](https://github.com/CMA-ES/pycma/releases/tag/r2.6.0) allows initial solution `x0` to be a callable.

* Version ``2.4.2`` added the function `cma.fmin2` which, similar to `cma.purecma.fmin`, 
  returns ``(x_best:numpy.ndarray, es:cma.CMAEvolutionStrategy)``  instead of a 10-tuple
  like `cma.fmin`.
  
* Version ``2.4.1`` included ``bbob`` testbed.

* Version ``2.2.0`` added VkD CMA-ES to the master branch.

* Version ``2.*`` is a multi-file split-up of the original module.

* Version ``1.x.*`` is a one file implementation and not available in the history of
  this repository. The latest ``1.*`` version ```1.1.7`` can be found
  [here](https://pypi.python.org/pypi/cma/1.1.7).
  
