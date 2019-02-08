# pycma &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;[![DOI](https://zenodo.org/badge/68926339.svg)](https://doi.org/10.5281/zenodo.2559634)

<!--- 34 points to the latest, this is 35: https://zenodo.org/badge/latestdoi/68926339 --->

A Python implementation of CMA-ES and a few related numerical optimization tools.

The [Covariance Matrix Adaptation Evolution Strategy](https://en.wikipedia.org/wiki/CMA-ES) 
([CMA-ES](http://cma.gforge.inria.fr/)) is a stochastic derivative-free numerical optimization
algorithm for difficult (non-convex, ill-conditioned, multi-modal, rugged, noisy) optimization
problems in continuous search spaces.

Useful links:

* [A quick start guide with a few usage examples](https://pypi.python.org/pypi/cma)

* [The API Documentation](http://cma.gforge.inria.fr/apidocs-pycma)

* [Hints for how to use this (kind of) optimization module in practice](http://cma.gforge.inria.fr/cmaes_sourcecode_page.html#practical)

## Installation of the [latest release](https://pypi.python.org/pypi/cma)

Type
```
  python -m pip install cma
```
in a system shell to install the [latest _release_](https://pypi.python.org/pypi/cma)
from the [Python Package Index (PyPI)](https://pypi.python.org/pypi). The
release link also provides more installation hints and a quick start guide.

## Installation of the current master branch

The quick way (requires git to be installed):

     pip install git+https://github.com/CMA-ES/pycma.git@master

The long version: download and unzip the code (see green button above) or
``git clone https://github.com/CMA-ES/pycma.git``. 

- Either, copy (or move) the ``cma`` source code folder into a folder visible to Python, 
  namely a folder which is in the Python path (e.g. the current folder). Then, 
  ``import cma`` works without any further installation.

- Or, install the ``cma`` package by typing within the folder, where the ``cma`` source 
  code folder is visible,

      pip install -e cma

  Moving the ``cma`` folder away from its location would invalidate this
  installation.

It may be necessary to replace ``pip`` with ``python -m pip`` and/or prefixing
either of these with ``sudo``.

## Version History

* Version ``2.4.2`` added the function `cma.fmin2` which, similar to `cma.purecma.fmin`, 
  returns ``(x_best:numpy.ndarray, es:cma.CMAEvolutionStrategy)``  instead of a 10-tuple
  like `cma.fmin`.

* Version ``2.2.0`` added VkD CMA-ES to the master branch.

* Version ``2.*`` is a multi-file split-up of the original module.

* Version ``1.x.*`` is a one file implementation and not available in the history of
  this repository. The latest ``1.*`` version ```1.1.7`` can be found
  [here](https://pypi.python.org/pypi/cma/1.1.7).
  
