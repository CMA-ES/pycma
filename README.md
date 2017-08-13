# pycma

A Python implementation of CMA-ES and a few related numerical optimization tools.

The [Covariance Matrix Adaptation Evolution Strategy](https://en.wikipedia.org/wiki/CMA-ES) 
([CMA-ES](http://cma.gforge.inria.fr/)) is a stochastic derivative-free numerical optimization
algorithm for difficult (non-convex, ill-conditioned, multi-modal, rugged, noisy) optimization
problems in continuous search spaces.

A quick start guide with a few usage examples is given [here](https://pypi.python.org/pypi/cma).

The API Documentation is available [here](http://cma.gforge.inria.fr/apidocs-pycma).

Hints of how to use this (kind of) optimization module in practice can be found
[here](http://cma.gforge.inria.fr/cmaes_sourcecode_page.html#practical).

## Installation of the [latest release](https://pypi.python.org/pypi/cma)

Type
```
  python -m pip install cma
```
in a system shell to install the [latest _release_](https://pypi.python.org/pypi/cma)
from the [Python Package Index (PyPI)](https://pypi.python.org/pypi). The
release link also provides more installation hints and a quick start guide.

## Installation of the current master branch

Download and unzip the code (see green button above) or 
``git clone https://github.com/CMA-ES/pycma.git``. 

- Either, copy (or move) the ``cma`` source code folder into a folder visible to Python, 
  namely a folder which is in the Python path (e.g. the current folder). Then, 
  ``import cma`` works without any further installation.

- Or, install the ``cma`` package by typing within the folder, where the ``cma`` source 
  code folder is visible,

      python -m pip install -e cma

  Typing ``pip`` instead of ``python -m pip`` may be sufficient, prefixing with ``sudo`` 
  may be necessary. Moving the ``cma`` folder away from this location would invalidate the 
  installation.

## Version History

* Version ``2.2.0`` added VkD CMA-ES to the master branch.

* Version ``2.*`` is a multi-file split-up of the original module.

* Version ``1.x.*`` is a one file implementation and not available in the history of
  this repository. The latest ``1.*`` version ```1.1.7`` can be found
  [here](https://pypi.python.org/pypi/cma/1.1.7).
  
