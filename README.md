# pycma
A Python implementation of CMA-ES and a few related numerical optimization tools. 

The [Covariance Matrix Adaptation Evolution Strategy](https://en.wikipedia.org/wiki/CMA-ES) 
(CMA-ES) is a stochastic numerical optimization algorithm for difficult (non-convex, 
ill-conditioned, multi-modal, rugged, noisy) optimization problems in continuous search 
spaces. 

The API Documentation is available [here](http://cma.gforge.inria.fr/apidocs-pycma).

## Installation

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

## Installation of the [latest release](https://pypi.python.org/pypi/cma)

Typing
```
  python -m pip install cma
```
in a system shell installs the [most recent _release_](https://pypi.python.org/pypi/cma)
from the [Python Package Index (PyPI)](https://pypi.python.org/pypi). The [release link](https://pypi.python.org/pypi/cma)
also provides more installation hints and a quick start guide.
