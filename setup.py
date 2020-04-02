# from distutils.core import setup
from setuptools import setup
from cma import __version__  # assumes that the right module is visible first in path, i.e., cma folder is in current folder
from cma import __doc__ as long_description

# prevent the error when building Windows .exe
import codecs
try:
    codecs.lookup('mbcs')
except LookupError:
    ascii = codecs.lookup('ascii')
    func = lambda name, enc=ascii: {True: enc}.get(name=='mbcs')
    codecs.register(func)

# packages = ['cma'],  # indicates a multi-file module and that we have a cma folder and cma/__init__.py file

try:
    with open('README.txt') as file:
        long_description = file.read()  # now assign long_description=long_description below
except IOError:  # file not found
    pass

setup(name="cma",
      long_description=long_description,  # __doc__, # can be used in the cma.py file
      version=__version__.split()[0],
      description="CMA-ES, Covariance Matrix Adaptation " +
                  "Evolution Strategy for non-linear numerical " +
                  "optimization in Python",
      author="Nikolaus Hansen",
      author_email="authors firstname.lastname at inria dot fr",
      maintainer="Nikolaus Hansen",
      maintainer_email="authors_firstname.lastname@inria.fr",
      url="https://github.com/CMA-ES/pycma",
      # license="MIT",
      license="BSD",
      classifiers = [
          "Intended Audience :: Science/Research",
          "Intended Audience :: Education",
          "Intended Audience :: Other Audience",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 2.6",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Development Status :: 4 - Beta",
          "Environment :: Console",
          "License :: OSI Approved :: BSD License",
          # "License :: OSI Approved :: MIT License",
      ],
      keywords=["optimization", "CMA-ES", "cmaes"],
      packages=["cma", "cma.utilities"],
      requires=["numpy", "scikit-optimize"],
      package_data={'': ['LICENSE']},  # i.e. cma/LICENSE
      )
