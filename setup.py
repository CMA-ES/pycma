#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""setup for cma package distribution.

To prepare the docs from a dirty code folder::

    conda activate py27
    backup cma --move
    git checkout -- cma
    pip install -e .
    pydoctor --docformat=restructuredtext --make-html cma  > pydoctor-messages.txt ; less pydoctor-messages.txt
    backup --recover
    # see script-make-doc for syncing to gforge

To prepare a distribution from a dirty code folder::

    backup cma --move    # backup is a homebrew minitool
    git checkout -- cma
    python setup.py check
    python setup.py sdist bdist_wheel --universal > dist_call_output.txt ; less dist_call_output.txt  # bdist_wininst
    twdiff cma build/lib/cma/  # just checking
    backup --recover  # recover above moved folder (and backup current, just in case)

Check distribution and project description:

    tree build  # check that the build folders are clean
    twine check dist/*
    python setup.py --long-description | rst2html.py > long-description.html ; open long-description.html

Finally upload the distribution::

    twine upload dist/*3.0.3*  # to not upload outdated stuff

Anaconda::

    code tools/conda.recipe/meta.yaml  # edit version number
    conda-build -q tools/conda.recipe

"""
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
      long_description_content_type = 'text/x-rst', # 'text/markdown',
      version=__version__.split()[0],
      description="CMA-ES, Covariance Matrix Adaptation " +
                  "Evolution Strategy for non-linear numerical " +
                  "optimization in Python",
      author="Nikolaus Hansen",
      author_email="authors_firstname.lastname@inria.fr",
      maintainer="Nikolaus Hansen",
      # maintainer_email="authors_firstname.lastname@inria.fr",
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
          # "Programming Language :: Python :: 2.6",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Development Status :: 5 - Production/Stable",
          "Environment :: Console",
          "Framework :: IPython",
          "Framework :: Jupyter",
          "License :: OSI Approved :: BSD License",
          # "License :: OSI Approved :: MIT License",
      ],
      keywords=["optimization", "CMA-ES", "cmaes"],
      packages=["cma", "cma.utilities"],
      install_requires=["numpy"],
      extras_require={
            "plotting": ["matplotlib"],
            # "wrap-skopt": ["scikit-optimize"]  # who wants to wrap skopt has skopt already installed
      },
      package_data={'': ['LICENSE']},  # i.e. cma/LICENSE
      )
