#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Obsolete and replaced by pyproject.toml, still contains the howto. setup for cma package distribution.

Switch to the desired branch.

Run local tests

    ./script-test-all-all-arm.sh
    ruff check cma

Push to a test branch to trigger test:

    git push origin :test  # delete remote test branch if necessary
    git push origin HEAD:test

Check/edit version numbers into (new) commit vX.X.X::

    code cma/__init__.py  # edit version number
    code tools/conda.recipe/meta.yaml  # edit version number

Add a release note (based on git ls, same commit) in::

    ./README.md  # add release description and amend v.X.X.X. commit

To check the apidocs from a dirty code folder:

    ./script-make-apidocs.sh

        ==>
            backup apidocs --move
            backup cma --move
            git checkout -- cma
            pydoctor --docformat=restructuredtext --html-output=apidocs cma > pydoctor-messages.txt
            backup --recover
            less +G pydoctor-messages.txt  # check for errors (which are at the end!)

Make and check the distribution from a (usual) dirty code folder ==> install-folder::

    ./script-prepare-distribution.sh
    
        ==>
            backup install-folder --move  # CAVEAT: not the homebrew tool
            mkdir install-folder  # delete if necessary
            backup cma --move    # backup is a self-coded minitool
            git checkout -- cma
            cp -rp cma pyproject.toml LICENSE README.txt install-folder
            backup --recover  # recover above moved folder (and backup current, just in case)

    cd install-folder
    python -m build > dist_call_output.txt; less +G dist_call_output.txt
    twine check dist/*  # fails with Python 3.13
    less +G dist_call_output.txt  # errors are shown in the end
    tar -tf dist/cma-4.2.0.tar.gz | tree --fromfile | less
                #   ==> 36 files, check that the distribution folders are clean
                # not really useful anymore as we copy into a clean folder

# see https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html#summary

Loop over tests and distribution and fix code until everything is fine.

Upload the distribution::

    twine upload dist/*4.2.0*  # to not upload outdated stuff

Push new docs to github

    cp -r apidocs/* /Users/hansen/git/CMA-ES.github.io/apidocs-pycma
    cd /Users/hansen/git/CMA-ES.github.io
    git add apidocs-pycma  # there may be new files
    git ci
    git push

Tag and push git branch::

    git tag -a r4.2.0  # refactor boundary handling code, add UnboundDomain stand-alone class
    git push origin r4.2.0

Create a release on GitHub (click on releases and then new draft, or an r4... tag will do?).

Anaconda::

    # edit version number in tools/conda.recipe/meta.yaml
    conda-build -q tools/conda.recipe  # takes about 1/2 hour

"""
# from distutils.core import setup
from setuptools import setup
from cma import __version__  # assumes that the right module is visible first in path, i.e., cma folder is in current folder
from cma import __doc__ as long_description  # is overwritten below

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
            "constrained-solution-tracking": ["moarchiving"],
            # "wrap-skopt": ["scikit-optimize"]  # who wants to wrap skopt has skopt already installed
      },
      package_data={'': ['LICENSE']},  # i.e. cma/LICENSE
      )
