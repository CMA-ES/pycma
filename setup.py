#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""setup for cma package distribution.

Switch to master branch.

Run local tests

    ./script-test-all.py

Push to origin/development to trigger test:

    git push origin HEAD:development

Edit version numbers into (new) commit vX.X.X::

    code cma/__init__.py  # edit version number
    code tools/conda.recipe/meta.yaml  # edit version number

Add a release note (based on git ls, same commit) in::

    ./README.md  # add release description

To prepare the docs from a dirty code folder::

    backup cma --move
    git checkout -- cma
    pip install -e .
    pydoctor --docformat=restructuredtext --html-output=apidocs cma > pydoctor-messages.txt ; less pydoctor-messages.txt  # check for errors (which are at the end!)
    backup --recover

    # push new docs to github
    cp -r apidocs/* /Users/hansen/git/CMA-ES.github.io/apidocs-pycma
    cd /Users/hansen/git/CMA-ES.github.io; git add apidocs-pycma  # there may be new files
    git ci
    git push

To prepare a distribution from a (usual) dirty code folder::

    backup cma --move    # backup is a homebrew minitool
    git checkout -- cma
    python setup.py check  # obsolete but why not
      # python build was: python setup.py sdist bdist_wheel --universal > dist_call_output.txt ; less dist_call_output.txt  # bdist_wininst
    python -m build > dist_call_output.txt ; less dist_call_output.txt  # see https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html#summary
    # bbdiff cma build/lib/cma/  # just checking
    backup --recover  # recover above moved folder (and backup current, just in case)

Check distribution and project description:

    tree build | less # check that the build folders are clean
    twine check dist/*
    # python setup.py --long-description | rst2html.py > long-description.html ; open long-description.html

Push branch to master::

    git push

Finally upload the distribution::

    twine upload dist/*3.x.x*  # to not upload outdated stuff

Anaconda::

    conda-build -q tools/conda.recipe  # takes about 1/2 hour

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
