#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Obsolete and replaced by pyproject.toml, still contains the howto. setup for cma package distribution.

Switch to the desired branch.

0.
Check/edit version numbers into (new) commit vX.X.X::

    code cma/__init__.py  # edit version number
    code tools/conda.recipe/meta.yaml  # edit version number

1.
Run local tests

    ruff check cma
    ./script-test-all-all-arm.sh

2.
Push to a test branch to trigger test:

    g push origin :test; g push origin HEAD:test

3.
Add a release note (based on git ls, same commit) in::

    ./README.md  # add release description and amend v.X.X.X. commit

4.
To check the apidocs from a dirty code folder:

    ./script-make-apidocs.sh

        ==>
            backup apidocs --move
            backup cma --move
            git checkout -- cma
            pydoctor --docformat=restructuredtext --html-output=apidocs cma > pydoctor-messages.txt
            backup --recover
            less pydoctor-messages.txt  # less +G = check for errors (which are at the end!)

5.
Make and check the distribution assuming a clean src/ folder (ln -s ../cma src/cma, was: from a (usual) dirty code folder ==> install-folder)::

CAVEAT trouble shooting: remove folders (/bin/rm -r) when they have been created
    - ./cma-4.4...
    - src/cma.egg-info... (may not be necessary)
    - cma.egg-info (may not be necessary)

    python -m build > dist_call_output.txt; less dist_call_output.txt
    git tag -f last-build  # not sure whether this is useful

    twine check dist/*  # needs py314
    tar -tf dist/cma-4.4.0.tar.gz | tree --fromfile | less
                #   ==> 7 directories, 40 files, check that the distribution folders are clean
                #   was: ==> 5 directories, 36 files, check that the distribution folders are clean

# see https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html#summary

6.
Loop over tests and distribution and fix code until everything is fine.

7.a
Create a tag to pick later for the Github release::

    git tag -a r4.4.1 -m 'r4.4.1 bug fixes and...'

(Alternatively: push code to development and/or main (this must be done at some
point anyways, if only to update the README.md.) and ``git fetch`` or with
``--tags`` to fetch tags not followed remotely.)

7b.
Draft a release on Github (click on releases and then new draft) and use the
text above added in README.md.

7.c
Upload the distribution (was: in ``install-folder``)::

    # optional (test upload):
        in pyproject.toml: change "name = cma" to = cmae
        python -m build > dist_call_outpute.txt
        twine upload --repository testpypi dist/cmae*
        open https://test.pypi.org/project/cmae

    twine upload dist/*4.4.4*
    git push origin r4.4.1

8.
On Github: select the tag, review and publish the release.

9.
Push new docs to github

    cp -r apidocs/* /Users/hansen/git/CMA-ES.github.io/apidocs-pycma
    cd /Users/hansen/git/CMA-ES.github.io
    git add apidocs-pycma  # there may be new files
    git ci
    git push

10. (if not already done)
Update main and push main and development to remote


conda-forge::

    # edit version number in tools/conda.recipe/meta.yaml
    ./script-prepare-distribution.sh  # make clean install-folder
    conda-build -q tools/conda.recipe  # takes about 4 minutes

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
