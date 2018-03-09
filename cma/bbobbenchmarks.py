#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""BBOB noiseless testbed.

The optimisation test functions are represented as classes `F1` to
`F24` (and `F101` to `F130`).

This module implements the class `BBOBFunction` and
sub-classes:

* class `BBOBNfreeFunction` which have all the methods common to the
  classes `F1` to `F24`
* classes `BBOBGaussFunction`, `BBOBCauchyFunction`,
  `BBOBUniformFunction` which have methods in classes from
  `F101` to `F130`

Module attributes:

* `dictbbob` is a dictionary such that dictbbob[2] contains
  the test function class F2 and f2 = dictbbob[2]() returns
  the instance 0 of the test function that can be
  called as f2([1,2,3]).
* `nfreeIDs` == range(1,25) indices for the noiseless functions that can be
  found in dictbbob
* `noisyIDs` == range(101, 131) indices for the noisy functions that can be
  found in dictbbob. We have nfreeIDs + noisyIDs == sorted(dictbbob.keys())
* `nfreeinfos` function infos

Examples:

>>> from cma import bbobbenchmarks as bn
>>> for s in bn.nfreeinfos:
...    print(s)
1: Noise-free Sphere function
2: Separable ellipsoid with monotone transformation
<BLANKLINE>
    Parameter: condition number (default 1e6)
<BLANKLINE>
<BLANKLINE>
3: Rastrigin with monotone transformation separable "condition" 10
4: skew Rastrigin-Bueche, condition 10, skew-"condition" 100
5: Linear slope
6: Attractive sector function
7: Step-ellipsoid, condition 100, noise-free
8: Rosenbrock noise-free
9: Rosenbrock, rotated
10: Ellipsoid with monotone transformation, condition 1e6
11: Discus (tablet) with monotone transformation, condition 1e6
12: Bent cigar with asymmetric space distortion, condition 1e6
13: Sharp ridge
14: Sum of different powers, between x^2 and x^6, noise-free
15: Rastrigin with asymmetric non-linear distortion, "condition" 10
16: Weierstrass, condition 100
17: Schaffers F7 with asymmetric non-linear transformation, condition 10
18: Schaffers F7 with asymmetric non-linear transformation, condition 1000
19: F8F2 sum of Griewank-Rosenbrock 2-D blocks, noise-free
20: Schwefel with tridiagonal variable transformation
21: Gallagher with 101 Gaussian peaks, condition up to 1000, one global rotation, noise-free
22: Gallagher with 21 Gaussian peaks, condition up to 1000, one global rotation
23: Katsuura function
24: Lunacek bi-Rastrigin, condition 100
<BLANKLINE>
    in PPSN 2008, Rastrigin part rotated and scaled
<BLANKLINE>
<BLANKLINE>
>>> f3 = bn.F3(13)  # instantiate instance 13 of function f3
>>> f3([0, 1, 2]) # short-cut for f3.evaluate([0, 1, 2]) # doctest:+ELLIPSIS
59.8733529...
>>> print(bn.instantiate(5)[1])  # returns function instance and optimal f-value
51.53
>>> print(bn.nfreeIDs) # list noise-free functions
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
>>> for i in bn.nfreeIDs: # evaluate all noiseless functions once
...    print(bn.instantiate(i)[0]([0., 0., 0., 0.])) # doctest:+ELLIPSIS
-77.2745459...
6180022.8217...
92.987750752...
92.987750752...
140.51011761...
70877.955412...
-72.550520219...
33355.792472...
-339.94
4374717.4934...
15631566.348...
4715481.086...
550.59978390...
-17.299175622...
27.363312851...
-227.82783352...
-24.330591878...
131.42015934...
40.710373742...
6160.8178292...
376.74688954...
107.83042676...
220.48226655...
106.09476738...

"""

# TODO: define interface for this module.
# TODO: funId is expected to be a number since it is used as rseed.

from __future__ import print_function
try:
    xrange
except NameError:
    xrange = range
import warnings
from pdb import set_trace
import numpy as np
from math import floor as floor
from numpy import dot, linspace, diag, tile, zeros, sign, resize
from numpy.random import standard_normal as _randn # TODO: may bring confusion
from numpy.random import random as _rand # TODO: may bring confusion

"""
% VAL = BENCHMARKS(X, FUNCID)
% VAL = BENCHMARKS(X, STRFUNC)
%    Input:
%       X -- solution column vector or matrix of column vectors
%       FUNCID -- number of function to be executed with X as input,
%                 by default 8.
%       STRFUNC -- function as string to be executed with X as input
%    Output: function value(s) of solution(s)
%    Examples:
%      F = BENCHMARKS([1 2 3]', 17);
%      F = BENCHMARKS([1 2 3]', 'f1');
%
% NBS = BENCHMARKS()
% NBS = BENCHMARKS('FunctionIndices')
%    Output:
%      NBS -- array of valid benchmark function numbers,
%             presumably 1:24
%
% FHS = BENCHMARKS('handles')
%    Output:
%      FHS -- cell array of function handles
%    Examples:
%      FHS = BENCHMARKS('handles');
%      f = FHS{1}(x);  % evaluates x on the sphere function f1
%      f = feval(FHS{1}, x);  % ditto
%
% see also: functions FGENERIC, BENCHMARKINFOS, BENCHMARKSNOISY

% Authors (copyright 2009): Nikolaus Hansen, Raymond Ros, Steffen Finck
%    Version = 'Revision: $Revision: 1115 $'
%    Last Modified: $Date: 2009-02-09 19:22:42 +0100 (Mon, 09 Feb 2009) $

% INTERFACE OF BENCHMARK FUNCTIONS
% FHS = BENCHMARKS('handles');
% FUNC = FHS{1};
%
% [FVALUE, FTRUE] = FUNC(X)
% [FVALUE, FTRUE] = FUNC(X, [], IINSTANCE)
%   Input: X -- matrix of column vectors
%          IINSTANCE -- instance number of the function, sets function
%             instance (XOPT, FOPT, rotation matrices,...)
%             up until a new number is set, or the function is
%             cleared. Default is zero.
%   Output: row vectors with function value for each input column
%     FVALUE -- function value
%     FTRUE -- noise-less, deterministic function value
% [FOPT STRFUNCTION] = FUNC('any_even_empty_string', ...)
%   Output:
%     FOPT -- function value at optimum
%     STRFUNCTION -- not yet implemented: function description string, ID before first whitespace
% [FOPT STRFUNCTION] = FUNC('any_even_empty_string', DIM, NTRIAL)
%   Sets rotation matrices and xopt depending on NTRIAL (by changing the random seed).
%   Output:
%     FOPT -- function value at optimum
%     STRFUNCTION -- not yet implemented: function description string, ID before first whitespace
% [FOPT, XOPT] = FUNC('xopt', DIM)
%   Output:
%     FOPT -- function value at optimum XOPT
%     XOPT -- optimal solution vector in DIM-D
% [FOPT, MATRIX] = FUNC('linearTF', DIM)  % might vanish in future
%   Output:
%     FOPT -- function value at optimum XOPT
%     MATRIX -- used transformation matrix

"""

### FUNCTION DEFINITION ###

def compute_xopt(rseed, dim):
    """Generate a random vector used as optimum argument.

    Rounded by four digits, but never to zero.

    """
    xopt = 8 * np.floor(1e4 * unif(dim, rseed)) / 1e4 - 4
    idx = (xopt == 0)
    xopt[idx] = -1e-5
    return xopt

def compute_rotation(seed, dim):
    """Returns an orthogonal basis."""

    B = np.reshape(gauss(dim * dim, seed), (dim, dim))
    for i in range(dim):
        for j in range(0, i):
            B[i] = B[i] - dot(B[i], B[j]) * B[j]
        B[i] = B[i] / (np.sum(B[i]**2) ** .5)
    return B

def monotoneTFosc(f):
    """Maps [-inf,inf] to [-inf,inf] with different constants
    for positive and negative part.

    """
    if np.isscalar(f):
        if f > 0.:
            f = np.log(f) / 0.1
            f = np.exp(f + 0.49 * (np.sin(f) + np.sin(0.79 * f))) ** 0.1
        elif f < 0.:
            f = np.log(-f) / 0.1
            f = -np.exp(f + 0.49 * (np.sin(0.55 * f) + np.sin(0.31 * f))) ** 0.1
        return f
    else:
        f = np.asarray(f)
        g = f.copy()
        idx = (f > 0)
        g[idx] = np.log(f[idx]) / 0.1
        g[idx] = np.exp(g[idx] + 0.49 * (np.sin(g[idx]) + np.sin(0.79 * g[idx])))**0.1
        idx = (f < 0)
        g[idx] = np.log(-f[idx]) / 0.1
        g[idx] = -np.exp(g[idx] + 0.49 * (np.sin(0.55 * g[idx]) + np.sin(0.31 * g[idx])))**0.1
        return g

def defaultboundaryhandling(x, fac):
    """Returns a float penalty for being outside of boundaries [-5, 5]"""
    xoutside = np.maximum(0., np.abs(x) - 5) * sign(x)
    fpen = fac * np.sum(xoutside**2, -1) # penalty
    return fpen

def gauss(N, seed):
    """Samples N standard normally distributed numbers
    being the same for a given seed

    """
    r = unif(2 * N, seed)
    g = np.sqrt(-2 * np.log(r[:N])) * np.cos(2 * np.pi * r[N:2*N])
    if np.any(g == 0.):
        g[g == 0] = 1e-99
    return g

def unif(N, inseed):
    """Generates N uniform numbers with starting seed."""

    # initialization
    inseed = np.abs(inseed)
    if inseed < 1.:
        inseed = 1.

    rgrand = 32 * [0.]
    aktseed = inseed
    for i in xrange(39, -1, -1):
        tmp = floor(aktseed / 127773.)
        aktseed = 16807. * (aktseed - tmp * 127773.) - 2836. * tmp
        if aktseed < 0:
            aktseed = aktseed + 2147483647.
        if i < 32:
            rgrand[i] = aktseed
    aktrand = rgrand[0]

    # sample numbers
    r = int(N) * [0.]
    for i in xrange(int(N)):
        tmp = floor(aktseed / 127773.)
        aktseed = 16807. * (aktseed - tmp * 127773.) - 2836. * tmp
        if aktseed < 0:
            aktseed = aktseed + 2147483647.
        tmp = int(floor(aktrand / 67108865.))
        aktrand = rgrand[tmp]
        rgrand[tmp] = aktseed
        r[i] = aktrand / 2.147483647e9
    r = np.asarray(r)
    if (r == 0).any():
        warnings.warn('zero sampled(?), set to 1e-99')
        r[r == 0] = 1e-99
    return r

# for testing and comparing to other implementations,
#   myrand and myrandn are used only for sampling the noise
#   Rename to myrand and myrandn to rand and randn and
#   comment lines 24 and 25.

_randomnseed = 30. # warning this is a global variable...
def _myrandn(size):
    """Normal random distribution sampling.

    For testing and comparing purpose.

    """

    global _randomnseed
    _randomnseed = _randomnseed + 1.
    if _randomnseed > 1e9:
        _randomnseed = 1.
    res = np.reshape(gauss(np.prod(size), _randomnseed), size)
    return res

_randomseed = 30. # warning this is a global variable...
def _myrand(size):
    """Uniform random distribution sampling.

    For testing and comparing purpose.

    """

    global _randomseed
    _randomseed = _randomseed + 1
    if _randomseed > 1e9:
        _randomseed = 1
    res = np.reshape(unif(np.prod(size), _randomseed), size)
    return res

def fGauss(ftrue, beta):
    """Returns Gaussian model noisy value."""
    # expects ftrue to be a np.array
    popsi = np.shape(ftrue)
    fval = ftrue * np.exp(beta * _randn(popsi)) # with gauss noise
    tol = 1e-8
    fval = fval + 1.01 * tol
    idx = ftrue < tol
    try:
        fval[idx] = ftrue[idx]
    except (IndexError, TypeError): # fval is a scalar
        if idx:
            fval = ftrue
    return fval

def fUniform(ftrue, alpha, beta):
    """Returns uniform model noisy value."""
    # expects ftrue to be a np.array
    popsi = np.shape(ftrue)
    fval = (_rand(popsi) ** beta * ftrue *
            np.maximum(1., (1e9 / (ftrue + 1e-99)) ** (alpha * _rand(popsi))))
    tol = 1e-8
    fval = fval + 1.01 * tol
    idx = ftrue < tol
    try:
        fval[idx] = ftrue[idx]
    except (IndexError, TypeError): # fval is a scalar
        if idx:
            fval = ftrue
    return fval

def fCauchy(ftrue, alpha, p):
    """Returns Cauchy model noisy value

    Cauchy with median 1e3*alpha and with p=0.2, zero otherwise

    P(Cauchy > 1,10,100,1000) = 0.25, 0.032, 0.0032, 0.00032

    """
    # expects ftrue to be a np.array
    popsi = np.shape(ftrue)
    fval = ftrue + alpha * np.maximum(0., 1e3 + (_rand(popsi) < p) *
                                          _randn(popsi) / (np.abs(_randn(popsi)) + 1e-199))
    tol = 1e-8
    fval = fval + 1.01 * tol
    idx = ftrue < tol
    try:
        fval[idx] = ftrue[idx]
    except (IndexError, TypeError): # fval is a scalar
        if idx:
            fval = ftrue
    return fval

### CLASS DEFINITION ###

class AbstractTestFunction(object):
    """Abstract class for test functions.

    Defines methods to be implemented in test functions which are to be
    provided to method setfun of class Logger.
    In particular, (a) the attribute fopt and (b) the method _evalfull.

    The _evalfull method returns two values, the possibly noisy value and
    the noise-free value. The latter is only meant to be for recording purpose.

    """
    def __call__(self, x): # makes the instances callable
        """Returns the objective function value of argument x.

        Example:

            >>> from cma import bbobbenchmarks as bn
            >>> f3 = bn.F3(13) # instantiate function 3 on instance 13
            >>> 59.8733529 < f3([0, 1, 2]) < 59.87335292 # call f3, same as f3.evaluate([0, 1, 2])
            True

        """
        return self.evaluate(x)

    def evaluate(self, x):
        """Returns the objective function value (in case noisy).

        """
        return self._evalfull(x)[0]
    # TODO: is it better to leave evaluate out and check for hasattr('evaluate') in ExpLogger?

    def _evalfull(self, x):
        """return noisy and noise-free value, the latter for recording purpose. """
        raise NotImplementedError

    def getfopt(self):
        """Returns the best function value of this instance of the function."""
        # TODO: getfopt error:
        # import bbobbenchmarks as bb
        # bb.instantiate(1)[0].getfopt()
        # AttributeError: F1 instance has no attribute '_fopt'

        if not hasattr(self, 'iinstance'):
            raise Exception('This function class has not been instantiated yet.')
        return self._fopt

    def setfopt(self, fopt):
        try:
            self._fopt = float(fopt)
        except ValueError:
            raise Exception('Optimal function value must be cast-able to a float.')

    fopt = property(getfopt, setfopt)

class BBOBFunction(AbstractTestFunction):
    """Abstract class of BBOB test functions.

    Implements some base functions that are used by the test functions
    of BBOB such as initialisations of class attributes.

    """
    def __init__(self, iinstance=0, zerox=False, zerof=False, param=None, **kwargs):
        """Common initialisation.

        Keyword arguments:
        iinstance -- instance of the function (int)
        zerox -- sets xopt to [0, ..., 0]
        zerof -- sets fopt to 0
        param -- parameter of the function (if applicable)
        kwargs -- additional attributes

        """
        # Either self.rrseed or self.funId have to be defined for BBOBFunctions
        # TODO: enforce
        try:
            rrseed = self.rrseed
        except AttributeError:
            rrseed = self.funId

        try:
            self.rseed = rrseed + 1e4 * iinstance
        except TypeError:
            # rrseed AND iinstance have to be float
            warnings.warn('self.rseed could not be set, reset to 1 instead.')
            self.rseed = 1

        self.zerox = zerox
        if zerof:
            self.fopt = 0.
        else:
            self.fopt = min(1000, max(-1000, (np.round(100 * 100 * gauss(1, self.rseed)[0] / gauss(1, self.rseed + 1)[0]) / 100)))
        self.iinstance = iinstance
        self.dim = None
        self.lastshape = None
        self.param = param
        for i, v in kwargs.items():
            setattr(self, i, v)
        self._xopt = None

    def shape_(self, x):
        # this part is common to all evaluate function
        # it is assumed x are row vectors
        curshape = np.shape(x)
        dim = np.shape(x)[-1]
        return curshape, dim

    def getiinstance(self):
        """Designates the instance of the function class.

        An instance in this case means a given target function value, a
        given optimal argument x, and given transformations for the
        function. It needs to have a string representation. Preferably
        it should be a number or a string.

        """
        return self._iinstance

    def setiinstance(self, iinstance):
        self._iinstance = iinstance

    iinstance = property(getiinstance, setiinstance)

    def shortstr(self):
        """Gives a short string self representation (shorter than str(self))."""

        res = 'F%s' % str(self.funId)
        if hasattr(self, 'param'):
            res += '_p%s' % str(self.param)  # NH param -> self.param
        return res

    def __eq__(self, obj):
        return (self.funId == obj.funId
                and (not hasattr(self, 'param') or self.param == obj.param))
        # TODO: make this test on other attributes than param?

#    def dimensionality(self, dim):
#        """Return the availability of dimensionality dim."""
#        return True

    # GETTERS
#    def getfopt(self):
#        """Optimal Function Value."""
#        return self._fopt

#    fopt = property(getfopt)

    def _setxopt(self, xopt):
        """Return the argument of the optimum of the function."""
        self._xopt = xopt

    def _getxopt(self):
        """Return the argument of the optimum of the function."""
        if self._xopt is None:
            warnings.warn('You need to evaluate object to set dimension first.')
        return self._xopt

    xopt = property(_getxopt, _setxopt)

#    def getrange(self):
#        """Return the domain of the function."""
#        #TODO: could depend on the dimension
#        # TODO: return exception NotImplemented yet
#        pass

#    range = property(getrange)

#    def getparam(self):
#        """Optional parameter value."""
#        return self._param

#    param = property(getparam)

#    def getitrial(self):
#        """Instance id number."""
#        return self._itrial

#    itrial = property(getitrial)

#    def getlinearTf(self):
#        return self._linearTf

#    linearTf = property(getlinearTf)

#    def getrotation(self):
#        return self._rotation

#    rotation = property(getrotation)



class BBOBNfreeFunction(BBOBFunction):
    """Class of the noise-free functions of BBOB."""

    def noise(self, ftrue):
        """Returns the noise-free function values."""

        return ftrue.copy()

class BBOBGaussFunction(BBOBFunction):
    """Class of the Gauss noise functions of BBOB.

    Attribute gaussbeta needs to be defined by inheriting classes.

    """

    # gaussbeta = None

    def noise(self, ftrue):
        """Returns the noisy function values."""

        return fGauss(ftrue, self.gaussbeta)

    def boundaryhandling(self, x):
        return defaultboundaryhandling(x, 100.)

class BBOBUniformFunction(BBOBFunction, object):
    """Class of the uniform noise functions of BBOB.

    Attributes unifalphafac and unifbeta need to be defined by inheriting
    classes.

    """
    # unifalphafac = None
    # unifbeta = None

    def noise(self, ftrue):
        """Returns the noisy function values."""

        return fUniform(ftrue, self.unifalphafac * (0.49 + 1. / self.dim), self.unifbeta)

    def boundaryhandling(self, x):
        return defaultboundaryhandling(x, 100.)

class BBOBCauchyFunction(BBOBFunction):
    """Class of the Cauchy noise functions of BBOB.

    Attributes cauchyalpha and cauchyp need to be defined by inheriting
    classes.

    """
    # cauchyalpha = None
    # cauchyp = None

    def noise(self, ftrue):
        """Returns the noisy function values."""

        return fCauchy(ftrue, self.cauchyalpha, self.cauchyp)

    def boundaryhandling(self, x):
        return defaultboundaryhandling(x, 100.)

class _FSphere(BBOBFunction):
    """Abstract Sphere function.

    Method boundaryhandling needs to be defined.

    """
    rrseed = 1

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        fadd = fadd + self.boundaryhandling(x)

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!

        # COMPUTATION core
        ftrue = np.sum(x**2, -1)
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F1(_FSphere, BBOBNfreeFunction):
    """Noise-free Sphere function"""
    funId = 1
    def boundaryhandling(self, x):
        return 0.

class F101(_FSphere, BBOBGaussFunction):
    """Sphere with moderate Gauss noise"""
    funId = 101
    gaussbeta = 0.01

class F102(_FSphere, BBOBUniformFunction):
    """Sphere with moderate uniform noise"""
    funId = 102
    unifalphafac = 0.01
    unifbeta = 0.01

class F103(_FSphere, BBOBCauchyFunction):
    """Sphere with moderate Cauchy noise"""
    funId = 103
    cauchyalpha = 0.01
    cauchyp = 0.05

class F107(_FSphere, BBOBGaussFunction):
    """Sphere with  Gauss noise"""
    funId = 107
    gaussbeta = 1.

class F108(_FSphere, BBOBUniformFunction):
    """Sphere with uniform noise"""
    funId = 108
    unifalphafac = 1.
    unifbeta = 1.

class F109(_FSphere, BBOBCauchyFunction):
    """Sphere with Cauchy noise"""
    funId = 109
    cauchyalpha = 1.
    cauchyp = 0.2

class F2(BBOBNfreeFunction):
    """Separable ellipsoid with monotone transformation

    Parameter: condition number (default 1e6)

    """

    funId = 2
    paramValues = (1e0, 1e6)
    condition = 1e6

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            if hasattr(self, 'param') and self.param: # not self.param is None
                tmp = self.param
            else:
                tmp = self.condition
            self.scales = tmp ** linspace(0, 1, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!

        # COMPUTATION core
        ftrue = dot(monotoneTFosc(x)**2, self.scales)
        fval = self.noise(ftrue) # without noise

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F3(BBOBNfreeFunction):
    """Rastrigin with monotone transformation separable "condition" 10"""

    funId = 3
    condition = 10.
    beta = 0.2

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialisation
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.scales = (self.condition ** .5) ** linspace(0, 1, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)
            self.arrscales = resize(self.scales, curshape)
            self.arrexpo = resize(self.beta * linspace(0, 1, dim), curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt
        x = monotoneTFosc(x)
        idx = (x > 0)
        x[idx] = x[idx] ** (1 + self.arrexpo[idx] * np.sqrt(x[idx]))
        x = self.arrscales * x

        # COMPUTATION core
        ftrue = 10 * (self.dim - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)
        fval = self.noise(ftrue) # without noise

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F4(BBOBNfreeFunction):
    """skew Rastrigin-Bueche, condition 10, skew-"condition" 100"""

    funId = 4
    condition = 10.
    alpha = 100.
    maxindex = np.inf # 1:2:min(DIM,maxindex) are the skew variables
    rrseed = 3

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.xopt[:min(dim, self.maxindex):2] = abs(self.xopt[:min(dim, self.maxindex):2])
            self.scales = (self.condition ** .5) ** linspace(0, 1, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)
            self.arrscales = resize(self.scales, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        xoutside = np.maximum(0., np.abs(x) - 5) * sign(x)
        fpen = 1e2 * np.sum(xoutside**2, -1) # penalty
        fadd = fadd + fpen # self.fadd becomes an array

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # shift optimum to zero
        x = monotoneTFosc(x)
        try:
            tmpx = x[:, :min(self.dim, self.maxindex):2] # tmpx is a reference to a part of x
        except IndexError:
            tmpx = x[:min(self.dim, self.maxindex):2] # tmpx is a reference to a part of x
        tmpx[tmpx > 0] = self.alpha ** .5 * tmpx[tmpx > 0] # this modifies x
        x = self.arrscales * x # scale while assuming that Xopt == 0

        # COMPUTATION core
        ftrue = 10 * (self.dim - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F5(BBOBNfreeFunction):
    """Linear slope"""

    funId = 5
    alpha = 100.

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim) # TODO: what happens here?
            else:
                self.xopt = 5 * sign(compute_xopt(self.rseed, dim))
            self.scales = -sign(self.xopt) * (self.alpha ** .5) ** linspace(0, 1, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)
        fadd = fadd + 5 * np.sum(np.abs(self.scales))

        # BOUNDARY HANDLING
        # move "too" good coordinates back into domain
        x = np.array(x) # convert x and make a copy of x.
        # The following may modify x directly.
        idx_out_of_bounds = (x * self.arrxopt) > 25 # 25 == 5 * 5
        x[idx_out_of_bounds] = sign(x[idx_out_of_bounds]) * 5

        # TRANSFORMATION IN SEARCH SPACE

        # COMPUTATION core
        ftrue = dot(x, self.scales)
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F6(BBOBNfreeFunction):
    """Attractive sector function"""

    funId = 6
    condition = 10.
    alpha = 100.

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.rotation = compute_rotation(self.rseed + 1e6, dim)
            self.scales = (self.condition ** .5) ** linspace(0, 1, dim)
            self.linearTF = dot(compute_rotation(self.rseed, dim), diag(self.scales))
            # decouple scaling from function definition
            self.linearTF = dot(self.linearTF, self.rotation)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.linearTF) # TODO: check

        # COMPUTATION core
        idx = (x * self.arrxopt) > 0
        x[idx] = self.alpha * x[idx]
        ftrue = monotoneTFosc(np.sum(x**2, -1)) ** .9
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class _FStepEllipsoid(BBOBFunction):
    """Abstract Step-ellipsoid, condition 100

    Method boundaryhandling needs to be defined.

    """
    rrseed = 7
    condition = 100.
    alpha = 10.

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.rotation = compute_rotation(self.rseed + 1e6, dim)
            self.scales = self.condition ** linspace(0, 1, dim)
            self.linearTF = dot(compute_rotation(self.rseed, dim),
                                diag(((self.condition / 10.)**.5) ** linspace(0, 1, dim)))

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        fadd = fadd + self.boundaryhandling(x)

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.linearTF)
        try:
            x1 = x[:, 0]
        except IndexError:
            x1 = x[0]
        idx = np.abs(x) > .5
        x[idx] = np.round(x[idx])
        x[~idx] = np.round(self.alpha * x[~idx]) / self.alpha
        x = dot(x, self.rotation)

        # COMPUTATION core
        ftrue = .1 * np.maximum(1e-4 * np.abs(x1), dot(x ** 2, self.scales))
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F7(_FStepEllipsoid, BBOBNfreeFunction):
    """Step-ellipsoid, condition 100, noise-free"""
    funId = 7
    def boundaryhandling(self, x):
        return defaultboundaryhandling(x, 1.)

class F113(_FStepEllipsoid, BBOBGaussFunction):
    """Step-ellipsoid with gauss noise, condition 100"""
    funId = 113
    gaussbeta = 1.

class F114(_FStepEllipsoid, BBOBUniformFunction):
    """Step-ellipsoid with uniform noise, condition 100"""
    funId = 114
    unifalphafac = 1.
    unifbeta = 1.

class F115(_FStepEllipsoid, BBOBCauchyFunction):
    """Step-ellipsoid with Cauchy noise, condition 100"""
    funId = 115
    cauchyalpha = 1.
    cauchyp = 0.2

class _FRosenbrock(BBOBFunction):
    """Abstract Rosenbrock, non-rotated

    Method boundaryhandling needs to be defined.

    """
    rrseed = 8

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = .75 * compute_xopt(self.rseed, dim) # different from all others
            self.scales = max(1, dim ** .5 / 8.)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        fadd = fadd + self.boundaryhandling(x)

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= self.arrxopt!
        x = self.scales * x
        x = x + 1 # shift zero to factual optimum 1

        # COMPUTATION core
        try:
            ftrue = (1e2 * np.sum((x[:, :-1] ** 2 - x[:, 1:]) ** 2, -1) +
                     np.sum((x[:, :-1] - 1.) ** 2, -1))
        except IndexError:
            ftrue = (1e2 * np.sum((x[:-1] ** 2 - x[1:]) ** 2) +
                     np.sum((x[:-1] - 1.) ** 2))
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F8(_FRosenbrock, BBOBNfreeFunction):
    """Rosenbrock noise-free"""
    funId = 8
    def boundaryhandling(self, x):
        return 0.

class F104(_FRosenbrock, BBOBGaussFunction):
    """Rosenbrock non-rotated with moderate Gauss noise"""
    funId = 104
    gaussbeta = 0.01

class F105(_FRosenbrock, BBOBUniformFunction):
    """Rosenbrock non-rotated with moderate uniform noise"""
    funId = 105
    unifalphafac = 0.01
    unifbeta = 0.01

class F106(_FRosenbrock, BBOBCauchyFunction):
    """Rosenbrock non-rotated with moderate Cauchy noise"""
    funId = 106
    cauchyalpha = 0.01
    cauchyp = 0.05

class F110(_FRosenbrock, BBOBGaussFunction):
    """Rosenbrock non-rotated with Gauss noise"""
    funId = 110
    gaussbeta = 1.

class F111(_FRosenbrock, BBOBUniformFunction):
    """Rosenbrock non-rotated with uniform noise"""
    funId = 111
    unifalphafac = 1.
    unifbeta = 1.

class F112(_FRosenbrock, BBOBCauchyFunction):
    """Rosenbrock non-rotated with Cauchy noise"""
    funId = 112
    cauchyalpha = 1.
    cauchyp = 0.2

class F9(BBOBNfreeFunction):
    """Rosenbrock, rotated"""
    funId = 9

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            scale = max(1, dim ** .5 / 8.) # nota: different from scales in F8
            self.linearTF = scale * compute_rotation(self.rseed, dim)
            self.xopt = np.hstack(dot(.5 * np.ones((1, dim)), self.linearTF.T)) / scale ** 2

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING

        # TRANSFORMATION IN SEARCH SPACE
        x = dot(x, self.linearTF) + 0.5 # different from F8

        # COMPUTATION core
        try:
            ftrue = (1e2 * np.sum((x[:, :-1] ** 2 - x[:, 1:]) ** 2, -1) +
                     np.sum((x[:, :-1] - 1.) ** 2, -1))
        except IndexError:
            ftrue = (1e2 * np.sum((x[:-1] ** 2 - x[1:]) ** 2) +
                     np.sum((x[:-1] - 1.) ** 2))
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class _FEllipsoid(BBOBFunction):
    """Abstract Ellipsoid with monotone transformation.

    Method boundaryhandling needs to be defined.

    """
    rrseed = 10
    condition = 1e6

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.rotation = compute_rotation(self.rseed + 1e6, dim)
            self.scales = self.condition ** linspace(0, 1, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        fadd = fadd + self.boundaryhandling(x)

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.rotation)
        x = monotoneTFosc(x)

        # COMPUTATION core
        ftrue = dot(x ** 2, self.scales)
        try:
            ftrue = np.hstack(ftrue)
        except TypeError: # argument 2 to map() must support iteration
            pass
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F10(_FEllipsoid, BBOBNfreeFunction):
    """Ellipsoid with monotone transformation, condition 1e6"""
    funId = 10
    condition = 1e6
    def boundaryhandling(self, x):
        return 0.

class F116(_FEllipsoid, BBOBGaussFunction):
    """Ellipsoid with Gauss noise, monotone x-transformation, condition 1e4"""
    funId = 116
    condition = 1e4
    gaussbeta = 1.

class F117(_FEllipsoid, BBOBUniformFunction):
    """Ellipsoid with uniform noise, monotone x-transformation, condition 1e4"""
    funId = 117
    condition = 1e4
    unifalphafac = 1.
    unifbeta = 1.

class F118(_FEllipsoid, BBOBCauchyFunction):
    """Ellipsoid with Cauchy noise, monotone x-transformation, condition 1e4"""
    funId = 118
    condition = 1e4
    cauchyalpha = 1.
    cauchyp = 0.2

class F11(BBOBNfreeFunction):
    """Discus (tablet) with monotone transformation, condition 1e6"""
    funId = 11
    condition = 1e6

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.rotation = compute_rotation(self.rseed + 1e6, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.rotation)
        x = monotoneTFosc(x)

        # COMPUTATION core
        try:
            ftrue = np.sum(x**2, -1) + (self.condition - 1.) * x[:, 0] ** 2
        except IndexError:
            ftrue = np.sum(x**2) + (self.condition - 1.) * x[0] ** 2
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F12(BBOBNfreeFunction):
    """Bent cigar with asymmetric space distortion, condition 1e6"""
    funId = 12
    condition = 1e6
    beta = .5

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed + 1e6, dim) # different from others
            self.rotation = compute_rotation(self.rseed + 1e6, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)
            self.arrexpo = resize(self.beta * linspace(0, 1, dim), curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.rotation) # no scaling here, because it would go to the arrExpo
        idx = x > 0
        x[idx] = x[idx] ** (1 + self.arrexpo[idx] * np.sqrt(x[idx]))
        x = dot(x, self.rotation)

        # COMPUTATION core
        try:
            ftrue = self.condition * np.sum(x**2, -1) + (1 - self.condition) * x[:, 0]**2
        except IndexError:
            ftrue = self.condition * np.sum(x**2) + (1 - self.condition) * x[0]**2
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F13(BBOBNfreeFunction):
    """Sharp ridge"""
    funId = 13
    condition = 10.
    alpha = 100. # slope

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.rotation = compute_rotation(self.rseed + 1e6, dim)
            self.scales = (self.condition ** .5) ** linspace(0, 1, dim)
            self.linearTF = dot(compute_rotation(self.rseed, dim), diag(self.scales))
            self.linearTF = dot(self.linearTF, self.rotation)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.linearTF)

        # COMPUTATION core
        try:
            ftrue = x[:, 0] ** 2 + self.alpha * np.sqrt(np.sum(x[:, 1:] ** 2, -1))
        except IndexError:
            ftrue = x[0] ** 2 + self.alpha * np.sqrt(np.sum(x[1:] ** 2, -1))
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class _FDiffPow(BBOBFunction):
    """Abstract Sum of different powers, between x^2 and x^6.

    Method boundaryhandling needs to be defined.

    """
    alpha = 4.
    rrseed = 14

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.rotation = compute_rotation(self.rseed + 1e6, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)
            self.arrexpo = resize(2. + self.alpha * linspace(0, 1, dim), curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        fadd = fadd + self.boundaryhandling(x)

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.rotation)

        # COMPUTATION core
        ftrue = np.sqrt(np.sum(np.abs(x) ** self.arrexpo, -1))
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F14(_FDiffPow, BBOBNfreeFunction):
    """Sum of different powers, between x^2 and x^6, noise-free"""
    funId = 14
    def boundaryhandling(self, x):
        return 0.

class F119(_FDiffPow, BBOBGaussFunction):
    """Sum of different powers with Gauss noise, between x^2 and x^6"""
    funId = 119
    gaussbeta = 1.

class F120(_FDiffPow, BBOBUniformFunction):
    """Sum of different powers with uniform noise, between x^2 and x^6"""
    funId = 120
    unifalphafac = 1.
    unifbeta = 1.

class F121(_FDiffPow, BBOBCauchyFunction):
    """Sum of different powers with seldom Cauchy noise, between x^2 and x^6"""
    funId = 121
    cauchyalpha = 1.
    cauchyp = 0.2

class F15(BBOBNfreeFunction):
    """Rastrigin with asymmetric non-linear distortion, "condition" 10"""
    funId = 15
    condition = 10.
    beta = 0.2

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.rotation = compute_rotation(self.rseed + 1e6, dim)
            self.scales = (self.condition ** .5) ** linspace(0, 1, dim)
            self.linearTF = dot(compute_rotation(self.rseed, dim), diag(self.scales))
            # decouple scaling from function definition
            self.linearTF = dot(self.linearTF, self.rotation)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)
            self.arrexpo = resize(self.beta * linspace(0, 1, dim), curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.rotation) # no scaling here, because it would go to the arrexpo
        x = monotoneTFosc(x)
        idx = x > 0.
        x[idx] = x[idx] ** (1. + self.arrexpo[idx] * np.sqrt(x[idx])) # smooth in zero
        x = dot(x, self.linearTF)

        # COMPUTATION core
        ftrue = 10. * (dim - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F16(BBOBNfreeFunction):
    """Weierstrass, condition 100"""
    funId = 16
    condition = 100.

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.rotation = compute_rotation(self.rseed + 1e6, dim)
            self.scales = (1. / self.condition ** .5) ** linspace(0, 1, dim) # CAVE?
            self.linearTF = dot(compute_rotation(self.rseed, dim), diag(self.scales))
            # decouple scaling from function definition
            self.linearTF = dot(self.linearTF, self.rotation)
            K = np.arange(0, 12)
            self.aK = np.reshape(0.5 ** K, (1, 12))
            self.bK = np.reshape(3. ** K, (1, 12))
            self.f0 = np.sum(self.aK * np.cos(2 * np.pi * self.bK * 0.5)) # optimal value

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        xoutside = np.maximum(0, np.abs(x) - 5.) * sign(x)
        fpen = (10. / dim) * np.sum(xoutside ** 2, -1)
        fadd = fadd + fpen

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.rotation)
        x = monotoneTFosc(x)
        x = dot(x, self.linearTF)

        # COMPUTATION core
        if len(curshape) < 2: # popsize is one
            ftrue = np.sum(dot(self.aK, np.cos(dot(self.bK.T, 2 * np.pi * (np.reshape(x, (1, len(x))) + 0.5)))))
        else:
            ftrue = np.zeros(curshape[0]) # curshape[0] is popsize
            for k, i in enumerate(x):
                # TODO: simplify next line
                ftrue[k] = np.sum(dot(self.aK, np.cos(dot(self.bK.T, 2 * np.pi * (np.reshape(i, (1, len(i))) + 0.5)))))
        ftrue = 10. * (ftrue / dim - self.f0) ** 3
        try:
            ftrue = np.hstack(ftrue)
        except TypeError:
            pass
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class _FSchaffersF7(BBOBFunction):
    """Abstract Schaffers F7 with asymmetric non-linear transformation, condition 10

    Class attribute condition and method boundaryhandling need to be defined.

    """
    rrseed = 17
    condition = None
    beta = 0.5

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.rotation = compute_rotation(self.rseed + 1e6, dim)
            self.scales = (self.condition ** .5) ** linspace(0, 1 , dim)
            self.linearTF = dot(compute_rotation(self.rseed, dim), diag(self.scales))

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)
            self.arrexpo = resize(self.beta * linspace(0, 1, dim), curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        fadd = fadd + self.boundaryhandling(x)

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.rotation)
        idx = x > 0
        x[idx] = x[idx] ** (1 + self.arrexpo[idx] * np.sqrt(x[idx]))
        x = dot(x, self.linearTF)

        # COMPUTATION core
        try:
            s = x[:, :-1] ** 2 + x[:, 1:] ** 2
        except IndexError:
            s = x[:-1] ** 2 + x[1:] ** 2
        ftrue = np.mean(s ** .25 * (np.sin(50 * s ** .1) ** 2 + 1), -1) ** 2
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F17(_FSchaffersF7, BBOBNfreeFunction):
    """Schaffers F7 with asymmetric non-linear transformation, condition 10"""
    funId = 17
    condition = 10.
    def boundaryhandling(self, x):
        return defaultboundaryhandling(x, 10.)

class F18(_FSchaffersF7, BBOBNfreeFunction):
    """Schaffers F7 with asymmetric non-linear transformation, condition 1000"""
    funId = 18
    condition = 1000.
    def boundaryhandling(self, x):
        return defaultboundaryhandling(x, 10.)

class F122(_FSchaffersF7, BBOBGaussFunction):
    """Schaffers F7 with Gauss noise, with asymmetric non-linear transformation, condition 10"""
    funId = 122
    condition = 10.
    gaussbeta = 1.

class F123(_FSchaffersF7, BBOBUniformFunction):
    """Schaffers F7 with uniform noise, asymmetric non-linear transformation, condition 10"""
    funId = 123
    condition = 10.
    unifalphafac = 1.
    unifbeta = 1.

class F124(_FSchaffersF7, BBOBCauchyFunction): # TODO: check boundary handling
    """Schaffers F7 with seldom Cauchy noise, asymmetric non-linear transformation, condition 10"""
    funId = 124
    condition = 10.
    cauchyalpha = 1.
    cauchyp = 0.2

class _F8F2(BBOBFunction):
    """Abstract F8F2 sum of Griewank-Rosenbrock 2-D blocks

    Class attribute facftrue and method boundaryhandling need to be defined.

    """
    facftrue = None
    rrseed = 19

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            scale = max(1, dim ** .5 / 8.)
            self.linearTF = scale * compute_rotation(self.rseed, dim)
            # if self.zerox:
            #    self.xopt = zeros(dim) # does not work here
            # else:
            # TODO: clean this line
            self.xopt = np.hstack(dot(self.linearTF, 0.5 * np.ones((dim, 1)) / scale ** 2))

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        fadd = fadd + self.boundaryhandling(x)

        # TRANSFORMATION IN SEARCH SPACE
        x = dot(x, self.linearTF) + 0.5 # cannot be replaced with x -= arrxopt!

        # COMPUTATION core
        try:
            f2 = 100. * (x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (1. - x[:, :-1]) ** 2
        except IndexError:
            f2 = 100. * (x[:-1] ** 2 - x[1:]) ** 2 + (1. - x[:-1]) ** 2
        ftrue = self.facftrue + self.facftrue * np.sum(f2 / 4000. - np.cos(f2), -1) / (dim - 1.)
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F19(_F8F2, BBOBNfreeFunction):
    """F8F2 sum of Griewank-Rosenbrock 2-D blocks, noise-free"""
    funId = 19
    facftrue = 10.
    def boundaryhandling(self, x):
        return 0.

class F125(_F8F2, BBOBGaussFunction):
    """F8F2 sum of Griewank-Rosenbrock 2-D blocks with Gauss noise"""
    funId = 125
    facftrue = 1.
    gaussbeta = 1.

class F126(_F8F2, BBOBUniformFunction):
    """F8F2 sum of Griewank-Rosenbrock 2-D blocks with uniform noise"""
    funId = 126
    facftrue = 1.
    unifalphafac = 1.
    unifbeta = 1.

class F127(_F8F2, BBOBCauchyFunction):
    """F8F2 sum of Griewank-Rosenbrock 2-D blocks with seldom Cauchy noise"""
    funId = 127
    facftrue = 1.
    cauchyalpha = 1.
    cauchyp = 0.2

class F20(BBOBNfreeFunction):
    """Schwefel with tridiagonal variable transformation"""
    funId = 20
    condition = 10.

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = 0.5 * sign(unif(dim, self.rseed) - 0.5) * 4.2096874633
            self.scales = (self.condition ** .5) ** np.linspace(0, 1, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(2 * np.abs(self.xopt), curshape)
            self.arrscales = resize(self.scales, curshape)
            self.arrsigns = resize(sign(self.xopt), curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # TRANSFORMATION IN SEARCH SPACE
        x = 2 * self.arrsigns * x # makes the below boundary handling effective for coordinates
        try:
            x[:, 1:] = x[:, 1:] + .25 * (x[:, :-1] - self.arrxopt[:, :-1])
        except IndexError:
            x[1:] = x[1:] + .25 * (x[:-1] - self.arrxopt[:-1])
        x = 100. * (self.arrscales * (x - self.arrxopt) + self.arrxopt)

        # BOUNDARY HANDLING
        xoutside = np.maximum(0., np.abs(x) - 500.) * sign(x) # in [-500, 500]
        fpen = 0.01 * np.sum(xoutside ** 2, -1)
        fadd = fadd + fpen

        # COMPUTATION core
        ftrue = 0.01 * ((418.9828872724339) - np.mean(x * np.sin(np.sqrt(np.abs(x))), -1))
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class _FGallagher(BBOBFunction):
    """Abstract Gallagher with nhighpeaks Gaussian peaks, condition up to 1000, one global rotation

    Attribute fac2, nhighpeaks, highpeakcond and method boundary
    handling need to be defined.

    """
    rrseed = 21
    maxcondition = 1000.
    fitvalues = (1.1, 9.1)
    fac2 = None # added: factor for xopt not too close to boundaries, used by F22
    nhighpeaks = None
    highpeakcond = None

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            self.rotation = compute_rotation(self.rseed, dim)
            arrcondition = self.maxcondition ** linspace(0, 1, self.nhighpeaks - 1)
            idx = np.argsort(unif(self.nhighpeaks - 1, self.rseed)) # random permutation
            arrcondition = np.insert(arrcondition[idx], 0, self.highpeakcond)
            self.arrscales = []
            for i, e in enumerate(arrcondition):
                s = e ** linspace(-.5, .5, dim)
                idx = np.argsort(unif(dim, self.rseed + 1e3 * i)) # permutation instead of rotation
                self.arrscales.append(s[idx]) # this is inverse Cov
            self.arrscales = np.vstack(self.arrscales)
            # compute peak values, 10 is global optimum
            self.peakvalues = np.insert(linspace(self.fitvalues[0], self.fitvalues[1], self.nhighpeaks - 1), 0, 10.)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.xlocal = dot(self.fac2 * np.reshape(10. * unif(dim * self.nhighpeaks, self.rseed) - 5., (self.nhighpeaks, dim)),
                              self.rotation)
            if self.zerox:
                self.xlocal[0, :] = zeros(dim)
            else:
                # global optimum not too close to boundary
                self.xlocal[0, :] = 0.8 * self.xlocal[0, :]
            self.xopt = dot(self.xlocal[0, :], self.rotation.T)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        fadd = fadd + self.boundaryhandling(x)

        # TRANSFORMATION IN SEARCH SPACE
        x = dot(x, self.rotation)

        # COMPUTATION core
        fac = -0.5 / dim
        # f = NaN(nhighpeaks, popsi)
        # TODO: optimize
        if len(curshape) < 2: # popsize is 1 in this case
            f = np.zeros(self.nhighpeaks)
            xx = tile(x, (self.nhighpeaks, 1)) - self.xlocal
            f[:] = self.peakvalues * np.exp(fac * np.sum(self.arrscales * xx ** 2, 1))
        elif curshape[0] < .5 * self.nhighpeaks:
            f = np.zeros((curshape[0], self.nhighpeaks))
            for k, e in enumerate(x):
                xx = tile(e, (self.nhighpeaks, 1)) - self.xlocal
                f[k, :] = self.peakvalues * np.exp(fac * np.sum(self.arrscales * xx ** 2, 1))
        else:
            f = np.zeros((curshape[0], self.nhighpeaks))
            for i in range(self.nhighpeaks):
                xx = (x - tile(self.xlocal[i, :], (curshape[0], 1)))
                f[:, i] = self.peakvalues[i] * np.exp(fac * (dot(xx ** 2, self.arrscales[i, :])))
        ftrue = monotoneTFosc(10 - np.max(f, -1)) ** 2
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F21(_FGallagher, BBOBNfreeFunction):
    """Gallagher with 101 Gaussian peaks, condition up to 1000, one global rotation, noise-free"""
    funId = 21
    nhighpeaks = 101
    fac2 = 1.
    highpeakcond = 1000. ** .5
    def boundaryhandling(self, x):
        return defaultboundaryhandling(x, 1.)

class F22(_FGallagher, BBOBNfreeFunction):
    """Gallagher with 21 Gaussian peaks, condition up to 1000, one global rotation"""
    funId = 22
    rrseed = 22
    nhighpeaks = 21
    fac2 = 0.98
    highpeakcond = 1000.
    def boundaryhandling(self, x):
        return defaultboundaryhandling(x, 1.)

class F128(_FGallagher, BBOBGaussFunction): # TODO: check boundary handling
    """Gallagher with 101 Gaussian peaks with Gauss noise, condition up to 1000, one global rotation"""
    funId = 128
    nhighpeaks = 101
    fac2 = 1.
    highpeakcond = 1000. ** .5
    gaussbeta = 1.

class F129(_FGallagher, BBOBUniformFunction):
    """Gallagher with 101 Gaussian peaks with uniform noise, condition up to 1000, one global rotation"""
    funId = 129
    nhighpeaks = 101
    fac2 = 1.
    highpeakcond = 1000. ** .5
    unifalphafac = 1.
    unifbeta = 1.

class F130(_FGallagher, BBOBCauchyFunction):
    """Gallagher with 101 Gaussian peaks with seldom Cauchy noise, condition up to 1000, one global rotation"""
    funId = 130
    nhighpeaks = 101
    fac2 = 1.
    highpeakcond = 1000. ** .5
    cauchyalpha = 1.
    cauchyp = 0.2

class F23(BBOBNfreeFunction):
    """Katsuura function"""
    funId = 23
    condition = 100.
    arr2k = np.reshape(2. ** (np.arange(1, 33)), (1, 32)) # bug-fix for 32-bit (NH): 2 -> 2. (relevance is minor)

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)
            self.rotation = compute_rotation(self.rseed + 1e6, dim)
            self.scales = (self.condition ** .5) ** linspace(0, 1, dim)
            self.linearTF = dot(compute_rotation(self.rseed, dim), diag(self.scales))
            # decouple scaling from function definition
            self.linearTF = dot(self.linearTF, self.rotation)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        xoutside = np.maximum(0, np.abs(x) - 5.) * sign(x)
        fpen = np.sum(xoutside ** 2, -1)
        fadd = fadd + fpen

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!
        x = dot(x, self.linearTF)

        # COMPUTATION core
        if len(curshape) < 2: # popsize is 1 in this case
            arr = dot(np.reshape(x, (dim, 1)), self.arr2k) # dim times d array
            ftrue = (-10. / dim ** 2. +
                     10. / dim ** 2. *
                     np.prod(1 + np.arange(1, dim + 1) * np.dot(np.abs(arr - np.round(arr)), self.arr2k.T ** -1.).T) ** (10. / dim ** 1.2))
        else:
            ftrue = zeros(curshape[0])
            for k, e in enumerate(x):
                arr = dot(np.reshape(e, (dim, 1)), self.arr2k) # dim times d array
                ftrue[k] = (-10. / dim ** 2. +
                            10. / dim ** 2. *
                            np.prod(1 + np.arange(1, dim + 1) * np.dot(np.abs(arr - np.round(arr)), self.arr2k.T ** -1.).T) ** (10. / dim ** 1.2))
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

class F24(BBOBNfreeFunction):
    """Lunacek bi-Rastrigin, condition 100

    in PPSN 2008, Rastrigin part rotated and scaled

    """
    funId = 24
    condition = 100.
    _mu1 = 2.5

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = .5 * self._mu1 * sign(gauss(dim, self.rseed))
            self.rotation = compute_rotation(self.rseed + 1e6, dim)
            self.scales = (self.condition ** .5) ** linspace(0, 1, dim)
            self.linearTF = dot(compute_rotation(self.rseed, dim), diag(self.scales))
            # decouple scaling from function definition
            self.linearTF = dot(self.linearTF, self.rotation)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            # self.arrxopt = resize(self.xopt, curshape)
            self.arrscales = resize(2. * sign(self.xopt), curshape) # makes up for xopt

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING
        xoutside = np.maximum(0, np.abs(x) - 5.) * sign(x)
        fpen = 1e4 * np.sum(xoutside ** 2, -1)
        fadd = fadd + fpen

        # TRANSFORMATION IN SEARCH SPACE
        x = self.arrscales * x

        # COMPUTATION core
        s = 1 - .5 / ((dim + 20)**0.5 - 4.1) # tested up to DIM = 160 p in [0.25,0.33]
        d = 1 # shift [1,3], smaller is more difficult
        mu2 = -((self._mu1 ** 2 - d) / s) ** .5
        ftrue = np.minimum(np.sum((x - self._mu1) ** 2, -1),
                           d * dim + s * np.sum((x - mu2) ** 2, -1))
        ftrue = ftrue + 10 * (dim - np.sum(np.cos(2 * np.pi * dot(x - self._mu1, self.linearTF)), -1))
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

# dictbbob = {'sphere': F1, 'ellipsoid': F2, 'Rastrigin': F3}
nfreefunclasses = (F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14,
                   F15, F16, F17, F18, F19, F20, F21, F22, F23, F24) # hard coded
noisyfunclasses = (F101, F102, F103, F104, F105, F106, F107, F108, F109, F110,
                   F111, F112, F113, F114, F115, F116, F117, F118, F119, F120,
                   F121, F122, F123, F124, F125, F126, F127, F128, F129, F130)
dictbbobnfree = dict((i.funId, i) for i in nfreefunclasses)
nfreeIDs = sorted(dictbbobnfree.keys())  # was: "nfreenames"
nfreeinfos = [str(i) + ': ' + dictbbobnfree[i].__doc__ for i in nfreeIDs]

dictbbobnoisy = dict((i.funId, i) for i in noisyfunclasses)
noisyIDs = sorted(dictbbobnoisy.keys())  # was noisynames

funclasses = list(nfreefunclasses) + list(noisyfunclasses)
dictbbob = dict((i.funId, i) for i in funclasses)

# TODO: pb xopt f9, 21, 22
class _FTemplate(BBOBNfreeFunction):
    """Template based on F1"""

    funId = 421337

    def initwithsize(self, curshape, dim):
        # DIM-dependent initialization
        if self.dim != dim:
            if self.zerox:
                self.xopt = zeros(dim)
            else:
                self.xopt = compute_xopt(self.rseed, dim)

        # DIM- and POPSI-dependent initialisations of DIM*POPSI matrices
        if self.lastshape != curshape:
            self.dim = dim
            self.lastshape = curshape
            self.arrxopt = resize(self.xopt, curshape)

        self.linearTf = None
        self.rotation = None

    def _evalfull(self, x):
        fadd = self.fopt
        curshape, dim = self.shape_(x)
        # it is assumed x are row vectors

        if self.lastshape != curshape:
            self.initwithsize(curshape, dim)

        # BOUNDARY HANDLING

        # TRANSFORMATION IN SEARCH SPACE
        x = x - self.arrxopt # cannot be replaced with x -= arrxopt!

        # COMPUTATION core
        ftrue = np.sum(x**2, 1)
        fval = self.noise(ftrue)

        # FINALIZE
        ftrue += fadd
        fval += fadd
        return fval, ftrue

def instantiate(ifun, iinstance=0, param=None, **kwargs):
    """Returns test function ifun, by default instance 0,
    and its optimal f-value."""
    res = dictbbob[ifun](iinstance = iinstance, param = param, **kwargs) # calling BBOBFunction.__init__(iinstance, param,...)
    return res, res.fopt

def get_param(ifun):
    """Returns the parameter values of the function ifun."""
    try:
        return dictbbob[ifun].paramValues
    except AttributeError:
        return (None,)

if __name__ == "__main__":
    import doctest
    doctest.testmod() # run all doctests in this module
