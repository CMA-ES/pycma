# -*- coding: utf-8 -*-
""" various math utilities, notably `eig` and a collection of simple
functions in `Mh`
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
# from future.builtins.disabled import *  # don't use any function which could lead to different results in Python 2 vs 3
import numpy as np
from .python3for2 import range
del absolute_import, division, print_function  #, unicode_literals

def randhss(n, dim, norm_=lambda x: len(x)**0.5, randn=np.random.randn):
    """`n` iid `dim`-dimensional vectors with length ``norm_(vector)``.

    The vectors are uniformly distributed on a hypersphere surface.

    CMA-ES diverges with popsize 100 in 15-D without option
    'CSA_clip_length_value': [0,0].

    >>> from cma.utilities.math import randhss
    >>> dim = 3
    >>> assert dim - 1e-7 < sum(randhss(1, dim)[0]**2) < dim + 1e-7

    """
    arv = randn(n, dim)
    for v in arv:
        v *= norm_(v) / np.sum(v**2)**0.5
    return arv

def randhss_mixin(n, dim, norm_=lambda x: len(x)**0.5,
                  c=lambda d: 1. / d, randn=np.random.randn):
    """`n` iid vectors uniformly distributed on the hypersphere surface with
    mixing in of normal distribution, which can be beneficial in smaller
    dimension.
    """
    arv = randhss(n, dim, norm_, randn)
    c = min((1, c(dim)))
    if c > 0:
        if c > 1:  # can never happen
            raise ValueError("c(dim)=%f should be <=1" % c)
        for v in arv:
            v *= (1 - c**2)**0.5 # has 2 / c longer time horizon than 1 - c
            v += c * randn(1, dim)[0]  # c is sqrt(2/c) times smaller than sqrt(c * (2 - c))
    return arv

# ____________________________________________________________
# ____________________________________________________________
#
# C and B are arrays rather than matrices, because they are
# addressed via B[i][j], matrices can only be addressed via B[i,j]

# tred2(N, B, diagD, offdiag);
# tql2(N, diagD, offdiag, B);


# Symmetric Householder reduction to tridiagonal form, translated from JAMA package.
def eig(C):
    """eigendecomposition of a symmetric matrix, much slower than
    `numpy.linalg.eigh`, return ``(EVals, Basis)``, the eigenvalues
    and an orthonormal basis of the corresponding eigenvectors, where

        ``Basis[i]``
            the i-th row of ``Basis``
        columns of ``Basis``, ``[Basis[j][i] for j in range(len(Basis))]``
            the i-th eigenvector with eigenvalue ``EVals[i]``

    """

# class eig(object):
#     def __call__(self, C):

# Householder transformation of a symmetric matrix V into tridiagonal form.
    # -> n             : dimension
    # -> V             : symmetric nxn-matrix
    # <- V             : orthogonal transformation matrix:
    #                    tridiag matrix == V * V_in * V^t
    # <- d             : diagonal
    # <- e[0..n-1]     : off diagonal (elements 1..n-1)

    # Symmetric tridiagonal QL algorithm, iterative
    # Computes the eigensystem from a tridiagonal matrix in roughtly 3N^3 operations
    # -> n     : Dimension.
    # -> d     : Diagonale of tridiagonal matrix.
    # -> e[1..n-1] : off-diagonal, output from Householder
    # -> V     : matrix output von Householder
    # <- d     : eigenvalues
    # <- e     : garbage?
    # <- V     : basis of eigenvectors, according to d


    #  tred2(N, B, diagD, offdiag); B=C on input
    #  tql2(N, diagD, offdiag, B);

    #  private void tred2 (int n, double V[][], double d[], double e[]) {
    def tred2 (n, V, d, e):
        #  This is derived from the Algol procedures tred2 by
        #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        #  Fortran subroutine in EISPACK.

        num_opt = False  # factor 1.5 in 30-D

        for j in range(n):
            d[j] = V[n - 1][j]  # d is output argument

        # Householder reduction to tridiagonal form.

        for i in range(n - 1, 0, -1):
            # Scale to avoid under/overflow.
            h = 0.0
            if not num_opt:
                scale = 0.0
                for k in range(i):
                    scale = scale + abs(d[k])
            else:
                scale = sum(abs(d[0:i]))

            if scale == 0.0:
                e[i] = d[i - 1]
                for j in range(i):
                    d[j] = V[i - 1][j]
                    V[i][j] = 0.0
                    V[j][i] = 0.0
            else:

                # Generate Householder vector.
                if not num_opt:
                    for k in range(i):
                        d[k] /= scale
                        h += d[k] * d[k]
                else:
                    d[:i] /= scale
                    h = np.dot(d[:i], d[:i])

                f = d[i - 1]
                g = h**0.5

                if f > 0:
                    g = -g

                e[i] = scale * g
                h = h - f * g
                d[i - 1] = f - g
                if not num_opt:
                    for j in range(i):
                        e[j] = 0.0
                else:
                    e[:i] = 0.0

                # Apply similarity transformation to remaining columns.

                for j in range(i):
                    f = d[j]
                    V[j][i] = f
                    g = e[j] + V[j][j] * f
                    if not num_opt:
                        for k in range(j + 1, i):
                            g += V[k][j] * d[k]
                            e[k] += V[k][j] * f
                        e[j] = g
                    else:
                        e[j + 1:i] += V.T[j][j + 1:i] * f
                        e[j] = g + np.dot(V.T[j][j + 1:i], d[j + 1:i])

                f = 0.0
                if not num_opt:
                    for j in range(i):
                        e[j] /= h
                        f += e[j] * d[j]
                else:
                    e[:i] /= h
                    f += np.dot(e[:i], d[:i])

                hh = f / (h + h)
                if not num_opt:
                    for j in range(i):
                        e[j] -= hh * d[j]
                else:
                    e[:i] -= hh * d[:i]

                for j in range(i):
                    f = d[j]
                    g = e[j]
                    if not num_opt:
                        for k in range(j, i):
                            V[k][j] -= (f * e[k] + g * d[k])
                    else:
                        V.T[j][j:i] -= (f * e[j:i] + g * d[j:i])

                    d[j] = V[i - 1][j]
                    V[i][j] = 0.0

            d[i] = h
        # end for i--

        # Accumulate transformations.

        for i in range(n - 1):
            V[n - 1][i] = V[i][i]
            V[i][i] = 1.0
            h = d[i + 1]
            if h != 0.0:
                if not num_opt:
                    for k in range(i + 1):
                        d[k] = V[k][i + 1] / h
                else:
                    d[:i + 1] = V.T[i + 1][:i + 1] / h

                for j in range(i + 1):
                    if not num_opt:
                        g = 0.0
                        for k in range(i + 1):
                            g += V[k][i + 1] * V[k][j]
                        for k in range(i + 1):
                            V[k][j] -= g * d[k]
                    else:
                        g = np.dot(V.T[i + 1][0:i + 1], V.T[j][0:i + 1])
                        V.T[j][:i + 1] -= g * d[:i + 1]

            if not num_opt:
                for k in range(i + 1):
                    V[k][i + 1] = 0.0
            else:
                V.T[i + 1][:i + 1] = 0.0


        if not num_opt:
            for j in range(n):
                d[j] = V[n - 1][j]
                V[n - 1][j] = 0.0
        else:
            d[:n] = V[n - 1][:n]
            V[n - 1][:n] = 0.0

        V[n - 1][n - 1] = 1.0
        e[0] = 0.0


    # Symmetric tridiagonal QL algorithm, taken from JAMA package.
    # private void tql2 (int n, double d[], double e[], double V[][]) {
    # needs roughly 3N^3 operations
    def tql2 (n, d, e, V):

        #  This is derived from the Algol procedures tql2, by
        #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        #  Fortran subroutine in EISPACK.

        num_opt = False  # using vectors from numpy makes it faster

        if not num_opt:
            for i in range(1, n):  # (int i = 1; i < n; i++):
                e[i - 1] = e[i]
        else:
            e[0:n - 1] = e[1:n]
        e[n - 1] = 0.0

        f = 0.0
        tst1 = 0.0
        eps = 2.0**-52.0
        for l in range(n):  # (int l = 0; l < n; l++) {

            # Find small subdiagonal element

            tst1 = max(tst1, abs(d[l]) + abs(e[l]))
            m = l
            while m < n:
                if abs(e[m]) <= eps * tst1:
                    break
                m += 1

            # If m == l, d[l] is an eigenvalue,
            # otherwise, iterate.

            if m > l:
                iiter = 0
                while 1:  # do {
                    iiter += 1  # (Could check iteration count here.)

                    # Compute implicit shift

                    g = d[l]
                    p = (d[l + 1] - g) / (2.0 * e[l])
                    r = (p**2 + 1)**0.5  # hypot(p,1.0)
                    if p < 0:
                        r = -r

                    d[l] = e[l] / (p + r)
                    d[l + 1] = e[l] * (p + r)
                    dl1 = d[l + 1]
                    h = g - d[l]
                    if not num_opt:
                        for i in range(l + 2, n):
                            d[i] -= h
                    else:
                        d[l + 2:n] -= h

                    f = f + h

                    # Implicit QL transformation.

                    p = d[m]
                    c = 1.0
                    c2 = c
                    c3 = c
                    el1 = e[l + 1]
                    s = 0.0
                    s2 = 0.0

                    # hh = V.T[0].copy()  # only with num_opt
                    for i in range(m - 1, l - 1, -1):  # (int i = m-1; i >= l; i--) {
                        c3 = c2
                        c2 = c
                        s2 = s
                        g = c * e[i]
                        h = c * p
                        r = (p**2 + e[i]**2)**0.5  # hypot(p,e[i])
                        e[i + 1] = s * r
                        s = e[i] / r
                        c = p / r
                        p = c * d[i] - s * g
                        d[i + 1] = h + s * (c * g + s * d[i])

                        # Accumulate transformation.

                        if not num_opt:  # overall factor 3 in 30-D
                            for k in range(n):  # (int k = 0; k < n; k++) {
                                h = V[k][i + 1]
                                V[k][i + 1] = s * V[k][i] + c * h
                                V[k][i] = c * V[k][i] - s * h
                        else:  # about 20% faster in 10-D
                            hh = V.T[i + 1].copy()
                            # hh[:] = V.T[i+1][:]
                            V.T[i + 1] = s * V.T[i] + c * hh
                            V.T[i] = c * V.T[i] - s * hh
                            # V.T[i] *= c
                            # V.T[i] -= s * hh

                    p = -s * s2 * c3 * el1 * e[l] / dl1
                    e[l] = s * p
                    d[l] = c * p

                    # Check for convergence.
                    if abs(e[l]) <= eps * tst1:
                        break
                # } while (Math.abs(e[l]) > eps*tst1);

            d[l] = d[l] + f
            e[l] = 0.0


        # Sort eigenvalues and corresponding vectors.
        if 11 < 3:
            for i in range(n - 1):  # (int i = 0; i < n-1; i++) {
                k = i
                p = d[i]
                for j in range(i + 1, n):  # (int j = i+1; j < n; j++) {
                    if d[j] < p:  # NH find smallest k>i
                        k = j
                        p = d[j]

                if k != i:
                    d[k] = d[i]  # swap k and i
                    d[i] = p
                    for j in range(n):  # (int j = 0; j < n; j++) {
                        p = V[j][i]
                        V[j][i] = V[j][k]
                        V[j][k] = p
    # tql2

    N = len(C[0])
    if 11 < 3:
        V = np.array([x[:] for x in C])  # copy each "row"
        N = V[0].size
        d = np.zeros(N)
        e = np.zeros(N)
    else:
        V = [[x[i] for i in range(N)] for x in C]  # copy each "row"
        d = N * [0.]
        e = N * [0.]

    tred2(N, V, d, e)
    tql2(N, d, e, V)
    return np.array(d), np.array(V)

class MathHelperFunctions(object):
    """static convenience math helper functions, if the function name
    is preceded with an "a", a numpy array is returned

    TODO: there is probably no good reason why this should be a class and not a
    module.

    """
    @staticmethod
    def aclamp(x, upper):
        return -MathHelperFunctions.apos(-x, -upper)
    @staticmethod
    def equals_approximately(a, b, eps=1e-12):
        if a < 0:
            a, b = -1 * a, -1 * b
        return (a - eps < b < a + eps) or ((1 - eps) * a < b < (1 + eps) * a)
    @staticmethod
    def vequals_approximately(a, b, eps=1e-12):
        a, b = np.array(a), np.array(b)
        idx = np.nonzero(a < 0)[0]  # find
        if len(idx):
            a[idx], b[idx] = -1 * a[idx], -1 * b[idx]
        return (np.all(a - eps < b) and np.all(b < a + eps)
                ) or (np.all((1 - eps) * a < b) and np.all(b < (1 + eps) * a))
    @staticmethod
    def expms(A, eig=np.linalg.eigh):
        """matrix exponential for a symmetric matrix"""
        # TODO: check that this works reliably for low rank matrices
        # first: symmetrize A
        D, B = eig(A)
        return np.dot(B, (np.exp(D) * B).T)
    @staticmethod
    def amax(vec, vec_or_scalar):
        return np.array(MathHelperFunctions.max(vec, vec_or_scalar))
    @staticmethod
    def max(vec, vec_or_scalar):
        b = vec_or_scalar
        if np.isscalar(b):
            m = [max(x, b) for x in vec]
        else:
            m = [max(vec[i], b[i]) for i in range(len((vec)))]
        return m
    @staticmethod
    def minmax(val, min_val, max_val):
        assert min_val <= max_val
        return min((max_val, max((val, min_val))))
    @staticmethod
    def aminmax(val, min_val, max_val):
        return np.array([min((max_val, max((v, min_val)))) for v in val])
    @staticmethod
    def amin(vec_or_scalar, vec_or_scalar2):
        return np.array(MathHelperFunctions.min(vec_or_scalar, vec_or_scalar2))
    @staticmethod
    def min(a, b):
        iss = np.isscalar
        if iss(a) and iss(b):
            return min(a, b)
        if iss(a):
            a, b = b, a
        # now only b can be still a scalar
        if iss(b):
            return [min(x, b) for x in a]
        else:  # two non-scalars must have the same length
            return [min(a[i], b[i]) for i in range(len((a)))]
    @staticmethod
    def norm(vec, expo=2):
        return sum(vec**expo)**(1 / expo)
    @staticmethod
    def apos(x, lower=0):
        """clips argument (scalar or array) from below at lower"""
        if lower == 0:
            return (x > 0) * x
        else:
            return lower + (x > lower) * (x - lower)

    @staticmethod
    def apenalty_quadlin(x, lower=0, upper=None):
        """Huber-like smooth penality which starts at lower.

        The penalty is zero below lower and affine linear above upper.

        Return::

            0, if x <= lower
            quadratic in x, if lower <= x <= upper
            affine linear in x with slope upper - lower, if x >= upper

        `upper` defaults to ``lower + 1``.

        """
        if upper is None:
            upper = np.asarray(lower) + 1
        z = np.asarray(x) - lower
        del x  # assert that x is not used anymore accidentally
        u = np.asarray(upper) - lower
        return (z > 0) * ((z <= u) * (z ** 2 / 2) + (z > u) * u * (z - u / 2))

    @staticmethod
    def prctile(data, p_vals=[0, 25, 50, 75, 100], sorted_=False):
        """``prctile(data, 50)`` returns the median, but p_vals can
        also be a sequence.

        Provides for small samples or extremes IMHO better values than
        matplotlib.mlab.prctile or np.percentile, however also slower.

        """
        ps = [p_vals] if np.isscalar(p_vals) else p_vals

        if not sorted_:
            data = sorted(data)
        n = len(data)
        d = []
        for p in ps:
            fi = p * n / 100 - 0.5
            if fi <= 0:  # maybe extrapolate?
                d.append(data[0])
            elif fi >= n - 1:
                d.append(data[-1])
            else:
                i = int(fi)
                d.append((i + 1 - fi) * data[i] + (fi - i) * data[i + 1])
        return d[0] if np.isscalar(p_vals) else d
    @staticmethod
    def iqr(data, percentile_function=np.percentile):  # MathHelperFunctions.prctile
        """interquartile range"""
        q25, q75 = percentile_function(data, [25, 75])
        return np.asarray(q75) - np.asarray(q25)
    def interdecilerange(data, percentile_function=np.percentile):
        """return 10% to 90% range width"""
        q10, q90 = percentile_function(data, [10, 90])
        return np.asarray(q90) - np.asarray(q10)
    @staticmethod
    def logit10(x, lower=0, upper=1):
        """map [lower, upper] -> R such that::

            upper - 10^-x  ->   x, and
            lower + 10^-x  ->  -x

        for large enough x. By default, simplifies close to `log10(x / (1 - x))`.

        >>> from cma.utilities.math import Mh
        >>> l, u = -1, 2
        >>> print(Mh.logit10([l+0.01, 0.5, u-0.01], l, u))
        [-1.9949189  0.         1.9949189]

        """
        x = np.asarray(x)
        z = (x - lower) / (upper - lower)  # between 0 and 1
        return np.log10((x - lower)**(1-z) / (upper - x)**z)
        return (1 - z) * np.log10(x - lower) - z * np.log10(upper - x)
    @staticmethod
    def sround(nb):  # TODO: to be vectorized
        """return stochastic round: int(nb) + (rand()<remainder(nb))"""
        return int(nb) + (1 if np.random.rand(1)[0] < (nb % 1) else 0)
    @staticmethod
    def cauchy_with_variance_one():
        n = np.random.randn() / np.random.randn()
        while abs(n) > 1000:
            n = np.random.randn() / np.random.randn()
        return n / 25
    @staticmethod
    def standard_finite_cauchy(size=1):
        try:
            l = len(size)
        except TypeError:
            l = 0

        if l == 0:
            return np.array([MathHelperFunctions.cauchy_with_variance_one() for _i in range(size)])
        elif l == 1:
            return np.array([MathHelperFunctions.cauchy_with_variance_one() for _i in range(size[0])])
        elif l == 2:
            return np.array([[MathHelperFunctions.cauchy_with_variance_one() for _i in range(size[1])]
                         for _j in range(size[0])])
        else:
            raise ValueError('len(size) cannot be larger than two')

Mh = MathHelperFunctions
