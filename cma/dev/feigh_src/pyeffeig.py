from __future__ import division  # use // for integer division
from __future__ import absolute_import  # use from . import
from __future__ import print_function  # use print("...") instead of print "..."
from __future__ import unicode_literals  # all the strings are unicode

"""Module for Efficient Computation of Eigen Decomposition of Real Matrix

This module provide functionarities to compute the eigen decomposition of
a real-valued matrix of the form

    C = Q * B * Q' + X * X' - Y * Y',

where
    Q: m-by-n with orthonormal columns
    B: n-by-n symmetric matrix
    X: m-by-nx arbitrary matrix
    Y: m-by-ny arbitrary matrix

If n + nx + ny < m, C is not full rank. In this case, `efficient_eigh`
computes the non-zero eigenvalues and the corresponding eigenvectors of C
with O(m * (n + nx + ny)^2) floating point operations.
"""

__author__ = 'Youhei Akimoto'

import numpy as np
import numpy.linalg as la

EPS_DEFAULT = 1e-12


def thinsvd(X):
    """Compute the thin SVD of X O(m*n^2)

    The thin SVD of X is
        X = U * S * V,
    where
        U: the left singular vectors (m-by-r, orthonormal columns)
        S: the singular values (r-by-r, diagonal)
        V: transpose of the right singular vectors (r-by-n, orthonormal rows)

    The output of the function satisfies X = np.dot(U * S, V).

    Parameters
    ----------
    X: 2d array
        m-by-n with rank r

    Returns
    -------
    U (2d array) : m-by-r matrix with orthonormal columns
    s (1d array) : array of r singular values (sorted in descending order)
    V (2D array) : r-by-n matrix with orthonormal rows
    """
    U, s, V = la.svd(X, full_matrices=False, compute_uv=True)
    bfil = s ** 2 < EPS_DEFAULT
    s[bfil] = 0.0
    U[:, bfil] = 0.0
    V[bfil, :] = 0.0
    return U, s, V

def qb_decomposition(Q, B, X, sign=1):
    """Compute a decomposition of Q*B*Q' + X*X'

    Decomposition is defined as
        Qo*Bo*Qo' = Q*B*Q' + sign * X*X.

    Let r <= n + k be the rank of the matrix above. Then,
    Qo is an m-by-r matrix with orthonormal columns,
    Bo is an r-by-r full-rank symmetric matrix.

    Parameters
    ----------
    Q : 2d array
        m-by-n matrix with orthonormal columns (n < m).
    B : 2d array
        n-by-n full-rank symmetric matrix.
    X : 2d array
        m-by-k dimensional matrix (k < m)
    sign : float, default = 1

    Returns
    -------
    Qo (2d array) : m-by-r matrix with orthonormal columns.
    Bo (2d array) : r-by-r full-rank symmetric matrix.
    """

    QX = np.dot(Q.T, X)
    Y = - np.dot(Q, QX)
    Y += X
    Qqr, d, v = thinsvd(Y)
    Rqr = (v.T * d).T

    m = Q.shape[0]
    n = Q.shape[1]
    r = Qqr.shape[1]

    Qo = np.empty((m, n + r))
    Qo[:, :n] = Q
    Qo[:, n:] = Qqr

    Bo = np.empty((n + r, n + r))
    Bo[:n, :n] = np.dot(QX, QX.T)
    Bo[:n, :n] *= sign
    Bo[:n, :n] += B
    Bo[n:, :n] = np.dot(Rqr, QX.T)
    Bo[n:, :n] *= sign
    Bo[:n, n:] = Bo[n:, :n].T
    Bo[n:, n:] = np.dot(Rqr, Rqr.T)
    Bo[n:, n:] *= sign

    return Qo, Bo


def qb_eigh(Q, B):
    """Compute the thin eigen decomposition of Q*B*Q'

    Parameters
    ----------
    Q : 2d array
        matrix with orthonormal columns.
    B : 2d array
        full-rank symmetric matrix.

    Returns
    -------
    d (1d array) : eigenvalues (in ascending order).
    E (2d array) : eigenvectors.
    """
    D, Qeig = la.eigh(B)
    bfil = np.abs(D) < EPS_DEFAULT
    D[bfil] = 0.0
    Qeig[:, bfil] = 0.0
    return D, np.dot(Q, Qeig)


def feigh(alpha, Q, B, beta, X, gamma, Y, E, D):
    """Eigen Decomposition of alpha * Q*B*Q' + beta * X*X' + gamma * Y*Y'

    Its complexity is O(m*(n+nx+ny)^2).

    Parameters
    ----------
    alpha, beta, gamma : float
        coefficients
    Q : 2d array
        matrix with orthonormal columns.
    B : 2d array
        full-rank symmetric matrix.
    X, Y : 2d array
        any matrix with the same row dimension as Q
    E : 2d array, output
        eigenvectors
    D : 1d array, output
        eigenvalues (in ascending order)
    """
    m = Q.shape[0]
    n = Q.shape[1]
    nx = X.shape[1]
    ny = Y.shape[1]
    no = E.shape[1]
    E[:, :] = 0.0
    D[:] = 0.0

    if no >= m:
        # Use the standard eigh if the resulting matrix will be full rank
        mat = alpha * np.dot(Q, np.dot(B, Q.T))
        mat += beta * np.dot(X, X.T)
        mat += gamma * np.dot(Y, Y.T)
        dd, ee = la.eigh(mat)
        bfil = np.abs(dd) < EPS_DEFAULT
        D[bfil] = 0.0
        E[:, bfil] = 0.0
        return
    else:
        if alpha == 0 and beta == 0 and gamma == 0:
            return 

        elif alpha == 0 and beta == 0 and gamma != 0:
            U, s, V = thinsvd(Y)
            dd = gamma * s ** 2
            D[:ny] = dd
            E[:m, :ny] = U
            return

        elif alpha == 0 and beta != 0 and gamma == 0:
            U, s, V = thinsvd(X)
            dd = beta * s ** 2
            D[:nx] = dd[::-1]
            E[:m, :nx] = U[:, ::-1]
            return

        elif alpha == 0 and beta != 0 and gamma != 0:
            Q1, s, V = thinsvd(X)
            B1 = np.diag(beta * s ** 2)
            Q2, B2 = qb_decomposition(Q1, B1, Y, sign=gamma)
            D[:nx+ny], E[:m, :nx+ny] = qb_eigh(Q2, B2)
            return
        
        elif alpha != 0 and beta == 0 and gamma == 0:
            D[:n], E[:m, :n] = qb_eigh(Q, B)
            D *= alpha
            return
        
        elif alpha != 0 and beta == 0 and gamma != 0:
            Q2, B2 = qb_decomposition(Q, alpha * B, Y, sign=gamma)
            D[:n+ny], E[:m, :n+ny] = qb_eigh(Q2, B2)
            return

        elif alpha != 0 and beta != 0 and gamma == 0:
            Q1, B1 = qb_decomposition(Q, alpha * B, X, sign=beta)
            D[:n+nx], E[:m, :n+nx] = qb_eigh(Q1, B1)
            return
        
        elif alpha != 0 and beta != 0 and gamma != 0:
            Q1, B1 = qb_decomposition(Q, alpha * B, X, sign=beta)
            Q2, B2 = qb_decomposition(Q1, B1, Y, sign=gamma)
            D[:n+nx+ny], E[:m, :n+nx+ny] = qb_eigh(Q2, B2)
            return

        else:
            raise RuntimeError()
