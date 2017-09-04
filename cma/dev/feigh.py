import numpy as np
try:  # import already compiled module
    from .feigh_src.effeig import feigh
except Exception:
    import os    
    try:  # compile and import module
        from numpy import f2py
        cwd = os.getcwd()
        srcdir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.join(srcdir, "feigh_src"))
        with open("effeig.f90", 'rb') as sourcefile:
            sourcecode = sourcefile.read()
        f2py.compile(sourcecode, modulename='effeig', extra_args='--link-lapack_opt', verbose=False, extension='.f90')
        os.chdir(cwd)
        from .feigh_src.effeig import feigh
    except Exception:  # import python implementation
        if os.getcwd() != cwd:
            os.chdir(cwd)
        from .feigh_src.pyeffeig import feigh


def pyfeigh(alpha, Q, B, beta, Y, gamma, Z, E, D):
    """Eigen Decomposition of alpha * Q*B*Q' + beta * Y*Y' + gamma * Y*Y'

    Its complexity is O(m*(n+nx+ny)^2).

    Parameters
    ----------
    alpha, beta, gamma : float
        coefficients
    Q : 2d array
        matrix with orthonormal columns.
    B : 2d array
        full-rank symmetric matrix.
    Y, Z : 2d array
        any matrix with the same row dimension as Q
    E : 2d array, output
        eigenvectors
    D : 1d array, output
        eigenvalues (in ascending order)
    """
    assert Q.flags['F_CONTIGUOUS']
    assert B.flags['F_CONTIGUOUS']
    assert Y.flags['F_CONTIGUOUS']
    assert Z.flags['F_CONTIGUOUS']
    assert E.flags['F_CONTIGUOUS']
    assert D.flags['F_CONTIGUOUS']
    M = Q.shape[0]
    N = Q.shape[1]
    Ny = Y.shape[1]
    Nz = Z.shape[1]
    myQ = Q if N > 0 else np.empty((M, 1), order='F')
    myB = B if N > 0 else np.empty((1, 1), order='F')
    myY = Y if Ny > 0 else np.empty((M, 1), order='F')
    myZ = Z if Nz > 0 else np.empty((M, 1), order='F')
    myalpha = alpha if N > 0 else 0.0
    mybeta = beta if Ny > 0 else 0.0
    mygamma = gamma if Nz > 0 else 0.0
    return feigh(myalpha, myQ, myB, mybeta, myY, mygamma, myZ, E, D)


if __name__ == '__main__':
    import time
    import math
    timer = time.process_time
    M = 10000             # number of rows
    N = int(math.log(M))  # number of columns of Q
    Nx = N                # number of columns of X
    Ny = N                # number of columns of Y
    No = N + Ny + Nx      # number of columns of output matrix
    # Prepare input and output matrices
    # Note that the arrays need to be column major
    Q = np.random.randn(N, M)
    for j in range(Q.shape[0]):
        for k in range(j):
            Q[j] -= np.dot(Q[k], Q[j]) * Q[k]
        Q[j] /= np.linalg.norm(Q[j])
    B = np.random.randn(N, N)
    B = (B + B.T) / 2.
    X = np.random.randn(Nx, M)
    Y = np.random.randn(Ny, M)
    E = np.empty((No, M))        # matrix of eigenvectors
    D = np.empty(No)             # vector of eigenvalues
    t1 = timer()
    info = pyfeigh(1.0, Q.T, B.T, 1.0, X.T, -1.0, Y.T, E.T, D)
    t2 = timer()
    print(t2 - t1)    
