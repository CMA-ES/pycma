! Module for Computationally Efficient Eigen Decomposition of Low Rank Matrix
!
! ALGORITHM
! ---------
! This module provides functionarities to compute the eigen decomposition of
! a real-valued matrix of the form
!     C = alpha * Q * B * Q' + beta * X * X' + gamma * Y * Y',
! where
!     Q: m-by-n with orthonormal columns
!     B: n-by-n symmetric matrix
!     X: m-by-nx arbitrary matrix
!     Y: m-by-ny arbitrary matrix
!     alpha, beta, gamma: arbitrary real numbers
!
! If n + nx + ny < m, C is not full rank. In this case, `feigh`
! computes the non-zero eigenvalues and the corresponding eigenvectors of C
! with O(m * (n + nx + ny)^2) floating point operations and
! 3 * (m + n + nx + ny + 1) * (n + nx + ny) extra memory.
!
! If n + nx + ny >= m, C is full rank. Hence, we use the LAPACK `dsyev`
! function. Indeed, if 3 * (m + n + nx + ny + 1) * (n + nx + ny) > m * m,
! `dsyev` of LAPACK is used.
!
! REQUIREMENT
! -----------
! Optimized LAPACK (Accerelate/vecLib for older Mac OS, ATLAS for Linux)
! Type the following to check your system
!   f2py --help-link lapack_opt
!
! COMPILE
! -------
! To compile the module as a Python module, one can use `f2py` command
! that comes with `scipy`. The compile can be done as follows
!
!   f2py -c -m effeig --link-lapack_opt effeig.f90
!
! If the program is compiled successfully, you will find .so file.
!
! USAGE
! -----
! A python sample code is given in the following
!
! #---------- Python Sample Code ----------#
! import time
! import math
! import numpy as np
! import effeig
!
! def pyfeigh(alpha, Q, B, beta, Y, gamma, Z, E, D):
!     assert Q.flags['F_CONTIGUOUS']
!     assert B.flags['F_CONTIGUOUS']
!     assert Y.flags['F_CONTIGUOUS']
!     assert Z.flags['F_CONTIGUOUS']
!     assert E.flags['F_CONTIGUOUS']
!     assert D.flags['F_CONTIGUOUS']
!     M = Q.shape[0]
!     N = Q.shape[1]
!     Ny = Y.shape[1]
!     Nz = Z.shape[1]
!     myQ = Q if N > 0 else np.empty((M, 1), order='F')
!     myB = B if N > 0 else np.empty((1, 1), order='F')
!     myY = Y if Ny > 0 else np.empty((M, 1), order='F')
!     myZ = Z if Nz > 0 else np.empty((M, 1), order='F')
!     myalpha = alpha if N > 0 else 0.0
!     mybeta = beta if Ny > 0 else 0.0
!     mygamma = gamma if Nz > 0 else 0.0
!     return effeig.feigh(myalpha, myQ, myB, mybeta, myY, mygamma, myZ, E, D)
!
! #timer = time.perf_counter
! timer = time.process_time
! M = 10000             # number of rows
! N = int(math.log(M))  # number of columns of Q
! Nx = N                # number of columns of X
! Ny = N                # number of columns of Y
! No = N + Ny + Nx      # number of columns of output matrix
! # Prepare input and output matrices
! # Note that the arrays need to be column major
! Q = np.random.randn(N, M)
! for j in range(Q.shape[0]):
!     for k in range(j):
!         Q[j] -= np.dot(Q[k], Q[j]) * Q[k]
!     Q[j] /= np.linalg.norm(Q[j])
! B = np.random.randn(N, N)
! B = (B + B.T) / 2.
! X = np.random.randn(Nx, M)
! Y = np.random.randn(Ny, M)
! E = np.empty((No, M))        # matrix of eigenvectors
! D = np.empty(No)             # vector of eigenvalues
! t1 = timer()
! info = pyfeigh(1.0, Q.T, B.T, 1.0, X.T, -1.0, Y.T, E.T, D)
! t2 = timer()
! print(t2 - t1)
! #----------------------------------------#
!
! LAST UPDATE
! -----------
! 2017/05/22: first release
!
! Author
! ------
! Youhei Akimoto, Ph.D., Assistant Professor,
! Shinshu University, Nagano, Japan.
! y_akimoto [at] shinshu-u.ac.jp

module effeig
  implicit none
  double precision, parameter :: EPS = 1.0d-12
contains

  subroutine thinsvd(M, N, X, U, S, VT, work)
    ! Compute the thin SVD of X O(m*n^2)
    ! The thin SVD of X is
    !     X = U * S * VT,
    ! where
    !     U: the left singular vectors (m-by-r, orthonormal columns)
    !     S: the singular values (r-by-r, diagonal)
    !     VT: transpose of the right singular vectors (r-by-n, orthonormal rows)
    implicit none
    integer, intent(in) :: M, N
    double precision, intent(in) :: X(M, N)
    double precision, intent(out) :: U(M, N), S(N), VT(N, N), work(:)
    integer :: lda, ldu, ldvt, lwork, info, i
    character(1), parameter :: jobu = 'S'
    character(1), parameter :: jobvt = 'S'
    lda = M
    ldu = M
    ldvt = N
    lwork = size(work)
    call dgesvd(jobu, jobvt, M, N, X(1:M, 1:N), lda, S(1:N), U(1:M, 1:N), ldu,&
         VT(1:N, 1:N), ldvt, work(1:lwork), lwork, info)
    do i = 1, N
       if ( abs(S(i)) < EPS ) then
          S(i) = 0.0d0
          U(1:M, i) = 0.0d0
          VT(i, 1:N) = 0.0d0
       end if
    end do
  end subroutine thinsvd

  subroutine qb(M, N, R, Qi, Bi, Xi, Qo, Bo, sign, work)
    ! Compute a decomposition of Q*B*Q' + X*X'
    ! Decomposition is defined as
    !     Qo*Bo*Qo' = Qi*Bi*Qi' + sign * Xi*Xi.
    ! Let r <= n + k be the rank of the matrix above. Then,
    ! Qo is an m-by-r matrix with orthonormal columns,
    ! Bo is an r-by-r full-rank symmetric matrix.
    implicit none
    integer, intent(in) :: M, N, R
    double precision, intent(in) :: Qi(M, N)   
    double precision, intent(in) :: Bi(N, N)   
    double precision, intent(in) :: Xi(M, R)    
    double precision, intent(out) :: Qo(M, N+R)  
    double precision, intent(out) :: Bo(N+R, N+R) 
    double precision, intent(in) :: sign
    double precision, intent(out) :: work(:) 
    integer :: i, lwork
    lwork = size(work)
    do i = 1, R
       work(R*R+R+1+M*(i-1):R*R+R+M*i) = Xi(1:M, i)
    end do
    Qo(1:M, 1:N) = Qi(1:M, 1:N)
    call dgemm('T', 'N', N, R, M, 1.0d0, Qi(1:M, 1:N), M, Xi(1:M, 1:R), M, 0.0d0, Bo(1:N, N+1:N+R), N)
    call dgemm('N', 'N', M, R, N, -1.0d0, Qi(1:M, 1:N), M, Bo(1:N, N+1:N+R), N, 1.0d0, work(R*R+R+1:R*R+R+M*R), M)
    call thinsvd(M, R, work(R*R+R+1:R*R+R+M*R), Qo(1:M, N+1:N+R), work(R*R+1:R*R+R), work(1:R*R), work(R*(R+M+1)+1:lwork))
    do i = 1, R
       work(1+R*(i-1):R*i) = work(1+R*(i-1):R*i) * work(R*R+1:R*R+R)
    end do
    Bo(1:N, 1:N) = Bi(1:N, 1:N)
    call dgemm('N', 'T', N, N, R, sign, Bo(1:N, N+1:N+R), N, Bo(1:N, N+1:N+R), N, 1.0d0, Bo(1:N, 1:N), N)
    call dgemm('N', 'T', R, R, R, sign, work(1:R*R), R, work(1:R*R), R, 0.0d0, Bo(N+1:N+R, N+1:N+R), R)    
    call dgemm('N', 'T', R, N, R, sign, work(1:R*R), R, Bo(1:N, N+1:N+R), N, 0.0d0, Bo(N+1:N+R, 1:N), R)
    do i = 1, R
       Bo(1:N, N+i) = Bo(N+i, 1:N)
    end do
  end subroutine qb

  subroutine qbeigh(M, N, Qio, Bio, work)
    implicit none
    integer, intent(in) :: M, N
    double precision, intent(inout) :: Qio(M, N)
    double precision, intent(inout) :: Bio(N, N)    
    double precision, intent(out) :: work(:)
    integer :: lwork, info, i
    character(1), parameter :: jobz = 'V', uplo = 'U'
    lwork = size(work)
    call dsyev(jobz, uplo, N, Bio(1:N, 1:N), N, work(1:N), work(N+1:lwork), lwork-N, info)
    do i = 1, N
       if ( abs(work(i)) < EPS ) then
          work(i) = 0.0d0
          Bio(1:N, i) = 0.0d0
       end if
    end do
    call dgemm('N', 'N', M, N, N, 1.0d0, Qio(1:M, 1:N), M, Bio(1:N, 1:N), N, 0.0d0, work(N+1:N+M*N), M)
    Bio(1:N, 1) = work(1:N)
    Bio(1:N, 2:N) = 0.0d0
    do i = 1, N
       Qio(1:M, i) = work(N+M*(i-1)+1:N+M*i)
    end do
  end subroutine qbeigh
  
  subroutine fasteigh(alpha, Q, B, beta, Y, gamma, Z, E, D, info)
    implicit none
    double precision, intent(in) :: alpha, beta, gamma
    double precision, intent(in) :: Q(:, :), B(:, :), Y(:, :), Z(:, :)
    double precision, intent(out) :: E(:, :), D(:)
    integer, intent(out) :: info
    double precision, allocatable :: work(:)
    integer :: M, N, ny, nz, no, i, lwork
    character(1), parameter :: jobz = 'V', uplo = 'U'    
    M = size(Q, 1)
    N = size(Q, 2)
    ny = size(Y, 2)
    nz = size(Z, 2)
    no = size(E, 2)
    if (alpha == 0.0d0) then
       N = 0
    end if
    if (beta == 0.0d0) then
       ny = 0
    end if
    if (gamma == 0.0d0) then
       nz = 0
    end if

    if (size(Y, 1) /= M .or. size(Z, 1) /= M .or. size(E, 1) /= M) then
       print *, 'row dimension mismatch'
       info = -1
       return
    else if (size(B, 1) /= N .and. alpha /= 0.0d0) then
       print *, 'column dimension of Q and row dimension of B are different'
       info = -2       
       return
    else if (size(B, 2) /= N .and. alpha /= 0.0d0) then
       print *, 'B not square'
       info = -3       
       return
    else if (size(D) /= no) then
       print *, 'column dimension of E and length of D are different'
       info = -4       
       return
    else if (no < min(N + ny + nz, M)) then
       print *, 'length of D smaller than min. of M and the sum of columns of Q, Y, and Z'
       info = -5       
       return
    end if

    if (no >= M) then
       lwork = M * M
    else
       lwork = 3 * (M + N + ny + nz + 1) * (N + ny + nz)
    end if
    allocate(work(lwork))           
    E(:, :) = 0.0d0
    D(:) = 0.0d0
    
    if (no >= M) then

       if (N > 0) then
          call dgemm('N', 'T', N, M, N, 1.0d0, B(1:N, 1:N), N, Q(1:M, 1:N), M, 0.0d0, work(1:N*M), N)
          call dgemm('N', 'N', M, M, N, 1.0d0, Q(1:M, 1:N), M, work(1:N*M), N, 0.0d0, E(1:M, 1:M), M)
       end if
       if (ny > 0) then
          call dgemm('N', 'T', M, M, ny, beta, Y(1:M, 1:ny), M, Y(1:M, 1:ny), M, alpha, E(1:M, 1:M), M)
       else
          E(1:M, 1:M) = E(1:M, 1:M) * alpha
       end if
       if (nz > 0) then
          call dgemm('N', 'T', M, M, nz, gamma, Z(1:M, 1:nz), M, Z(1:M, 1:nz), M, 1.0d0, E(1:M, 1:M), M)
       end if
       call dsyev(jobz, uplo, M, E(1:M, 1:M), M, D(1:M), work(:), lwork, info)
       do i = 1, M
          if ( abs(D(i)) < EPS ) then
             work(i) = 0.0d0
             E(1:M, i) = 0.0d0
          end if
       end do
       info = 0
       
    else if (alpha /= 0.0d0 .and. beta /= 0.0d0 .and. gamma /= 0.0d0) then

       call qb(M, N, ny, Q, B, Y, work(1:M*(N+ny)), work(M*(N+ny)+1:M*(N+ny)+(N+ny)**2), beta / alpha, &
            work(M*(N+ny)+(N+ny)**2+1:lwork))
       call qb(M, N+ny, nz, work(1:M*(N+ny)), work(M*(N+ny)+1:M*(N+ny)+(N+ny)**2), Z, E(1:M, 1:N+ny+nz), &
            work(M*(N+ny)+(N+ny)**2+1:M*(N+ny)+(N+ny)**2+(N+ny+nz)**2), gamma / alpha, &
            work(M*(N+ny)+(N+ny)**2+(N+ny+nz)**2+1:lwork))
       work(1:(N+ny+nz)**2) = work(M*(N+ny)+(N+ny)**2+1:M*(N+ny)+(N+ny)**2+(N+ny+nz)**2)
       call qbeigh(M, N+ny+nz, E, work(1:(N+ny+nz)**2), work((N+ny+nz)**2+1:lwork))
       D(1:N+ny+nz) = work(1:N+ny+nz) * alpha
       info = 1

    else if (alpha /= 0.0d0 .and. beta /= 0.0d0 .and. gamma == 0.0d0) then

       call qb(M, N, ny, Q, B, Y, E(1:M, 1:N+ny), work(1:(N+ny)**2), beta / alpha, work((N+ny)**2+1:lwork))
       call qbeigh(M, N+ny, E(1:M, 1:N+ny), work(1:(N+ny)**2), work((N+ny)**2+1:lwork))
       D(1:N+ny) = work(1:N+ny) * alpha
       info = 2

    else if (alpha /= 0.0d0 .and. beta == 0.0d0 .and. gamma /= 0.0d0) then

       call qb(M, N, nz, Q, B, Z, E(1:M, 1:N+nz), work(1:(N+nz)**2), gamma / alpha, work((N+nz)**2+1:lwork))
       call qbeigh(M, N+nz, E(1:M, 1:N+nz), work(1:(N+nz)**2), work((N+nz)**2+1:lwork))
       D(1:N+nz) = work(1:N+nz) * alpha
       info = 3

    else if (alpha /= 0.0d0 .and. beta == 0.0d0 .and. gamma == 0.0d0) then

       E(1:M, 1:N) = Q(1:M, 1:N)
       do i = 1, N
          work(N*(i-1)+1:N*i) = B(1:M, i)
       end do
       call qbeigh(M, N, E, work(1:N*N), work(N*N+1:lwork))
       D(1:N) = work(1:N) * alpha
       info = 4

    else if (alpha == 0.0d0 .and. beta /= 0.0d0 .and. gamma /= 0.0d0) then

       call thinsvd(M, ny, Y, work(1:M*ny), D(1:ny), work(M*ny+1:M*ny+ny*ny), work(M*ny+ny*ny+1:lwork))
       work(M*ny+1:M*ny+ny*ny) = 0.0d0
       do i = 1, ny
          work(M * ny + i + ny * (i - 1)) = D(i) ** 2 * beta
       end do
       call qb(M, ny, nz, work(1:M*ny), work(M*ny+1:M*ny+ny*ny), Z, E(1:M, 1:ny+nz),&
            work(M*ny+ny*ny+1:M*ny+ny*ny+(ny+nz)**2), gamma, work(M*ny+ny*ny+(ny+nz)**2+1:lwork))
       work(1:(ny+nz)**2) = work(M*ny+ny*ny+1:M*ny+ny*ny+(ny+nz)**2)
       call qbeigh(M, ny+nz, E(1:M, 1:ny+nz), work(1:(ny+nz)**2), work((ny+nz)**2+1:lwork))
       D(1:ny+nz) = work(1:ny+nz)
       info = 5

    else if (alpha == 0.0d0 .and. beta /= 0.0d0 .and. gamma == 0.0d0) then

       call thinsvd(M, ny, Y, E(1:M, 1:ny), D(1:ny), work(1:ny*ny), work(ny*ny+1:lwork))
       D(1:ny) = D(ny:1:-1) * D(ny:1:-1) * beta
       E(1:M, 1:ny) = E(1:M, ny:1:-1)
       info = 6

    else if (alpha == 0.0d0 .and. beta == 0.0d0 .and. gamma /= 0.0d0) then

       call thinsvd(M, nz, Z, E(1:M, 1:nz), D(1:nz), work(1:nz*nz), work(nz*nz+1:lwork))
       D(1:nz) = D(1:nz) * D(1:nz) * gamma
       info = 7

    else if (alpha == 0.0d0 .and. beta == 0.0d0 .and. gamma == 0.0d0) then

       info = 8

    end if
    deallocate(work)
  end subroutine fasteigh

  subroutine checkerr(alpha, Q, B, beta, Y, gamma, Z, err)
    implicit none
    double precision, intent(in) :: alpha, beta, gamma
    double precision, intent(in) :: Q(:, :), B(:, :), Y(:, :), Z(:, :)
    double precision, intent(out) :: err    
    integer :: M, N, ny, nz, i, info
    double precision, allocatable :: A(:, :), C(:, :), E(:, :), D(:), F(:, :)

    M = size(Q, 1)
    N = size(B, 2)
    ny = size(Y, 2)
    nz = size(Z, 2)
    allocate(A(M, M), C(M, M), E(M, N+ny+nz), D(N+ny+nz), F(M, M))
    if ( alpha /= 0.0d0 ) then
       call dgemm('N', 'N', M, N, N, alpha, Q, M, B, N, 0.0d0, C(1:M, 1:N), M)
       call dgemm('N', 'T', M, M, N, 1.0d0, C(1:M, 1:N), M, Q, M, 0.0d0, A, M)
    else
       A(:, :) = 0.0d0
    end if
    call dgemm('N', 'T', M, M, ny, beta, Y, M, Y, M, 1.0d0, A, M)
    call dgemm('N', 'T', M, M, nz, gamma, Z, M, Z, M, 1.0d0, A, M)
    call fasteigh(alpha, Q, B, beta, Y, gamma, Z, E, D, info)
    do i = 1, N+ny+nz
       F(:, i) = E(:, i) * D(i)
    end do
    call dgemm('N', 'T', M, M, N+ny+nz, 1.0d0, F(1:M, 1:N+ny+nz), M, E, M, 0.0d0, C, M)
    err = maxval(abs(A - C))    
    !print *, info, M, N, ny, nz, err
    deallocate(A, C, E, D, F)
  end subroutine checkerr

  subroutine checkerrqb(alpha, Q, B, beta, Y, err1, err2)
    implicit none
    double precision, intent(in) :: alpha, beta
    double precision, intent(in) :: Q(:, :), B(:, :), Y(:, :)
    double precision, intent(out) :: err1, err2
    integer :: M, N, ny, i
    double precision, allocatable :: A(:, :), C(:, :), E(:, :), D(:), F(:, :), G(:, :), work(:)
    M = size(Q, 1)
    N = size(B, 2)
    ny = size(Y, 2)
    allocate(A(M, M), C(M, M), E(M, N+ny), D(N+ny), F(M, M), G(N+ny, N+ny), work(2*(M+2)*max(M, N+ny)))
    ! Compute a*Q*B*Qt + b*Y*Yt + g*Z*Zt
    call dgemm('N', 'N', M, N, N, alpha, Q, M, B, N, 0.0d0, C(1:M, 1:N), M)
    call dgemm('N', 'T', M, M, N, 1.0d0, C(1:M, 1:N), M, Q, M, 0.0d0, A, M)
    call dgemm('N', 'T', M, M, ny, beta, Y, M, Y, M, 1.0d0, A, M)
    ! Compute qb
    call qb(M, N, ny, Q, alpha * B, Y, E, G, beta, work)
    call dgemm('N', 'N', M, N+ny, N+ny, 1.0d0, E, M, G, N+ny, 0.0d0, F(1:M, 1:N+ny), M)
    call dgemm('N', 'T', M, M, N+ny, 1.0d0, F(1:M, 1:N+ny), M, E, M, 0.0d0, C, M)
    err1 = maxval(abs(A - C))
    !print *, '  QB', M, N, ny, err1
    ! Compute qbeigh
    call qbeigh(M, N+ny, E, G, work)
    do i = 1, N+ny
       F(:, i) = E(:, i) * G(i, 1)
    end do
    call dgemm('N', 'T', M, M, N+ny, 1.0d0, F(:, 1:N+ny), M, E, M, 0.0d0, C, M)
    err2 = maxval(abs(A - C))
    !print *, '  QBEIGH', M, N, ny, err2
    deallocate(A, C, E, D, F, G, work)
  end subroutine checkerrqb

end module effeig

subroutine feigh(alpha, Q, m, n, B, beta, Y, ny, gamma, Z, nz, E, no, D, info)
  use effeig, only : fasteigh
  implicit none
  integer, intent(in) :: m, n, ny, nz, no
  double precision, intent(in) :: alpha, beta, gamma
  double precision, intent(in) :: Q(m, n), B(n, n), Y(m, ny), Z(m, nz)
  double precision, intent(inout) :: E(m, no), D(no)
  integer, intent(out) :: info
  call fasteigh(alpha, Q, B, beta, Y, gamma, Z, E, D, info)  
end subroutine feigh

subroutine checkerror(alpha, Q, m, n, B, beta, Y, ny, gamma, Z, nz, err)
  use effeig, only : checkerr
  implicit none
  integer, intent(in) :: m, n, ny, nz  
  double precision, intent(in) :: alpha, beta, gamma
  double precision, intent(in) :: Q(m, n), B(n, n), Y(m, ny), Z(m, nz)
  double precision, intent(out) :: err
  call checkerr(alpha, Q, B, beta, Y, gamma, Z, err)
end subroutine checkerror
