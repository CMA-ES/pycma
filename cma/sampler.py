"""Collection of classes that sample from parametrized distributions and
provide an update mechanism of the distribution parameters.

All classes are supposed to follow the base class
`StatisticalModelSamplerWithZeroMeanBaseClass` interface in module
`interfaces`.
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
from .utilities.python3for2 import range
import numpy as np
from .utilities.utils import rglen, print_warning
from .interfaces import StatisticalModelSamplerWithZeroMeanBaseClass
del absolute_import, division, print_function  #, unicode_literals

_assertions_quadratic = True

class GaussStandardConstant(StatisticalModelSamplerWithZeroMeanBaseClass):
    """Standard Multi-variate normal distribution with zero mean.

    No update/change of distribution parameters.
    """
    def __init__(self, dimension,
                 randn=np.random.randn,
                 quadratic=False,
                 **kwargs):
        try:
            self.dimension = len(dimension)
            self.standard_deviations = np.asarray(dimension)
        except TypeError:
            self.dimension = dimension
        self.randn = randn
        self.quadratic = quadratic

    @property
    def variances(self):
        if not hasattr(self, 'standard_deviations'):
            return np.ones(self.dimension)
        return self.standard_deviations**2

    def sample(self, number, same_length=False):
        arz = self.randn(number, self.dimension)
        if same_length:
            if same_length is True:
                len_ = self.chiN
            else:
                len_ = same_length # presumably N**0.5, useful if self.opts['CSA_squared']
            for i in rglen(arz):
                ss = sum(arz[i]**2)
                if 1 < 3 or ss > self.dimension + 10.1:
                    arz[i] *= len_ / ss**0.5
        if hasattr(self, 'standard_deviations'):
            arz *= self.standard_deviations
        return arz

    def update(self, vectors, weights):
        """do nothing"""
        pass

    def transform(self, x):
        if hasattr(self, 'standard_deviations'):
            return self.standard_deviations * x
        return x

    def transform_inverse(self, x):
        if hasattr(self, 'standard_deviations'):
            return x / self.standard_deviations
        return x

    def norm(self, x):
        return sum(self.transform_inverse(x)**2)**0.5

    def __imul__(self, factor):
        """variance multiplier"""
        try:
            self.standard_deviations *= factor**0.5
        except AttributeError:
            self.standard_deviations = factor**0.5 * np.ones(self.dimension)
        return self

    @property
    def condition_number(self):
        if hasattr(self, 'standard_deviations'):
            return max(self.standard_deviations) / min(self.standard_deviations)
        return 1.0

    @property
    def covariance_matrix(self):
        if not self.quadratic:
            return None
        try:
            return np.diag(self.standard_deviations**2)
        except AttributeError:
            return np.diag(np.ones(self.dimension))

    @property
    def correlation_matrix(self):
        return np.diag(np.ones(self.dimension)) if self.quadratic else None

    @property
    def chin(self):
        """approximation of the expected length.

        The exact value could be computed by::

            from scipy.special import gamma
            return 2**0.5 * gamma((self.dimension+1) / 2) / gamma(self.dimension / 2)

        The approximation obeys ``chin < chin_hat < (1 + 5e-5) * chin``.
        """
        values = {1: 0.7978845608028656, 2: 1.2533141373156,
                  3: 1.59576912160574,   4: 1.87997120597326}
        try:
            val = values[self.dimension]
        except KeyError:
            # for dim > 4 we have chin < chin_hat < (1 + 5e-5) * chin
            N = self.dimension
            val = N**0.5 * (1 - 1. / (4 * N) + 1. / (26 * N**2)) # was: 21
        return val

class GaussFullSampler(StatisticalModelSamplerWithZeroMeanBaseClass):
    """Multi-variate normal distribution with zero mean.

    Provides methods to `sample` from and `update` a multi-variate
    normal distribution with zero mean and full covariance matrix.

    :param dimension: (required) define the dimensionality (attribute
        ``dimension``) of the normal distribution. If ``dimension`` is a
        vector, it sets the diagonal of the initial covariance matrix.

    :param lazy_update_gap=0: is the number of iterations to wait between
        the O(n^3) updates of the sampler. All values <=1 behave
        identically.

    :param constant_trace='': 'arithmetic'/'aeigen' or 'geometric'
        or 'geigen' (geometric mean of eigenvalues) are available to be
        constant.

    :param randn=np.random.randn: is used to generate N(0,1) numbers.

    :param eigenmethod=np.linalg.eigh: function returning eigenvalues
        and -vectors of symmetric matrix

    >>> import cma, numpy as np
    >>> g = cma.sampler.GaussFullSampler(np.ones(4))
    >>> z = g.sample(1)[0]
    >>> assert g.norm([1,0,0,0]) == 1
    >>> g.update([[1., 0., 0., 0]], [.9])
    >>> g.update_now()
    >>> assert g.norm([1,0,0,0]) == 1
    >>> g.update([[4., 0., 0.,0]], [.5])
    >>> g.update_now()
    >>> g *= 2
    >>> assert cma.utilities.math.Mh.equals_approximately(g.variances[0], 17)
    >>> assert cma.utilities.math.Mh.equals_approximately(g.D[-1]**2, 17)

    TODO
    ----

    o Clean up CMAEvolutionStrategy attributes related to sampling
    (like usage of B, C, D, dC, sigma_vec, these are pretty
    substantial changes). In particular this should become
    compatible with any StatisticalModelSampler. Plan: keep B, C,
    D, dC for the time being as output-info attributes,
    DONE: keep sigma_vec (55 appearances) as a class.

    o combination of sigma_vec and C:
       - update sigma_vec with y (this is wrong: use "z")
       - rescale y according to the inverse update of sigma_vec (as
         if y is expressed in the new sigma_vec while C in the old)
       - update C with the "new" y.
    """
    def __init__(self, dimension,
                 lazy_update_gap=0,
                 constant_trace='',
                 condition_limit=None,
                 randn=np.random.randn,
                 eigenmethod=np.linalg.eigh):
        try:
            self.dimension = len(dimension)
            standard_deviations = np.asarray(dimension)
        except TypeError:
            self.dimension = dimension
            standard_deviations = np.ones(dimension)
        assert len(standard_deviations) == self.dimension

        # prevent equal eigenvals, a hack for np.linalg:
        self.C = np.diag(standard_deviations**2
                    * np.exp((1e-4 / self.dimension) *
                             np.arange(self.dimension)))
        "covariance matrix"
        self.lazy_update_gap = lazy_update_gap
        self.constant_trace = constant_trace
        self.condition_limit = condition_limit if condition_limit else np.inf
        self.randn = randn
        self.eigenmethod = eigenmethod
        self.B = np.eye(self.dimension)
        "columns, B.T[i] == B[:, i], are eigenvectors of C"
        self.D = np.diag(self.C)**0.5  # we assume that C is yet diagonal
        idx = self.D.argsort()
        self.D = self.D[idx]
        self.B = self.B[:, idx]
        "axis lengths, roots of eigenvalues, sorted"
        self._inverse_root_C = None  # see transform_inv...
        self.last_update = 0
        self.count_tell = 0
        self.count_eigen = 0

    def reset(self, standard_deviations=None):
        """reset distribution while keeping all other parameters.

        If `standard_deviations` is not given, `np.ones` is used,
        which might not be the original initial setting.
        """
        if standard_deviations is None:
            standard_deviations = np.ones(self.dimension)
        self.__init__(standard_deviations,
                      lazy_update_gap=self.lazy_update_gap,
                      constant_trace=self.constant_trace,
                      randn=self.randn,
                      eigenmethod=self.eigenmethod)

    @property
    def variances(self):
        return np.diag(self.C)

    def sample(self, number, lazy_update_gap=None, same_length=False):
        self.update_now(lazy_update_gap)
        arz = self.randn(number, self.dimension)
        if same_length:
            if same_length is True:
                len_ = self.chiN
            else:
                len_ = same_length # presumably N**0.5, useful if self.opts['CSA_squared']
            for i in rglen(arz):
                ss = sum(arz[i]**2)
                if 1 < 3 or ss > self.dimension + 10.1:
                    arz[i] *= len_ / ss**0.5
            # or to average
            # arz *= 1 * self.const.chiN / np.mean([sum(z**2)**0.5 for z in arz])
        ary = np.dot(self.B, (self.D * arz).T).T
        # self.ary = ary  # needed whatfor?
        return ary

    def update(self, vectors, weights, c1_times_delta_hsigma=0):
        """update/learn by natural gradient ascent.

        The natural gradient used for the update is::

            np.dot(weights * vectors.T, vectors)

        and equivalently::

            sum([outer(weights[i] * vec, vec)
                 for i, vec in enumerate(vectors)], axis=0)

        Details:

        - The weights include the learning rate and ``-1 <= sum(
          weights[idx]) <= 1`` must be `True` for ``idx = weights > 0``
          and for ``idx = weights < 0``.

        - The content (length) of ``vectors`` with negative weights
          is changed!

        """
        weights = np.array(weights, copy=True)
        vectors = np.asarray(vectors) # row vectors
        assert np.isfinite(vectors[0][0])
        assert len(weights) == len(vectors)

        self.C *= 1 + c1_times_delta_hsigma - sum(weights)

        for k in np.nonzero(weights < 0)[0]:
            # normalize and hence limit ||weight * vector|| to a
            # weight-dependent constant; prevents harm if `vector` is
            # very long while no real harm is done even if `vector` is
            # very short (hence divided by a small number)
            norm = self.norm(vectors[k])
            assert np.isfinite(norm)  # otherwise we later compute 0 * inf
            weights[k] *= len(vectors[k]) / (norm + 1e-9)**2
            assert np.isfinite(weights[k])

        self.C += np.dot(weights * vectors.T, vectors)

        self.count_tell += 1

    def update_now(self, lazy_update_gap=None):
        """update internal variables for sampling the distribution
        with the current covariance matrix C.

        This method is O(dim^3) by calling ``_decompose_C``.

        If ``lazy_update_gap is None`` the lazy_update_gap from init
        is taken. If ``lazy_update_gap < 0`` the (possibly expensive)
        update is done even when the model seems to be up to date.
        """
        if lazy_update_gap is None:
            lazy_update_gap = self.lazy_update_gap
        if (self.count_tell < self.last_update + lazy_update_gap or
            lazy_update_gap == self.count_tell - self.last_update == 0
            ):
            return
        self._updateC()
        self._decompose_C()
        self.last_update = self.count_tell

        if _assertions_quadratic and any(abs(sum(
                self.B[:, 0:self.dimension - 1]
                        * self.B[:, 1:], 0)) > 1e-6):
            print('B is not orthogonal')
            print(self.D)
            print(sum(self.B[:, 0:self.dimension - 1] * self.B[:, 1:], 0))
        # is O(N^3)
        # assert(sum(abs(self.C - np.dot(self.D * self.B,  self.B.T))) < N**2*1e-11)

    def _updateC(self):
        pass

    def _decompose_C(self):
        """eigen-decompose self.C thereby updating self.B and self.D.

        self.C is made symmetric.

        Know bugs: if update is not called before decompose, the
        state variables can get into an inconsistent state.

        """
        self.C = (self.C + self.C.T) / 2
        D_old = self.D
        try:
            self.D, self.B = self.eigenmethod(self.C)
            if any(self.D <= 0):
                raise ValueError(
                    "covariance matrix was not positive definite"
                    " with a minimal eigenvalue of %e." % min(self.D))
        except Exception as e:  # "as" is available since Python 2.6
            # raise RuntimeWarning(  # raise doesn't recover
            print_warning(
                "covariance matrix eigen decomposition failed with \n"
                + str(e) +
                "\nConsider to reformulate the objective function")
            # try again with diag(C) = diag(C) + min(eigenvalues(C_old))
            min_di2 = min(D_old)**2
            for i in range(self.dimension):
                self.C[i][i] += min_di2
            self.D = (D_old**2 + min_di2)**0.5
            self._decompose_C()
        else:
            self.count_eigen += 1
            assert all(np.isfinite(self.D))
            if 1 < 3:  # is only n*log(n) compared to n**3 of eig right above
                idx = np.argsort(self.D)
                self.D = self.D[idx]
                # self.B[i] is a row, column B[:,i] == B.T[i] is eigenvector
                self.B = self.B[:, idx]
                assert (min(self.D), max(self.D)) == (self.D[0], self.D[-1])

            self.limit_condition()
            try:
                if not self.constant_trace:
                    s = 1
                elif self.constant_trace in (1, True) or self.constant_trace.startswith(('ar', 'mean')):
                    s = 1 / np.mean(self.variances)
                elif self.constant_trace.startswith(('geo')):
                    s = np.exp(-np.mean(np.log(self.variances)))
                elif self.constant_trace.startswith('aeig'):
                    s = 1 / np.mean(self.D)  # same as arith
                elif self.constant_trace.startswith('geig'):
                    s = np.exp(-np.mean(np.log(self.D)))
                else:
                    print_warning("trace normalization option setting '%s' not recognized (further warnings will be surpressed)" %
                                  repr(self.constant_trace),
                                  class_name='GaussFullSampler', maxwarns=1, iteration=self.count_eigen + 1)
                    s = 1
            except AttributeError:
                raise ValueError("Value '%s' not allowed for constant trace setting" % repr(self.constant_trace))
            if s != 1:
                self.C *= s
                self.D *= s
            self.D **= 0.5
            assert all(np.isfinite(self.D))
            self._inverse_root_C = None

        # self.dC = np.diag(self.C)

        if 11 < 3:  # not needed for now
            self.inverse_root_C = np.dot(self.B / self.D, self.B.T)
            self.inverse_root_C = (self.inverse_root_C + self.inverse_root_C.T) / 2

    def limit_condition(self, limit=None):
        """bound condition number to `limit` by adding eps to the trace.

        This method only changes the sampling distribution, but not the
        underlying covariance matrix.

        We add ``eps = (a - limit * b) / (limit - 1)`` to the diagonal
        variances, derived from ``limit = (a + eps) / (b + eps)`` with
        ``a, b = lambda_max, lambda_min``.

        >>> import cma
        >>> es = cma.CMAEvolutionStrategy(3 * [1], 1, {'verbose':-9})
        >>> _ = es.optimize(cma.ff.elli)
        >>> assert es.sm.condition_number > 1e4
        >>> es.sm.limit_condition(1e4 - 1)
        >>> assert es.sm.condition_number < 1e4

        """
        if limit is None:
            limit = self.condition_limit
        elif limit <= 1:
            raise ValueError("condition limit was %f<=1 but should be >1"
                             % limit)
        if not np.isfinite(limit) or self.condition_number <= limit:
            return

        eps = (self.D[-1]**2 - limit * self.D[0]**2) / (limit - 1)
        if eps <= 0:  # should never happen, because cond > limit
            raise RuntimeWarning("cond=%e, limit=%e, eps=%e" %
                (self.condition_number, limit, eps))
            return

        for i in range(self.dimension):
            self.C[i][i] += eps
        self.D **= 2
        self.D += eps
        self.D **= 0.5

    def multiply_C(self, factor):
        """multiply ``self.C`` with ``factor`` updating internal states.

        ``factor`` can be a scalar, a vector or a matrix. The vector
        is used as outer product and multiplied element-wise, i.e.,
        ``multiply_C(diag(C)**-0.5)`` generates a correlation matrix.

        Details:
        """
        self._updateC()
        if np.isscalar(factor):
            self.C *= factor
            self.D *= factor**0.5
            try:
                self.inverse_root_C /= factor**0.5
            except AttributeError:
                pass
        elif len(np.asarray(factor).shape) == 1:
            self.C *= np.outer(factor, factor)
            self._decompose_C()
        elif len(factor.shape) == 2:
            self.C *= factor
            self._decompose_C()
        else:
            raise ValueError(str(factor))
        # raise NotImplementedError('never tested')

    def __imul__(self, factor):
        """``sm *= factor`` is a shortcut for ``sm = sm.__imul__(factor)``.

        Multiplies the covariance matrix with `factor`.
        """
        self.multiply_C(factor)
        return self

    def to_linear_transformation(self, reset=False):
        """return associated linear transformation.

        If ``B = sm.to_linear_transformation()`` and z ~ N(0, I), then
        np.dot(B, z) ~ Normal(0, sm.C) and sm.C and B have the same
        eigenvectors. With `reset=True`, ``np.dot(B, sm.sample(1)[0])``
        obeys the same distribution after the call.

        See also: `to_unit_matrix`
        """
        tf = np.dot(self.B * self.D, self.B.T)
        if reset:
            self.reset()
        return tf

    def to_linear_transformation_inverse(self, reset=False):
        """return inverse of associated linear transformation.

        If ``B = sm.to_linear_transformation_inverse()`` and z ~
        Normal(0, sm.C), then np.dot(B, z) ~ Normal(0, I) and sm.C and
        B have the same eigenvectors. With `reset=True`,
        also ``sm.sample(1)[0] ~ Normal(0, I)`` after the call.

        See also: `to_unit_matrix`
        """
        tf = np.dot(self.B / self.D, self.B.T)
        if reset:
            self.reset()
        return tf

    @property
    def covariance_matrix(self):
        return self.C

    @property
    def correlation_matrix(self):
        """return correlation matrix of the distribution.
        """
        c = self.C.copy()
        for i in range(c.shape[0]):
            fac = c[i, i]**0.5
            c[:, i] /= fac
            c[i, :] /= fac
        c = (c + c.T) / 2.0
        return c

    def to_correlation_matrix(self):
        """"re-scale" C to a correlation matrix and return the scaling
         factors as standard deviations.

         See also: `to_linear_transformation`.
        """
        self.update_now(0)
        sigma_vec = np.diag(self.C)**0.5
        self.C = self.correlation_matrix
        self._decompose_C()
        return sigma_vec

    def correlation(self, i, j):
        """return correlation between variables i and j.
        """
        return self.C[i][j] / (self.C[i][i] * self.C[j][j])**0.5

    def transform(self, x):
        """apply linear transformation ``C**0.5`` to `x`."""
        return np.dot(self.B, self.D * np.dot(self.B.T, x))

    def transform_inverse(self, x):
        """apply inverse linear transformation ``C**-0.5`` to `x`."""
        if 22 < 3:
            if self._inverse_root_C is None:
                # is O(N^3)
                self._inverse_root_C = np.dot(self.B / self.D, self.B.T)
                self._inverse_root_C = (self._inverse_root_C + self._inverse_root_C.T) / 2
            return np.dot(self._inverse_root_C, x)
        # works only if x is a vector:
        return np.dot(self.B, np.dot(self.B.T, x) / self.D)
        # should work regardless:
        # return np.dot(np.dot(self.B, (self.B / self.D).T, x))

    @property
    def condition_number(self):
        assert (min(self.D), max(self.D)) == (self.D[0], self.D[-1])
        return (self.D[-1] / self.D[0])**2

    def norm(self, x):
        """compute the Mahalanobis norm that is induced by the
        statistical model / sample distribution, specifically by
        covariance matrix ``C``. The expected Mahalanobis norm is
        about ``sqrt(dimension)``.

        Example
        -------
        >>> import cma, numpy as np
        >>> sm = cma.sampler.GaussFullSampler(np.ones(10))
        >>> x = np.random.randn(10)
        >>> d = sm.norm(x)

        `d` is the norm "in" the true sample distribution,
        sampled points have a typical distance of ``sqrt(2*sm.dim)``,
        where ``sm.dim`` is the dimension, and an expected distance of
        close to ``dim**0.5`` to the sample mean zero. In the example,
        `d` is the Euclidean distance, because C = I.
        """
        return sum((np.dot(self.B.T, x) / self.D)**2)**0.5

    @property
    def chin(self):
        """approximation of the expected length.

        The exact value could be computed by::

            from scipy.special import gamma
            return 2**0.5 * gamma((self.dimension+1) / 2) / gamma(self.dimension / 2)

        The approximation obeys ``chin < chin_hat < (1 + 5e-5) * chin``.
        """
        values = {1: 0.7978845608028656, 2: 1.2533141373156,
                  3: 1.59576912160574,   4: 1.87997120597326}
        try:
            val = values[self.dimension]
        except KeyError:
            # for dim > 4 we have chin < chin_hat < (1 + 5e-5) * chin
            N = self.dimension
            val = N**0.5 * (1 - 1. / (4 * N) + 1. / (26 * N**2)) # was: 21
        return val

    def inverse_hessian_scalar_correction(self, mean, sigma, f):
        # find points to evaluate
        fac = 10  # try to go beyond the true optimum such that
                  # the mean inaccuracy becomes irrelevant
        X = [mean - fac * sigma * self.D[0] * self.B[0], mean,
             mean + fac * sigma * self.D[0] * self.B[0]]
        F = [f(x) for x in X]
        raise NotImplementedError

class GaussDiagonalSampler(StatisticalModelSamplerWithZeroMeanBaseClass):
    """Multi-variate normal distribution with zero mean and diagonal
    covariance matrix.

    Provides methods to `sample` from and `update` a multi-variate
    normal distribution with zero mean and diagonal covariance matrix.

    Arguments to `__init__`
    -----------------------

    `standard_deviations` (required) define the diagonal of the
    initial  covariance matrix, and consequently also the
    dimensionality (attribute `dim`) of the normal distribution. If
    `standard_deviations` is an `int`, ``np.ones(standard_deviations)``
    is used.

    `constant_trace='None'`: 'arithmetic' or 'geometric' or 'aeigen'
    or 'geigen' (geometric mean of eigenvalues) are available to be
    constant.

    `randn=np.random.randn` is used to generate N(0,1) numbers.

    >>> import cma, numpy as np
    >>> s = cma.sampler.GaussDiagonalSampler(np.ones(4))
    >>> z = s.sample(1)[0]
    >>> assert s.norm([1,0,0,0]) == 1
    >>> s.update([[1., 0., 0., 0]], [.9])
    >>> assert s.norm([1,0,0,0]) == 1
    >>> s.update([[4., 0., 0.,0]], [.5])
    >>> g *= 2

    TODO
    ----

    o DONE implement CMA_diagonal with samplers

    o Clean up CMAEvolutionStrategy attributes related to sampling
    (like usage of B, C, D, dC, sigma_vec, these are pretty
    substantial changes). In particular this should become
    compatible with any StatisticalModelSampler. Plan: keep B, C,
    D, dC for the time being as output-info attributes,
    keep sigma_vec (55 appearances) either as constant scaling or
    as a class. Current favorite: make a class (DONE) .

    o combination of sigma_vec and C:
       - update sigma_vec with y (this is wrong: use "z")
       - rescale y according to the inverse update of sigma_vec (as
         if y is expressed in the new sigma_vec while C in the old)
       - update C with the "new" y.
    """
    def __init__(self, dimension,
                 constant_trace='None',
                 randn=np.random.randn,
                 quadratic=False,
                 **kwargs):
        try:
            self.dimension = len(dimension)
            standard_deviations = np.asarray(dimension)
        except TypeError:
            self.dimension = dimension
            standard_deviations = np.ones(dimension)
        assert self.dimension == len(standard_deviations)
        assert len(standard_deviations) == self.dimension

        self.C = standard_deviations**2
        "covariance matrix diagonal"
        self.constant_trace = constant_trace
        self.randn = randn
        self.quadratic = quadratic
        self.count_tell = 0

    def reset(self):
        """reset distribution while keeping all other parameters
        """
        self.__init__(self.dimension,
                      constant_trace=self.constant_trace,
                      randn=self.randn,
                      quadratic=self.quadratic)

    @property
    def variances(self):
        return self.C

    def sample(self, number, same_length=False):
        arz = self.randn(number, self.dimension)
        if same_length:
            if same_length is True:
                len_ = self.chin
            else:
                len_ = same_length # presumably N**0.5, useful if self.opts['CSA_squared']
            for i in rglen(arz):
                ss = sum(arz[i]**2)
                if 1 < 3 or ss > self.dimension + 10.1:
                    arz[i] *= len_ / ss**0.5
            # or to average
            # arz *= 1 * self.const.chiN / np.mean([sum(z**2)**0.5 for z in arz])
        ary = self.C**0.5 * arz
        # self.ary = ary  # needed whatfor?
        return ary

    def update(self, vectors, weights, c1_times_delta_hsigma=0):
        """update/learn by natural gradient ascent.

        The natural gradient used for the update of the coordinate-wise
        variances is::

            np.dot(weights, vectors**2)

        Details: The weights include the learning rate and
        ``-1 <= sum(weights[idx]) <= 1`` must be `True` for
        ``idx = weights > 0`` and for ``idx = weights < 0``.
        The content of `vectors` with negative weights is changed.
        """
        weights = np.array(weights, copy=True)
        vectors = np.asarray(vectors) # row vectors
        assert np.isfinite(vectors[0][0])
        assert len(weights) == len(vectors)

        self.C *= 1 + c1_times_delta_hsigma - sum(weights)

        for k in np.nonzero(weights < 0)[0]:
            # normalize and hence limit ||weight * vector|| to a
            # weight-dependent constant; prevents harm if `vector` is
            # very long while no real harm is done even if `vector` is
            # very short (hence divided by a small number)
            norm = self.norm(vectors[k])
            assert np.isfinite(norm)  # otherwise we later compute 0 * inf
            weights[k] *= len(vectors[k]) / (norm + 1e-9)**2
            assert np.isfinite(weights[k])

        self.C += np.dot(weights, vectors**2)

        self.count_tell += 1

    def multiply_C(self, factor):
        """multiply `self.C` with `factor` updating internal states.

        `factor` can be a scalar, a vector or a matrix. The vector
        is used as outer product, i.e. ``multiply_C(diag(C)**-0.5)``
        generates a correlation matrix."""
        self.C *= factor

    def __imul__(self, factor):
        """``sm *= factor`` is a shortcut for ``sm = sm.__imul__(factor)``.

        Multiplies the covariance matrix with `factor`.
        """
        self.multiply_C(factor)
        return self

    def to_linear_transformation(self, reset=False):
        """return associated linear transformation.

        If ``B = sm.to_linear_transformation()`` and z ~ N(0, I), then
        np.dot(B, z) ~ Normal(0, sm.C) and sm.C and B have the same
        eigenvectors. With `reset=True`, also ``np.dot(B, sm.sample(1)[0])``
        obeys the same distribution after the call.

        See also: `to_unit_matrix`
        """
        tf = self.C**0.5
        if reset:
            self.reset()
        return tf

    def to_linear_transformation_inverse(self, reset=False):
        """return associated inverse linear transformation.

        If ``B = sm.to_linear_transformation_inverse()`` and z ~
        Normal(0, sm.C), then np.dot(B, z) ~ Normal(0, I) and sm.C and
        B have the same eigenvectors. With `reset=True`,
        also ``sm.sample(1)[0] ~ Normal(0, I)`` after the call.

        See also: `to_unit_matrix`
        """
        tf = self.C**-0.5
        if reset:
            self.reset()
        return tf

    @property
    def covariance_matrix(self):
        return np.diag(self.C) if self.quadratic else None

    @property
    def correlation_matrix(self):
        """return correlation matrix of the distribution.
        """
        return np.eye(self.dimension) if self.quadratic else None

    def to_correlation_matrix(self):
        """"re-scale" C to a correlation matrix and return the scaling
         factors as standard deviations.

         See also: `to_linear_transformation`.
        """
        sigma_vec = self.C**0.5
        self.C = np.ones(self.dimension)
        return sigma_vec

    def correlation(self, i, j):
        """return correlation between variables i and j.
        """
        return 0

    def transform(self, x):
        """apply linear transformation ``C**0.5`` to `x`."""
        return self.C**0.5 * x

    def transform_inverse(self, x):
        """apply inverse linear transformation ``C**-0.5`` to `x`."""
        return x / self.C**0.5

    @property
    def condition_number(self):
        return max(self.C) / min(self.C)

    def norm(self, x):
        """compute the Mahalanobis norm that is induced by the
        statistical model / sample distribution, specifically by
        covariance matrix ``C``. The expected Mahalanobis norm is
        about ``sqrt(dimension)``.

        Example
        -------
        >>> import cma, numpy as np
        >>> sm = cma.sampler.GaussFullSampler(np.ones(10))
        >>> x = np.random.randn(10)
        >>> d = sm.norm(x)

        `d` is the norm "in" the true sample distribution,
        sampled points have a typical distance of ``sqrt(2*sm.dim)``,
        where ``sm.dim`` is the dimension, and an expected distance of
        close to ``dim**0.5`` to the sample mean zero. In the example,
        `d` is the Euclidean distance, because C = I.
        """
        return sum(np.asarray(x)**2 / self.C)**0.5

    @property
    def chin(self):
        """approximation of the expected length.

        The exact value could be computed by::

            from scipy.special import gamma
            return 2**0.5 * gamma((self.dimension+1) / 2) / gamma(self.dimension / 2)

        The approximation obeys ``chin < chin_hat < (1 + 5e-5) * chin``.

        """
        values = {1: 0.7978845608028656, 2: 1.2533141373156,
                  3: 1.59576912160574,   4: 1.87997120597326}
        try:
            val = values[self.dimension]
        except KeyError:
            # for dim > 4 we have chin < chin_hat < (1 + 5e-5) * chin
            N = self.dimension
            val = N**0.5 * (1 - 1. / (4 * N) + 1. / (26 * N**2)) # was: 21
        return val

