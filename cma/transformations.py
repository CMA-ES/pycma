"""Search space transformation and encoding/decoding classes"""
from __future__ import absolute_import, division, print_function  #, unicode_literals

import numpy as np
from numpy import array, isfinite, log

from .utilities.utils import rglen, print_warning
from .utilities.python3for2 import range
del absolute_import, division, print_function  #, unicode_literals

class ConstRandnShift(object):
    """``ConstRandnShift()(x)`` adds a fixed realization of
    ``stddev * randn(len(x))`` to the vector ``x``.

    By default, the realized shift is the same for each instance of
    `ConstRandnShift`, see ``seed`` argument. This class is used in
    class `Shifted` as default transformation.

    :See also: class `Shifted`
    """
    def __init__(self, stddev=3, seed=1):
        """with ``seed=None`` each instance realizes a different shift"""
        self.seed = seed
        self.stddev = stddev
        self._xopt = {}
    def __call__(self, x):
        """return "shifted" ``x - shift``

        """
        try:
            x_opt = self._xopt[len(x)]
        except KeyError:
            if self.seed is None:
                shift = np.random.randn(len(x))
            else:
                rstate = np.random.get_state()
                np.random.seed(self.seed)
                shift = np.random.randn(len(x))
                np.random.set_state(rstate)
            x_opt = self._xopt.setdefault(len(x), self.stddev * shift)
        return array(x, copy=False) - x_opt
    def get(self, dimension):
        """return shift applied to ``zeros(dimension)``

            >>> import numpy as np, cma
            >>> s = cma.transformations.ConstRandnShift()
            >>> assert all(s(-s.get(3)) == np.zeros(3))
            >>> assert all(s.get(3) == s(np.zeros(3)))

        """
        return self.__call__(np.zeros(dimension))

class Rotation(object):
    """implement an orthogonal linear transformation for each dimension.

    By default each `Rotation` instance provides a different "random"
    but fixed rotation. This class is used to implement non-separable
    test functions, most conveniently via `Rotated`.

    Example:

    >>> import cma, numpy as np
    >>> R = cma.transformations.Rotation()
    >>> R2 = cma.transformations.Rotation() # another rotation
    >>> x = np.array((1,2,3))
    >>> list(np.round(R(R(x), inverse=1), 9))
    [1.0, 2.0, 3.0]

    :See also: `Rotated`

    """
    def __init__(self, seed=None):
        """same ``seed`` means same rotation, by default a random but
        fixed once and for all rotation, different for each instance
        """
        self.seed = seed
        self.dicMatrices = {}
    def __call__(self, x, inverse=False):
        """Rotates the input array `x` with a fixed rotation matrix
           (``self.dicMatrices[len(x)]``)
        """
        x = np.array(x, copy=False)
        N = x.shape[0]  # can be an array or matrix, TODO: accept also a list of arrays?
        if N not in self.dicMatrices:  # create new N-basis once and for all
            rstate = np.random.get_state()
            np.random.seed(self.seed) if self.seed else np.random.seed()
            self.state = np.random.get_state()  # only keep last state
            B = np.random.randn(N, N)
            np.random.set_state(rstate)  # keep untouched/good sequence from outside view
            for i in range(N):
                for j in range(0, i):
                    B[i] -= np.dot(B[i], B[j]) * B[j]
                B[i] /= sum(B[i]**2)**0.5
            self.dicMatrices[N] = B
        if inverse:
            return np.dot(self.dicMatrices[N].T, x)  # compute rotation
        else:
            return np.dot(self.dicMatrices[N], x)  # compute rotation

class BoxConstraintsTransformationBase(object):
    """Implements a transformation into boundaries and is used in
    top level boundary handling classes.

    Example::

        tf = BoxConstraintsTransformationAnyDerivedClass([[1, 4]])
        x = [3, 2, 4.4]
        y = tf(x)  # "repaired" solution
        print(tf([2.5]))  # middle value is never changed
        [2.5]
        assert all([yi <= 4 for yi in y])

    :See also: `BoundTransform`

    """
    def __init__(self, bounds):
        try:
            if len(bounds[0]) != 2:
                raise ValueError
        except:
            raise ValueError(' bounds must be either [[lb0, ub0]] or [[lb0, ub0], [lb1, ub1],...], \n where in both cases the last entry is reused for all remaining dimensions')
        self.bounds = bounds
        self.initialize()

    def initialize(self):
        """initialize in base class"""
        self._lb = [b[0] for b in self.bounds]  # can be done more efficiently?
        self._ub = [b[1] for b in self.bounds]

    def _lowerupperval(self, a, b, c):
        return np.max([np.max(a), np.min([np.min(b), c])])
    def bounds_i(self, i):
        """return ``[ith_lower_bound, ith_upper_bound]``"""
        return self.bounds[self._index(i)]
    def __call__(self, solution_in_genotype):
        res = [self._transform_i(x, i) for i, x in enumerate(solution_in_genotype)]
        return res
    transform = __call__
    def inverse(self, solution_in_phenotype, *args, **kwars):
        return [self._inverse_i(y, i) for i, y in enumerate(solution_in_phenotype)]
    def _index(self, i):
        return min((i, len(self.bounds) - 1))
    def _transform_i(self, x, i):
        raise NotImplementedError('this is an abstract method that should be implemented in the derived class')
    def _inverse_i(self, y, i):
        raise NotImplementedError('this is an abstract method that should be implemented in the derived class')
    def shift_or_mirror_into_invertible_domain(self, solution_genotype):
        """return the reference solution that has the same ``box_constraints_transformation(solution)``
        value, i.e. ``tf.shift_or_mirror_into_invertible_domain(x) = tf.inverse(tf.transform(x))``.
        This is an idempotent mapping (leading to the same result independent how often it is
        repeatedly applied).

        """
        return self.inverse(self(solution_genotype))
        raise NotImplementedError('this is an abstract method that should be implemented in the derived class')

class _BoxConstraintsTransformationTemplate(BoxConstraintsTransformationBase):
    """copy/paste this template to implement a new boundary handling
    transformation"""
    def __init__(self, bounds):
        # BoxConstraintsTransformationBase.__init__(self, bounds)
        super(_BoxConstraintsTransformationTemplate, self).__init__(bounds)
    def initialize(self):
        BoxConstraintsTransformationBase.initialize(self)  # likely to be removed
    def _transform_i(self, x, i):
        raise NotImplementedError('this is an abstract method that should be implemented in the derived class')
    def _inverse_i(self, y, i):
        raise NotImplementedError('this is an abstract method that should be implemented in the derived class')
    __doc__ = BoxConstraintsTransformationBase.__doc__ + __doc__

class BoxConstraintsLinQuadTransformation(BoxConstraintsTransformationBase):
    """implement a bijective, monotonous transformation between
    ``[lb - al, ub + au]`` and ``[lb, ub]``.

    Generally speaking, this transformation aims to resemble ``sin`` to
    be a continuous differentiable (ie. C^1) transformation over R into
    a bounded interval; then, it also aims to improve over ``sin`` in
    the following two ways: (i) resemble the identity over an interval
    as large possible while keeping the second derivative in reasonable
    limits, and (ii) numerical stability in "pathological" corner cases
    of the boundary limit values.

    The transformation is the identity (and therefore linear) in ``[lb
    + al, ub - au]`` (typically about 90% of the interval) and
    quadratic in ``[lb - 3*al, lb + al]`` and in ``[ub - au,
    ub + 3*au]``. The transformation is periodically expanded beyond
    the limits (somewhat resembling the shape ``sin(x-pi/2))`` with a
    period of ``2 * (ub - lb + al + au)``.

    Details
    =======

    Partly due to numerical considerations depend the values ``al`` and
    ``au`` on ``abs(lb)`` and ``abs(ub)`` which makes the
    transformation non-translation invariant. In particular, the linear
    proportion decreases to zero when ``ub-lb`` becomes small. In
    contrast to ``sin(.)``, the transformation is also robust to
    "arbitrary" large values for boundaries, e.g. a lower bound of
    ``-1e99`` or upper bound of ``np.Inf`` or bound ``None``.

    Examples
    ========
    Example to use with cma:

    >>> import warnings
    >>> import cma
    >>> from cma.transformations import BoxConstraintsLinQuadTransformation
    >>> # only the first variable has an upper bound
    >>> tf = BoxConstraintsLinQuadTransformation([[1,2], [1,None]]) # second==last pair is re-cycled
    >>> with warnings.catch_warnings(record=True) as warns:
    ...     x, es = cma.fmin2(cma.ff.elli, 9 * [2], 1,
    ...                 {'transformation': [tf.transform, tf.inverse],
    ...                  'verb_disp':0, 'verbose': -2})
    >>> str(warns[0].message).startswith(('in class GenoPheno: user defi',
    ...                                   'flat fitness'))
    True

    or:

    >>> es = cma.CMAEvolutionStrategy(9 * [2], 1)  # doctest: +ELLIPSIS
    (5_w,10)-aCMA-ES (mu_w=...
    >>> with warnings.catch_warnings(record=True) as warns:
    ...     while not es.stop():
    ...         X = es.ask()
    ...         f = [cma.ff.elli(tf(x)) for x in X]  # tf(x)==tf.transform(x)
    ...         es.tell(X, f)
    >>> warns[0].message  # doctest: +ELLIPSIS
    UserWarning('flat fitness (f=...

    Example of the internal workings:

    >>> from cma.transformations import BoxConstraintsLinQuadTransformation
    >>> tf = BoxConstraintsLinQuadTransformation([[1,2], [1,11], [1,11]])
    >>> tf.bounds
    [[1, 2], [1, 11], [1, 11]]
    >>> tf([1.5, 1.5, 1.5])
    [1.5, 1.5, 1.5]
    >>> list(np.round(tf([1.52, -2.2, -0.2, 2, 4, 10.4]), 9))
    [1.52, 4.0, 2.0, 2.0, 4.0, 10.4]
    >>> res = np.round(tf._au, 2)
    >>> assert list(res[:4]) == [ 0.15, 0.6, 0.6, 0.6]
    >>> res = [round(x, 2) for x in tf.shift_or_mirror_into_invertible_domain([1.52, -12.2, -0.2, 2, 4, 10.4])]
    >>> assert res == [1.52, 9.2, 2.0, 2.0, 4.0, 10.4]
    >>> tmp = tf([1])  # call with lower dimension

    """
    def __init__(self, bounds):
        """``x`` is defined in ``[lb - 3*al, ub + au + r - 2*al]`` with
        ``r = ub - lb + al + au``, and ``x == transformation(x)`` in
        ``[lb + al, ub - au]``.

        ``beta*x - alphal = beta*x - alphau`` is then defined in
        ``[lb, ub]``.

        ``alphal`` and ``alphau`` represent the same value,
        but respectively numerically better suited for values close to
        lb and ub.

        todo: revise this to be more comprehensible.
        """
        # BoxConstraintsTransformationBase.__init__(self, bounds)
        super(BoxConstraintsLinQuadTransformation, self).__init__(bounds)
        # super().__init__(bounds) # only available since Python 3.x
        # super(BB, self).__init__(bounds) # is supposed to call initialize

    def initialize(self, length=None):
        """see ``__init__``"""
        if length is None:
            length = len(self.bounds)
        max_i = min((len(self.bounds) - 1, length - 1))
        self._lb = array([self.bounds[min((i, max_i))][0]
                          if self.bounds[min((i, max_i))][0] is not None
                          else -np.Inf
                          for i in range(length)], copy=False)
        self._ub = array([self.bounds[min((i, max_i))][1]
                          if self.bounds[min((i, max_i))][1] is not None
                          else np.Inf
                          for i in range(length)], copy=False)
        lb = self._lb
        ub = self._ub
        # define added values for lower and upper bound
        self._al = array([min([(ub[i] - lb[i]) / 2, (1 + np.abs(lb[i])) / 20])
                             if isfinite(lb[i]) else 1 for i in rglen(lb)], copy=False)
        self._au = array([min([(ub[i] - lb[i]) / 2, (1 + np.abs(ub[i])) / 20])
                             if isfinite(ub[i]) else 1 for i in rglen(ub)], copy=False)

    def __call__(self, solution_genotype, copy=True):
        # about four times faster version of array([self._transform_i(x, i) for i, x in enumerate(solution_genotype)])
        # still, this makes a typical run on a test function two times slower, but there might be one too many copies
        # during the transformations in gp
        if len(self._lb) != len(solution_genotype):
            self.initialize(len(solution_genotype))
        lb = self._lb
        ub = self._ub
        al = self._al
        au = self._au

        if not isinstance(solution_genotype[0], float):
            # transformed value is likely to be a float
            y = np.array(solution_genotype, copy=True, dtype=float)
            # if solution_genotype is not a float, copy value is disregarded
            copy = False
        else:
            y = solution_genotype
        idx = (y < lb - 2 * al - (ub - lb) / 2.0) | (y > ub + 2 * au + (ub - lb) / 2.0)
        if idx.any():
            r = 2 * (ub[idx] - lb[idx] + al[idx] + au[idx])  # period
            s = lb[idx] - 2 * al[idx] - (ub[idx] - lb[idx]) / 2.0  # start
            if copy:
                y = np.array(y, copy=True)
                copy = False
            y[idx] -= r * ((y[idx] - s) // r)  # shift
        idx = y > ub + au
        if idx.any():
            if copy:
                y = np.array(y, copy=True)
                copy = False
            y[idx] -= 2 * (y[idx] - ub[idx] - au[idx])
        idx = y < lb - al
        if idx.any():
            if copy:
                y = np.array(y, copy=True)
                copy = False
            y[idx] += 2 * (lb[idx] - al[idx] - y[idx])
        idx = y < lb + al
        if idx.any():
            if copy:
                y = np.array(y, copy=True)
                copy = False
            y[idx] = lb[idx] + (y[idx] - (lb[idx] - al[idx]))**2 / 4 / al[idx]
        idx = y > ub - au
        if idx.any():
            if copy:
                y = np.array(y, copy=True)
                copy = False
            y[idx] = ub[idx] - (y[idx] - (ub[idx] + au[idx]))**2 / 4 / au[idx]
        # assert Mh.vequals_approximately(y, BoxConstraintsTransformationBase.__call__(self, solution_genotype))
        return y
    __call__.doc = BoxConstraintsTransformationBase.__doc__
    transform = __call__
    def idx_infeasible(self, solution_genotype):
        """return indices of "infeasible" variables, that is,
        variables that do not directly map into the feasible domain such that
        ``tf.inverse(tf(x)) == x``.

        """
        res = [i for i, x in enumerate(solution_genotype)
                                if not self.is_feasible_i(x, i)]
        return res
    def is_feasible_i(self, x, i):
        """return True if value ``x`` is in the invertible domain of
        variable ``i``

        """
        lb = self._lb[self._index(i)]
        ub = self._ub[self._index(i)]
        al = self._al[self._index(i)]
        au = self._au[self._index(i)]
        return lb - al < x < ub + au
    def is_loosely_feasible_i(self, x, i):
        """never used"""
        lb = self._lb[self._index(i)]
        ub = self._ub[self._index(i)]
        al = self._al[self._index(i)]
        au = self._au[self._index(i)]
        return lb - 2 * al - (ub - lb) / 2.0 <= x <= ub + 2 * au + (ub - lb) / 2.0

    def shift_or_mirror_into_invertible_domain(self, solution_genotype,
                                               copy=False):
        """parameter ``solution_genotype`` is changed.

        The domain is
        ``[lb - al, ub + au]`` and in ``[lb - 2*al - (ub - lb) / 2, lb - al]``
        mirroring is applied.

        """
        assert solution_genotype is not None
        if copy:
            y = [val for val in solution_genotype]
        else:
            y = solution_genotype
        if isinstance(y, np.ndarray) and not isinstance(y[0], float):
            y = array(y, dtype=float)
        for i in rglen(y):
            lb = self._lb[self._index(i)]
            ub = self._ub[self._index(i)]
            al = self._al[self._index(i)]
            au = self._au[self._index(i)]
            # x is far from the boundary, compared to ub - lb
            if y[i] < lb - 2 * al - (ub - lb) / 2.0 or y[i] > ub + 2 * au + (ub - lb) / 2.0:
                r = 2 * (ub - lb + al + au)  # period
                s = lb - 2 * al - (ub - lb) / 2.0  # start
                y[i] -= r * ((y[i] - s) // r)  # shift
            if y[i] > ub + au:
                y[i] -= 2 * (y[i] - ub - au)
            if y[i] < lb - al:
                y[i] += 2 * (lb - al - y[i])
        return y
    shift_or_mirror_into_invertible_domain.__doc__ = BoxConstraintsTransformationBase.shift_or_mirror_into_invertible_domain.__doc__ + shift_or_mirror_into_invertible_domain.__doc__

    def _shift_or_mirror_into_invertible_i(self, x, i):
        """shift into the invertible domain [lb - ab, ub + au], mirror close to
        boundaries in order to get a smooth transformation everywhere

        """
        assert x is not None
        lb = self._lb[self._index(i)]
        ub = self._ub[self._index(i)]
        al = self._al[self._index(i)]
        au = self._au[self._index(i)]
        # x is far from the boundary, compared to ub - lb
        if x < lb - 2 * al - (ub - lb) / 2.0 or x > ub + 2 * au + (ub - lb) / 2.0:
            r = 2 * (ub - lb + al + au)  # period
            s = lb - 2 * al - (ub - lb) / 2.0  # start
            x -= r * ((x - s) // r)  # shift
        if x > ub + au:
            x -= 2 * (x - ub - au)
        if x < lb - al:
            x += 2 * (lb - al - x)
        return x
    def _transform_i(self, x, i):
        """return transform of x in component i"""
        x = self._shift_or_mirror_into_invertible_i(x, i)
        lb = self._lb[self._index(i)]
        ub = self._ub[self._index(i)]
        al = self._al[self._index(i)]
        au = self._au[self._index(i)]
        if x < lb + al:
            return lb + (x - (lb - al))**2 / 4 / al
        elif x < ub - au:
            return x
        elif x < ub + 3 * au:
            return ub - (x - (ub + au))**2 / 4 / au
        else:
            assert False  # shift removes this case
            return ub + au - (x - (ub + au))
    def _inverse_i(self, y, i):
        """return inverse of y in component i"""
        lb = self._lb[self._index(i)]
        ub = self._ub[self._index(i)]
        al = self._al[self._index(i)]
        au = self._au[self._index(i)]
        if 1 < 3:
            if not lb <= y <= ub:
                raise ValueError('argument of inverse must be within the given bounds')
        else:
            y -= 2 * (ub - lb) * int((y - lb) / (2 * (ub - lb)))  # comes close
            while y > ub:
                y -= 2 * (ub - lb)
            while y < lb:
                y += 2 * (ub - lb)
            if y > ub:
                y = ub - (y - ub)  # mirror
        if y < lb + al:
            return (lb - al) + 2 * (al * (y - lb))**0.5
        elif y < ub - au:
            return y
        else:
            return (ub + au) - 2 * (au * (ub - y))**0.5

class AdaptiveDecoding(object):
    """base class for adaptive decoding.

    The adaptive decoding class is "dual" to the StasticalModel class,
    in that for linear transformations adapting either one or the other
    is equivalent.

    TODO: this is a stump

    """
    def __init__(self, scaling):
        """``len(scaling)`` determines the dimension.

        The initial transformation is (typically) `np.diag(scaling)`.

        """
        raise NotImplementedError

    def transform(self, x):
        """apply the transformation / decoding AKA geno-pheno tf"""
        raise NotImplementedError

    def transform_inverse(self, x):
        """inverse transformation (encoding), might return None"""
        pass

    def __mul__(self, x):
        """A linear transformation expressed by multiplication"""
        return self.transform(x)

    # def __rmul__(self, x):
    #     """``x * self`` should at least work if `x` is a scalar"""
    #     raise NotImplementedError

    # def __rtruediv__(self, x):
    #     """``x / ad`` applies the inverse of `ad` to `x`. """
    #     return self.transform_inverse(x)

    def update(self, vectors, weights):
        """AKA update.

        :param vectors: is a list of samples.
        :param weights: define a learning rate for each vector.

        ``vectors`` are "isotropic", e.g.::

            sm = StatisticalModel...()
            ad = AdaptiveDecoding...()
            z = sm.sample(1)[0]
            y = ad * z  # decoding applied
            x = m + y  # candidate solution
            ad.tell([sm.transform_inverse(z)], [0.1])
            sm.update([y / ad], [0.01]) # remark that y / ad != z

        where the symmetric transformation ``sm.transform_inverse(z)``
        makes ``z`` isotropic.

        TODO: what exactly does this mean, is this a generic
        construction, is this even the right construction?
        """
        raise NotImplementedError

    def norm(self, x):
        """return norm of ``x`` prior to the transformation"""
        return sum(self.transform_inverse(x)**2)**0.5

    def update_now(self, lazy_update_gap=None):
        """update model here, if lazy update is implemented"""
        pass  # implementation can be omitted

    @property
    def correlation_matrix(self):
        """return correlation matrix or None"""
        pass  # implementation can be omitted

    @property
    def condition_number(self):
        """return condition number of the squared transformation matrix"""
        raise NotImplementedError
        return 1  # simple but rather meaningless implementation

class DiagonalDecoding(AdaptiveDecoding):
    """Diagonal linear transformation.

    Supports ``self * a``, ``a * self``, ``a / self``, ``self *= a``,
    as if ``self`` is an np.array. Problem: ``np.array`` does
    broadcasting.

    >>> import cma
    >>> from cma.transformations import DiagonalDecoding as DD

    References: N. Hansen (2008). Adaptive Encoding: How to render search
    coordinate system invariant. In PPSN Parallel Problem Solving from
    Nature X, pp. 205-214.

    """
    def __init__(self, scaling):
        if isinstance(scaling, int):
            scaling = scaling * [1.0]
        self.scaling = np.array(scaling, dtype=float)
        self.dim = np.size(self.scaling)
        self.is_identity = False
        if np.all(self.scaling == 1):
            self.is_identity = True

    def transform(self, x):
        return self.scaling * x
        return x if self.is_identity else self.scaling * x

    def transform_inverse(self, x):
        return x / self.scaling
        return x if self.is_identity else x / self.scaling

    def __array__(self):
        """``sigma * self`` tries to call ``self.__array__()`` if
        ``isinstance(sigma, np.float64)``.
        """
        return self.scaling

    def equals(self, x):
        """return `True` if the diagonal equals to ``x``"""
        return all(self.scaling == x)

    def __imul__(self, factor):
        """define ``self *= factor``.

        As a shortcut for::

            self = self.__imul__(factor)

        """
        try:
            if factor == 1:
                return self
        except: pass
        try:
            if (np.size(factor) == np.size(self.scaling) and
                    all(factor == 1)):
                return self
        except: pass
        if self.is_identity and np.size(self.scaling) == 1:
            self.scaling = np.ones(np.size(factor))
        self.is_identity = False
        self.scaling *= factor
        self.dim = np.size(self.scaling)
        return self

    def __mul__(self, x):
        """multiplication with array or scalar"""
        # return self.scaling * x
        return x if self.is_identity else self.scaling * x

    def __rmul__(self, x):
        """``x * self`` works (only) if `x` is a scalar"""
        # return x * self.scaling
        return x if self.is_identity else self.scaling * x

    def __rdiv__(self, x):  # caveat: div vs truediv
        raise NotImplementedError('use ``this**-1 * x`` or ' +
            '``this.transform_inverse(x)`` instead of ``x / this``')
        return x / self.scaling

    def __rtruediv__(self, x):  # caveat: div vs truediv
        return x if self.is_identity else self.scaling**-1 * x

    def __pow__(self, power):
        # return self.scaling**power
        return 1 if self.is_identity else self.scaling**power

    # def __iter__(self):
    #     """allows multiplication with numpy array"""
    #     # TODO: this doesn't work if scaling is a scalar
    #     return self.scaling.__iter__()

    def update(self, vectors, weights):
        if self.is_identity and np.size(self.scaling) == 1:
            self.scaling = np.ones(len(vectors[0]))
        self.is_identity = False
        weights = np.asarray(weights)
        if sum(abs(weights)) > 3:
            raise ValueError("sum of weights %f + %f is too large"
                             % (sum(weights[weights>0]),
                                -sum(weights[weights<0])))
        z2 = np.asarray(vectors)**2  # popsize x dim array
        if 11 < 3 and np.max(z2) > 50:
            print(np.max(z2))
        # z2 = 1.96 * np.tanh(np.asarray(vectors) / 1.4)**2  # popsize x dim array
        z2_average = np.dot(weights, z2)  # dim-dimensional vector
        # 1 + w (z2 - 1) ~ exp(w (z2 - 1)) = exp(w z2 - w)
        facs = np.exp(np.log(2) * (z2_average - sum(weights)))
        # z2=0, w=-1, d=log(2) => exp(d w (0 - 1)) = 2 = 1 + w (0 - 1)
        # z2=2, w=1, d=log(2) => exp(d w (2 - 1)) = 2 = 1 + w (2 - 1)
        # because 1 + eta (z^2 - 1) < max(z^2, 1) if eta < 1
        # we want for exp(eta (z^2 - 1)) ~ 1 + eta (z^2 - 1):
        #   exp(eta (z^2 - 1)) < z^2  <=>  eta < log z^2 / (z^2 - 1)
        # where eta := sum w^+, z^2 := sum w^+ zi^2 / eta
        # remark: for z^2 \to+ 1, eta_max |to- log z^2 / (z^2 - 1) = 1

        #if np.any(facs > 10):
            #print(np.sum(z2, axis=1))
            #print(weights)
            #rint(facs)
        # idxx = np.argmax(z2.flatten())
        # idxxx = (idxx // z2.shape[1], idxx - (idxx // z2.shape[1]) * z2.shape[1])
        # print(idxxx, z2[idxxx])

        
        if 1 < 3:  # bound increment to observed value
            idx = weights > 0  # for negative weights w (z^2 - 1) <= w
            # Remark: z2 - 1 can never be < -1, i.e. eta_max >= log(2) ~ 0.7
            eta = sum(abs(weights[idx]))
            z2_pos_average = np.dot(weights[idx], z2[idx]) / eta
            z2_large_pos = z2_pos_average[z2_pos_average > 1]
            if np.size(z2_large_pos):
                if 1 < 3:
                    eta_max = max(np.log(z2_large_pos) /  # TODO: review/approve this
                                    (z2_large_pos - 1))
                    if eta > eta_max:
                        facs **= (eta_max / eta)
                elif 1 < 3:
                    raise NotImplementedError("this was never tested")
                    correction = max(log(z2) / log(facs))
                    if correction < 1:
                        facs **= correction  # could rather be applied only on positive update?
                else:
                    # facs = (scaling_tt / scaling_t)**2
                    # assure facs <= z**2
                    idx = facs > z2_pos_average
                    if any(idx):
                        facs[idx] = z2_pos_average[idx]
        self.scaling *= facs
        # print(facs)

    @property
    def condition_number(self):
        return (max(self.scaling) / min(self.scaling))**2

    @property
    def correlation_matrix(self):
        return np.eye(self.dim)

    def tolist(self):
        return self.scaling.tolist()

class GenoPheno(object):
    """Genotype-phenotype transformation.

    Method `pheno` provides the transformation from geno- to phenotype,
    that is from the internal representation to the representation used
    in the objective function. Method `geno` provides the "inverse" pheno-
    to genotype transformation. The geno-phenotype transformation comprises,
    in this order:

       - insert fixed variables (with the phenotypic values)
       - affine linear transformation (first scaling then shift)
       - user-defined transformation
       - repair (e.g. into feasible domain due to boundaries)
       - re-assign fixed variables their original phenotypic value

    By default all transformations are the identity. The repair is only
    applied, if the transformation is given as argument to the method
    `pheno`.

    `geno` is only necessary, if solutions have been injected.

    """
    def __init__(self, dim, scaling=None, typical_x=None,
                 fixed_values=None, tf=None):
        """return `GenoPheno` instance with phenotypic dimension `dim`.

        Keyword Arguments
        -----------------
            `scaling`
                the diagonal of a scaling transformation matrix, multipliers
                in the genotyp-phenotyp transformation, see `typical_x`
            `typical_x`
                ``pheno = scaling*geno + typical_x``
            `fixed_values`
                a dictionary of variable indices and values, like ``{0:2.0, 2:1.1}``,
                that are not subject to change, negative indices are ignored
                (they act like incommenting the index), values are phenotypic
                values.
            `tf`
                list of two user-defined transformation functions, or `None`.

                ``tf[0]`` is a function that transforms the internal representation
                as used by the optimizer into a solution as used by the
                objective function. ``tf[1]`` does the back-transformation.
                For example::

                    tf_0 = lambda x: [xi**2 for xi in x]
                    tf_1 = lambda x: [abs(xi)**0.5 fox xi in x]

                or "equivalently" without the `lambda` construct::

                    def tf_0(x):
                        return [xi**2 for xi in x]
                    def tf_1(x):
                        return [abs(xi)**0.5 fox xi in x]

                ``tf=[tf_0, tf_1]`` is a reasonable way to guaranty that only positive
                values are used in the objective function.

        Details
        -------
        If ``tf_0`` is not the identity and ``tf_1`` is ommitted,
        the genotype of ``x0`` cannot be computed consistently and
        "injection" of phenotypic solutions is likely to lead to
        unexpected results.

        """
        self.N = dim
        self.fixed_values = fixed_values
        if tf is not None:
            self.tf_pheno = tf[0]
            self.tf_geno = tf[1]  # TODO: should not necessarily be needed
            # r = np.random.randn(dim)
            # assert all(tf[0](tf[1](r)) - r < 1e-7)
            # r = np.random.randn(dim)
            # assert all(tf[0](tf[1](r)) - r > -1e-7)
            print_warning("in class GenoPheno: user defined transformations have not been tested thoroughly")
        else:
            self.tf_geno = None
            self.tf_pheno = None

        if fixed_values:
            if not isinstance(fixed_values, dict):
                raise ValueError("fixed_values must be a dictionary {index:value,...}, found: %s, %s"
                                 % (str(type(fixed_values)), fixed_values))
            if max(fixed_values.keys()) >= dim:
                raise ValueError("max(fixed_values.keys()) = " + str(max(fixed_values.keys())) +
                    " >= dim=N=" + str(dim) + " is not a feasible index")
            # convenience commenting functionality: drop negative keys
            for k in list(fixed_values.keys()):
                if k < 0:
                    fixed_values.pop(k)

        def vec_is_default(vec, default_val=0):
            """return True if `vec` has the value `default_val`,
            None or [None] are also recognized as default

            """
            # TODO: rather let default_val be a list of default values,
            # cave comparison of arrays
            try:
                if len(vec) == 1:
                    vec = vec[0]  # [None] becomes None and is always default
            except TypeError:
                pass  # vec is a scalar

            if vec is None or all(vec == default_val):
                return True

            if all([val is None or val == default_val for val in vec]):
                    return True

            return False

        self.scales = array(scaling) if scaling is not None else None
        if vec_is_default(self.scales, 1):
            self.scales = 1  # CAVE: 1 is not array(1)
        elif self.scales.shape is not () and len(self.scales) != self.N:
            raise ValueError('len(scales) == ' + str(len(self.scales)) +
                         ' does not match dimension N == ' + str(self.N))

        self.typical_x = array(typical_x) if typical_x is not None else None
        if vec_is_default(self.typical_x, 0):
            self.typical_x = 0
        elif self.typical_x.shape is not () and len(self.typical_x) != self.N:
            raise ValueError('len(typical_x) == ' + str(len(self.typical_x)) +
                         ' does not match dimension N == ' + str(self.N))

        if (self.scales is 1 and
                self.typical_x is 0 and
                self.fixed_values is None and
                self.tf_pheno is None):
            self.isidentity = True
        else:
            self.isidentity = False
        if self.tf_pheno is None:
            self.islinear = True
        else:
            self.islinear = False

    def pheno(self, x, into_bounds=None, copy=True,
              archive=None, iteration=None):
        """maps the genotypic input argument into the phenotypic space,
        see help for class `GenoPheno`

        Details
        -------
        If ``copy``, values from ``x`` are copied if changed under the
        transformation.

        """
        input_type = type(x)
        if into_bounds is None:
            into_bounds = (lambda x, copy=False:
                                x if not copy else array(x, copy=copy))
        if self.isidentity:
            y = into_bounds(x) # was into_bounds(x, False) before (bug before v0.96.22)
        else:
            if self.fixed_values is None:
                y = array(x, copy=copy)  # make a copy, in case
            else:  # expand with fixed values
                y = list(x)  # is a copy
                for i in sorted(self.fixed_values.keys()):
                    y.insert(i, self.fixed_values[i])
                y = array(y, copy=False)
            copy = False

            if self.scales is not 1:  # just for efficiency
                y *= self.scales

            if self.typical_x is not 0:
                y += self.typical_x

            if self.tf_pheno is not None:
                y = array(self.tf_pheno(y), copy=False)

            y = into_bounds(y, copy)  # copy is False

            if self.fixed_values is not None:
                for i, k in list(self.fixed_values.items()):
                    y[i] = k

        if input_type is np.ndarray:
            y = array(y, copy=False)
        if archive is not None:
            archive.insert(y, geno=x, iteration=iteration)
        return y

    def geno(self, y, from_bounds=None,
             copy=True,
             repair=None, archive=None):
        """maps the phenotypic input argument into the genotypic space,
        that is, computes essentially the inverse of ``pheno``.

        By default a copy is made only to prevent to modify ``y``.

        The inverse of the user-defined transformation (if any)
        is only needed if external solutions are injected, it is not
        applied to the initial solution x0.

        Details
        =======
        ``geno`` searches first in ``archive`` for the genotype of
        ``y`` and returns the found value, typically unrepaired.
        Otherwise, first ``from_bounds`` is applied, to revert a
        projection into the bound domain (if necessary) and ``pheno``
        is reverted. ``repair`` is applied last, and is usually the
        method ``CMAEvolutionStrategy.repair_genotype`` that limits the
        Mahalanobis norm of ``geno(y) - mean``.

        """
        if from_bounds is None:
            from_bounds = lambda x, copy=False: x  # not change, no copy

        if archive is not None:
            try:
                x = archive[y]['geno']
            except (KeyError, TypeError):
                x = None
            if x is not None:
                if archive[y]['iteration'] < archive.last_iteration \
                        and repair is not None:
                    x = repair(x, copy_if_changed=copy)
                return x

        input_type = type(y)
        x = y

        x = from_bounds(x, copy)

        if self.isidentity:
            if repair is not None:
                x = repair(x, copy)
            return x

        if copy:  # could be improved?
            x = array(x, copy=True)
            copy = False

        # user-defined transformation
        if self.tf_geno is not None:
            x = np.asarray(self.tf_geno(x))
        elif self.tf_pheno is not None:
            raise ValueError('t1 of options transformation was not defined but is needed as being the inverse of t0')

        # affine-linear transformation: shift and scaling
        if self.typical_x is not 0:
            x -= self.typical_x
        if self.scales is not 1:  # just for efficiency
            x /= self.scales

        # kick out fixed_values
        if self.fixed_values is not None:
            # keeping the transformed values does not help much
            # therefore it is omitted
            if 1 < 3:
                keys = sorted(self.fixed_values.keys())
                x = array([x[i] for i in range(len(x)) if i not in keys],
                          copy=False)
            else:  # TODO: is this more efficient?
                x = list(x)
                for key in sorted(list(self.fixed_values.keys()), reverse=True):
                    x.remove(key)
                x = array(x, copy=False)

        # repair injected solutions
        if repair is not None:
            x = repair(x, copy)
        if input_type is np.ndarray:
            x = array(x, copy=False)
        return x

