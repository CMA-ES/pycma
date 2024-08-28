"""Search space transformation and encoding/decoding classes"""
from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import array, isfinite, log
import warnings as _warnings

from .utilities.utils import rglen, print_warning, is_one, SolutionDict as _SolutionDict
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
        return np.asarray(x) - x_opt
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
    >>> np.round(R(R(x), inverse=1), 9).tolist()
    [1.0, 2.0, 3.0]

    :See also: `Rotated`

    """
    def __init__(self, seed=None):
        """same ``seed`` means same rotation, by default a random but
        fixed once and for all rotation, different for each instance
        """
        self.seed = seed
        self.dicMatrices = {}
    def __call__(self, x, inverse=False, **kwargs):
        """Rotates the input array `x` with a fixed rotation matrix
           (``self.dicMatrices[len(x)]``)
        """
        x = np.asarray(x)
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
    try: __doc__ = BoxConstraintsTransformationBase.__doc__ + __doc__
    except: pass

def margin_width1(bound):
    """return quadratic domain image width ``(1 + abs(bound)) / 20``"""
    return (np.abs(bound) + 1) / 20

def margin_width2(bound):
    """return quadratic domain image width ``max(1, abs(bound)) / 20``"""
    return max((1, np.abs(bound))) / 20

linquad_margin_width = margin_width2
'''used in BoxConstraintsLinQuadTransformation.initialize'''

class BoxConstraintsLinQuadTransformation(BoxConstraintsTransformationBase):
    """implement a periodic transformation that is bijective from

    ``[lb - al, ub + au]`` -> ``[lb, ub]``, where either
    ``al = min((ub-lb) / 2, 0.05 * (|lb| + 1))`` and
    ``ul = min((ub-lb) / 2, 0.05 * (|ub| + 1))`` or (default)
    ``al = min((ub-lb) / 2, 0.05 * max(|lb|, 1))`` and
    ``ul = min((ub-lb) / 2, 0.05 * max(|ub|, 1))``
    depending on the method `cma.transformations.linquad_margin_width`
    assigned as `margin_width1` or `margin_width2`.

    Generally speaking, this transformation aims to resemble ``sin`` to
    be a continuous differentiable (ie. C^1) transformation over R into
    a bounded interval; then, it also aims to improve over ``sin`` in
    the following two ways: (i) resemble the identity over an interval
    as large possible while keeping the second derivative in reasonable
    limits, and (ii) numerical stability in "pathological" corner cases
    of the boundary limit values.

    The transformation resembles the shape ``sin(2*pi * x / a - pi/2))``
    with a period length of ``a = 2 * ((ub + au) - (lb - al)) = 2 * (ub -
    lb + al + au)``.

    The transformation is the identity in ``[lb + al, ub - au]`` (typically
    about 90% of the interval) and it is quadratic to the left of
    ``lb + al`` down to ``lb - 3*al`` and to the right of ``ub - au`` up to
    ``ub + 3*au``.

    Details
    =======
    Partly due to numerical considerations depend the values ``al`` and
    ``au`` on ``abs(lb)`` and ``abs(ub)`` which makes the transformation
    non-translation invariant. When ``ub-lb`` is small compared to
    ``min(|lb|, |ub|)``, the linear proportion becomes zero.

    In contrast to ``sin(.)``, the transformation is robust to
    "arbitrary" large values for boundaries, e.g. a lower bound of
    ``-1e99`` or upper bound of ``np.inf`` or bound ``None``.

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
    ...                  'verb_disp':0, 'tolflatfitness': 1e9, 'verbose': -2})
    >>> not warns or str(warns[0].message).startswith(('in class GenoPheno: user defi',
    ...                                   'flat fitness'))
    True

    or:

    >>> es = cma.CMAEvolutionStrategy(4 * [2], 1, {'verbose':0, 'verb_log':0})  # doctest: +ELLIPSIS
    (4_w,8)-aCMA-ES (mu_w=...
    >>> with warnings.catch_warnings(record=True) as warns:  # flat fitness warning, not necessary anymore
    ...     while not es.stop():
    ...         X = es.ask()
    ...         f = [cma.ff.elli(tf(x)) for x in X]  # tf(x)==tf.transform(x)
    ...         es.tell(X, f)
    >>> assert 'tolflatfitness' in es.stop(), str(es.stop())

    Example of the internal workings:

    >>> import cma.transformations as ts
    >>> ts.linquad_margin_width = ts.margin_width1
    >>> tf = ts.BoxConstraintsLinQuadTransformation([[1,2], [1,11], [1,11]])
    >>> tf.bounds
    [[1, 2], [1, 11], [1, 11]]
    >>> tf([1.5, 1.5, 1.5])
    [1.5, 1.5, 1.5]
    >>> np.round(tf([1.52, -2.2, -0.2, 2, 4, 10.4]), 9).tolist()
    [1.52, 4.0, 2.0, 2.0, 4.0, 10.4]
    >>> res = np.round(tf._au, 2)
    >>> assert res[:4].tolist() == [ 0.15, 0.6, 0.6, 0.6], res[:4].tolist()
    >>> res = [round(x, 2) for x in tf.shift_or_mirror_into_invertible_domain([1.52, -12.2, -0.2, 2, 4, 10.4])]
    >>> assert res == [1.52, 9.2, 2.0, 2.0, 4.0, 10.4], res
    >>> tmp = tf([1])  # call with lower dimension

    >>> ts.linquad_margin_width = ts.margin_width2
    >>> tf = ts.BoxConstraintsLinQuadTransformation([[1,2], [1,11], [1,11]])
    >>> tf.bounds
    [[1, 2], [1, 11], [1, 11]]
    >>> tf([1.5, 1.5, 1.5])
    [1.5, 1.5, 1.5]
    >>> np.round(tf([1.52, -2.2, -0.2, 2, 4, 10.4]), 9).tolist()
    [1.52, 4.1, 2.1, 2.0, 4.0, 10.4]
    >>> res = np.round(tf._au, 2)
    >>> assert list(res[:4]) == [ 0.1, 0.55, 0.55, 0.55], list(res[:4])
    >>> res = [round(x, 2) for x in tf.shift_or_mirror_into_invertible_domain([1.52, -12.2, -0.2, 2, 4, 10.4])]
    >>> assert res == [1.52, 9.0, 2.1, 2.0, 4.0, 10.4], res
    >>> tmp = tf([1])  # call with lower dimension
    >>> for i in range(5):
    ...     lb = np.random.randn(4) - 1000 * i * np.random.rand()
    ...     ub = lb + (1e-7 + np.random.rand(4)) / (1e-9 + np.random.rand(4)) + 1001 * i * np.random.rand()
    ...     lb[-1], ub[-1] = lb[-2], ub[-2]
    ...     b = ts.BoxConstraintsLinQuadTransformation([[l, u] for (l, u) in zip(lb[:3], ub[:3])])
    ...     for x in [(ub - lb) * np.random.randn(4) / np.sqrt(np.random.rand(4)) for _ in range(11)]:
    ...         assert all(lb <= b.transform(x)), (lb, ub, b.transform(x), b.__dict__)
    ...         assert all(b.transform(x) <= ub), (lb, ub, b.transform(x), b.__dict__)
    ...         assert all(b.transform(lb - b._al) == lb), (lb, ub, b.transform(x), b.__dict__)
    ...         assert all(b.transform(ub + b._au) == ub), (lb, ub, b.transform(x), b.__dict__)

    """

    def initialize(self, length=None):
        """see ``__init__``"""
        if length is None:
            length = len(self.bounds)
        max_i = min((len(self.bounds) - 1, length - 1))
        self._lb = np.asarray([self.bounds[min((i, max_i))][0]
                          if self.bounds[min((i, max_i))][0] is not None
                          else -np.inf
                          for i in range(length)])
        self._ub = np.asarray([self.bounds[min((i, max_i))][1]
                          if self.bounds[min((i, max_i))][1] is not None
                          else np.inf
                          for i in range(length)])
        lb = self._lb
        ub = self._ub
        if any(lb >= ub):
            raise ValueError('Lower bounds need to be smaller than upper bounds. They'
                             ' were not at idx={0} where lb={1}, ub={2}'
                             .format(np.where(lb >= ub)[0], lb, ub))
        # define added values for lower and upper bound
        self._al = np.asarray([min([(ub[i] - lb[i]) / 2, linquad_margin_width(lb[i])])
                             if isfinite(lb[i]) else 1 for i in rglen(lb)])
        self._au = np.asarray([min([(ub[i] - lb[i]) / 2, linquad_margin_width(ub[i])])
                             if isfinite(ub[i]) else 1 for i in rglen(ub)])

    def __call__(self, solution_genotype, copy=True):
        # about four times faster version of array([self._transform_i(x, i) for i, x in enumerate(solution_genotype)])
        # still, this makes a typical run on a test function two times slower, but there might be one too many copies
        # during the transformations in gp
        # the boundary handling adds [0.2, 0.35] ms to [0.4, 0.5] ms per non-verbose iteration without CMA
        # in dimension [10, 40] as of 2023.
        # return np.array([self._transform_i(x, i) for i, x in enumerate(solution_genotype)])
        # adds [.35, 1.9] ms
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
    try: shift_or_mirror_into_invertible_domain.__doc__ = BoxConstraintsTransformationBase.shift_or_mirror_into_invertible_domain.__doc__ + shift_or_mirror_into_invertible_domain.__doc__
    except: pass

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
    """Diagonal linear transformation with exponential update.

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
        if is_one(self.scaling):
            self.is_identity = True
        self._parameters = {}

    def transform(self, x):
        return self.scaling * x
        return x if self.is_identity else self.scaling * x

    def transform_inverse(self, x):
        return x / self.scaling
        return x if self.is_identity else x / self.scaling

    def transform_covariance_matrix(self, C):
        """return the covariance matrix D * C * D"""
        if self.is_identity:
            return C
        return (self.scaling * C).T * self.scaling

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

    def parameters(self, mueff, c1_factor=1, cmu_factor=1):
        """learning rate parameter suggestions.

        TODO: either input popsize or input something like fac = 1 + (2...5) / popsize
              cmu has already 1/7 as popsize correction
    """
        N = self.dim
        input_parameters = N, mueff, c1_factor, cmu_factor
        try: return self._parameters[input_parameters]
        except KeyError: pass

        # conedf(df) = 1 / (df + 2. * sqrt(df) + mu / N) = c1_sep is WRONG?
        # c1_default = min(1, sp.popsize / 6) * 2 / (
        #         (N + 1.3)**2 + mueff))
        # 2020 diagonal decoding/acceleration paper (Akimoto & Hansen, ECJ):
        # c1DDacc = 1 / (2 * (df / N + 1) * (N + 1)**(3/4) + mueff/2)

        c1dd_orig = 1 / (4 * (N + 1)**(3/4) + mueff/2)  # if df=N
        c1 = 1 / (5 + 2 * N + mueff / 2)

        if 11 < 3:  # side note
            # the squared length of the mu-average vector on the linear fct is close to
            #  N / mu * (1 + 0.6321 * mu / N) = N / mu + 0.6321
            # (and 0.63212... = (1 - exp(-1))). Remark also that the mu-average is
            # is multiplied by sqrt(mu), hence the squared length would be
            # N + 0.6321 mu.
            # Here we learn single components/projections which are roughly
            # of size mu / 2 and 1. Code to see the values:
            mu, N = 10000, 100
            z = np.random.randn(mu, N)
            z[:,0] = abs(z[:,0])  # selected vectors of the (2xmu, 4xmu)-ES on the linear function
            z2 = np.mean(z, 0)**2
            print(z2[0] * mu / (mu / 1.55), np.mean(z2[1:] * mu))
            print(sum(z2) / (N / mu + 0.6321))  # are all close to one

        # cmudf(df) = (0.25 + mu + 1 / mu - 2) / (df + 4 * sqrt(df) + mu / 2)
        # cmu_default = alphacov * # a simpler nominator would be: (mu - 0.75)
        #      (0.25 + mu + 1 / mu - 2) / (
        #       (N + 2)**2 + alphacov * mu / 2))  # alphacov = 2
        #              # cmu_default -> 1 for mu -> N**2 * (2 / alphacov)

        cmudd_orig = min((1 - c1dd_orig,
                          c1dd_orig * (mueff + 1/mueff - 2 + 1/7)))  # 0.5 * lam/(lam+5)))
        cmu = (mueff + 1./mueff - 2 + 1/7) / (
                5 + 2 * N + mueff / 2)
        # print("c1/c1_org={}, cmu/cmu_orig={}".format(c1 / c1dd_orig, cmu / cmudd_orig))

        if 11 < 3:
            # print("original activated")
            c1 = c1dd_orig
            cmu = cmudd_orig
            cc = np.sqrt(mueff * c1) / 2

        cc = np.sqrt(mueff * c1) / 2  # because mueff/2 is in the denominator
        c1 *= c1_factor  # should cc be reset?
        cmu *= cmu_factor
        cmu = min((cmu, 1 - c1))

        self._parameters[input_parameters] = {'c1': c1, 'cmu': cmu, 'cc': cc}
        def check_values(d, input_parameters=None):
            """`d` is the parameters dictionary"""
            if not (0 <= d['c1'] < 0.75 and 0 <= d['cmu'] <= 1 and
                    d['c1'] <= d['cc'] <= 1):
                raise ValueError("On input {0},\n"
                    "the values {1}\n"
                    "do not satisfy\n"
                    "  `0 <= c1 < 0.75 and 0 <= cmu <= 1 and"
                    " c1 <= cc <= 1`".format(str(input_parameters), str(d)))
        check_values(self._parameters[input_parameters], input_parameters)
        return self._parameters[input_parameters]

    def _init_(self, int_or_vector):
        """init scaling (only) when not yet done"""
        if not self.is_identity or not np.size(self.scaling) == 1:
            return self
        try: int_ = len(int_or_vector)
        except TypeError: int_ = int_or_vector
        self.scaling = np.ones(int_)
        return self

    def set_i(self, index, value):
        """set ``scaling[index] = value``.

        To guaranty initialization to non-identity, the use pattern::

            de = cma.transformations.DiagonalDecoding()
            de._init_(dimension).set_i(3, 4.4)

        is available.
        """
        if np.size(self.scaling) == 1:
            raise ValueError("not yet initialized (dimension needed)")
        self.is_identity = False
        self.scaling[index] = value

    def update(self, vectors, weights, ignore_indices=None):
        """exponential update of the scaling factors.

        `vectors` have shape popsize x dimension and are assumed to be
        standard normal before selection.

        `weights` may be negative and include the learning rate(s).

        Variables listed in `ignore_indices` are not updated.
        """
        self._init_(vectors[0])
        self.is_identity = False
        weights = np.asarray(weights)
        # weights[weights < 0] = 0  # only positive weights
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
        facs = np.exp((z2_average - sum(weights)) / 2)
            # remark that exp(log(2) * x) = 2**x
            # without log(2) we have that exp(z2 - 1) = z2 iff z2 = 1
            #     and always exp(z2 - 1) >= z2
            # with log(2) we also have that exp(z2 - 1) = z2 if z2 = 2
            #   (and exp(z2 - 1) <= z2 iff z2 in [1, 2]
            #    and also exp(z2 - 1) = 1/2 if z2 = min z2 = 0)

        # z2=0, w=-1, d=log(2) => exp(d w (0 - 1)) = 2 = 1 + w (0 - 1)
        # z2=2, w=1, d=log(2) => exp(d w (2 - 1)) = 2 = 1 + w (2 - 1)

        if 1 < 3:  # bound increment to observed value
            if 1 < 3:
                idx = facs > 1
                if any(idx):
                    # TODO: generally, a percentile instead of the max seems preferable
                    max_z2 = np.max(np.abs(z2[:,idx] - 1), axis=0) / 2 + 1  # dim-dimensional vector
                    if 11 < 3 and any(max_z2 < 1):
                        print()
                        print(max_z2)
                        print(z2[:,idx])
                        print(z2)
                        print()
                        1/0
                    idx2 = facs[idx] > max_z2
                    if any(idx2):
                        print_warning("clipped exponential update in indices {0}\n"
                                    "from {1} to max(|z^2-1| + 1)={2}".format(
                                        np.where(idx)[0][idx2], facs[idx][idx2], max_z2[idx2]))
                        facs[idx][idx2] = max_z2[idx2]
            else:  # previous attempts
                # because 1 + eta (z^2 - 1) < max(z^2, 1) if eta < 1
                # we want for exp(eta (z^2 - 1)) ~ 1 + eta (z^2 - 1):
                #   exp(eta (z^2 - 1)) < z^2  <=>  eta < log z^2 / (z^2 - 1)
                # where eta := sum w^+, z^2 := sum w^+ zi^2 / eta
                # remark: for z^2 \to+ 1, eta_max \to- log z^2 / (z^2 - 1) = 1

                idx = weights > 0  # for negative weights w (z^2 - 1) <= w
                # Remark: z2 - 1 can never be < -1, i.e. eta_max >= log(2) ~ 0.7
                eta = sum(abs(weights[idx]))
                z2_pos_average = np.dot(weights[idx], z2[idx]) / eta
                z2_large_pos = z2_pos_average[z2_pos_average > 1]
                if np.size(z2_large_pos):
                    if 1 < 3:
                        eta_max = max(np.log(z2_large_pos) /  # DONEish: review/approve this
                                        (z2_large_pos - 1))
                        if eta > eta_max:
                            facs **= (eta_max / eta)
                            _warnings.warn("corrected exponential update by {0} from {1}".format(eta_max/eta, eta))
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
        if ignore_indices is None or len(ignore_indices) == 0:
            self.scaling *= facs
        else:  # do not update all variables
            for i in range(len(self.scaling)):
                if i not in ignore_indices:
                    self.scaling[i] *= facs[i]
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
        self.repaired_solutions = _SolutionDict()
        if tf is not None:
            self.tf_pheno = tf[0]
            self.tf_geno = tf[1]  # TODO: should not necessarily be needed
            # r = np.random.randn(dim)
            # assert all(tf[0](tf[1](r)) - r < 1e-7)
            # r = np.random.randn(dim)
            # assert all(tf[0](tf[1](r)) - r > -1e-7)
            print_warning("in class GenoPheno: user defined transformations have not been tested thoroughly", maxwarns=1)
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
        elif self.scales.shape != () and len(self.scales) != self.N:
            raise ValueError('len(scales) == ' + str(len(self.scales)) +
                         ' does not match dimension N == ' + str(self.N))

        self.typical_x = array(typical_x) if typical_x is not None else None
        if vec_is_default(self.typical_x, 0):
            self.typical_x = 0
        elif self.typical_x.shape != () and len(self.typical_x) != self.N:
            raise ValueError('len(typical_x) == ' + str(len(self.typical_x)) +
                         ' does not match dimension N == ' + str(self.N))

        if (is_one(self.scales) and
                not np.any(self.typical_x) and
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
            def into_bounds(x, copy=False):
                return x if not copy else np.array(x, copy=True)
        if self.isidentity:
            y = into_bounds(x) # was into_bounds(x, False) before (bug before v0.96.22)
        else:
            if self.fixed_values is None:
                y = array(x, copy=copy)  # make a copy, in case
            else:  # expand with fixed values
                y = list(x)  # is a copy
                for i in sorted(self.fixed_values.keys()):
                    y.insert(i, self.fixed_values[i])
                y = np.asarray(y)
            copy = False

            if not is_one(self.scales):  # just for efficiency
                y *= self.scales

            if np.any(self.typical_x):
                y += self.typical_x

            if self.tf_pheno is not None:
                y = np.asarray(self.tf_pheno(y))

            y = into_bounds(y, copy)  # copy is False

            if self.fixed_values is not None:
                for i, k in list(self.fixed_values.items()):
                    y[i] = k

        if input_type is np.ndarray:
            y = np.asarray(y)
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
        def repair_and_flag_change(self, repair, x, copy):
            if repair is None:
                return x
            x2 = repair(x, copy_if_changed=copy)  # need to ignore copy?
            if 11 < 3 and not np.all(np.asarray(x) == x2):  # assumes that dimension does not change
                self.repaired_solutions[x2] = {'count': len(self.repaired_solutions)}
            return x2

        if from_bounds is None:
            def from_bounds(x, copy=False):
                return x  # not change, no copy

        if archive is not None:
            try:
                x = archive[y]['geno']
            except (KeyError, TypeError):
                x = None
            if x is not None:
                if archive[y]['iteration'] < archive.last_iteration:
                    x = repair_and_flag_change(self, repair, x, copy)
                    # x = repair(x, copy_if_changed=copy)
                return x

        input_type = type(y)
        x = y

        x = from_bounds(x, copy)

        if self.isidentity:
            x = repair_and_flag_change(self, repair, x, copy)
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
        if np.any(self.typical_x):
            x -= self.typical_x
        if not is_one(self.scales):  # just for efficiency
            x /= self.scales

        # kick out fixed_values
        if self.fixed_values is not None:
            x = np.asarray([x[i] for i in range(len(x)) if i not in self.fixed_values])

        # repair injected solutions
        x = repair_and_flag_change(self, repair, x, copy)
        if input_type is np.ndarray:
            x = np.asarray(x)
        return x
