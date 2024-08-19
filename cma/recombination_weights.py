# -*- coding: utf-8 -*-
"""`RecombinationWeights` is a list of recombination weights for the CMA-ES.

The most delicate part is the correct setting of negative weights depending
on learning rates to prevent negative definite matrices when using the
weights in the covariance matrix update.

The dependency chain is

lambda -> weights -> mueff -> c1, cmu -> negative weights

"""
# https://gist.github.com/nikohansen/3eb4ef0790ff49276a7be3cdb46d84e9
from __future__ import division, print_function
import math

class RecombinationWeights(list):
    """a list of decreasing (recombination) weight values.

    To be used in the update of the covariance matrix C in CMA-ES as
    ``w_i``::

        C <- (1 - c1 - cmu * sum w_i) C + c1 ... + cmu sum w_i y_i y_i^T

    After calling `finalize_negative_weights`, the weights
    ``w_i`` let ``1 - c1 - cmu * sum w_i = 1`` and guaranty positive
    definiteness of C if ``y_i^T C^-1 y_i <= dimension`` for all
    ``w_i < 0``.

    Class attributes/properties:

    - ``lambda_``: number of weights, alias for ``len(self)``
    - ``mu``: number of strictly positive weights, i.e.
      ``sum([wi > 0 for wi in self])``
    - ``mueff``: variance effective number of positive weights, i.e.
      ``1 / sum([self[i]**2 for i in range(self.mu)])`` where
      ``1 == sum([self[i] for i in range(self.mu)])**2``
    - `mueffminus`: variance effective number of negative weights
    - `positive_weights`: `np.array` of the strictly positive weights
    - ``finalized``: `True` if class instance is ready to use

    Class methods not inherited from `list`:

    - `finalize_negative_weights`: main method
    - `zero_negative_weights`: set negative weights to zero, leads to
      ``finalized`` to be `True`.
    - `set_attributes_from_weights`: useful when weight values are
      "manually" changed, removed or inserted
    - `asarray`: alias for ``np.asarray(self)``
    - `do_asserts`: check consistency of weight values, passes also when
      not yet ``finalized``

    Usage:

    >>> # from recombination_weights import RecombinationWeights
    >>> from cma.recombination_weights import RecombinationWeights
    >>> dimension, popsize = 5, 7
    >>> weights = RecombinationWeights(popsize)
    >>> c1 = 2. / (dimension + 1)**2  # caveat: __future___ division
    >>> cmu = weights.mueff / (weights.mueff + dimension**2)
    >>> weights.finalize_negative_weights(dimension, c1, cmu)
    >>> print('weights = [%s]' % ', '.join("%.2f" % w for w in weights))
    weights = [0.59, 0.29, 0.12, 0.00, -0.31, -0.57, -0.79]
    >>> print("sum=%.2f, c1+cmu*sum=%.2f" % (sum(weights),
    ...                                      c1 + cmu * sum(weights)))
    sum=-0.67, c1+cmu*sum=0.00
    >>> print('mueff=%.1f, mueffminus=%.1f, mueffall=%.1f' % (
    ...       weights.mueff,
    ...       weights.mueffminus,
    ...       sum(abs(w) for w in weights)**2 /
    ...         sum(w**2 for w in weights)))
    mueff=2.3, mueffminus=2.7, mueffall=4.8
    >>> weights = RecombinationWeights(popsize)
    >>> print("sum=%.2f, mu=%d, sumpos=%.2f, sumneg=%.2f" % (
    ...       sum(weights),
    ...       weights.mu,
    ...       sum(weights[:weights.mu]),
    ...       sum(weights[weights.mu:])))
    sum=0.00, mu=3, sumpos=1.00, sumneg=-1.00
    >>> print('weights = [%s]' % ', '.join("%.2f" % w for w in weights))
    weights = [0.59, 0.29, 0.12, 0.00, -0.19, -0.34, -0.47]
    >>> weights = RecombinationWeights(21)
    >>> weights.finalize_negative_weights(3, 0.081, 0.28)
    >>> weights.insert(weights.mu, 0)  # add zero weight in the middle
    >>> weights = weights.set_attributes_from_weights()  # change lambda_
    >>> assert weights.lambda_ == 22
    >>> print("sum=%.2f, mu=%d, sumpos=%.2f" %
    ...       (sum(weights), weights.mu, sum(weights[:weights.mu])))
    sum=0.24, mu=10, sumpos=1.00
    >>> print('weights = [%s]%%' % ', '.join(["%.1f" % (100*weights[i])
    ...                                     for i in range(0, 22, 5)]))
    weights = [27.0, 6.8, 0.0, -6.1, -11.7]%
    >>> weights.zero_negative_weights()  #  doctest:+ELLIPSIS
    [0.270...
    >>> "%.2f, %.2f" % (sum(weights), sum(weights[weights.mu:]))
    '1.00, 0.00'
    >>> mu = int(weights.mu / 2)
    >>> for i in range(len(weights)):
    ...     weights[i] = 1. / mu if i < mu else 0
    >>> weights = weights.set_attributes_from_weights()
    >>> 5 * "%.1f  " % (sum(w for w in weights if w > 0),
    ...                 sum(w for w in weights if w < 0),
    ...                 weights.mu,
    ...                 weights.mueff,
    ...                 weights.mueffminus)
    '1.0  0.0  5.0  5.0  0.0  '

    The optimal weights on the sphere and other functions are closer
    to exponent 0.75:

    >>> for expo, w in [(expo, RecombinationWeights(5, exponent=expo))
    ...                 for expo in [1, 0.9, 0.8, 0.7, 0.6, 0.5]]:
    ...    assert all([len(w(i)) == i for i in range(3, 8)])
    ...    print(7 * "%.2f " % tuple([expo, w.mueff] + w))
    1.00 1.65 0.73 0.27 0.00 -0.36 -0.64 
    0.90 1.70 0.71 0.29 0.00 -0.37 -0.63 
    0.80 1.75 0.69 0.31 0.00 -0.39 -0.61 
    0.70 1.80 0.67 0.33 0.00 -0.40 -0.60 
    0.60 1.84 0.65 0.35 0.00 -0.41 -0.59 
    0.50 1.89 0.62 0.38 0.00 -0.43 -0.57 

    >>> for lam in [8, 8**2, 8**3, 8**4]:
    ...     if lam == 8:
    ...         print(" lam expo mueff        w[i] / w[i](1)")
    ...         print("          /mu(1) 1   2    3    4    5    6    7    8")
    ...     w1 = RecombinationWeights(lam, exponent=1)
    ...     for expo, w in [(expo, RecombinationWeights(lam, exponent=expo))
    ...                     for expo in [1, 0.8, 0.6]]:
    ...         print('%4d ' % lam + 10 * "%.2f " % tuple([expo, w.mueff / w1.mueff] + [w[i] / w1[i] for i in range(8)]))
     lam expo mueff        w[i] / w[i](1)
              /mu(1) 1   2    3    4    5    6    7    8
       8 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 
       8 0.80 1.11 0.90 1.02 1.17 1.50 1.30 1.07 0.98 0.93 
       8 0.60 1.24 0.80 1.02 1.35 2.21 1.68 1.13 0.95 0.85 
      64 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 
      64 0.80 1.17 0.82 0.86 0.88 0.91 0.93 0.95 0.97 0.98 
      64 0.60 1.36 0.65 0.72 0.76 0.80 0.84 0.87 0.91 0.94 
     512 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 
     512 0.80 1.20 0.76 0.78 0.79 0.80 0.81 0.82 0.83 0.83 
     512 0.60 1.42 0.56 0.59 0.61 0.63 0.64 0.65 0.67 0.68 
    4096 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 
    4096 0.80 1.21 0.71 0.73 0.74 0.74 0.75 0.75 0.76 0.76 
    4096 0.60 1.44 0.50 0.52 0.53 0.54 0.55 0.55 0.56 0.56 

    Reference: Hansen 2016, arXiv:1604.00772.
"""
    def __init__(self, len_, exponent=1):
        """return recombination weights `list`, post condition is
        ``sum(self) == 0 and sum(self.positive_weights) == 1``.

        Positive and negative weights sum to 1 and -1, respectively.
        The number of positive weights, ``self.mu``, is about
        ``len_/2``. Weights are strictly decreasing.

        `finalize_negative_weights` (...) or `zero_negative_weights` ()
        should be called to finalize the negative weights.

        :param `len_`: AKA ``lambda`` is the number of weights, see
            attribute `lambda_` which is an alias for ``len(self)``.
            Alternatively, a list of "raw" weights can be provided.

        """
        weights = len_
        self.exponent = exponent  # for the record
        if exponent is None:
            self.exponent = 1  # shall become 4/5 or 3/4?
        try:
            len_ = len(weights)
        except TypeError:
            try:  # iterator without len
                len_ = len(list(weights))
            except TypeError:  # create from scratch
                def signed_power(x, expo):
                    if expo == 1: return x
                    s = (x != 0) * (-1 if x < 0 else 1)
                    return s * math.fabs(x)**expo
                weights = [signed_power(math.log((len_ + 1) / 2.) - math.log(i), self.exponent)
                           for i in range(1, len_ + 1)]  # raw shape
        if len_ < 2:
            raise ValueError("number of weights must be >=2, was %d"
                             % (len_))
        self.debug = False

        # self[:] = weights  # should do, or
        # super(RecombinationWeights, self).__init__(weights)
        list.__init__(self, weights)

        self.set_attributes_from_weights(do_asserts=False)
        sum_neg = sum(self[self.mu:])
        if sum_neg != 0:
            for i in range(self.mu, len(self)):
                self[i] /= -sum_neg
        self.do_asserts()
        self.finalized = False

    def __call__(self, lambda_):
        """return a cut or expanded weight list with similar mueff if possible"""
        if lambda_ <= self.mu:
            return self[:lambda_]
        if lambda_ < self.lambda_:
            return self[:self.mu] + self[self.mu - lambda_:]
        if lambda_ > self.lambda_:
            return self[:self.mu] + (lambda_ - self.lambda_) * [0] + self[self.mu:]
        return self

    def trim_middle(self, lambda_):
        """return weight list of len `lambda_` from the extreme weights.

        This obeys the constraint ``sign(w[i]) == sign((lambda_-1)/2 - i)``
        if the original weights do.

        >>> import cma
        >>> for lam in [2, 3, 5, 11, 33]:
        ...     w = cma.recombination_weights.RecombinationWeights(lam)
        ...     for i in range(1, len(w) + 2):
        ...         w2 = w.trim_middle(i)
        ...         assert len(w2) == i, (len(w2), i)
        ...         if i > 1:
        ...             assert w2[0] == w[0], (w2, w)
        ...             assert w2[-1] == w[-1], (w2, w)
        ...             i_middle = (i - 1) / 2
        ...             assert i > len(w) or (
        ...                  w2[int(i_middle - .1)] * w2[int(i_middle + 1.1)] < 0), (
        ...                        lam, i, i_middle)

        """
        if lambda_ >= self.lambda_:
            return self(lambda_)
        if lambda_ < 2:  # somewhat a hack, FIXME
            return int(lambda_) * [0.]  # maybe this should be 1.?
        mu_ = int(lambda_ / 2)
        assert mu_ > 0
        return self[:mu_] + (lambda_ > 2 * mu_) * [0.] + self[-mu_:]

    def set_attributes_from_weights(self, weights=None, do_asserts=True):
        """make the class attribute values consistent with weights, in
        case after (re-)setting the weights from input parameter ``weights``,
        post condition is also ``sum(self.postive_weights) == 1``.

        This method allows to set or change the weight list manually,
        e.g. like ``weights[:] = new_list`` or using the `pop`,
        `insert` etc. generic `list` methods to change the list.
        Currently, weights must be non-increasing and the first weight
        must be strictly positive and the last weight not larger than
        zero. Then all ``weights`` are normalized such that the
        positive weights sum to one.
        """
        if weights is not None:
            if not weights[0] > 0:
                raise ValueError(
                    "the first weight must be >0 but was %f" % weights[0])
            if weights[-1] > 0:
                raise ValueError(
                    "the last weight must be <=0 but was %f" %
                    weights[-1])
            self[:] = weights
        weights = self
        assert all(weights[i] >= weights[i+1]
                        for i in range(len(weights) - 1))
        self.mu = sum(w > 0 for w in weights)
        spos = sum(weights[:self.mu])
        assert spos > 0
        for i in range(len(self)):
            self[i] /= spos
        # variance-effectiveness of sum^mu w_i x_i
        self.mueff = 1**2 / sum(w**2 for w in
                                   weights[:self.mu])
        sneg = sum(weights[self.mu:])
        assert (sneg - sum(w for w in weights if w < 0))**2 < 1e-11
        not do_asserts or self.do_asserts()
        return self

    def finalize_negative_weights(self, dimension, c1, cmu, pos_def=True):
        """finalize negative weights using ``dimension`` and learning
        rates ``c1`` and ``cmu``.

        This is a rather intricate method which makes this class
        useful. The negative weights are scaled to achieve
        in this order:

        1. zero decay, i.e. ``c1 + cmu * sum w == 0``,
        2. a learning rate respecting mueff, i.e. ``sum |w|^- / sum |w|^+
           <= 1 + 2 * self.mueffminus / (self.mueff + 2)``,
        3. if `pos_def` guaranty positive definiteness when sum w^+ = 1
           and all negative input vectors used later have at most their
           dimension as squared Mahalanobis norm. This is accomplished by
           guarantying ``(dimension-1) * cmu * sum |w|^- < 1 - c1 - cmu``
           via setting ``sum |w|^- <= (1 - c1 -cmu) / dimension / cmu``.

        The latter two conditions do not change the weights with default
        population size.

        Details:

        - To guaranty 3., the input vectors associated to negative
          weights must obey ||.||^2 <= dimension in Mahalanobis norm.
        - The third argument, ``cmu``, usually depends on the
          (raw) weights, in particular it depends on ``self.mueff``.
          For this reason the calling syntax
          ``weights = RecombinationWeights(...).finalize_negative_weights(...)``
          is not supported.

        """
        if dimension <= 0:
            raise ValueError("dimension must be larger than zero, was " +
                             str(dimension))
        self._c1 = c1  # for the record
        self._cmu = cmu

        if self[-1] < 0:
            if cmu > 0:
                if c1 > 10 * cmu:
                    print("""WARNING: c1/cmu = %f/%f seems to assume a
                    too large value for negative weights setting"""
                          % (c1, cmu))
                self._negative_weights_set_sum(1 + c1 / cmu)
                if pos_def:
                    self._negative_weights_limit_sum((1 - c1 - cmu) / cmu
                                                     / dimension)
            self._negative_weights_limit_sum(1 + 2 * self.mueffminus
                                             / (self.mueff + 2))
        self.do_asserts()
        self.finalized = True

        if self.debug:
            print("sum w = %.2f (final)" % sum(self))

    def zero_negative_weights(self):
        """finalize by setting all negative weights to zero"""
        for k in range(len(self)):
            self[k] *= 0 if self[k] < 0 else 1
        self.finalized = True
        return self

    def _negative_weights_set_sum(self, value):
        """set sum of negative weights to ``-abs(value)``

        Precondition: the last weight must no be greater than zero.

        Details: if no negative weight exists, all zero weights with index
        lambda / 2 or greater become uniformely negative.
        """
        weights = self  # simpler to change to data attribute and nicer to read
        value = abs(value)  # simplify code, prevent erroneous assertion error
        assert weights[self.mu] <= 0
        if not weights[-1] < 0:
            # breaks if mu == lambda
            # we could also just return here
            # return
            istart = max((self.mu, int(self.lambda_ / 2)))
            for i in range(istart, self.lambda_):
                weights[i] = -value / (self.lambda_ - istart)
        factor = abs(value / sum(weights[self.mu:]))
        for i in range(self.mu, self.lambda_):
            weights[i] *= factor
        assert 1 - value - 1e-5 < sum(weights) < 1 - value + 1e-5
        if self.debug:
            print("sum w = %.2f, sum w^- = %.2f" %
                  (sum(weights), -sum(weights[self.mu:])))

    def _negative_weights_limit_sum(self, value):
        """lower bound the sum of negative weights to ``-abs(value)``.
        """
        weights = self  # simpler to change to data attribute and nicer to read
        value = abs(value)  # simplify code, prevent erroneous assertion error
        if sum(weights[self.mu:]) >= -value:  # nothing to limit
            return  # needed when sum is zero
        assert weights[-1] < 0 and weights[self.mu] <= 0
        factor = abs(value / sum(weights[self.mu:]))
        if factor < 1:
            for i in range(self.mu, self.lambda_):
                weights[i] *= factor
            if self.debug:
                print("sum w = %.2f (with correction %.2f)" %
                      (sum(weights), value))
        assert sum(weights) + 1e-5 >= 1 - value

    def do_asserts(self):
        """assert consistency.

        Assert:

        - attribute values of ``lambda_, mu, mueff, mueffminus``
        - value of first and last weight
        - monotonicity of weights
        - sum of positive weights to be one

        """
        weights = self
        assert 1 >= weights[0] > 0
        assert weights[-1] <= 0
        assert len(weights) == self.lambda_
        assert all(weights[i] >= weights[i+1]
                        for i in range(len(weights) - 1))  # monotony
        assert self.mu > 0  # needed for next assert
        assert weights[self.mu-1] > 0 >= weights[self.mu]
        assert 0.999 < sum(w for w in weights[:self.mu]) < 1.001
        assert (self.mueff / 1.001 <
                sum(weights[:self.mu])**2 / sum(w**2 for w in weights[:self.mu]) <
                1.001 * self.mueff)
        assert (self.mueffminus == 0 == sum(weights[self.mu:]) or
                self.mueffminus / 1.001 <
                sum(weights[self.mu:])**2 / sum(w**2 for w in weights[self.mu:]) <
                1.001 * self.mueffminus)

    @property
    def lambda_(self):
        """alias for ``len(self)``"""
        return len(self)
    @property
    def mueffminus(self):
        weights = self
        sneg = sum(weights[self.mu:])
        assert (sneg - sum(w for w in weights if w < 0))**2 < 1e-11
        return (0 if sneg == 0 else
                sneg**2 / sum(w**2 for w in weights[self.mu:]))
    @property
    def positive_weights(self):
        """all (strictly) positive weights as ``np.array``.

        Useful to implement recombination for the new mean vector.
        """
        try:
            from numpy import asarray
            return asarray(self[:self.mu])
        except:
            return self[:self.mu]
    @property
    def asarray(self):
        """return weights as numpy array"""
        from numpy import asarray
        return asarray(self)
