# -*- coding: utf-8 -*-
"""Integer handling class `IntegerCentering` to be used in combination with CMA-ES.

Reference: Marty et al, LB+IC-CMA-ES: Two simple modiﬁcations of CMA-ES to
handle mixed-integer problems. PPSN 2024.
"""
import warnings as _warnings
import numpy as np

class IntegerCentering(object):
    """round values of int-variables that are different from the int-mean.

    The callable class instance changes a population of solutions in place.
    This assumes that int-variables change the fitness only if their
    rounded (genotype) value change (which may be wrong if a geno-pheno
    transformation is applied). The class corrects for the bias introduced
    by the rounding.

    This class should generally be used in combination with a lower bound
    on the integer variables along the lines of ``min((0.2,
    mueff/dimension))``, as it is induced by passing a nonempty
    ``'integer_variables'`` option to `CMAEvolutionStrategy`.

    This class has no dynamically changing state variables!

    When integer variable indices are given with the
    ``'integer_variables'`` option, an `IntegerCentering` class instance is
    called in `cma.CMAEvolutionStrategy.tell` with (genotypic) solutions
    like ``int_centering(self.pop_sorted[:self.sp.weights.mu], self.mean)``
    before the CMA update of the state variables. The call changes the
    numpy arrays of `self.pop_sorted` in place. The call tries to access
    the (phenotypic) bounds as defined in ``es.boundary_handler`` from the
    constructor argument. Hence it is expected to fail with a combination
    of ``bounds`` and ``fixed_variables`` set in `cma.CMAOptions`.

    Pseudocode usage hints::

        >> es = cma.CMA(...)
        >> ic = cma.integer_centering.IntegerCentering(es)
        >> [...]
        >> ic(es.pop_sorted[0:es.sp.weights.mu], es.mean)  # before the mean update, use es.mean_old after
        >> es.mean = np.dot(es.sp.weights.positive_weights,  # (re-)assign mean
        ..                  es.pop_sorted[0:es.sp.weights.mu])

    Pseudocode example via ask-and-tell::

        >> es = cma.CMA(...)  # set integer_variables option here
        >> ic = cma.integer_centering.IntegerCentering(es)  # read integer_variables
        >> while not es.stop():
        >>     X = es.ask()
        >>     F = [fun(x) for x in X]
        >>     ic([X[i] for i in np.argsort(F)[:es.sp.weights.mu]], es.mean)
        >>     es.tell(X, F)
        >>     es.logger.add()
        >>     es.disp()

    Working code example:

    >>> import cma
    >>>
    >>> def int1_sphere(x):
    ...     return int(x[0] + 0.5 - (x[0] < 0))**2 + 1000 * (sum(xi**2 for xi in x[1:]))
    >>>
    >>> es = cma.CMA([2, 0.1], 0.22, {'integer_variables': [0],
    ...         'tolfun': 0, 'tolfunhist': 0, 'tolflatfitness': 60, 'verbose': -9})
    >>> _ = es.optimize(int1_sphere)
    >>> assert int1_sphere(es.mean) < 1e-6, es.stop()

    Details: The default `method2` was used in the below reference and, as
    of version 4.0.0, is activated in `cma.CMAEvolutionStrategy` when
    `integer_variables` are given as option.

    Reference: Marty et al, LB+IC-CMA-ES: Two simple modiﬁcations of
    CMA-ES to handle mixed-integer problems. PPSN 2024.
    """
    def __init__(self, int_indices, method=2, correct_bias=True,
                 repair_into_bounds=True, **kwargs):
        """`int_indices` can also be a `CMAEvolutionStrategy` or `CMAOptions` instance.
        `correct_bias` can be in [0, 1] indicating the fraction of the bias
        that should be corrected (but only up to the mean center).

        `repair_into_bounds` repairs all solutions before centering is
        applied. This does not guaranty solutions to be feasible after
        centering in the case when `np.round` can make a solution
        infeasible.

        Details: When the bias computation and correction is applied only
        to the better half of the population (as by default), the value of
        `correct_bias` must be smaller than 1 to move the mean toward a
        solution that has rank lambda/5 or worse, because ``w_{lambda/5}``
        equals roughly ``1/mu`` (``w_{lambda/3}`` equals roughly
        ``1/mu/2``).

        """
        self.params = dict(kwargs)
        self.params.update({n:v for n, v in locals().items() if n != 'self'})
        self.params.setdefault('bounds_offset', 1e-12)
        '''absolute and relative epsilon offset on the bounds such that
           ``np.round`` should never make a solution infeasible that was
           set on a ``int+-1/2`` bound'''
        if not 0 <= self.params['correct_bias'] <= 1:
            new_value = True if self.params['correct_bias'] else False
            _warnings.warn("correct_bias should be in [0, 1],"
                           " the given value {0} is now interpreted as {1}"
                           .format(self.params['correct_bias'], new_value))
            self.params['correct_bias'] = new_value
        self._int_mask = None
        '''the mask will be set from `int_indices` for coding convenience'''
        self.int_indices = self.get_int_indices(int_indices)
        self._has_bounds = None  # set once in has_bounds
        self._lower_bounds = None  # set once
        self._upper_bounds = None  # set once
        try:
            self.center = getattr(self, 'method' + str(method))
        except AttributeError:
            raise ValueError("`method` argument must be 1 or 2, was: {0}"
                             .format(method))

        # for the record / debugging:
        self._record = False
        self._print = False
        self.mahalanobis0 = []  # only for the record
        self.mahalanobis1 = []  # after modification
        self.last_changes = []
        self.last_changes_iteration = []

    def bound(self, i):
        """return lower and upper bound on variable `i`, not in use.
        """
        try:
            return self.params['es'].boundary_handler.get_bound(i)
        except Exception:
            return [-np.inf, np.inf]
    def has_bounds(self, which='both'):
        # TODO: make which='lower' and 'upper' work?
        #       It's probably only a minor speedup.
        if which != 'both':
            raise NotImplementedError(
                "which={0} is invalid. Currently, only 'both' bounds can be checked"
                " with `has_bounds`".format(which))
        if self._has_bounds is not None:
            return self._has_bounds
        self._has_bounds = False
        try:
            bh = self.params['es'].boundary_handler
        except Exception as e:
            _warnings.warn("``bh = self.params['es'].boundary_handler`` failed with"
                           "\n\n  {0}".format(e))
        try:
            if bh is not None and bh.has_bounds():
                self._has_bounds = True
        except Exception as e:
            _warnings.warn("``bh.has_bounds()`` failed with"
                           "\n\n  {0}".format(e))
        return self._has_bounds
    def lower_bounds(self, dimension):
        """return lower bounds of dimension `dimension` or a scalar.

        `dimension` remains the dimension from the first call unless
        the ``_lower_bounds`` attribute is reset to `None`.
        """
        if self._lower_bounds is None:
            self._lower_bounds = self._set_bounds('lower',
                    np.zeros(dimension) - np.inf)  # np.asarray is slower than np.zeros for dimension >30
        return self._lower_bounds
    def upper_bounds(self, dimension):
        """return upper bounds of dimension `dimension` or a scalar.

        `dimension` remains the dimension from the first call unless
        the ``_upper_bounds`` attribute is reset to `None`.
        """
        if self._upper_bounds is None:
            self._upper_bounds = self._set_bounds('upper',
                    np.zeros(dimension) + np.inf)
        return self._upper_bounds
    def _set_bounds(self, which, bounds):
        """return in place modified `bounds`"""
        assert which in ('lower', 'upper'), which
        if self.has_bounds():
            bounds[:] = self.params['es'].boundary_handler.get_bounds(which, len(bounds))
            if self.params['bounds_offset']:
                # mitigate corner cases such as ``np.round([.5, 1.5]) == [0, 2]``
                # pick variables with value at ..., -0.5, 0.5, 1.5, ...
                idx = np.mod(bounds, 1) == 0.5
                bounds[idx] += (1 if which == 'lower' else
                    -1) * self.params['bounds_offset'] * np.maximum(1, np.abs(bounds[idx]))
        return bounds

    def get_int_indices(self, es_opts_indices=None):
        """determine integer variable indices from es or es.opts or

        a variable index list or self.
        """
        if es_opts_indices is None:  # check self.int_indices and self.params
            return self.int_indices if (
                hasattr(self, 'int_indices') and self.int_indices is not None
                ) else self.params['int_indices']
        if hasattr(es_opts_indices, 'opts'):  # check es
            if 'es' not in self.params:
                self.params['es'] = es_opts_indices
            opts = es_opts_indices.opts
        else:  # assume that arg was an opts dict
            opts = es_opts_indices
        names = ['integer_variables', 'integer_indices', 'int_indices']
        for name in names:
            if name in opts:
                indices = opts[name]
                break  # key found
        else:
            indices = opts
        if 'int_indices' not in self.params:  # can't currently happen
            self.params['int_indices'] = indices
        return indices

    def callback(self, es):
        """change `es.pop_sorted` and update `es.mean` accordingly.

        Not in use, as this must be ideally done in the middle of `tell`,
        that is, neither before nor after `tell`.
        """
        self(es.pop_sorted, es.mean_old)
        es.mean = np.dot(es.sp.weights.positive_weights,
                         es.pop_sorted[0:es.sp.weights.mu])
    @property
    def int_mask(self):
        if self._int_mask is None:
            raise ValueError("dimension is not known yet")
        return self._int_mask

    def __call__(self, solution_list, mean):
        """round values of int-variables in `solution_list` ideally without bias to the mean"""
        if self._int_mask is None:
            self._int_mask = np.asarray([i in self.int_indices
                                         for i in range(len(mean))])
        if self.params.get('repair_into_bounds', False):
            self.repair(solution_list)  # create a bias towards feasible solutions
            # this does not guaranty that round(x) is also in bounds which
            # may be a good feature when the bounds are chosen suboptimally
        self.center(solution_list, mean)
        if 11 < 3 and self.params.get('repair_into_bounds', False):
            self.repair(solution_list, randomized=False)
        return solution_list

    def method1(self, solution_list, mean):
        """DEPRECATED (experimental and outdated) round values of int-variables in `solution_list` and reduce bias to the mean"""
        m_int = np.round(mean)
        mutated_down = np.mean([np.round(x) < m_int for x in solution_list], axis=0)
        mutated_up = np.mean([np.round(x) > m_int for x in solution_list], axis=0)
        maxmut_ratio = np.where(mutated_down > mutated_up, mutated_down, mutated_up)
        # assert all(maxmut_ratio == np.max([mutated_down, mutated_up], axis=0))  # 3x slower
        self.last_changes = []
        for x in solution_list:
            for i in self.int_indices:
                if (np.round(x[i]) != m_int[i] or (self.params['correct_bias'] and
                    (maxmut_ratio[i] > 0.5 or  # reduce bias
                     np.random.rand() < maxmut_ratio[i] / (1 - mutated_up[i] - mutated_down[i])))):
                     if self._record:
                        self.mahalanobis0.append(self.params['es'].mahalanobis_norm(x - mean))
                     x[i] = np.round(x[i])
                     if self._record:
                        self.mahalanobis1.append(self.params['es'].mahalanobis_norm(x - mean))
                        n0, n1 = self.mahalanobis0[-1], self.mahalanobis1[-1]
                        self.last_changes.append([i, n0, n1])
                        self.last_changes_iteration.append(self.params['es'].countiter)
        return solution_list  # declarative, elements of solution_list have changed in place
    def method2(self, solution_list, mean):
        """center (round) values of int-variables of solutions in `solution_list`.

        Elements of `solution_list` must accept boolean indexing like ``np.arrays``.

        Values are centered iff the centered value differs from the
        centered mean. If `correct_bias`, the introduced bias is amended by
        changing the other (noncentered) coordinates towards their
        int-center (which is the int-center of the mean) too, up to the
        fraction `correct_bias`.

        CAVEAT/TODO: the bias correction currently doesn't check bounds,
        hence it may push feasible solutions out-of-bounds by (i) centering
        single solutions and (ii) shifting coordinates towards their
        centered value during bias correction. In itself and overall, this
        may not be a problem, in particular when `repair` is applied before
        or after calling `method2`.
        """
        repair_int = False
        if self._print:
            mean0 = np.mean(solution_list, axis=0)
        # m_int = np.round(self.repair([mean], randomized=False, copy=True)[0])
        m_int = np.round(mean)
        '''to check whether x_int == m_int and to limit the bias correction move'''
        # Should we check that m_int is in-bounds!? The problem to
        # repair m_int here is that it breaks the ``x_int == m_int``
        # condition. Generally, the mean doesn't need to be feasible and if
        # it is updated only with feasible solutions it becomes feasible.
        dim = len(solution_list[0])
        if self.params['correct_bias']:
            biases = np.zeros(dim)
            '''bias (sum) created from centering solutions'''
            mneg = np.zeros(dim)
            '''sum of possible move to the left for unbiasing'''
            mpos = np.zeros(dim)
            '''sum of possible move to the right for unbiasing'''
        for i, x in enumerate(solution_list):
            x_int = np.round(x)
            ism = (x_int == m_int)
            # The below repairs are only effective (i.e. x_int < lbs is
            # only possible) when x < lbs or the bound has an unreasonable
            # value like 0.25. Then, (i) the below would prevent centering
            # a solution with the boundary domain value at all. (ii)
            # without the below, the bias correct introduces a bias toward
            # the feasible domain (which is not necessarily bad).
            # if self.params.get('repair_into_bounds', False) and self.has_bounds():
            #     # TODO: this breaks the bias correct condition
            #     #       x_int == m_int, which may be fine
            #     ism = np.logical_or(ism, x_int < self.lower_bounds(dim))
            #     ism = np.logical_or(ism, x_int > self.upper_bounds(dim))
            if repair_int:
                self.repair([x_int])
            if self.params['correct_bias']:
                mpos += ism * ((x_int - x) > 0) * (x_int - x)
                mneg += ism * ((x_int - x) < 0) * (x_int - x)
                biases += ~ism * (x_int - x)
            x[~ism * self.int_mask] = x_int[~ism * self.int_mask]  # round off-mean values
        if self.params['correct_bias']:
            # change noncentered variables (those with the int-value of the mean)
            # compare biases with mpos or mneg and set alpha like bias/mpos
            frac = self.params['correct_bias']

            _time_code = False
            if _time_code:  # compare two versions
                import collections, time
                try: self.alpha_timings
                except: self.alpha_timings = collections.defaultdict(float)
                # takes 0.03s (~1.5%) in 200D for 1e4 evaluations
                t0 = time.time()
                alphas = []
                for i, (b, p, n) in enumerate(zip(biases, mpos, mneg)):
                    if not self.int_mask[i]:
                        alphas += [0.]
                    elif b * p < 0:
                        alphas += [-frac * b / p if -frac * b < p else 1.0]
                    elif b * n < 0:
                        alphas += [-frac * b / n if frac * b < -n else 1.0]
                    else:
                        alphas += [0.]
                self.alpha_timings['alpha'] += time.time() - t0
                t0 = time.time()

            # about 4 times faster in 200 D
            alphas2 = np.zeros(len(biases))
            for moves in [mpos, mneg]:
                idx = self.int_mask * biases * moves < 0
                alphas2[idx] = np.minimum(1, -frac * biases[idx] / moves[idx])
                assert moves is mneg or np.random.rand() < 0.98 or (
                    np.all(idx + (biases * mneg < 0) < 2)), (idx, biases, mpos, mneg)

            if _time_code:
                self.alpha_timings['alpha2'] += time.time() - t0
                assert np.all(alphas2 - 1e-11 <= alphas) and (
                       np.all(alphas2 + 1e-11 >= alphas)), (
                            alphas2, alphas, alphas2 - alphas)

            # this does not guaranty feasibility unless above x_int was repaired
            m_int_repaired = (self.repair([np.array(m_int, copy=True)])[0]
                        if repair_int else m_int)
            for x in solution_list:
                x += ((np.round(x) == m_int)
                        * (biases * (m_int_repaired - x) < 0)
                        * alphas2 * (m_int_repaired - x))  # move towards m_int == x_int
        if self._print:  # print remaining biases
            mean1 = np.mean(solution_list, axis=0)
            biases2 = [(i, mean1[i] - mean0[i]) for i in range(len(mean0))
                    if (mean1[i] - mean0[i])**2 > 1e-22]
            if biases2 or not self.params['es'].countiter % 100:
                print(self.params['es'].countiter, alphas2, biases2)
        return solution_list  # declarative, elements of solution_list have changed in place

    def repair(self, solution_list, randomized=True):
        """set values of int-variables of solutions of `solution_list` into bounds.

        Elements of `solution_list` are changed in place after passing them
        through ``np.asarray`` and `solution_list` is changed in place too.

        When ``randomized is True`` sample the value uniformly between the
        bound and the value center (the rounded bound) when the latter is
        feasible.
        """
        if not self.has_bounds():
            return solution_list
        copy = False  # is a constant for the time being
        dim = len(solution_list[0])
        lbs = self.lower_bounds(dim)
        ubs = self.upper_bounds(dim)
        for i, x in enumerate(solution_list[:]):
            x = np.asarray(x)  # this may detach x from solution_list
            # set int-variables of x onto bounds
            islow = np.logical_and(x < lbs, self.int_mask)
            ishigh = np.logical_and(x > ubs, self.int_mask)
            assert np.random.rand() < 0.95 or not np.any(islow * ishigh), (
                i, islow, ishigh)  # rand makes it ~5 times faster
            if np.any(islow):
                if copy:  # FIXME: this is a second copy if asarray made one
                    x = np.array(x, copy=True)
                x[islow] = lbs[islow]
            if np.any(ishigh):
                if copy:  # FIXME: this is a second copy if asarray made one
                    x = np.array(x, copy=True)
                x[ishigh] = ubs[ishigh]
            # move modified variables randomly towards the center
            if randomized:
                x_int = np.round(x)
                idx = np.logical_or(np.logical_and(islow, x_int > x),
                                    np.logical_and(ishigh, x_int < x))
                if np.any(idx):
                    x[idx] += (x_int[idx] - x[idx]) * np.random.rand(sum(idx))
            solution_list[i] = x  # needed when np.asarray above made a copy
        return solution_list
