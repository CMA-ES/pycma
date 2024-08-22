"""Wrapper for objective functions like noise, rotation, gluing args
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals, with_statement
import warnings
from functools import partial
import numpy as np
import time
from .utilities import utils
from .utilities.math import Mh as _Mh
from .transformations import ConstRandnShift, Rotation
from .constraints_handler import BoundTransform
from .optimization_tools import EvalParallel2  # for backwards compatibility
from .utilities.python3for2 import range
del absolute_import, division, print_function  #, unicode_literals, with_statement

rotate = Rotation()

class Function(object):
    """a declarative base class, indicating that a derived class instance
    "is" a (fitness/objective) function.

    A `callable` passed to `__init__` is called as the fitness
    `Function`, otherwise the `_eval` method is called, if defined in the
    derived class, when the `Function` instance is called. If the input
    argument is a matrix or a list of vectors, the method is called for
    each vector individually like
    ``_eval(numpy.asarray(vector)) for vector in matrix``.

    >>> import cma
    >>> from cma.fitness_transformations import  Function
    >>> f = Function(cma.ff.rosen)
    >>> assert f.evaluations == 0
    >>> assert f([2, 3]) == cma.ff.rosen([2, 3])
    >>> assert f.evaluations == 1
    >>> assert f([[1], [2]]) == [cma.ff.rosen([1]), cma.ff.rosen([2])]
    >>> assert f.evaluations == 3
    >>> class Fsphere(Function):
    ...     def _eval(self, x):
    ...         return sum(x**2)
    >>> fsphere = Fsphere()
    >>> assert fsphere.evaluations == 0
    >>> assert fsphere([2, 3]) == 4 + 9 and fsphere([[2], [3]]) == [4, 9]
    >>> assert fsphere.evaluations == 3
    >>> Fsphere.__init__ = lambda self: None  # overwrites Function.__init__
    >>> assert Fsphere()([2]) == 4  # which is perfectly fine to do

    Details:

    - When called, a class instance calls either the function passed to
      `__init__` or, if none was given, tries to call any of the
      `function_names_to_evaluate_first_found`, first come first serve.
      By default, ``function_names_to_evaluate_first_found == ["_eval"]``.

    - This class cannot move to module `fitness_functions`, because the
      latter uses `fitness_transformations.rotate`.

    """
    _function_names_to_evaluate_first_found = ["_eval"]
    @property
    def function_names_to_evaluate_first_found(self):
        """attributes which are searched for to be called if no function
        was given to `__init__`.

        The first present match is used.
        """  # % str(Function._function_names_to_evaluate_first_found)
        return Function._function_names_to_evaluate_first_found

    def __init__(self, fitness_function=None):
        """allows to define the fitness_function to be called, doesn't
        need to be ever called
        """
        Function.initialize(self, fitness_function)
    def initialize(self, fitness_function):
        """initialization of `Function`
        """
        self.__callable = fitness_function  # this naming prevents interference with a derived class variable
        self.evaluations = 0
        self.ftarget = -np.inf
        self.target_hit_at = 0  # evaluation counter when target was first hit
        self.__initialized = True

    def __call__(self, *args, **kwargs):
        # late initialization if necessary
        try:
            if not self.__initialized:
                raise AttributeError
        except AttributeError:
            Function.initialize(self, None)
        # find the "right" callable
        callable_ = self.__callable
        if callable_ is None:
            for name in self.function_names_to_evaluate_first_found:
                try:
                    callable_ = getattr(self, name)
                    break
                except AttributeError:
                    pass
        # call with each vector
        if callable_ is not None:
            X, list_revert = utils.as_vector_list(args[0])
            self.evaluations += len(X)
            F = [callable_(np.asarray(x), *args[1:], **kwargs) for x in X]
            if not self.target_hit_at and any(np.asarray(F) <= self.ftarget):
                self.target_hit_at = self.evaluations - len(X) + 1 + list(np.asarray(F) <= self.ftarget).index(True)
            return list_revert(F)
        else:
            self.evaluations += 1  # somewhat bound to fail

class ComposedFunction(Function, list):
    """compose an arbitrary number of functions.

    A class instance is a list of functions. Calling the instance executes
    the composition of these functions (evaluating from right to left as
    in math notation). Functions can be added to or removed from the list
    at any time with the obvious effect. To remain consistent (if needed),
    the ``list_of_inverses`` attribute must be updated respectively.

    >>> import numpy as np
    >>> from cma.fitness_transformations import ComposedFunction
    >>> f1, f2, f3, f4 = lambda x: 2*x, lambda x: x**2, lambda x: 3*x, lambda x: x**3
    >>> f = ComposedFunction([f1, f2, f3, f4])
    >>> assert isinstance(f, list) and isinstance(f, ComposedFunction)
    >>> assert f[0] == f1  # how I love Python indexing
    >>> assert all(f(x) == f1(f2(f3(f4(x)))) for x in np.random.rand(10))
    >>> assert f4 == f.pop()
    >>> assert len(f) == 3
    >>> f.insert(1, f4)
    >>> f.append(f4)
    >>> assert all(f(x) == f1(f4(f2(f3(f4(x))))) for x in range(5))

    A more specific example:

    >>> from cma.fitness_transformations import ComposedFunction
    >>> from cma.constraints_handler import BoundTransform
    >>> from cma import ff
    >>> f = ComposedFunction([ff.elli,
    ...                       BoundTransform([[0], [1]]).transform])
    >>> assert max(f([2, 3]), f([1, 1])) <= ff.elli([1, 1])

    Details:

    - This class can serve as basis for a more transparent
      alternative to a ``scaling_of_variables`` CMA option or for any
      necessary transformation of the fitness/objective function
      (genotype-phenotype transformation).

    - The parallelizing call with a list of solutions of the `Function`
      class is not inherited. The inheritence from `Function` is rather
      declarative than funtional and could be omitted.

    """
    def __init__(self, list_of_functions, list_of_inverses=None):
        """Caveat: to remain consistent, the ``list_of_inverses`` must be
        updated explicitly, if the list of function was updated after
        initialization.
        """
        list.__init__(self, list_of_functions)
        Function.__init__(self)
        self.list_of_inverses = list_of_inverses

    def __call__(self, x, *args, **kwargs):
        Function.__call__(self, x, *args, **kwargs)  # for the possible side effects only
        for i in range(-1, -len(self) - 1, -1):
            x = self[i](x, *args, **kwargs)
        return x

    def inverse(self, x, *args, **kwargs):
        """evaluate the composition of inverses on ``x``.

        Return `None`, if no list was provided.
        """
        if self.list_of_inverses is None:
            utils.print_warning("inverses were not given")
            return
        for i in range(len(self.list_of_inverses)):
            x = self.list_of_inverses[i](x, *args, **kwargs)
        return x

class StackFunction(Function):
    """a function that returns ``f1(x[:n1]) + f2(x[n1:])``.

    >>> import functools
    >>> import numpy as np
    >>> import cma
    >>> def elli48(x):
    ...     return 1e-4 * functools.partial(cma.ff.elli, cond=1e8)(x)
    >>> fcigtab = cma.fitness_transformations.StackFunction(
    ...     elli48, cma.ff.sphere, 2)
    >>> x = [1, 2, 3, 4]
    >>> assert np.isclose(fcigtab(x), cma.ff.cigtab(np.asarray(x)))

"""
    def __init__(self, f1, f2, n1):
        self.f1 = f1
        self.f2 = f2
        self.n1 = n1
    def _eval(self, x, *args, **kwargs):
        return self.f1(x[:self.n1], *args, **kwargs) + self.f2(x[self.n1:], *args, **kwargs)

class GlueArguments(Function):
    """deprecated, use `functools.partial` or
    `cma.fitness_transformations.partial` instead, which has the same
    functionality and interface.

    from a `callable` return a `callable` with arguments attached.


    An ellipsoid function with condition number ``1e4`` is created by
    ``felli1e4 = cma.s.ft.GlueArguments(cma.ff.elli, cond=1e4)``.

    >>> import cma
    >>> f = cma.fitness_transformations.GlueArguments(cma.ff.elli,
    ...                                               cond=1e1)
    >>> assert f([1, 2]) == 1**2 + 1e1 * 2**2

    """
    def __init__(self, fitness_function, *args, **kwargs):
        """define function, ``args``, and ``kwargs``.

        ``args`` are appended to arguments passed in the call, ``kwargs``
        are updated with keyword arguments passed in the call.
        """
        warnings.warn("GlueArguments is deprecated.\n"
                      "Use `functools.partial` (`cma.fitness_transformations.partial`) instead.",
                      DeprecationWarning)
        Function.__init__(self, fitness_function)
        self.fitness_function = fitness_function  # never used
        self.args = args
        self.kwargs = kwargs
    def __call__(self, x, *args, **kwargs):
        """call function with at least one additional argument and
        attached args and kwargs.
        """
        joined_kwargs = dict(self.kwargs)
        joined_kwargs.update(kwargs)
        x = np.asarray(x)
        return Function.__call__(self, x, *(args + self.args),
                                 **joined_kwargs)

class FBoundTransform(ComposedFunction):
    """shortcut for ``ComposedFunction([f, BoundTransform(bounds).transform])``,
    see also below.

    Maps the argument into bounded or half-bounded (feasible) domain
    before evaluating ``f``.

    Example with lower bound at 0, which becomes the image of -0.05 in
    `BoundTransform.transform`:

    >>> import cma, numpy as np
    >>> f = cma.fitness_transformations.FBoundTransform(cma.ff.elli,
    ...                                                 [[0], None])
    >>> assert all(f[1](np.random.randn(200)) >= 0)
    >>> assert all(f[1]([-0.05, -0.05]) == 0)
    >>> assert f([-0.05, -0.05]) == 0

    A slightly more verbose version to implement the lower bound at zero
    in the very same way:

        >>> import cma
        >>> felli_in_bound = cma.s.ft.ComposedFunction(
        ...    [cma.ff.elli, cma.BoundTransform([[0], None]).transform])

    """
    def __init__(self, fitness_function, bounds):
        """`bounds[0]` are lower bounds, `bounds[1]` are upper bounds
        """
        self.bound_tf = BoundTransform(bounds)  # not strictly necessary
        ComposedFunction.__init__(self,
                    [fitness_function, self.bound_tf.transform])

class Rotated(ComposedFunction):
    """return a rotated version of a function for testing purpose.

    This class is a convenience shortcut for the litte more verbose
    composition of a function with a rotation:

    >>> import cma
    >>> from cma import fitness_transformations as ft
    >>> f1 = ft.Rotated(cma.ff.elli)
    >>> f2 = ft.ComposedFunction([cma.ff.elli, ft.Rotation()])
    >>> assert f1([2]) == f2([2])  # same rotation only in 1-D
    >>> assert f1([1, 2]) != f2([1, 2])

    """
    def __init__(self, f, rotate=None, seed=None):
        """optional argument ``rotate(x)`` must return a (stable) rotation
        of ``x``.
        """
        if rotate is None:
            rotate = Rotation(seed=seed)
        ComposedFunction.__init__(self, [f, rotate])

class Shifted(ComposedFunction):
    """compose a function with a shift in x-space.

    >>> import cma
    >>> f = cma.s.ft.Shifted(cma.ff.elli)

    Details: this class solely provides as default second argument to
    `ComposedFunction`, namely a random shift in search space.
    ``shift=lambda x: x`` would provide "no shift", ``None``
    expands to ``cma.transformations.ConstRandnShift()``.
    """
    def __init__(self, f, shift=None):
        """``shift(x)`` must return a (stable) shift of x"""
        if shift is None:
            shift = ConstRandnShift()
        ComposedFunction.__init__(self, [f, shift])

class ScaleCoordinates(ComposedFunction):
    """compose a (fitness) function with a preceding scaling and offset.

    Scaling interface
    -----------------
    After ``fun2 = cma.ScaleCoordinates(fun, multipliers, zero)``, we have
    ``fun2(x) == fun(multipliers * (x - zero))``, where the vector size of
    `multipliers` and `zero` is adapated to the size of `x`, in case by
    recycling their last entry. This awkwardly asks to pass the `zero`
    argument of the preimage space where it has little meaning. Hence more
    conveniently,
    ``fun2 = cma.ScaleCoordinates(fun, multipliers, lower=lower)`` gets
    ``fun2(x) == fun(multipliers * x + lower)``.

    Domain interface (lower and upper variable values)
    --------------------------------------------------
    Let ``u`` and ``l`` be vectors (or a scalar) of (approximate) lower and
    upper variable values, respectively. After
    ``fun2 = cma.ScaleCoordinates(fun, upper=u, lower=l)`` we have
    ``fun2(x) == fun(l + (u - l) * x)``. Now, passing 0 as ``x[i]`` to
    ``fun2`` evaluates ``fun`` at ``l[i]`` while passing 1 evaluates
    ``fun`` at ``u[i]``.

    To match the size of ``x``, the sizes of ``u`` and ``l`` are shortened
    or their last entry is recycled if necessary.

    The default value for `lower` is zero in which case `upper` just
    becomes a scaling multiplier.

    Bounding the search domain of ``fun2`` to ``[0, 1]`` now bounds ``fun``
    to the domain ``[l, u]``. The ``'bounds'`` option of `CMAOptions`
    allows to set these bounds.

    More general, the affine transformation is defined such that
    ``x[i]==from_lower_upper[0]`` evaluates ``fun`` at ``l[i]`` and
    ``x[i]==from_lower_upper[1]`` evaluates ``fun`` at ``u[i]`` where
    ``from_lower_upper == [0, 1]`` by default.

    Examples and Doctest
    --------------------

    >>> import numpy as np
    >>> import cma
    >>> fun = cma.ScaleCoordinates(cma.ff.sphere, upper=[30, 1])
    >>> bool(fun([1, 1]) == 30**2 + 1**2)
    True
    >>> fun.transform([1, 1]).tolist(), fun.transform([0.2, 0.2]).tolist()
    ([30.0, 1.0], [6.0, 0.2])
    >>> fun.inverse(fun.transform([0.1, 0.3])).tolist()
    [0.1, 0.3]
    >>> fun = cma.ScaleCoordinates(cma.ff.sphere, upper=[31, 3], lower=[1, 2])
    >>> bool(-1e-9 < fun([1, -1]) - (31**2 + 1**2) < 1e-9)
    True
    >>> f = cma.ScaleCoordinates(cma.ff.sphere, [100, 1])
    >>> assert f[0] == cma.ff.sphere  # first element of f-composition
    >>> assert f(range(1, 6)) == 100**2 + sum([x**2 for x in range(2, 6)])
    >>> assert f([2.1]) == 210**2 == f(2.1)
    >>> assert f(20 * [1]) == 100**2 + 19
    >>> assert np.all(f.inverse(f.scale_and_offset([1, 2, 3, 4])) ==
    ...               np.asarray([1, 2, 3, 4]))
    >>> f = cma.ScaleCoordinates(f, [-2, 7], [2, 3, 4]) # last is recycled
    >>> bool(f([5, 6]) == sum(x**2 for x in [100 * -2 * (5 - 2), 7 * (6 - 3)]))
    True

    See also these [Practical Hints](https://cma-es.github.io/cmaes_sourcecode_page.html#practical)
    for encoding variables.
    """
    def __init__(self, fitness_function, multipliers=None, zero=None,
                 upper=None, lower=None, from_lower_upper=(0, 1)):
        """
        :param fitness_function: a `callable` object
        :param multipliers: coordinate-wise multipliers.
        :param zero: defines a new zero in preimage space, that is,
            calling the `ScaleCoordinates` instance returns
            ``fitness_function(multipliers * (x - zero))``.
        :param upper: variable value to which from_lower_upper[1] maps.
        :param lower: variable value to which from_lower_upper[0] maps.

        Only either `multipliers` or 'upper` can be passed. If `zero` is
        passed then `upper` and `lower` are ignored. The arguments
        ``multipliers``, ``zero``, ``upper`` and ``lower`` can be vectors
        or scalars, superfluous trailing elements are ignored and the last
        element is recycled if needed to fit the length of the later given
        input.

        `from_lower_upper` is `(0, 1)` by default and defines the preimage
        values which are mapped to `lower` and `upper`, respectively
        (unless `multipliers` or `zero` are given). These two preimage
        values are always the same for all coordinates.

        Details
        -------
        The `upper` and `lower` and `from_lower_upper` parameters are used
        to assign `multipliers` and `zero` such that the transformation is
        then always computed from the latter two only.
        """
        ComposedFunction.__init__(self,
                [fitness_function, self.scale_and_offset])
        self.transform = self.scale_and_offset  # a useful alias
        self.multiplier = multipliers
        self.zero = zero
        # the following settings are only used in __init__:
        self.lower = lower
        self.upper = upper
        self.from_lower_upper = from_lower_upper

        def align(a1, a2):
            """align shorter to longer array such that a1*a2 doesn't bail"""
            try:
                l1, l2 = len(a1), len(a2)
            except: pass
            else:
                if l1 < l2:
                    a1 = utils.recycled(a1, as_=a2)
                elif l2 < l1:
                    a2 = utils.recycled(a2, as_=a1)
            return a1, a2

        if multipliers is not None:
            if upper is not None:
                raise ValueError('Either `multipliers` or `upper` argument must be None')
            if from_lower_upper not in (None, (0, 1)):
                warnings.warn("from_lower_upper={0} ignored because multipliers={1} were given"
                              .format(from_lower_upper, multipliers))
            self.multiplier = np.asarray(self.multiplier, dtype=float)
            if zero is None and lower is not None:
                self.multiplier, self.lower = align(self.multiplier, self.lower)
                self.zero = - np.asarray(self.lower) / self.multiplier
        elif zero is None:
            if upper is None and lower is None:
                raise ValueError('Either `multipliers` or `zero` or `upper` or `lower`'
                                 ' argument must be given.')
            if from_lower_upper is None:
                self.from_lower_upper = (0, 1)
            if lower is None:
                self.lower = 0
            if upper is None:
                self.upper = np.asarray(self.lower) + 1
            idx = np.where(np.asarray(self.upper, dtype=float) - self.lower <= 0)[0]
            if len(idx):
                raise ValueError('`upper` value(s) must be stricly larger than'
                                    ' `lower` value(s); values were:'
                                 '\n upper={0}'
                                 '\n lower={1}'
                                 '\n offending indices = {2}'
                                 .format(self.upper, self.lower, list(idx)))
            self.lower, self.upper = align(self.lower, self.upper)

            dx = self.from_lower_upper[1] - self.from_lower_upper[0]
            self.multiplier = (np.asarray(self.upper, dtype=float) - self.lower) / dx
            self.zero = self.from_lower_upper[0] - self.multiplier**-1 * self.lower
            zero_from_upper = self.from_lower_upper[1] - self.multiplier**-1 * self.upper
            if not _Mh.vequals_approximately(self.zero, zero_from_upper):
                warnings.warn('zero computed from upper and lower differ'
                              '\n from upper={0}'
                              '\n from lower={1}'
                              '\n This may be a bug or due to small numerical deviations'
                              .format(zero_from_upper, self.zero))
        if zero is not None:
            self.zero = np.asarray(zero, dtype=float)
            if lower is not None:
                warnings.warn("lower={0} is ignored because zero={1} was given"
                              .format(lower, zero))
            if upper is not None:
                warnings.warn("upper={0} is ignored because zero={1} was given"
                              .format(upper, zero))

    def scale_and_offset(self, x):
        x = np.asarray(x)
        def r(vec):
            return utils.recycled(vec, as_=x)
        if self.zero is not None and self.multiplier is not None:
            x = r(self.multiplier) * (x - r(self.zero))
        elif self.zero is not None:
            x = x - r(self.zero)
        elif self.multiplier is not None:
            x = r(self.multiplier) * x
        return x

    def inverse(self, x):
        """inverse of coordinate-wise affine transformation
        ``y / multipliers + zero``
        """
        x = np.asarray(x)
        def r(vec):
            return utils.recycled(vec, as_=x)
        if self.zero is not None and self.multiplier is not None:
            x = x / r(self.multiplier) + r(self.zero)
        elif self.zero is not None:
            x = x + r(self.zero)
        elif self.multiplier is not None:
            x = x / r(self.multiplier)
        return x

class FixVariables(ComposedFunction):
    """Insert variables with given values, thereby reducing the
    dimensionality of the resulting composed function.

    The constructor takes ``index_value_pairs``, a `dict` or `list` of
    pairs, as input and returns a function with smaller preimage space
    than input function ``f``.

    Fixing variable 3 and 5 works like

        >>> from cma.fitness_transformations import FixVariables
        >>> index_value_pairs = [[2, 0.2], [4, 0.4]]
        >>> fun = FixVariables(cma.ff.elli, index_value_pairs)
        >>> fun[1](4 * [1]) == [ 1.,  1.,  0.2,  1.,  0.4, 1.]
        True

    Or starting from a given current solution in the larger space from
    which we pick the fixed values:

        >>> from cma.fitness_transformations import FixVariables
        >>> current_solution = [0.1 * i for i in range(5)]
        >>> fixed_indices = [2, 4]
        >>> index_value_pairs = [[i, current_solution[i]]  # fix these
        ...                                     for i in fixed_indices]
        >>> fun = FixVariables(cma.ff.elli, index_value_pairs)
        >>> fun[1](4 * [1]) == [ 1.,  1.,  0.2,  1.,  0.4, 1.]
        True
        >>> assert (current_solution ==  # list with same values
        ...            fun.transform(fun.insert_variables(current_solution)))
        >>> assert (current_solution ==  # list with same values
        ...            fun.insert_variables(fun.transform(current_solution)))

    Details: this might replace the ``fixed_variables`` option in
    `CMAOptions` in future, but hasn't been thoroughly tested yet.

    Supersedes `ExpandSolution`.

    """
    def __init__(self, f, index_value_pairs):
        """return `f` with reduced dimensionality.

        ``index_value_pairs``:
            variables
        """
        # super(FixVariables, self).__init__(
        ComposedFunction.__init__(self, [f, self.insert_variables])
        self.index_value_pairs = dict(index_value_pairs)
    def transform(self, x):
        """transform `x` such that it could be used as argument to `self`.

        Return a list or array, usually dismissing some elements of
        `x`. ``fun.transform`` is the inverse of
        ``fun.insert_variables == fun[1]``, that is
        ``np.all(x == fun.transform(fun.insert_variables(x))) is True``.
        """
        res = [x[i] for i in range(len(x))
                if i not in self.index_value_pairs]
        return res if isinstance(x, list) else np.asarray(res)
    def insert_variables(self, x):
        """return `x` with inserted fixed values"""
        if len(self.index_value_pairs) == 0:
            return x
        y = list(x)
        for i in sorted(self.index_value_pairs):
            y.insert(i, self.index_value_pairs[i])
        if not isinstance(x, list):
            y = np.asarray(y)  # doubles the necessary time
        return y

class Expensify(Function):
    """Add waiting time to each evaluation, to simulate "expensive"
    behavior"""
    def __init__(self, callable_, time=1):
        """add time in seconds"""
        Function.__init__(self)  # callable_ could go here
        self.time = time
        self.callable = callable_
    def __call__(self, *args, **kwargs):
        time.sleep(self.time)
        Function.__call__(self, *args, **kwargs)
        return self.callable(*args, **kwargs)

class SomeNaNFitness(Function):
    """transform ``fitness_function`` to return sometimes ``NaN``"""
    def __init__(self, fitness_function, probability_of_nan=0.1):
        Function.__init__(self)
        self.fitness_function = fitness_function
        self.p = probability_of_nan
    def __call__(self, x, *args):
        Function.__call__(self, x, *args)
        if np.random.rand(1) <= self.p:
            return np.nan
        else:
            return self.fitness_function(x, *args)

class NoisyFitness(Function):
    """apply noise via ``f += rel_noise(dim) * f + abs_noise(dim)``"""
    def __init__(self, fitness_function,
                 rel_noise=lambda dim: 1.1 * np.random.randn() / dim,
                 abs_noise=lambda dim: 1.1 * np.random.randn()):
        """attach relative and absolution noise to ``fitness_function``.

        Relative noise is by default computed using the length of the
        input argument to ``fitness_function``. Both noise functions take
        ``dimension`` as input.

        >>> import cma
        >>> from cma.fitness_transformations import NoisyFitness
        >>> fn = NoisyFitness(cma.ff.elli)
        >>> assert fn([1, 2]) != cma.ff.elli([1, 2])
        >>> assert fn.evaluations == 1

        """
        Function.__init__(self, fitness_function)
        self.rel_noise = rel_noise
        self.abs_noise = abs_noise

    def __call__(self, x, *args):
        f = Function.__call__(self, x, *args)
        if self.rel_noise:
            f += f * self.rel_noise(len(x))
            assert np.isscalar(f)
        if self.abs_noise:
            f += self.abs_noise(len(x))
        return f

class IntegerMixedFunction2(ComposedFunction):
    """compose fitness function with some integer variables using `np.round` by default.

    >>> import numpy as np
    >>> import cma
    >>> f = cma.s.ft.IntegerMixedFunction2(cma.ff.elli, [0, 3, 5])
    >>> assert f([-0.2, 2]) == f([0.4, 2]) != f([0.8, 2])
    >>> f = cma.s.ft.IntegerMixedFunction2(cma.ff.elli, [0])
    >>> assert f([-0.2, 2]) == f(np.array([0.4, 2])) != f(np.array([0.8, 2]))

    Related: Option ``'integer_variables'`` of `cma.CMAOptions` sets
    ``'minstd'`` of integer variables, see
    `cma.options_parameters.integer_std_lower_bound` and rounds the better
    solutions, see `cma.integer_centering`.
    """
    def __init__(self, function, integer_variable_indices, operator=np.round, copy=True):
        """apply operator(x[i]) for i in integer_variable_indices before to call function(x).

        If `copy`, return a copy iff a value is changed.
        """
        ComposedFunction.__init__(self, [function, self._flatten])
        self.integer_variable_indices = integer_variable_indices
        self.operator = operator
        self.copy = copy
    def _flatten2(self, x):
        values = x[self.integer_variable_indices]
        new_values = np.round(values)
        if not np.all(new_values == values):
            if self.copy:
                x = np.array(x, copy=True)
            x[self.integer_variable_indices] = new_values
        return x
    def _flatten(self, x):
        if isinstance(x, np.ndarray):
            return self._flatten2(x)  # hopefully faster
        copied = False
        for i in sorted(self.integer_variable_indices):
            if i < -len(x):
                continue
            if i >= len(x):
                break
            m = self.operator(x[i])
            if x[i] != m:
                if not copied and self.copy:
                    x = np.array(x, copy=True)
                    copied = True
                x[i] = m
        return x

class IntegerMixedFunction(ComposedFunction):
    """DEPRECATED compose fitness function with some integer variables using `np.floor` by default.

    >>> import cma
    >>> f = cma.s.ft.IntegerMixedFunction(cma.ff.elli, [0, 3, 6])
    >>> assert f([0.2, 2]) == f([0.4, 2]) != f([1.2, 2])

    It is advisable to set minstd of integer variables to
    ``1 / (2 * len(integer_variable_indices) + 1)``, in which case in
    an independent model at least 33% (1 integer variable) -> 39% (many
    integer variables) of the solutions should have an integer mutation
    on average. Option ``integer_variables`` of `cma.CMAOptions`
    implements this simple measure.
    """
    def __init__(self, function, integer_variable_indices, operator=np.floor, copy_arg=True):
        """apply operator(x[i]) for i in integer_variable_indices before to call function(x)"""
        ComposedFunction.__init__(self, [function, self._flatten])
        self.integer_variable_indices = integer_variable_indices
        self.operator = operator
        self.copy_arg = copy_arg
    def _flatten(self, x):
        if self.copy_arg:
            x = np.array(x, copy=True)
        else:
            x = np.asarray(x)
        for i in sorted(self.integer_variable_indices):
            if i < -len(x):
                continue
            if i >= len(x):
                break
            x[i] = self.operator(x[i])
        return x
