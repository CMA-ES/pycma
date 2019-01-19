"""Wrapper for objective functions like noise, rotation, gluing args
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals, with_statement
from functools import partial
import numpy as np
from multiprocessing import Pool as ProcessingPool
# from pathos.multiprocessing import ProcessingPool
import time
from .utilities import utils
from .transformations import ConstRandnShift, Rotation
from .constraints_handler import BoundTransform
from .utilities.python3for2 import range
del absolute_import, division, print_function  #, unicode_literals, with_statement

rotate = Rotation()

class EvalParallel(object):
    """A class and context manager for parallel evaluations.

    To be used with the `with` statement (otherwise `terminate` needs to
    be called to free resources)::

        with EvalParallel() as eval_all:
            fvals = eval_all(fitness, solutions)

    assigns a callable `EvalParallel` class instance to ``eval_all``.
    The instance can be called with a `list` (or `tuple` or any
    sequence) of solutions and returns their fitness values. That is::

        eval_all(fitness, solutions) == [fitness(x) for x in solutions]

    `EvalParallel.__call__` may take two additional optional arguments,
    namely `args` passed to ``fitness`` and `timeout` passed to the
    `multiprocessing.pool.ApplyResult.get` method which raises
    `multiprocessing.TimeoutError` in case.

    Examples:

    >>> import cma
    >>> from cma.fitness_transformations import EvalParallel
    >>> # class usage, don't forget to call terminate
    >>> ep = EvalParallel()
    >>> ep(cma.fitness_functions.elli, [[1,2], [3,4], [4, 5]])  # doctest:+ELLIPSIS
    [4000000.944...
    >>> ep.terminate()
    ...
    >>> # use with `with` statement (context manager)
    >>> es = cma.CMAEvolutionStrategy(3 * [1], 1, dict(verbose=-9))
    >>> with EvalParallel(12) as eval_all:
    ...     while not es.stop():
    ...         X = es.ask()
    ...         es.tell(X, eval_all(cma.fitness_functions.elli, X))
    >>> assert es.result[1] < 1e-13 and es.result[2] < 1500

    Parameters: the `EvalParallel` constructor takes the number of
    processes as optional input argument, which is by default
    ``multiprocessing.cpu_count()``.
    
    Limitations: The `multiprocessing` module (on which this class is
    based upon) does not work with class instance method calls.
    
    In some cases the execution may be considerably slowed down,
    as for example with test suites from coco/bbob.

    """
    def __init__(self, number_of_processes=None):
        self.processes = number_of_processes
        self.pool = ProcessingPool(self.processes)

    def __call__(self, fitness_function, solutions, args=(), timeout=None):
        """evaluate a list/sequence of solution-"vectors", return a list
        of corresponding f-values.

        Raises `multiprocessing.TimeoutError` if `timeout` is given and
        exceeded.
        """
        warning_str = ("WARNING: `fitness_function` must be a function,"
                       " not an instancemethod, to work with"
                       " `multiprocessing`")
        if isinstance(fitness_function, type(self.__init__)):
            print(warning_str)
        jobs = [self.pool.apply_async(fitness_function, (x,) + args)
                for x in solutions]
        try:
            return [job.get(timeout) for job in jobs]
        except:
            print(warning_str)
            raise

    def terminate(self):
        """free allocated processing pool"""
        # self.pool.close()  # would wait for job termination
        self.pool.terminate()  # terminate jobs regardless
        self.pool.join()  # end spawning

    def __enter__(self):
        # we could assign self.pool here, but then `EvalParallel` would
        # *only* work when using the `with` statement
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def __del__(self):
        """though generally not recommended `__del__` should be OK here"""
        self.terminate()


class EvalParallel2(object):
    """A class and context manager for parallel evaluations.

    This class is based on the ``Pool`` class of the `multiprocessing` module.
    
    The interface in v2 changed, such that the fitness function can be
    given once in the constructor. Hence the number of processes has 
    become the second (optional) argument of `__init__` and the function
    has become the second and optional argument of `__call__`.

    To be used with the `with` statement (otherwise `terminate` needs to
    be called to free resources)::

        with EvalParallel2(fitness_function) as eval_all:
            fvals = eval_all(solutions)

    assigns a callable `EvalParallel2` class instance to ``eval_all``.
    The instance can be called with a `list` (or `tuple` or any
    sequence) of solutions and returns their fitness values. That is::

        eval_all(solutions) == [fitness_function(x) for x in solutions]

    `EvalParallel2.__call__` may take three additional optional arguments,
    namely `fitness_function` (like this the function may change from call
    to call), `args` passed to ``fitness`` and `timeout` passed to the
    `multiprocessing.pool.ApplyResult.get` method which raises
    `multiprocessing.TimeoutError` in case.

    Examples:

    >>> import cma
    >>> from cma.fitness_transformations import EvalParallel2
    >>> # class usage, don't forget to call terminate
    >>> ep = EvalParallel2(cma.fitness_functions.elli, 4)
    >>> ep([[1,2], [3,4], [4, 5]])  # doctest:+ELLIPSIS
    [4000000.944...
    >>> ep.terminate()
    ...
    >>> # use with `with` statement (context manager)
    >>> es = cma.CMAEvolutionStrategy(3 * [1], 1, dict(verbose=-9))
    >>> with EvalParallel2(number_of_processes=12) as eval_all:
    ...     while not es.stop():
    ...         X = es.ask()
    ...         es.tell(X, eval_all(X, cma.fitness_functions.elli))
    >>> assert es.result[1] < 1e-13 and es.result[2] < 1500

    Parameters: the `EvalParallel2` constructor takes the number of
    processes as optional input argument, which is by default
    ``multiprocessing.cpu_count()``.

    Limitations: as of 2018, the `multiprocessing` module (on which
    this class is based upon) does not work with class instance method
    calls.

    In some cases the execution may be considerably slowed down,
    as for example with test suites from coco/bbob.

    """
    def __init__(self, fitness_function=None, number_of_processes=None):
        self.fitness_function = fitness_function
        self.processes = number_of_processes
        self.pool = ProcessingPool(self.processes)

    def __call__(self, solutions, fitness_function=None, args=(), timeout=None):
        """evaluate a list/sequence of solution-"vectors", return a list
        of corresponding f-values.

        Raises `multiprocessing.TimeoutError` if `timeout` is given and
        exceeded.
        """
        fitness_function = fitness_function or self.fitness_function
        if fitness_function is None:
            raise ValueError("`fitness_function` was never given, must be"
                             " passed in `__init__` or `__call__`")
        warning_str = ("WARNING: `fitness_function` must be a function,"
                       " not an instancemethod, in order to work with"
                       " `multiprocessing`")
        if isinstance(fitness_function, type(self.__init__)):
            warnings.warn(warning_str)
        jobs = [self.pool.apply_async(fitness_function, (x,) + args)
                for x in solutions]
        try:
            return [job.get(timeout) for job in jobs]
        except:
            warnings.warn(warning_str)
            raise

    def terminate(self):
        """free allocated processing pool"""
        # self.pool.close()  # would wait for job termination
        self.pool.terminate()  # terminate jobs regardless
        self.pool.join()  # end spawning

    def __enter__(self):
        # we could assign self.pool here, but then `EvalParallel2` would
        # *only* work when using the `with` statement
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def __del__(self):
        """though generally not recommended `__del__` should be OK here"""
        self.terminate()


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
            x = self[i](x)
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

class GlueArguments(Function):
    """from a `callable` return a `callable` with arguments attached.

    See also `functools.partial` which has the same functionality and
    interface.

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
    """compose a (fitness) function with a scaling for each variable
    (more concisely, a coordinate-wise affine transformation).

    >>> import numpy as np
    >>> import cma
    >>> f = cma.ScaleCoordinates(cma.ff.sphere, [100, 1])
    >>> assert f[0] == cma.ff.sphere  # first element of f-composition
    >>> assert f(range(1, 6)) == 100**2 + sum([x**2 for x in range(2, 6)])
    >>> assert f([2.1]) == 210**2 == f(2.1)
    >>> assert f(20 * [1]) == 100**2 + 19
    >>> assert np.all(f.inverse(f.scale_and_offset([1, 2, 3, 4])) ==
    ...               np.asarray([1, 2, 3, 4]))
    >>> f = cma.ScaleCoordinates(f, [-2, 7], [2, 3, 4]) # last is recycled
    >>> f([5, 6]) == sum(x**2 for x in [100 * -2 * (5 - 2), 7 * (6 - 3)])
    True

    """
    def __init__(self, fitness_function, multipliers=None, zero=None):
        """
        :param fitness_function: a `callable` object
        :param multipliers: coordinate-wise multipliers.
        :param zero: defines a new zero in preimage space, that is,
            calling the `ScaleCoordinates` instance returns
            ``fitness_function(multipliers * (x - zero))``.

        For both arguments, ``multipliers`` and ``zero``, to fit in
        case the length of the given input, superfluous trailing
        elements are ignored or the last element is recycled.
        """
        ComposedFunction.__init__(self,
                [fitness_function, self.scale_and_offset])
        self.multiplier = multipliers
        if self.multiplier is not None:
            self.multiplier = np.asarray(self.multiplier, dtype=float)
        self.zero = zero
        if zero is not None:
            self.zero = np.asarray(zero, dtype=float)

    def scale_and_offset(self, x):
        x = np.asarray(x)
        r = lambda vec: utils.recycled(vec, as_=x)
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
        r = lambda vec: utils.recycled(vec, as_=x)
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
            return np.NaN
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

class IntegerMixedFunction(ComposedFunction):
    """compose fitness function with some integer variables.

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
    def __init__(self, function, integer_variable_indices, copy_arg=True):
        ComposedFunction.__init__(self, [function, self._flatten])
        self.integer_variable_indices = integer_variable_indices
        self.copy_arg = copy_arg
    def _flatten(self, x):
        x = np.array(x, copy=self.copy_arg)
        for i in sorted(self.integer_variable_indices):
            if i < -len(x):
                continue
            if i >= len(x):
                break
            x[i] = np.floor(x[i])
        return x
