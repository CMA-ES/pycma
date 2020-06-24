# -*- coding: utf-8 -*-
'''Interface wrappers for the `cma` module.

The `SkoptCMAoptimizer` wrapper interfaces an optimizer aligned with
`skopt.optimizer`.
'''
# built-in
import pdb
import copy
import inspect
import tempfile
import os
import warnings

# external
import numpy as np
import cma  # caveat: does not import necessarily the code of this root folder?

try: import skopt
except ImportError: warnings.warn('install `skopt` ("pip install scikit-optimize") '
                                  'to use `SkoptCMAoptimizer`')
else:
    def SkoptCMAoptimizer(
        func, dimensions, n_calls, verbose=False, callback=(), x0=None, n_jobs=1,
        sigma0=.5, normalize=True,
    ):
        '''
        Optmizer based on CMA-ES algorithm.
        This is essentially a wrapper fuction for the cma library function
        to align the interface with skopt library.

        Args:
            func (callable): function to optimize
            dimensions: list of tuples like ``4 * [(-1., 1.)]`` for defining the domain.
            n_calls: the number of samples.
            verbose: if this func should be verbose
            callback: the list of callback functions.
            n_jobs: number of cores to run different calls to `func` in parallel.
            x0: inital values
                if None, random point will be sampled
            sigma0: initial standard deviation relative to domain width
            normalize: whether optimization domain should be normalized

        Returns:
            `res` skopt.OptimizeResult object
            The optimization result returned as a dict object.
            Important attributes are:
            - `x` [list]: location of the minimum.
            - `fun` [float]: function value at the minimum.
            - `x_iters` [list of lists]: location of function evaluation for each
            iteration.
            - `func_vals` [array]: function value for each iteration.
            - `space` [skopt.space.Space]: the optimization space.

        Example::

            import cma.wrapper
            res = cma.wrapper.SkoptCMAoptimizer(lambda x: sum([xi**2 for xi in x]),
                                                2 * [(-1.,1.)], 55)
            res['cma_es'].logger.plot()

        '''
        specs = {
            'args': copy.copy(inspect.currentframe().f_locals),
            'function': inspect.currentframe().f_code.co_name,
        }

        if normalize: dimensions = list(map(lambda x: skopt.space.check_dimension(x, 'normalize'), dimensions))
        space = skopt.space.Space(dimensions)
        if x0 is None: x0 = space.transform(space.rvs())[0]
        else: x0 = space.transform([x0])[0]

        tempdir = tempfile.mkdtemp()
        xi, yi = [], []
        options = {
            'bounds': np.array(space.transformed_bounds).transpose().tolist(),
            'verb_filenameprefix': tempdir,
        }

        def delete_tempdir(self, *args, **kargs):
            os.removedirs(tempdir)
            return

        model = cma.CMAEvolutionStrategy(x0, sigma0, options)
        model.logger.__del__ = delete_tempdir
        switch = { -1: None,  # use number of available CPUs
                    1: 0,     # avoid using multiprocessor for just one CPU
                 }
        with cma.optimization_tools.EvalParallel2(func,
                number_of_processes=switch.get(n_jobs, n_jobs)) as parallel_func:
            for _i in range(n_calls):
                if model.stop(): break
                new_xi = model.ask()
                new_xi_denorm = space.inverse_transform(np.array(new_xi))
                # new_yi = [func(x) for x in new_xi_denorm]
                new_yi = parallel_func(new_xi_denorm)

                model.tell(new_xi, new_yi)
                model.logger.add()
                if verbose: model.disp()

                xi += new_xi_denorm
                yi += new_yi
                results = skopt.utils.create_result(xi, yi)
                for f in callback: f(results)

        results = skopt.utils.create_result(xi, yi, space)
        model.logger.load()
        results.cma_es = model
        results.cma_logger = model.logger
        results.specs = specs
        return results
