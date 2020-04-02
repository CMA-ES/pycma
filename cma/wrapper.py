'''
provides various optimizers.
The interfaces of optimizers provided by this module will align with
skopt.optimizers.
'''
# built-in
import pdb
import copy
import inspect
import tempfile
import os

# external
import numpy as np
import cma
from cma.logger import CMADataLogger
from skopt.utils import create_result
from skopt.space import Space
from skopt.space import check_dimension

def SkoptCMAoptimizer(
    func, dimensions, n_calls, verbose=False, callback=(), x0=None,
    sigma0=.5, normalize=True,
):
    '''
    Optmizer based on CMA-ES algorithm.
    This is essentially a wrapper fuction for the cma library function
    to align the interface with skopt library.

    Args:
        func (callable): function to optimize
        dimensions: list of tuples.  search dimensions
        n_calls: the number of samples.
        verbose: if this func should be verbose
        callback: the list of callback functions.
        x0: inital values
            if None, random point will be sampled
        sigma0: initial standard deviation
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
        - `space` [Space]: the optimization space.
    '''
    specs = {
        'args': copy.copy(inspect.currentframe().f_locals),
        'function': inspect.currentframe().f_code.co_name,
    }

    if normalize: dimensions = list(map(lambda x: check_dimension(x, 'normalize'), dimensions))
    space = Space(dimensions)
    if x0 is None: x0 = space.transform(space.rvs())[0]

    tempdir = tempfile.mkdtemp()
    xi, yi = [], []
    options = {
        'bounds': np.array(space.transformed_bounds).transpose().tolist(),
        'verb_filenameprefix': tempdir,
    }

    def delete_tempdir(self, *args, **kargs):
        os.removedirs(tempdir)
        return
    CMADataLogger.__del__ = delete_tempdir

    model = cma.CMAEvolutionStrategy(x0, sigma0, options)
    for i in range(n_calls):
        if model.stop(): break
        new_xi = model.ask()
        new_xi_denorm = space.inverse_transform(np.array(new_xi))
        new_yi = [func(x) for x in new_xi_denorm]

        model.tell(new_xi, new_yi)
        model.logger.add()
        if verbose: model.disp()

        xi += new_xi_denorm
        yi += new_yi
        results = create_result(xi, yi)
        for f in callback: f(results)

    results = create_result(xi, yi, space)
    results.cma_logger = model.logger
    results.specs = specs
    return results
