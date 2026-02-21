"""Compact GA minimal implementation"""

from __future__ import division, print_function
import numpy as np  # to replace 'inf' with np.inf and vice versa


class CompactGA(object):
    """Randomized search on the binary domain {0,1}^n.

    cGA samples two different solutions per iteration.

    Minimal example::

        dimension = 20
        opt = CompactGA(dimension * [0.5])  # initialise in domain middle
        while opt.best.f > 0 and not opt.stop():
            opt.optimize(sum, 1)
        print("%d: fbest=%f" % (opt.evaluation, opt.best.f))

    finds argmin_{x in {0, 1}**20} sum(x) in ``opt.result``.

    Details: due to the incrememental update, cGA is expected to show inferior
    scaling with dimension on LeadingOnes and/or BinVal.

    Reference: Harik et al 1999.
    """

    def __init__(self, mean_):
        """takes as input the initial vector of probabilities to sample 1 vs 0.

        The vector length defines the dimension. Initial values of the
        probability vector, which is also the distribution mean, are
        usually chosen to be 0.5.
        """
        self.mean = np.asarray(mean_)
        self.dimension = len(self.mean)
        self.eta = 1 / self.dimension  # 1 / dim according to Harik et al 1999
        self.lower_p = 1 / self.dimension / 2
        """lower and 1 - upper bound for mean, avoid getting stuck "forever" """
        # bookkeeping
        self.best = BestSolution()
        self.fcurrent = np.nan
        self.evaluation = 0

    def ask(self):
        """return two candidate solutions"""
        x1 = np.random.rand(self.dimension) < self.mean
        x2 = np.random.rand(self.dimension) < self.mean
        while all(x2 == x1):
            x2 = np.random.rand(self.dimension) <= self.mean
        return [np.array(x1, np.int8), np.array(x2, np.int8)]

    def tell(self, solutions, fitness_values):
        """update ``self.mean``"""
        assert len(solutions) == len(fitness_values) == 2
        self.evaluation += 2
        x1, x2 = solutions
        f1, f2 = fitness_values
        if f2 < f1:  # swap such that f1 is better
            f1, f2 = f2, f1
            x1, x2 = x2, x1
        if f1 < f2:  # update mean
            self.mean += self.eta * (x1 - x2)
            self.mean[self.mean < self.lower_p] = self.lower_p
            self.mean[self.mean > 1 - self.lower_p] = 1 - self.lower_p
        # bookkeeping
        self.fcurrent = f1
        self.best.update(f1, x1, self.evaluation)

    def optimize(self, fun, iterations):
        for _ in range(iterations):
            X = self.ask()
            self.tell(X, [fun(X[0]), fun(X[1])])
            if self.stop():
                break
        return self

    def stop(self):
        """dictionary containing satisfied termination conditions
        """
        stop_dict = {}
        self.stop_stagnation_evals = 2000 * self.dimension
        if self.evaluation > self.best.evaluation + self.stop_stagnation_evals:
            stop_dict['stagnation'] = self.stop_stagnation_evals
        return stop_dict

    @property
    def result(self):
        """for the time being `result` is the best seen solution
        """
        return self.best.x

class BestSolution(object):
    """Helper class to keep track of the best solution (minimization).

    All is stored in attributes ``x, f, evaluation``. The only reason
    for this class is to prevent code duplication of the `update`
    method.
    """
    def __init__(self):
        self.x = None
        self.f = np.inf
        self.evaluation = 0
    def update(self, f, x, evaluation):
        """update attributes ``f, x, evaluation`` if ``f < self.f``"""
        if f < self.f:
            self.f = f
            self.x = x[:]
            self.evaluation = evaluation

