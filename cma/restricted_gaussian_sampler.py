"""VD-CMA and VkD-CMA

Usage examples, VD-CMA:

    >>> import cma
    >>> from cma import restricted_gaussian_sampler as rgs
    >>> es = cma.CMAEvolutionStrategy(20 * [1], 1,
    ...          rgs.GaussVDSampler.extend_cma_options({
    ...             'seed': 6,
    ...             'ftarget': 1e-8,
    ...             'verbose': -9,  # helpful for automatic testing
    ...     }))
    >>> es = es.optimize(cma.fitness_transformations.Rotated(cma.ff.cigar, seed=6), iterations=None)
    >>> assert es.result.fbest <= 1e-8
    >>> print(es.result.evaluations)
    6372

It is recommended to always use `extend_cma_options()` to set the options
appropriately, even when no other options are passed through.

    >>> len(rgs.GaussVDSampler.extend_cma_options())
    2
    >>> len(rgs.GaussVkDSampler.extend_cma_options())
    3

The use case for VkD-CMA looks identical:

    >>> es = cma.CMAEvolutionStrategy(20 * [1], 1,
    ...          rgs.GaussVkDSampler.extend_cma_options({
    ...             'seed': 7,
    ...             'ftarget': 1e-8,
    ...             'verbose': -9,  # helpful for automatic testing
    ...     }))
    >>> es = es.optimize(cma.fitness_transformations.Rotated(cma.ff.cigar, seed=3), iterations=None)
    >>> assert es.result.fbest <= 1e-8
    >>> print(es.result.evaluations)
    6204


TODO: correct the interface of __init__, remove unnecessaries

TODO:
2017/05/10: pass the option to sampler
2017/05/10: how to give sigma to update?

MEMO: 
2017/05/08: line 2958 of evolution_strategy.py: cc is assigned from sp.cc
2017/05/08: line 3021 of evolution_strategy.py: `weights` are multiplied by c1 and cmu
2017/05/08: line 3021 of evolution_strategy.py: first element of `vectors` is pc
2017/05/07: hsig interface
2017/05/07: `CMAAdaptSigmaNone` not working
2017/05/07: `dimension` passed to __init__ in not int.
2017/05/06: 'AdaptSigma = CMAAdaptSigmaTPA' won't work. AssertionError happens in `_update_ps`.
2017/05/06: `correlation_matrix` is not declared in `StatisticalModelSamplerWithZeroMeanBaseClass`. However, it is used in `evolution_strategy.py`.
2017/05/06: the following line of code in `ask_geno` assumes that the result of `sample` is an ndarray, rather than list. ary = self.sigma_vec * self.sm.sample(Niid)/

"""
import math
import warnings
import numpy as np
from .interfaces import StatisticalModelSamplerWithZeroMeanBaseClass


class GaussVDSampler(StatisticalModelSamplerWithZeroMeanBaseClass):
    """Restricted Gaussian Sampler for VD-CMA
    VD-CMA: Linear Time/Space Comparison-based Natural Gradient Optimization
    The covariance matrix is limited as C = D * (I + v*v^t) * D,
    where D is a diagonal, v is a vector.

    Reference
    ---------
    Youhei Akimoto, Anne Auger, and Nikolaus Hansen.
    Comparison-Based Natural Gradient Optimization in High Dimension.
    In Proc. of GECCO 2014, pp. 373 -- 380 (2014)
    """
    @staticmethod
    def extend_cma_options(opts=None):
        """return correct options to run `cma.fmin` or initialize
        `cma.CMAEvolutionStrategy` using the `GaussVDSampler` AKA VD-CMA-ES
        """
        opts = opts or {}
        opts.update({'CMA_active': False,
                     # 'AdaptSigma': None,  # not sure about that, False seems to work much worse
                     'CMA_sampler': GaussVDSampler})
        return opts

    def __init__(self, dimension, randn=np.random.randn, debug=False):
        """pass dimension of the underlying sample space
        """
        try:
            self.N = len(dimension)
            std_vec = np.array(dimension, copy=True)
        except TypeError:
            self.N = dimension
            std_vec = np.ones(self.N)
        if self.N < 10:
            print('Warning: Not advised to use VD-CMA for dimension < 10.')
        self.randn = randn
        self.dvec = std_vec
        self.vvec = self.randn(self.N) / math.sqrt(self.N)
        self.norm_v2 = np.dot(self.vvec, self.vvec)
        self.norm_v = np.sqrt(self.norm_v2)
        self.vn = self.vvec / self.norm_v
        self.vnn = self.vn**2
        self.pc = np.zeros(self.N)
        self._debug = debug  # plot covariance matrix

    def sample(self, number, update=None):
        """return list of i.i.d. samples.

        :param number: is the number of samples.
        :param update: controls a possibly lazy update of the sampler.
        """
        X = np.asarray(
            [self.transform(self.randn(self.N)) for i in range(number)])
        return X

    def update(self, vectors, weights, hsig=True):
        """``vectors`` is a list of samples, ``weights`` a corrsponding
        list of learning rates
        """

        ww = np.array(weights, copy=True)
        assert np.all(ww >= 0.0)
        ww = ww[1:] / np.sum(np.abs(ww[1:]))  # w[0] is the weight for pc
        cc, cone, cmu = self._get_params(ww)
        mu = np.sum(ww > 0, dtype=int)
        mueff = 1.0 / np.dot(ww, ww)
        idx = np.argsort(ww)[::-1]
        sary = np.asarray(vectors)[idx[:mu] + 1]
        w = ww[idx[:mu]]

        # Cumulation
        self.pc = (1. - cc) * self.pc + hsig * math.sqrt(cc * (
            2. - cc) * mueff) * np.dot(w, sary)

        # Alpha and related variables
        alpha, avec, bsca, invavnn = self._alpha_avec_bsca_invavnn(
            self.vnn, self.norm_v2)
        # Rank-mu
        if cmu == 0:
            pvec_mu = np.zeros(self.N)
            qvec_mu = np.zeros(self.N)
        else:
            pvec_mu, qvec_mu = self._pvec_and_qvec(self.vn, self.norm_v2,
                                                   sary / self.dvec, w)
        # Rank-one
        if cone == 0:
            pvec_one = np.zeros(self.N)
            qvec_one = np.zeros(self.N)
        else:
            pvec_one, qvec_one = self._pvec_and_qvec(self.vn, self.norm_v2,
                                                     self.pc / self.dvec)
        # Add rank-one and rank-mu before computing the natural gradient
        pvec = cmu * pvec_mu + hsig * cone * pvec_one
        qvec = cmu * qvec_mu + hsig * cone * qvec_one
        # Natural gradient
        if cmu + cone > 0:
            ngv, ngd = self._ngv_ngd(self.dvec, self.vn, self.vnn, self.norm_v,
                                     self.norm_v2, alpha, avec, bsca, invavnn,
                                     pvec, qvec)
            # truncation factor to guarantee at most 70 percent change
            upfactor = 1.0
            upfactor = min(upfactor,
                           0.7 * self.norm_v / math.sqrt(np.dot(ngv, ngv)))
            upfactor = min(upfactor, 0.7 * (self.dvec / np.abs(ngd)).min())
        else:
            ngv = np.zeros(self.N)
            ngd = np.zeros(self.N)
            upfactor = 1.0
        # Update parameters
        self.vvec += upfactor * ngv
        self.dvec += upfactor * ngd
        # update the constants
        self.norm_v2 = np.dot(self.vvec, self.vvec)
        self.norm_v = math.sqrt(self.norm_v2)
        self.vn = self.vvec / self.norm_v
        self.vnn = self.vn**2

    @staticmethod
    def _alpha_avec_bsca_invavnn(vnn, norm_v2):
        gamma = 1.0 / math.sqrt(1.0 + norm_v2)
        alpha = math.sqrt(norm_v2**2 + (1.0 + norm_v2) / max(vnn) * (
            2.0 - gamma)) / (2.0 + norm_v2)
        if alpha < 1.0:  # Compute beta = (1-alpha^2)*norm_v4/(1+norm_v2)
            beta = (4.0 - (2.0 - gamma) / max(vnn)) / (1.0 + 2.0 / norm_v2)**2
        else:
            alpha = 1.0
            beta = 0
        bsca = 2.0 * alpha**2 - beta
        avec = 2.0 - (bsca + 2.0 * alpha**2) * vnn
        invavnn = vnn / avec
        return alpha, avec, bsca, invavnn

    @staticmethod
    def _pvec_and_qvec(vn, norm_v2, y, weights=0):
        y_vn = np.dot(y, vn)
        if isinstance(weights, int) and weights == 0:
            pvec = y**2 - norm_v2 / (1.0 + norm_v2) * (y_vn * (y * vn)) - 1.0
            qvec = y_vn * y - ((y_vn**2 + 1.0 + norm_v2) / 2.0) * vn
        else:
            pvec = np.dot(weights, y**2 - norm_v2 / (1.0 + norm_v2) *
                          (y_vn * (y * vn).T).T - 1.0)
            qvec = np.dot(weights, (y_vn * y.T).T - np.outer(
                (y_vn**2 + 1.0 + norm_v2) / 2.0, vn))
        return pvec, qvec

    @staticmethod
    def _ngv_ngd(dvec, vn, vnn, norm_v, norm_v2, alpha, avec, bsca, invavnn,
                 pvec, qvec):
        rvec = pvec - alpha / (1.0 + norm_v2) * (
            (2.0 + norm_v2) * (qvec * vn) - norm_v2 * np.dot(vn, qvec) * vnn)
        svec = rvec / avec - bsca * np.dot(rvec, invavnn) / (
            1.0 + bsca * np.dot(vnn, invavnn)) * invavnn
        ngv = qvec / norm_v - alpha / norm_v * (
            (2.0 + norm_v2) * (vn * svec) - np.dot(svec, vnn) * vn)
        ngd = dvec * svec
        return ngv, ngd

    def _get_params2(self, mueff, **kwargs):
        cfactor = kwargs.get('cfactor', max((self.N - 5.) / 6.0, 0.5))
        cc = kwargs.get('cc', (4. + mueff / self.N) /
                        (self.N + 4. + 2. * mueff / self.N))
        cone = kwargs.get('cone', cfactor * 2. / ((self.N + 1.3)**2 + mueff))
        cmu = kwargs.get('cmu',
                         min(1. - cone,
                             cfactor * 2 * (mueff - 2. + 1. / mueff) / (
                                 (self.N + 2.)**2 + mueff)))
        return cc, cone, cmu

    def _get_params(self, weights, **kwargs):
        w = np.asarray(weights)
        mueff = 1.0 / np.dot(w, w)  # results in too large values with negative weights
        return self._get_params2(mueff, **kwargs)

    def parameters_old(self, weights):
        """return `dict` with (default) parameters, e.g., `c1` and `cmu`.

        :See also: `RecombinationWeights`"""
        try:
            if np.all(self.weights == weights):
                return self._parameters
        except AttributeError:
            pass
        self.weights = np.array(weights, copy=True)
        cc, c1, cmu = self._get_params(weights)
        self._parameters = dict(cc=cc, c1=c1, cmu=cmu)
        return self._parameters

    def parameters(self, mueff=None, **kwargs):
        """return `dict` with (default) parameters, e.g., `c1` and `cmu`.

        :See also: `RecombinationWeights`"""
        if (hasattr(self, '_mueff') and
                (mueff == self._mueff or mueff is None)):
            return self._parameters
        self._mueff = mueff
        cc, c1, cmu = self._get_params2(mueff)
        self._parameters = dict(cc=cc, c1=c1, cmu=cmu)
        return self._parameters

    def norm(self, x):
        """return Mahalanobis norm of `x` w.r.t. the statistical model"""
        return sum(self.transform_inverse(x)**2)**0.5

    @property
    def condition_number(self):
        raise NotImplementedError

    @property
    def covariance_matrix(self):
        if self._debug:
            # Expensive
            C = np.diag(self.dvec**2)
            dv = self.dvec * self.vvec
            C += np.outer(dv, dv)
            return C
        else:
            return None 

    @property
    def variances(self):
        """vector of coordinate-wise (marginal) variances"""
        dC = self.dvec**2 * (1.0 + self.vvec**2)
        return dC

    @property
    def correlation_matrix(self):
        if self._debug:
            # Expensive
            C = self.covariance_matrix
            sqrtdC = np.sqrt(self.variances)
            return (C / sqrtdC).T / sqrtdC
        else:
            return None  

    def transform(self, x):
        """transform ``x`` as implied from the distribution parameters"""
        y = self.dvec * (x + (math.sqrt(1.0 + self.norm_v2) - 1.0) * np.dot(
            x, self.vn) * self.vn)
        return y

    def transform_inverse(self, x):
        y = x / self.dvec
        y += (1.0 / math.sqrt(1.0 + self.norm_v2) - 1.0) * np.dot(
            y, self.vn) * self.vn
        return y

    def to_linear_transformation_inverse(self, reset=False):
        """return inverse of associated linear transformation"""
        raise NotImplementedError

    def to_linear_transformation(self, reset=False):
        """return associated linear transformation"""
        raise NotImplementedError

    def inverse_hessian_scalar_correction(self, mean, X, f):
        """return scalar correction ``alpha`` such that ``X`` and ``f``
        fit to ``f(x) = (x-mean) (alpha * C)**-1 (x-mean)``
        """
        raise NotImplementedError

    def __imul__(self, factor):
        self.dvec *= math.sqrt(factor)
        return self


class GaussVkDSampler(StatisticalModelSamplerWithZeroMeanBaseClass):
    """Restricted Gaussian Sampler for VkD-CMA
    O(N*k^2 + k^3) Time/Space Variant of CMA-ES with C = D * (I + V * V^T) * D

    References
    ----------
    [1] Youhei Akimoto and Nikolaus Hansen.
    Online Model Selection for Restricted Covariance Matrix Adaptation.
    In Proc. of PPSN 2016, pp. 3--13 (2016)
    [2] Youhei Akimoto and Nikolaus Hansen.
    Projection-Based Restricted Covariance Matrix Adaptation for High
    Dimension. In Proc. of GECCO 2016, pp. 197--204 (2016)
    """

    @staticmethod
    def extend_cma_options(opts=None):
        """return correct options to run `cma.fmin` or initialize
        `cma.CMAEvolutionStrategy` using the `GaussVkDSampler` AKA VkD-CMA-ES
        """
        opts = opts or {}
        opts.update({'CMA_active': False,
                     'AdaptSigma': False,
                     'CMA_sampler': GaussVkDSampler})
        return opts

    def __init__(self,
                 dimension,
                 randn=np.random.randn,
                 kadapt=True,
                 **kwargs):
        """pass dimension of the underlying sample space
        """
        try:
            self.N = len(dimension)
            std_vec = np.array(dimension, copy=True)
        except TypeError:
            self.N = dimension
            std_vec = np.ones(self.N)
        self.randn = randn
        self.sigma = 1.0
        self.sigma_fac = 1.0

        self.kadapt = kadapt
        # VkD Static Parameters
        self.k = kwargs.get('k_init', 0)  # alternatively, self.w.shape[0]
        self.k_active = 0
        if self.kadapt:
            self.kmin = kwargs.get('kmin', 0)
            self.kmax = kwargs.get('kmax', self.N - 1)
            assert (0 <= self.kmin <= self.kmax < self.N)
            self.k_inc_cond = kwargs.get('k_inc_cond', 30.0)
            self.k_dec_cond = kwargs.get('k_dec_cond', self.k_inc_cond)
            self.k_adapt_factor = kwargs.get('k_adapt_factor', 1.414)
            self.factor_sigma_slope = kwargs.get('factor_sigma_slope', 0.1)
            self.factor_diag_slope = kwargs.get(
                'factor_diag_slope', 2)  # 0.3 in PPSN (due to cc change)
            self.accepted_slowdown = max(1., self.k_inc_cond / 10.)
            self.k_adapt_decay = 1.0 / self.N
            self.k_adapt_wait = 2.0 / self.k_adapt_decay - 1

        # TPA Parameters
        self.cs = kwargs.get('cs', 0.3)
        self.ds = kwargs.get('ds', 4. - 3. / self.N)  #math.sqrt(self.N)) 
        self.flg_injection = False
        self.ps = 0.
        self._debug = kwargs.get('debug', False)

        # Initialize Dynamic Parameters
        self.D = std_vec
        self.V = np.zeros((self.k, self.N))
        self.S = np.zeros(self.N)
        self.pc = np.zeros(self.N)
        self.dx = np.zeros(self.N)
        self.U = None

    def sample(self, number, update=None):
        """return list of i.i.d. samples.

        :param number: is the number of samples.
        :param update: controls a possibly lazy update of the sampler.
        """
        # self.flg_injection = False  # no ssa inside this class
        if self.flg_injection:
            mnorm = self.norm(self.dx)
            dy = (np.linalg.norm(self.randn(self.N)) / mnorm) * self.dx
            X = np.asarray([dy, -dy] + [
                self.transform(self.randn(self.N)) for i in range(number - 2)
            ])
        else:
            X = np.asarray(
                [self.transform(self.randn(self.N)) for i in range(number)])
        return X

    def update(self, vectors, weights):
        """``vectors`` is a list of samples, ``weights`` a corrsponding
        list of learning rates
        """
        # self.flg_injection = False  # no ssa inside this class        
        ka = self.k_active
        k = self.k
        # Parameters
        ww = np.array(weights, copy=True)
        ww = ww[1:] / np.sum(np.abs(ww[1:]))  # w[0] is the weight for pc
        assert np.all(ww >= 0.0)
        cc, cone, cmu = self._get_params(ww, k)
        mu = np.sum(ww > 0, dtype=int)
        mueff = 1.0 / np.dot(ww, ww)
        idx = np.argsort(ww)[::-1]
        sary = np.asarray(vectors)[idx[:mu] + 1] / self.sigma
        w = ww[idx[:mu]]
        lam = len(weights) - 1

        if self.kadapt and not hasattr(self, 'opt_conv'):
            # VkD Dynamic Parameters
            self.opt_conv = 0.5 * min(1., float(lam) / self.N)
            self.last_log_sigma = np.log(self.sigma)
            self.last_log_d = 2.0 * np.log(self.D)
            self.last_log_cond_corr = np.zeros(self.N)
            self.ema_log_sigma = ExponentialMovingAverage(
                decay=self.opt_conv / self.accepted_slowdown, dim=1)
            self.ema_log_d = ExponentialMovingAverage(
                decay=self.k_adapt_decay, dim=self.N)
            self.ema_log_s = ExponentialMovingAverage(
                decay=self.k_adapt_decay, dim=self.N)
            self.itr_after_k_inc = 0

        # TPA (PPSN 2014 version
        if self.flg_injection:
            nlist = np.asarray([
                np.array(vectors[idx[i] + 1]) /
                np.linalg.norm(vectors[idx[i] + 1]) for i in range(lam)
            ])
            ndx = self.dx / np.linalg.norm(self.dx)
            if 11 < 3:
                for ip in range(lam):
                    if np.allclose(nlist[ip], ndx):
                        break
                    if ip == lam - 1:
                        raise RuntimeError("no first mirrored vector found for TPA")
                        warnings.warn("no first mirrored vector found for TPA",
                                      RuntimeWarning)
                for im in range(lam):
                    if np.allclose(nlist[im], -ndx):
                        break
                    if im == lam - 1:
                        raise RuntimeError("no second mirrored vector found for TPA")
                        warnings.warn("no second mirrored vector found for TPA",
                                      RuntimeWarning)
            else:
                inner = [np.dot(ny, ndx) for ny in nlist]
                ip = np.argmax(inner)
                im = np.argmin(inner)
                if inner[ip] < 0.99:
                    warnings.warn("no first mirrored vector found for TPA",
                                  RuntimeWarning)
                if inner[im] > -0.99:
                    warnings.warn("no second mirrored vector found for TPA",
                                  RuntimeWarning)
            alpha_act = im - ip
            alpha_act /= float(lam - 1)
            self.ps += self.cs * (alpha_act - self.ps)
            self.sigma *= math.exp(self.ps / self.ds)
            hsig = self.ps < 0.5
        else:
            self.flg_injection = True
            hsig = True
        self.dx = np.dot(w, sary)

        # Cumulation
        self.pc = (1. - cc) * self.pc + hsig * math.sqrt(cc * (2. - cc) *
                                                         mueff) * self.dx

        self.U = np.zeros((self.N, self.k + mu + 1))
        # Update V, S and D
        # Cov = D(alpha**2 * I + UU^t)D
        if cmu == 0.0:
            rankU = ka + 1
            alpha = math.sqrt(
                abs(1 - cmu - cone + cone * (1 - hsig) * cc * (2 - cc)))
            self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, rankU - 1] = math.sqrt(cone) * (self.pc / self.D)
        elif cone == 0.0:
            rankU = ka + mu
            alpha = math.sqrt(
                abs(1 - cmu - cone + cone * (1 - hsig) * cc * (2 - cc)))
            self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, ka:rankU] = np.sqrt(cmu * w) * (sary / self.D).T
        else:
            rankU = ka + mu + 1
            alpha = math.sqrt(
                abs(1 - cmu - cone + cone * (1 - hsig) * cc * (2 - cc)))
            self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, ka:rankU - 1] = np.sqrt(cmu * w) * (sary / self.D).T
            self.U[:, rankU - 1] = math.sqrt(cone) * (self.pc / self.D)

        if self.N > rankU:
            # O(Nk^2 + k^3)
            DD, R = np.linalg.eigh(
                np.dot(self.U[:, :rankU].T, self.U[:, :rankU]))
            idxeig = np.argsort(DD)[::-1]
            gamma = 0 if rankU <= k else DD[idxeig[k:]].sum() / (self.N - k)
            beta = alpha * alpha + gamma

            self.k_active = ka = min(np.sum(DD >= 0), k)
            self.S[:ka] = (DD[idxeig[:ka]] - gamma) / beta
            self.V[:ka] = (np.dot(self.U[:, :rankU], R[:, idxeig[:ka]]) /
                           np.sqrt(DD[idxeig[:ka]])).T
        else:
            # O(N^3 + N^2(k+mu+1))
            # If this is the case, the standard CMA is preferred
            DD, L = np.linalg.eigh(
                np.dot(self.U[:, :rankU], self.U[:, :rankU].T))
            idxeig = np.argsort(DD)[::-1]
            gamma = 0 if rankU <= k else DD[idxeig[k:]].sum() / (self.N - k)
            beta = alpha * alpha + gamma

            self.k_active = ka = min(np.sum(DD >= 0), k)
            self.S[:ka] = (DD[idxeig[:ka]] - gamma) / beta
            self.V[:ka] = L[:, idxeig[:ka]].T

        self.D *= np.sqrt(
            (alpha * alpha + np.sum(
                self.U[:, :rankU] * self.U[:, :rankU], axis=1)) /
            (1.0 + np.dot(self.S[:ka], self.V[:ka] * self.V[:ka])))

        # Covariance Normalization by Its Determinant
        gmean_eig = np.exp(self._get_log_determinant_of_cov() / self.N / 2.0)
        self.D /= gmean_eig
        self.pc /= gmean_eig

        # ======================================================================
        if self.kadapt is False:
            return
        # k-Adaptation (PPSN 2016)
        self.itr_after_k_inc += 1

        # Exponential Moving Average
        self.ema_log_sigma.update(math.log(self.sigma * self.sigma_fac) - self.last_log_sigma)
        self.lnsigma_change = self.ema_log_sigma.M / (self.opt_conv /
                                                      self.accepted_slowdown)
        self.last_log_sigma = math.log(self.sigma * self.sigma_fac)
        self.ema_log_d.update(2. * np.log(self.D) + np.log(1 + np.dot(
            self.S[:self.k], self.V[:self.k]**2)) - self.last_log_d)
        self.lndiag_change = self.ema_log_d.M / (cmu + cone)
        self.last_log_d = 2. * np.log(
            self.D) + np.log(1 + np.dot(self.S[:self.k], self.V[:self.k]**2))
        self.ema_log_s.update(np.log(1 + self.S) - self.last_log_cond_corr)
        self.lnlambda_change = self.ema_log_s.M / (cmu + cone)
        self.last_log_cond_corr = np.log(1 + self.S)

        # Check for adaptation condition
        flg_k_increase = self.itr_after_k_inc > self.k_adapt_wait
        flg_k_increase *= self.k < self.kmax
        flg_k_increase *= np.all((1 + self.S[:self.k]) > self.k_inc_cond)
        flg_k_increase *= (
            np.abs(self.lnsigma_change) < self.factor_sigma_slope)
        flg_k_increase *= np.all(
            np.abs(self.lndiag_change) < self.factor_diag_slope)
        # print(self.itr_after_k_inc > self.k_adapt_wait,
        #       self.k < self.kmax,
        #       np.all((1 + self.S[:self.k]) > self.k_inc_cond),
        #       np.abs(self.lnsigma_change) < self.factor_sigma_slope,
        #       np.percentile(np.abs(self.lndiag_change), [1, 50, 99]))

        flg_k_decrease = (self.k > self.kmin) * (
            1 + self.S[:self.k] < self.k_dec_cond)
        flg_k_decrease *= (self.lnlambda_change[:self.k] < 0.)

        if (self.itr_after_k_inc > self.k_adapt_wait) and flg_k_increase:
            # ----- Increasing k -----
            self.k_active = k
            self.k = newk = min(
                max(int(math.ceil(self.k * self.k_adapt_factor)), self.k + 1),
                self.kmax)
            self.V = np.vstack((self.V, np.zeros((newk - k, self.N))))
            self.U = np.empty((self.N, newk + mu + 1))
            # update constants
            (cone, cmu, cc) = self._get_params(w, self.k)
            self.itr_after_k_inc = 0

        elif self.itr_after_k_inc > k * self.k_adapt_wait and np.any(
                flg_k_decrease):
            # ----- Decreasing k -----
            flg_keep = np.logical_not(flg_k_decrease)
            new_k = max(np.count_nonzero(flg_keep), self.kmin)
            self.V = self.V[flg_keep]
            self.S[:new_k] = (self.S[:flg_keep.shape[0]])[flg_keep]
            self.S[new_k:] = 0
            self.k = self.k_active = new_k
            # update constants
            (cone, cmu, cc) = self._get_params(w, self.k)
        # ==============================================================================

        # Covariance Normalization by Its Determinant
        gmean_eig = math.exp(self._get_log_determinant_of_cov() / self.N / 2.0)
        self.D /= gmean_eig
        self.pc /= gmean_eig

    def _get_params(self, weights, k):
        """Return the learning rate cone, cmu, cc depending on k

        Parameters
        ----------
        weights : list of float
            the weight values for vectors used to update the distribution
        k : int
            the number of vectors for covariance matrix

        Returns
        -------
        cone, cmu, cc : float in [0, 1]. Learning rates for rank-one, rank-mu,
         and the cumulation factor for rank-one.
        """
        w = np.array(weights)
        mueff = np.sum(w[w > 0.])**2 / np.dot(w[w > 0.], w[w > 0.])
        return self._get_params2(mueff, k)

    def _get_params2(self, mueff, k):
        nelem = self.N * (k + 1)
        cone = 2.0 / (nelem + self.N + 2 * (k + 2) + mueff)  # PPSN 2016
        # cone = 2.0 / (nelem + 2 * (k + 2) + self.mueff)  # GECCO 2016
        # cc = (4 + self.mueff / self.N) / (
        #     (self.N + 2 * (k + 1)) / 3 + 4 + 2 * self.mueff / self.N)

        # New Cc and C1: Best Cc depends on C1, not directory on K.
        # Observations on Cigar (N = 3, 10, 30, 100, 300, 1000) by Rank-1 VkD.
        cc = math.sqrt(cone)
        cmu = min(1 - cone, 2.0 * (mueff - 2 + 1.0 / mueff) /
                  (nelem + 4 * (k + 2) + mueff))
        return cc, cone, cmu

    def parameters_old(self, weights):
        """return `dict` with (default) parameters, e.g., `c1` and `cmu`.

        :See also: `RecombinationWeights`"""
        try:
            if np.all(self.weights == weights):
                return self._parameters
        except AttributeError:
            pass
        self.weights = np.array(weights, copy=True)
        cc, c1, cmu = self._get_params(weights, self.k)
        self._parameters = dict(cc=cc, c1=c1, cmu=cmu)
        return self._parameters

    def parameters(self, mueff=None, **kwargs):
        """return `dict` with (default) parameters, e.g., `c1` and `cmu`.

        :See also: `RecombinationWeights`"""
        if mueff is not None:
            self._mueff = mueff
        if not hasattr(self, '_mueff'):
            print("""The first call of `parameters` method must specify
    the `mueff` argument! Otherwise an except will be raised. """)
        cc, c1, cmu = self._get_params2(self._mueff, self.k)
        self._parameters = dict(cc=cc, c1=c1, cmu=cmu)
        return self._parameters

    def norm(self, x):
        """return Mahalanobis norm of `x` w.r.t. the statistical model"""
        return np.sum(self.transform_inverse(x)**2)**0.5

    @property
    def condition_number(self):
        raise NotImplementedError

    @property
    def covariance_matrix(self):
        if self._debug:
            # return None
            ka = self.k_active
            if ka > 0:
                C = np.eye(self.N) + np.dot(self.V[:ka].T * self.S[:ka],
                                            self.V[:ka])
                C = (C * self.D).T * self.D
            else:
                C = np.diag(self.D**2)
            C *= self.sigma**2
        else:
            # Fake Covariance Matrix for Speed
            C = np.ones(1)
            self.B = np.ones(1)
        return C

    @property
    def variances(self):
        """vector of coordinate-wise (marginal) variances"""
        ka = self.k_active
        if ka == 0:
            return self.D**2 * self.sigma**2
        else:
            return self.D**2 * (
                1.0 + np.dot(self.S[:ka], self.V[:ka]**2)) * self.sigma**2

    @property
    def correlation_matrix(self):
        if self._debug:
            C = self.covariance_matrix
            sqrtdC = np.sqrt(self.variances)
            return (C / sqrtdC).T / sqrtdC
        else:
            return None

    def transform(self, x):
        """transform ``x`` as implied from the distribution parameters"""
        # Sampling with one normal vectors
        # Available even if S < 0 as long as V are orthogonal to each other
        ka = self.k_active
        y = x + np.dot(
            np.dot(x, self.V[:ka].T) *
            (np.sqrt(1.0 + self.S[:ka]) - 1.0), self.V[:ka])
        y *= self.D * self.sigma
        return y

    def transform_inverse(self, x):
        y = x / self.D / self.sigma
        if self.k_active == 0:
            return y
        else:
            return y + np.dot(
                np.dot(self.V[:self.k_active], y) *
                (1.0 / np.sqrt(1.0 + self.S[:self.k_active]) - 1.0
                 ), self.V[:self.k_active])

    def to_linear_transformation_inverse(self, reset=False):
        """return inverse of associated linear transformation"""
        raise NotImplementedError

    def to_linear_transformation(self, reset=False):
        """return associated linear transformation"""
        raise NotImplementedError

    def inverse_hessian_scalar_correction(self, mean, X, f):
        """return scalar correction ``alpha`` such that ``X`` and ``f``
        fit to ``f(x) = (x-mean) (alpha * C)**-1 (x-mean)``
        """
        raise NotImplementedError

    def __imul__(self, factor):
        self.sigma *= math.sqrt(factor)
        self.sigma_fac /= math.sqrt(factor)       
        return self

    def _get_log_determinant_of_cov(self):
        return 2.0 * np.sum(np.log(self.D)) + np.sum(
            np.log(1.0 + self.S[:self.k_active]))

    def get_condition_numbers(self):
        """get the condition numbers of D**2 and (I + VV')
        
        Theoretically, the condition number of the covariance matrix can be
        at most the product of the return values. It might be safe to stop 
        a run if the product of the return values reaches 1e14.

        Returns
        -------
        float
            condition number of D
        float 
            condition number of I + VV'
        """
        return (np.max(self.D) / np.min(self.D)) ** 2, np.max(1 + self.S[:self.k])

class ExponentialMovingAverage(object):
    """Exponential Moving Average, Variance, and SNR (Signal-to-Noise Ratio)

    See http://www-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
    """

    def __init__(self, decay, dim, flg_init_with_data=False):
        """

        The latest N steps occupy approximately 86% of the information when
        decay = 2 / (N - 1).
        """
        self.decay = decay
        self.M = np.zeros(dim)  # Mean Estimate
        self.S = np.zeros(dim)  # Variance Estimate
        self.flg_init = -flg_init_with_data

    def update(self, datum):
        a = self.decay if self.flg_init else 1.
        self.S += a * ((1 - a) * (datum - self.M)**2 - self.S)
        self.M += a * (datum - self.M)
