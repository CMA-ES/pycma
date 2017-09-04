"""VkS-CMA (Restricted Covariance Matrix Adaptation)

See module `cma.test` for a usage example.

LOG:
2017/09/04: file created

"""
import math
import numpy as np
import numpy.linalg as la
import numpy.random as mt
from ..interfaces import StatisticalModelSamplerWithZeroMeanBaseClass
from .optionparser import OptionBaseClass
from .feigh import pyfeigh


def findmaxinterval(sorted_, window_):
    sidx = 0
    maxlen = 1
    for j in range(len(sorted_)):
        if j + maxlen >= len(sorted_):
            break
        else:
            i = np.searchsorted(sorted_[j+maxlen:], sorted_[j] + window_, side='right')
            if i > 0:
                sidx = j
                maxlen += i
    return sidx, maxlen


def orthogonal_vectors(orthonormalrows):
    E = np.asarray(orthonormalrows)
    m, n = E.shape
    res = np.random.randn((n-m, n))
    for i in range(res.shape[0]):
        for j in range(E.shape[0]):
            res[i] -= np.dot(E[j], res[i]) * E[j]
        for j in range(i):
            res[i] -= np.dot(res[j], res[i]) * res[j]
        res[i] /= np.linalg.norm(res[i])
    return res


class VksOption(OptionBaseClass):

    def __init__(self,
                 N='0',
                 klong='1',
                 kshort='1',
                 batch_evaluation='False',
                 tpa_cs='0.3',
                 tpa_ds='4.0 - 3.6 / np.sqrt(N) # np.sqrt(N) # or 4.0 - 3.6 / np.sqrt(N) for ineffective-axes problem',
                 aneg='0.5 # factor to multiply the negative weights',
                 # k adaptation
                 flg_kadapt='True',
                 kmax='int(N)-1',
                 k_adapt_factor='1.1 # additive if it is 1.',
                 k_inc_cond='10.0',
                 k_dec_cond='3.0',
                 factor_sigma_slope='0.1 # TODO',
                 **kwargs):
        super(VksOption, self).__init__(**kwargs)
        self.setattr_from_local(locals())


class GaussVksSampler(StatisticalModelSamplerWithZeroMeanBaseClass):
    """Restricted Model Adaptive CMA-ES

    The covariance matrix is modeled by

    Cov = D * (I + V * (S - I) * V') * D,

    inv(I + V * (S - I) * V') = I + V * (S^{-1} - I)* V'

    sqrt(I + V * (S - I) * V') = I + V * (S^{1/2} - I) * V'

    sqrt(inv(I + V * (S - I) * V')) = I + V * (S^{-1/2} - I)* V'

    where
        D: diagonal, > 0
        S: diagonal, > 0
        V: orthonormal columns

    References
    ----------
    not published
    """

    def __init__(self,
                 dimension,
                 randn=np.random.randn,
                 debug=False,
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
        self._debug = debug
        # Optional Parameters
        opts_ = VksOption(kwargs)
        opts_.N = self.N
        self.opts = opts_.parse()
        self.klong = self.opts.klong
        self.kshort = self.opts.kshort
        self.kl_active = 0
        self.ks_active = 0
        self.k = self.klong + self.kshort
        self.k_inc_cond = self.opts.k_inc_cond
        self.k_dec_cond = self.opts.k_dec_cond
        self.k_adapt_factor = self.opts.k_adapt_factor
        self.kmax = self.opts.kmax
        assert 0 <= self.kmax < self.N, '0 <= kmax < N'
        # TPA
        self.flg_injection = False
        self.cs = self.opts.tpa_cs
        self.ds = self.opts.tpa_ds
        self.ps = 0
        # Others
        self.D = std_vec
        self.V = np.zeros((self.k, self.N))
        self.S = np.ones(self.N)
        self.pc = np.zeros(self.N)
        self.dx = np.zeros(self.N)
        # k adaptation
        self.mean_logsigma_slope = 0.0 
        self.var_logsigma_slope = 0.0
        self.itr_after_k_inc = 0

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
        k = self.k
        # NOTE: recombination weights passed to this method are ignored
        # TODO: correct the following lines
        ww = np.array(weights, copy=True)
        ww = ww[1:]
        idx = np.argsort(ww)[::-1]
        sary = np.asarray(vectors)[idx + 1] / self.sigma
        lam = len(idx)
        self.lam = lam
        self.w, self.wn, self.cc, self.cone, self.cmu = self._get_parameters()
        self.mu = self.w.shape[0]                
        self.sqrtw = np.sqrt(self.w)
        self.sqrtwn = np.sqrt(self.wn)
        #print(self.w, self.wn, self.cc, self.cone, self.cmu, flush=True)
        #print(self.k, self.klong, self.kl_active, self.kshort, self.ks_active)
        
        # k Adaptation
        self.opt_conv = 0.5 * self.lam / (self.N - 1 + self.lam)  # Optimal convergence rate on sphere
        self.accepted_slowdown = 2  # 2 for non-negative weights for m-update
        self.factor_sigma_slope = self.opts.factor_sigma_slope
        self.k_adapt_wait = 2.0 * self.N - 1  # 86% of the information
        self.last_log_sigma = np.log(self.sigma)
        self.decay_logsigma_slope = self.opt_conv / self.accepted_slowdown

        # Find injected points            
        if self.flg_injection:
            nlist = np.asarray([
                np.array(vectors[idx[i] + 1]) /
                np.linalg.norm(vectors[idx[i] + 1]) for i in range(lam)
            ])
            ndx = self.dx / np.linalg.norm(self.dx)
            for ip in range(lam):
                if np.allclose(nlist[ip], ndx):
                    break
                if ip == lam - 1:
                    print('error')
            for im in range(lam):
                if np.allclose(nlist[im], -ndx):
                    break
                if im == lam - 1:
                    print('error')
        #---------- VkD-CMA ----------
        ka = self.kl_active + self.ks_active
        oldS = self.S[:ka].copy()
        oldV = self.V[:ka].copy()
        gamma = 1.
        # Update xmean
        self.dx = np.dot(self.w, sary[:self.w.shape[0]])
        # Update sigma by TPA (PPSN 2014 version)
        if self.flg_injection:
            alpha_act = im - ip
            alpha_act /= self.lam - 1
            self.ps += self.cs * (alpha_act - self.ps)
            self.sigma *= math.exp(self.ps / self.ds)
            hsig = self.ps < 0.5
        else:
            self.flg_injection = True
            hsig = True

        # Cumulation
        self.pc *= (1 - self.cc)
        self.pc += (hsig * math.sqrt(self.cc * (2 - self.cc) / np.sum(self.w ** 2))) * self.dx
        
        # Update V and S
        fac = 1.0 - (self.cmu*(self.w.sum()-self.wn.sum()) + self.cone - self.cone*(1-hsig)*self.cc*(2-self.cc))

        # Input Matrices
        Yp = np.column_stack(((math.sqrt(self.cmu) * self.sqrtw) * (sary[:self.w.shape[0]] / self.D).T,
                              math.sqrt(self.cone) * (self.pc / self.D)))
        scaled_sqrtwn = self.sqrtwn * math.sqrt(self.N) / np.sqrt(self.square_mnorm(sary[-1:-self.wn.shape[0]-1:-1]))
        Yn = (math.sqrt(self.cmu) * scaled_sqrtwn) * (sary[-1:-self.wn.shape[0]-1:-1] / self.D).T

        # Eigen Decomposition
        rankSw = min(ka + len(self.w) + len(self.wn) + 1, self.N)
        Sw = np.empty(rankSw)
        Qw = np.empty((rankSw, self.N))
        pyfeigh(fac, self.V[:ka].T, np.diag(self.S[:ka] - 1.0).T, 1.0, Yp, -1.0, Yn, Qw.T, Sw)
        Qw = Qw[np.abs(Sw) > 1e-10]
        Sw = Sw[np.abs(Sw) > 1e-10]

        # Update S and V
        rankSw = Sw.shape[0]
        if rankSw > self.k:
            ka = self.k
            ks = self.kshort
            kl = ka - ks
            gamma = np.exp(((self.N - rankSw) * math.log(fac) + np.sum(np.log(fac + Sw[ks:rankSw-kl]))) / (self.N - ka))
        else:
            ka = rankSw
            ks = np.sum(Sw < fac)
            kl = ka - ks
            gamma = fac

        # Short directions
        self.S[:ks] = Sw[:ks] + fac
        self.V[:ks] = Qw[:ks]
        self.ks_active = ks

        # Long directions
        self.S[ks:ka] = Sw[rankSw-kl:] + fac
        self.V[ks:ka] = Qw[rankSw-kl:]
        self.kl_active = kl

        self.S[:ka] /= gamma        
        # force the change to be at most by the factor of 1.1
        # for numerical reason
        _S = np.ones(ka)
        _S[:np.sum(oldS < 1.0)] = oldS[oldS < 1.0]
        _S[ka-np.sum(oldS >= 1.0):] = oldS[oldS >= 1.0]
        self.S[:ka] = np.clip(self.S[:ka], _S / 1.1, _S * 1.1)

        # ---------- Adaptation of k ----------
        if self.opts.flg_kadapt and ka == self.k:
            self.itr_after_k_inc += 1
            # Update Exponential Moving Average
            diff = np.log(self.sigma) - self.last_log_sigma - self.mean_logsigma_slope
            self.mean_logsigma_slope += self.decay_logsigma_slope * diff          
            self.var_logsigma_slope += self.decay_logsigma_slope * ((1 - self.decay_logsigma_slope) * diff ** 2 - self.var_logsigma_slope)
            self.last_log_sigma = np.log(self.sigma)       

            # Check for adaptation condition
            flg_k_increase = self.itr_after_k_inc > self.k_adapt_wait
            flg_k_increase *= (self.klong + self.kshort) < self.kmax
            flg_k_increase *= np.abs(self.mean_logsigma_slope) < self.factor_sigma_slope * (self.opt_conv / self.accepted_slowdown)
            flg_kl_increase = flg_k_increase * np.all((self.S[self.kshort:self.kshort+self.klong]) > self.k_inc_cond)
            flg_ks_increase = flg_k_increase * np.all((self.S[:self.kshort]) < 1.0 / self.k_inc_cond)

            if (self.itr_after_k_inc > self.k_adapt_wait) and (flg_kl_increase or flg_ks_increase):
                # ----- Increasing k -----
                kl = self.klong
                ks = self.kshort
                if flg_kl_increase:
                    self.klong = min(max(int(np.ceil(self.klong * self.k_adapt_factor)), self.klong + 1), self.kmax)
                if flg_ks_increase:
                    self.kshort = min(max(int(np.ceil(self.kshort * self.k_adapt_factor)), self.kshort + 1), self.kmax)
                self.k = self.klong + self.kshort
                self.V = np.vstack((self.V, np.zeros((self.k - kl - ks, self.N))))
                #_, _, self.cc, self.cone, self.cmu = self._get_parameters()
                self.itr_after_k_inc = 0

            elif self.itr_after_k_inc > self.k_adapt_wait:
                # ----- Decreasing k -----
                self.lowerrankapprox(self.k_dec_cond)
                if not (kl == self.klong and ks == self.kshort):
                    #_, _, self.cc, self.cone, self.cmu = self._get_parameters()
                    self.itr_after_k_inc = 0
                if self.klong == self.kshort == 0:
                    self.k = self.klong = 1
                    self.V = np.zeros((1, self.N))

        # Covariance Normalization by Its Determinant
        gmean_eig = math.exp(self._get_log_determinant_of_cov() / self.N / 2.0)
        self.D /= gmean_eig
        self.pc /= gmean_eig

    def parameters(self, mueff=None, **kwargs):
        """return `dict` with (default) parameters, e.g., `c1` and `cmu`.

        :See also: `RecombinationWeights`"""
        _, _, cc, c1, cmu = self._get_parameters()
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
            raise NotImplementedError
        else:
            # Fake Covariance Matrix for Speed
            C = np.ones(1)
            self.B = np.ones(1)
        return C

    @property
    def variances(self):
        """vector of coordinate-wise (marginal) variances"""
        ka = self.kl_active + self.ks_active
        if ka == 0:
            return self.D**2 * self.sigma**2
        else:
            return self.D**2 * (1.0 + np.dot(self.S[:ka] - 1.0, self.V[:ka]**2)) * self.sigma**2

    @property
    def correlation_matrix(self):
        if self._debug:
            raise NotImplementedError
        else:
            return None

    def transform(self, x):
        """transform ``x`` as implied from the distribution parameters"""
        return self._transform(x) * self.sigma

    def transform_inverse(self, x):
        return self._transform_inverse(x) / self.sigma

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

    def _transform(self, x):
        """y = D * (I + V*(S-I)*V^t)^{1/2} z
        
        Parameter
        ---------
        x : 1d or 2d array-like
            a vector or a list of vectors to be transformed
        """
        ka = self.ks_active + self.kl_active
        y = np.asarray(x)
        if ka == 0:
            return y * self.D
        else:
            y += np.dot(np.dot(y, self.V[:ka].T) * (np.sqrt(self.S[:ka]) - 1.0), self.V[:ka])
            return y * self.D

    def _transform_inverse(self, x):
        """z = (I + E*(S-I)*E^t)^{-1/2} * D^{-1} * y
        sqrt(inv(I - EEt + ESEt))
        = sqrt(I - EEt + ES^-1Et)
        = I - EEt + E(S)^{-1/2}Et
        = I + E((S)^{-1/2} - I)Et

        Parameter
        ---------
        x : 1d or 2d array-like
            a vector or a list of vectors to be inversely transformed
        """
        ka = self.kl_active + self.ks_active        
        y = np.asarray(x) / self.D
        if ka == 0:
            return y
        else:
            return y + np.dot(np.dot(y, self.V[:ka].T) * (1.0 / np.sqrt(self.S[:ka]) - 1.0), self.V[:ka])

    def square_mnorm(self, dx):
        if np.ndim(dx) == 1:
            dy = self._transform_inverse(dx)            
            return np.dot(dy, dy)
        elif np.ndim(dx) == 2:
            dy = self._transform_inverse(dx / self.D)            
            return np.sum(dy * dy, axis=1)

    def _get_parameters(self):
        if not hasattr(self, 'lam'):
            self.lam = self.N  # XXX: face lambda. 
        wp = np.array([math.log((self.lam + 1) / 2.0) - math.log(1 + i) for i in range(self.lam // 2)])
        wp /= np.sum(wp)
        mueff = 1.0 / np.sum(wp * wp)
        k = self.k
        nelem = self.N * k
        cone = min(1.0, self.lam / 6.0) / (nelem + 2.0 * math.sqrt(nelem) + mueff / self.N)
        cmu = min(1 - cone, (0.3 + mueff - 2.0 + 1.0 / mueff) / (nelem + 4.0 * math.sqrt(nelem) + mueff / 2.0))
        cc = math.sqrt(cone)
        wn = np.array([math.log(self.lam - i) - math.log((self.lam + 1) / 2.0) for i in range(self.lam // 2)])
        wn /= np.sum(wn)
        mueffn = 1.0 / np.sum(wn * wn)
        amun = 1 + min(cone / cmu, 2 * mueffn / (mueff + 2))
        wn *= min(amun, (1 - cone - cmu) / (self.N * cmu)) * self.opts.aneg
        return wp, wn, cc, cone, cmu

    def _get_log_determinant_of_cov(self):
        return 2.0 * np.sum(np.log(self.D)) + np.sum(np.log(self.S[:self.kl_active+self.ks_active]))
    
    def lowerrankapprox(self, cond):
        """Find Even Lower Rank Approximation Satisfying the Following Condition
        OLD: Original I + E*S*Et
        NEW: Lower Rank I + E'*S'*E't
        Condition: Cond( OLD * NEW^{-1} ) <= 2 * `cond`
        
        Note: self.S is assumed to be sorted for the first self.klong + self.kshort elements.
        """
        kl = self.klong
        ks = self.kshort
        logCond = math.log(cond)
        idxS = np.argsort(self.S)
        logS = np.log(self.S[idxS])
        sidxS, maxlenS = findmaxinterval(logS, logCond)
        if self.N - maxlenS >= ks + kl:
            return
        else:
            fac = math.exp(np.mean(logS[sidxS:sidxS+maxlenS]))
            self.kshort = self.ks_active = sidxS
            self.klong = self.kl_active = self.N - (sidxS + maxlenS)
            self.k = self.kshort + self.klong
            if sidxS + maxlenS <= ks or sidxS >= self.N - kl:
                # Compute orthonormal basis of the null space of V
                E = orthogonal_vectors(self.V[:ks+kl])
            if sidxS < ks and sidxS + maxlenS <= ks:
                # ks <; kl >; ks + kl <
                # shift short
                self.S[self.kshort:self.kshort+(ks-maxlenS-sidxS)] = self.S[maxlenS+sidxS:ks]
                self.V[self.kshort:self.kshort+(ks-maxlenS-sidxS)] = self.V[maxlenS+sidxS:ks]
                # fill
                self.S[self.kshort:self.kshort+(self.N-ks-kl)] = 1.0
                self.V[self.kshort:self.kshort+(self.N-ks-kl)] = E
                # shift long 1
                self.S[self.kshort+(self.N-ks-kl):self.kshort+self.klong] = self.S[ks:ks+kl]
                self.V[self.kshort+(self.N-ks-kl):self.kshort+self.klong] = self.V[ks:ks+kl]
                # scale
                self.S[:self.kshort+self.klong] /= fac
                # empty
                self.S[self.kshort+self.klong:] = 1.0
                self.V = self.V[:self.kshort+self.klong]
            elif sidxS <= ks and sidxS + maxlenS >= self.N - kl:
                # ks <=; kl <=; ks + kl <
                # shift long
                self.S[self.kshort:self.kshort+self.klong] = self.S[ks+kl-self.klong:ks+kl]
                self.V[self.kshort:self.kshort+self.klong] = self.V[ks+kl-self.klong:ks+kl]
                # scale
                self.S[:self.kshort+self.klong] /= fac
                # empty
                self.S[self.kshort+self.klong:] = 1.0
                self.V = self.V[:self.kshort+self.klong]
            elif sidxS >= self.N - kl:
                # ks >; kl <; ks + kl <
                # fill
                self.S[self.kshort:self.kshort+(self.N-ks-kl)] = 1.0
                self.V[self.kshort:self.kshort+(self.N-ks-kl)] = E
                # shift long 1
                self.S[self.N-kl:sidxS] = self.S[ks:ks+sidxS-(self.N-kl)]
                self.V[self.N-kl:sidxS] = self.V[ks:ks+sidxS-(self.N-kl)]
                # shift long 2
                self.S[sidxS:sidxS+(self.N-sidxS-maxlenS)] = self.S[ks+kl-(self.N-sidxS-maxlenS):ks+kl]
                self.V[sidxS:sidxS+(self.N-sidxS-maxlenS)] = self.V[ks+kl-(self.N-sidxS-maxlenS):ks+kl]
                # scale
                self.S[:self.kshort+self.klong] /= fac
                # empty
                self.S[self.kshort+self.klong:] = 1.0
                self.V = self.V[:self.kshort+self.klong]
            else:
                raise RuntimeError("Bug.")
            return


if __name__ == '__main__':
    import numpy as np
    import cma
    from cma.dev.vks_sampler import GaussVksSampler, VksOption

    seedprob = 100
    seedalgo = 100
    N = 20
    es = cma.CMAEvolutionStrategy(N * [1], 1, {
        'seed':seedalgo,
        'CMA_active': False, 'AdaptSigma': None,
        'CMA_sampler': GaussVksSampler,
        'ftarget': 1e-8,
        'CMA_sampler_options': {},
        })
    es = es.optimize(cma.fitness_transformations.Rotated(cma.ff.cigar, seed=seedprob), iterations=None)    
