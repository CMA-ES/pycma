"""step-size adaptation classes, currently tightly linked to CMA,
because `hsig` is computed in the base class
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals, with_statement
import numpy as np
from numpy import square as _square, sqrt as _sqrt
from .utilities import utils
from .utilities.math import Mh
def _norm(x): return np.sqrt(np.sum(np.square(x)))
del absolute_import, division, print_function  #, unicode_literals, with_statement

class CMAAdaptSigmaBase(object):
    """step-size adaptation base class, implement `hsig` (for stalling
    distribution update) functionality via an isotropic evolution path.

    Details: `hsig` or `_update_ps` must be called before the sampling
    distribution is changed. `_update_ps` depends heavily on
    `cma.CMAEvolutionStrategy`.
    """
    def __init__(self, *args, **kwargs):
        self.is_initialized_base = False
        self._ps_updated_iteration = -1
        self.delta = 1
        "cumulated effect of adaptation"
    def initialize_base(self, es):
        """set parameters and state variable based on dimension,
        mueff and possibly further options.

        """
        ## meta_parameters.cs_exponent == 1.0
        b = 1.0
        ## meta_parameters.cs_multiplier == 1.0
        self.cs = 1.0 * (es.sp.weights.mueff + 2)**b / (es.N**b + (es.sp.weights.mueff + 3)**b)
        self.ps = np.zeros(es.N)
        self.is_initialized_base = True
        return self
    def _update_ps(self, es):
        """update the isotropic evolution path.

        Using ``es`` attributes ``mean``, ``mean_old``, ``sigma``,
        ``sigma_vec``, ``sp.weights.mueff``, ``cp.cmean`` and
        ``sm.transform_inverse``.

        :type es: CMAEvolutionStrategy
        """
        if not self.is_initialized_base:
            self.initialize_base(es)
        if self._ps_updated_iteration == es.countiter:
            return
        try:
            if es.countiter <= es.sm.itereigenupdated:
                # es.B and es.D must/should be those from the last iteration
                utils.print_warning('distribution transformation (B and D) have been updated before ps could be computed',
                              '_update_ps', 'CMAAdaptSigmaBase', verbose=es.opts['verbose'])
        except AttributeError:
            pass
        z = es.sm.transform_inverse((es.mean - es.mean_old) / es.sigma_vec.scaling)
        # assert Mh.vequals_approximately(z, np.dot(es.B, (1. / es.D) *
        #         np.dot(es.B.T, (es.mean - es.mean_old) / es.sigma_vec.scaling)))
        z *= es.sp.weights.mueff**0.5 / es.sigma / es.sp.cmean
        self.ps = (1 - self.cs) * self.ps + (self.cs * (2 - self.cs))**0.5 * z
        self._ps_updated_iteration = es.countiter
    def hsig(self, es):
        """return "OK-signal" for rank-one update, `True` (OK) or `False`
        (stall rank-one update), based on the length of an evolution path

        """
        self._update_ps(es)
        if self.ps is None:
            return True
        squared_sum = sum(self.ps**2) / (1 - (1 - self.cs)**(2 * es.countiter))
        # correction with self.countiter seems not necessary,
        # as pc also starts with zero
        return squared_sum / es.N - 1 < 1 + 4. / (es.N + 1)

    def update2(self, es, **kwargs):
        """return sigma change factor and update self.delta.

        ``self.delta == sigma/sigma0`` accumulates all past changes
        starting from `1.0`.

        Unlike `update`, `update2` is not supposed to change attributes
        in `es`, specifically it should not change `es.sigma`.
        """
        self._update_ps(es)
        raise NotImplementedError('must be implemented in a derived class')

    def update(self, es, **kwargs):
        """update ``es.sigma``

        :param es: `CMAEvolutionStrategy` class instance
        :param kwargs: whatever else is needed to update ``es.sigma``,
            which should be none.
        """
        self._update_ps(es)
        raise NotImplementedError('must be implemented in a derived class')
    def check_consistency(self, es):
        """make consistency checks with a `CMAEvolutionStrategy` instance
        as input
        """
class CMAAdaptSigmaNone(CMAAdaptSigmaBase):
    """constant step-size sigma"""
    def update(self, es, **kwargs):
        """no update, ``es.sigma`` remains constant.
        """
        pass
class CMAAdaptSigmaDistanceProportional(CMAAdaptSigmaBase):
    """artificial setting of ``sigma`` for test purposes, e.g.
    to simulate optimal progress rates.

    """
    def __init__(self, coefficient=1.2, **kwargs):
        """pass multiplier for normalized step-size"""
        super(CMAAdaptSigmaDistanceProportional, self).__init__() # base class provides method hsig()
        self.coefficient = coefficient
        self.is_initialized = True
    def update(self, es, **kwargs):
        """need attributes ``N``, ``sp.weights.mueff``, ``mean``,
        ``sp.cmean`` of input parameter ``es``
        """
        es.sigma = self.coefficient * es.sp.weights.mueff * _norm(es.mean) / es.N / es.sp.cmean
class CMAAdaptSigmaCSA(CMAAdaptSigmaBase):
    """CSA cumulative step-size adaptation AKA path length control.

    As of 2017, CSA is considered as the default step-size control method
    within CMA-ES.
    """
    def __init__(self, **kwargs):
        """postpone initialization to a method call where dimension and mueff should be known.

        """
        self.is_initialized = False
        self.delta = 1
    def initialize(self, es):
        """set parameters and state variable based on dimension,
        mueff and possibly further options.

        """
        self.disregard_length_setting = True if es.opts['CSA_disregard_length'] else False
        if es.opts['CSA_clip_length_value'] is not None:
            try:
                if len(es.opts['CSA_clip_length_value']) == 0:
                    es.opts['CSA_clip_length_value'] = [-np.Inf, np.Inf]
                elif len(es.opts['CSA_clip_length_value']) == 1:
                    es.opts['CSA_clip_length_value'] = [-np.Inf, es.opts['CSA_clip_length_value'][0]]
                elif len(es.opts['CSA_clip_length_value']) == 2:
                    es.opts['CSA_clip_length_value'] = np.sort(es.opts['CSA_clip_length_value'])
                else:
                    raise ValueError('option CSA_clip_length_value should be a number of len(.) in [1,2]')
            except TypeError:  # len(...) failed
                es.opts['CSA_clip_length_value'] = [-np.Inf, es.opts['CSA_clip_length_value']]
            es.opts['CSA_clip_length_value'] = list(np.sort(es.opts['CSA_clip_length_value']))
            if es.opts['CSA_clip_length_value'][0] > 0 or es.opts['CSA_clip_length_value'][1] < 0:
                raise ValueError('option CSA_clip_length_value must be a single positive or a negative and a positive number')
        ## meta_parameters.cs_exponent == 1.0
        b = 1.0
        ## meta_parameters.cs_multiplier == 1.0
        self.cs = 1.0 * (es.sp.weights.mueff + 2)**b / (es.N**b + (es.sp.weights.mueff + 3)**b)

        self.damps = es.opts['CSA_dampfac'] * (0.5 +
                                          0.5 * min([1, (es.sp.lam_mirr / (0.159 * es.sp.popsize) - 1)**2])**1 +
                                          2 * max([0, ((es.sp.weights.mueff - 1) / (es.N + 1))**es.opts['CSA_damp_mueff_exponent'] - 1]) +
                                          self.cs
                                          )
        self.max_delta_log_sigma = 1  # in symmetric use (strict lower bound is -cs/damps anyway)

        if self.disregard_length_setting:
            es.opts['CSA_clip_length_value'] = [0, 0]
            ## meta_parameters.cs_exponent == 1.0
            b = 1.0 * 0.5
            ## meta_parameters.cs_multiplier == 1.0
            self.cs = 1.0 * (es.sp.weights.mueff + 1)**b / (es.N**b + 2 * es.sp.weights.mueff**b)
            self.damps = es.opts['CSA_dampfac'] * 1  # * (1.1 - 1/(es.N+1)**0.5)
            if es.opts['verbose'] > 1:
                print('CMAAdaptSigmaCSA Parameters: ')
                for k, v in self.__dict__.items():
                    print('  ', k, ':', v)
        self.ps = np.zeros(es.N)
        self._ps_updated_iteration = -1
        self.is_initialized = True
    def _update_ps(self, es):
        """update path with isotropic delta mean, possibly clipped.

        From input argument `es`, the attributes isotropic_mean_shift,
        opts['CSA_clip_length_value'], and N are used.
        opts['CSA_clip_length_value'] can be a single value, the upper
        bound parameter, such that::

            max_len = sqrt(N) + opts['CSA_clip_length_value'] * N / (N+2)

        or a list with lower and upper bound parameters.
        """
        if not self.is_initialized:
            self.initialize(es)
        if self._ps_updated_iteration == es.countiter:
            return
        z = es.isotropic_mean_shift
        if es.opts['CSA_clip_length_value'] is not None:
            vals = es.opts['CSA_clip_length_value']
            try: len(vals)
            except TypeError: vals = [-np.inf, vals]
            if vals[0] > 0 or vals[1] < 0:
                raise ValueError(
                  """value(s) for option 'CSA_clip_length_value' = %s
                  not allowed""" % str(es.opts['CSA_clip_length_value']))
            min_len = es.N**0.5 + vals[0] * es.N / (es.N + 2)
            max_len = es.N**0.5 + vals[1] * es.N / (es.N + 2)
            act_len = _norm(z)
            new_len = Mh.minmax(act_len, min_len, max_len)
            if new_len != act_len:
                z *= new_len / act_len
                # z *= (es.N / sum(z**2))**0.5  # ==> sum(z**2) == es.N
                # z *= es.const.chiN / sum(z**2)**0.5
        self.ps = (1 - self.cs) * self.ps + _sqrt(self.cs * (2 - self.cs)) * z
        self._ps_updated_iteration = es.countiter
    def update2(self, es, **kwargs):
        """call ``self._update_ps(es)`` and update self.delta.

        Return change factor of self.delta.

        From input `es`, either attribute N or const.chiN is used.
        """
        delta_old = self.delta
        self._update_ps(es)  # caveat: if es.B or es.D are already updated and ps is not, this goes wrong!
        p = self.ps
        if 'pc for ps' in es.opts['vv']:
            # was: es.D**-1 * np.dot(es.B.T, es.pc)
            p = es.sm.transform_inverse(es.pc)
        if es.opts['CSA_squared']:
            s = (sum(_square(p)) / es.N - 1) / 2
            # sum(self.ps**2) / es.N has mean 1 and std sqrt(2/N) and is skewed
            # divided by 2 to have the derivative d/dx (x**2 / N - 1) for x**2=N equal to 1
        else:
            s = _norm(p) / es.const.chiN - 1
        s *= self.cs / self.damps
        s_clipped = Mh.minmax(s, -self.max_delta_log_sigma, self.max_delta_log_sigma)
        self.delta *= np.exp(s_clipped)
        # "error" handling
        if s_clipped != s:
            utils.print_warning('sigma change np.exp(' + str(s) + ') = ' + str(np.exp(s)) +
                          ' clipped to np.exp(+-' + str(self.max_delta_log_sigma) + ')',
                          'update',
                          'CMAAdaptSigmaCSA',
                                es.countiter, es.opts['verbose'])
        return self.delta / delta_old
    def update(self, es, **kwargs):
        """call ``self._update_ps(es)`` and update ``es.sigma``.

        Legacy method replaced by `update2`.
        """
        es.sigma *= self.update2(es, **kwargs)
        if 11 < 3:
            # derandomized MSR = natural gradient descent using mean(z**2) instead of mu*mean(z)**2
            fit = kwargs['fit']  # == es.fit
            slengths = np.array([sum(z**2) for z in es.arz[fit.idx[:es.sp.weights.mu]]])
            # print lengths[0::int(es.sp.weights.mu/5)]
            es.sigma *= np.exp(np.dot(es.sp.weights, slengths / es.N - 1))**(2 / (es.N + 1))
        if 11 < 3:
            es.more_to_write.append(10**((sum(self.ps**2) / es.N / 2 - 1 / 2 if es.opts['CSA_squared'] else _norm(self.ps) / es.const.chiN - 1)))
            es.more_to_write.append(10**(-3.5 + sum(self.ps**2) / es.N / 2 - _norm(self.ps) / es.const.chiN))
            # es.more_to_write.append(10**(-3 + sum(es.arz[es.fit.idx[0]]**2) / es.N))

class CMAAdaptSigmaMedianImprovement(CMAAdaptSigmaBase):
    """Compares median fitness to the 27%tile fitness of the
    previous iteration, see Ait ElHara et al, GECCO 2013.

    >>> import cma
    >>> es = cma.CMAEvolutionStrategy(3 * [1], 1,
    ... {'AdaptSigma':cma.sigma_adaptation.CMAAdaptSigmaMedianImprovement,
    ...  'verbose': -9})
    >>> assert es.optimize(cma.ff.elli).result[1] < 1e-9
    >>> assert es.result[2] < 2000

    """
    def __init__(self, **kwargs):
        CMAAdaptSigmaBase.__init__(self)  # base class provides method hsig()
        # super(CMAAdaptSigmaMedianImprovement, self).__init__()
    def initialize(self, es):
        """late initialization using attributes ``N`` and ``popsize``"""
        r = es.sp.weights.mueff / es.popsize
        self.index_to_compare = 0.5 * (r**0.5 + 2.0 * (1 - r**0.5) / np.log(es.N + 9)**2) * (es.popsize)  # TODO
        self.index_to_compare = 0.30 * es.popsize  # TODO
        self.damp = 2 - 2 / es.N  # sign-rule: 2
        self.c = 0.3  # sign-rule needs <= 0.3
        self.s = 0  # averaged statistics, usually between -1 and +1
    def update(self, es, **kwargs):
        if es.countiter < 2:
            self.initialize(es)
            self.fit = es.fit.fit
        else:
            ft1, ft2 = self.fit[int(self.index_to_compare)], self.fit[int(np.ceil(self.index_to_compare))]
            ftt1, ftt2 = es.fit.fit[(es.popsize - 1) // 2], es.fit.fit[int(np.ceil((es.popsize - 1) / 2))]
            pt2 = self.index_to_compare - int(self.index_to_compare)
            # ptt2 = (es.popsize - 1) / 2 - (es.popsize - 1) // 2  # not in use
            s = 0
            if 1 < 3:
                s += pt2 * sum(es.fit.fit <= self.fit[int(np.ceil(self.index_to_compare))])
                s += (1 - pt2) * sum(es.fit.fit < self.fit[int(self.index_to_compare)])
                s -= es.popsize / 2.
                s *= 2. / es.popsize  # the range was popsize, is 2
            elif 11 < 3:  # compare ft with median of ftt
                s += self.index_to_compare - sum(self.fit <= es.fit.fit[es.popsize // 2])
                s *= 2 / es.popsize  # the range was popsize, is 2
            else:  # compare ftt j-index of ft
                s += (1 - pt2) * np.sign(ft1 - ftt1)
                s += pt2 * np.sign(ft2 - ftt1)
            self.s = (1 - self.c) * self.s + self.c * s
            es.sigma *= np.exp(self.s / self.damp)
        # es.more_to_write.append(10**(self.s))

        #es.more_to_write.append(10**((2 / es.popsize) * (sum(es.fit.fit < self.fit[int(self.index_to_compare)]) - (es.popsize + 1) / 2)))
        # # es.more_to_write.append(10**(self.index_to_compare - sum(self.fit <= es.fit.fit[es.popsize // 2])))
        # # es.more_to_write.append(10**(np.sign(self.fit[int(self.index_to_compare)] - es.fit.fit[es.popsize // 2])))
        if 11 < 3:
            import scipy.stats.stats as stats
            zkendall = stats.kendalltau(list(es.fit.fit) + list(self.fit),
                                        len(es.fit.fit) * [0] + len(self.fit) * [1])[0]
            es.more_to_write.append(10**zkendall)
        self.fit = es.fit.fit
class CMAAdaptSigmaTPA(CMAAdaptSigmaBase):
    """two point adaptation for step-size sigma.

    Relies on a specific sampling of the first two offspring, whose
    objective function value ranks are used to decide on the step-size
    change, see `update` for the specifics.

    Example
    =======

    >>> import cma
    >>> cma.CMAOptions('adapt').pprint()  # doctest: +ELLIPSIS
     AdaptSigma='True...
    >>> es = cma.CMAEvolutionStrategy(10 * [0.2], 0.1,
    ...     {'AdaptSigma': cma.sigma_adaptation.CMAAdaptSigmaTPA,
    ...      'ftarget': 1e-8})  # doctest: +ELLIPSIS
    (5_w,10)-aCMA-ES (mu_w=3.2,w_1=45%) in dimension 10 (seed=...
    >>> es.optimize(cma.ff.rosen)  # doctest: +ELLIPSIS
    Iter...
    >>> assert 'ftarget' in es.stop()
    >>> assert es.result[1] <= 1e-8  # should coincide with the above
    >>> assert es.result[2] < 6500  # typically < 5500

    References: loosely based on Hansen 2008, CMA-ES with Two-Point
    Step-Size Adaptation, more tightly based on Hansen et al. 2014,
    How to Assess Step-Size Adaptation Mechanisms in Randomized Search.

    """
    def __init__(self, dimension=None, opts=None, **kwargs):
        super(CMAAdaptSigmaTPA, self).__init__() # base class provides method hsig()
        # CMAAdaptSigmaBase.__init__(self)
        self.initialized = False
        self.dimension = dimension
        self.opts = opts
    def initialize(self, N=None, opts=None):
        """late initialization.

        :param N: is used for the (minor) dependency on dimension,
        :param opts: is used for hacking
        """
        if self.initialized is True:
            return self
        self.initialized = False
        if N is None:
            N = self.dimension
        if opts is None:
            opts = self.opts
        try:
            damp_fac = opts['CSA_dampfac']  # should be renamed to sigma_adapt_dampfac or something
        except (TypeError, KeyError):
            damp_fac = 1

        self.sp = utils.BlancClass()  # just a container to have sp.name instead of sp['name'] to access parameters
        try:
            self.sp.damp = damp_fac * eval('N')**0.5  # (1) why do we need 10 <-> np.exp(1/10) == 1.1? 2 should be fine!?
            self.sp.damp = damp_fac * (4 - 3.6/eval('N')**0.5)  # (2) should become new default!?
            self.sp.damp = damp_fac * eval('N')**0.25
            self.sp.damp = 0.7 + np.log(eval('N'))  # between 2 and 9 very close to N**1/2, for N=7 equal to (1) and (2)
            # self.sp.damp = 100
        except:
            self.sp.damp = 4  # or 1 + np.log(10)
            self.initialized = 1/2
        try:
            self.sp.damp = opts['vv']['TPA_damp']
            print('damp set to %d' % self.sp.damp)
        except (KeyError, TypeError):
            pass

        self.sp.dampup = 0.5**0.0 * 1.0 * self.sp.damp  # 0.5 fails to converge on the Rastrigin function
        self.sp.dampdown = 2.0**0.0 * self.sp.damp
        if self.sp.dampup != self.sp.dampdown:
            print('TPA damping is asymmetric')
        self.sp.c = 0.3  # rank difference is asymetric and therefore the switch from increase to decrease takes too long
        self.sp.z_exponent = 0.5  # sign(z) * abs(z)**z_exponent, 0.5 seems better with larger popsize, 1 was default
        self.sp.sigma_fac = 1.0  # (obsolete) 0.5 feels better, but no evidence whether it is
        self.sp.relative_to_delta_mean = True  # (obsolete)
        self.s = 0  # the state/summation variable
        self.last = None
        if not self.initialized:
            self.initialized = True
        return self
    def update(self, es, function_values, **kwargs):
        """the first and second value in ``function_values``
        must reflect two mirrored solutions.

        Mirrored solutions must have been sampled
        in direction / in opposite direction of
        the previous mean shift, respectively.
        """
        # On the linear function, the two mirrored samples lead
        # to a sharp increase of the condition of the covariance matrix,
        # unless we have negative weights (which we have now by default).
        # Otherwise they should not be used to update the covariance
        # matrix, if the step-size inreases quickly.
        if self.initialized is not True:  # try again
            self.initialize(es.N, es.opts)
        if self.initialized is not True:
            utils.print_warning("dimension not known, damping set to 4",
                'update', 'CMAAdaptSigmaTPA')
            self.initialized = True
        if 1 < 3:
            f_vals = np.asarray(function_values)
            z = sum(f_vals < f_vals[1]) - sum(f_vals < f_vals[0])
            z /= len(f_vals) - 1  # z in [-1, 1]
        elif 1 < 3:
            # use the ranking difference of the mirrors for adaptation
            # damp = 5 should be fine
            z = np.nonzero(es.fit.idx == 1)[0][0] - np.nonzero(es.fit.idx == 0)[0][0]
            z /= es.popsize - 1  # z in [-1, 1]
        self.s = (1 - self.sp.c) * self.s + self.sp.c * np.sign(z) * np.abs(z)**self.sp.z_exponent
        if self.s > 0:
            es.sigma *= np.exp(self.s / self.sp.dampup)
        else:
            es.sigma *= np.exp(self.s / self.sp.dampdown)
        #es.more_to_write.append(10**z)

    def check_consistency(self, es):
        assert isinstance(es.adapt_sigma, CMAAdaptSigmaTPA)
        if es.countiter > 3:
            dm = es.mean[0] - es.mean_old[0]
            dx0 = es.pop[0][0] - es.mean_old[0]
            dx1 = es.pop[1][0] - es.mean_old[0]
            for i in np.random.randint(1, es.N, 1):
                if dx0 * dx1 * (es.pop[0][i] - es.mean_old[i]) * (
                    es.pop[1][i] - es.mean_old[i]):
                    dmi_div_dx0i = (es.mean[i] - es.mean_old[i]) \
                                    / (es.pop[0][i] - es.mean_old[i])
                    dmi_div_dx1i = (es.mean[i] - es.mean_old[i]) \
                                        / (es.pop[1][i] - es.mean_old[i])
                    if not Mh.equals_approximately(
                            dmi_div_dx0i, dm / dx0, 1e-4) or \
                            not Mh.equals_approximately(
                                    dmi_div_dx1i, dm / dx1, 1e-4):
                        utils.print_warning(
                            'TPA: apparent inconsistency with mirrored'
                            ' samples, where dmi_div_dx0i, dm/dx0=%f, %f'
                            ' and dmi_div_dx1i, dm/dx1=%f, %f' % (
                                dmi_div_dx0i, dm/dx0, dmi_div_dx1i, dm/dx1),
                            'check_consistency',
                            'CMAAdaptSigmaTPA', es.countiter)
                else:
                    utils.print_warning('zero delta encountered in TPA which' +
                                        ' \nshould be very rare and might be a bug' +
                                        ' (sigma=%f)' % es.sigma,
                                        'check_consistency', 'CMAAdaptSigmaTPA',
                                        es.countiter)

