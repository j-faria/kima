from dataclasses import dataclass
from functools import partial

import numpy as np
# from numpy import exp, sin, cos, sqrt, abs

from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.numpy import exp, sin, cos, sqrt
from jax import jit, vmap, jacfwd

from scipy.spatial.distance import squareform, pdist, cdist
from scipy.linalg import cholesky, cho_solve, solve_triangular


__all__ = [
    'QPkernel',
    'QPCkernel',
    'QPMatern32kernel',
    'QPMatern52kernel',
    'QPRQkernel',
    'SqExpkernel',
    'GP',
]


π = np.pi


class Kernel:

    @property
    def pars(self):
        return np.array([getattr(self, n) for n in self.names])

    @pars.setter
    def pars(self, newpars):
        newpars = np.atleast_1d(newpars).astype(float)
        if newpars.size != self.npars:
            name = self.__class__.__name__
            raise ValueError(
                f'{name} expects {self.npars} parameters, got {newpars.size}')
        for n, p in zip(self.names, newpars):
            setattr(self, n, p)

    # @partial(jit, static_argnums=(0,))
    # def metric(self, x1, x2=None):
    #     if x2 is None:
    #         return abs(x1[None, :] - x1[:, None])
    #     else:
    #         if x1.ndim == 1:
    #             x1 = x1.reshape(-1, 1)
    #         if x2.ndim == 1:
    #             x2 = x2.reshape(-1, 1)
    #         return abs(x1 - x2.T)

    @partial(jit, static_argnums=(0,))
    def __call__(self, xi, xj=None):
        if xj is None:
            return vmap(lambda x: vmap(lambda y: self.eval(x, y))(xi))(xi)
        else:
            return vmap(lambda x: vmap(lambda y: self.eval(x, y))(xi))(xj).T

    @partial(jit, static_argnums=(0,))
    def gradient_wrt_x1(self, xi, xj=None):
        i = jnp.arange(xi.size)
        return jacfwd(self.__call__, argnums=0)(xi, xj)[i, :, i]



@dataclass(eq=True, unsafe_hash=True)
class COSkernel(Kernel):
    """ The cosine kernel """
    η1: float
    P: float
    names = ['η1', 'P']
    npars = 2

    @partial(jit, static_argnums=(0,))
    def eval(self, xi, xj):
        r = xi - xj
        K = self.η1**2 * cos(2 * π * r / self.P)
        return K


@dataclass(eq=True, unsafe_hash=True)
class DopplerCOSkernel(Kernel):
    """ The cosine kernel """
    η1: float
    P: float
    RV: float
    names = ['η1', 'P', 'RV']
    npars = 3

    @partial(jit, static_argnums=(0,))
    def eval(self, xi, xj):
        α = 1 + self.RV / 299792458.0
        r = α * (xi - xj)
        K = self.η1**2 * cos(2 * π * r / self.P)
        return K


@dataclass(eq=True, unsafe_hash=True)
class RBFkernel(Kernel):
    """ The radial basis function (exponential squared) kernel """
    η1: float
    η2: float
    names = ['η1', 'η2']
    npars = 2

    @partial(jit, static_argnums=(0,))
    def eval(self, xi, xj):
        r = xi - xj
        sr = r / self.η2
        K = self.η1**2 * exp(-0.5 * sr**2)
        return K


@dataclass(eq=True, unsafe_hash=True)
class QPkernel(Kernel):
    """ The quasi-periodic kernel """
    η1: float
    η2: float
    η3: float
    η4: float
    names = ['η1', 'η2', 'η3', 'η4']
    npars = 4

    @partial(jit, static_argnums=(0,))
    def eval(self, xi, xj):
        r = xi - xj
        sr = r / self.η2
        pr = π * r / self.η3
        K = self.η1**2 * exp(-0.5 * sr**2 - 2 * (sin(pr) / self.η4)**2)
        return K


@dataclass(eq=True, unsafe_hash=True)
class QPCkernel(Kernel):
    """ The quasi-periodic-cosine kernel [Perger+2020]"""
    η1: float
    η2: float
    η3: float
    η4: float
    η5: float
    names = ['η1', 'η2', 'η3', 'η4', 'η5']
    npars = 5

    def eval(self, xi, xj):
        r = xi - xj
        sr = r / self.η2
        pr = π * r / self.η3
        η1 = self.η1**2
        η5 = self.η5**2
        K = exp(-0.5 * sr**2) * (η1 * exp(-2 * (sin(pr) / self.η4)**2) +
                                 η5 * cos(4 * pr))
        return K


@dataclass(eq=True, unsafe_hash=True)
class QPpCkernel(Kernel):
    """ The quasi-periodic *plus* cosine kernel """
    η1: float
    η2: float
    η3: float
    η4: float
    η5: float
    η6: float
    names = ['η1', 'η2', 'η3', 'η4', 'η5', 'η6']
    npars = 6

    @partial(jit, static_argnums=(0,))
    def eval(self, xi, xj):
        r = xi - xj
        sr = r / self.η2
        pr = π * r / self.η3
        K = self.η1**2 * exp(-0.5 * sr**2 - 2 * (sin(pr) / self.η4)**2) \
            + self.η5**2 * cos(2 * π * r / self.η6)
        return K


@dataclass(eq=True, unsafe_hash=True)
class PERkernel(Kernel):
    """ The periodic kernel [Rasmussen & Williams 2006]"""
    η1: float
    η3: float
    η4: float
    names = ['η1', 'η3', 'η4']
    npars = 3

    def eval(self, xi, xj):
        r = xi - xj
        pr = π * r / self.η3
        K = self.η1**2 * exp(- 2 * (sin(pr) / self.η4)**2)
        return K


class QPMatern32kernel():
    """ The quasi-periodic kernel based on the Matern 3/2 """
    def __init__(self, eta1, eta2, eta3, eta4):
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.eta4 = eta4

    def setpars(self, eta1=None, eta2=None, eta3=None, eta4=None):
        self.eta1 = eta1 if eta1 else self.eta1
        self.eta2 = eta2 if eta2 else self.eta2
        self.eta3 = eta3 if eta3 else self.eta3
        self.eta4 = eta4 if eta4 else self.eta4

    def __call__(self, x1, x2=None):
        if x1.ndim == 1:
            x1 = x1.reshape(-1, 1)

        if x2 is None:
            dists = squareform(pdist(x1, metric='euclidean'))
        else:
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            dists = cdist(x1, x2, metric='euclidean')

        argexp = dists / self.eta2
        argsin = np.pi * dists / self.eta3
        K = self.eta1**2 \
            * exp(-0.5 * argexp**2) \
            * (1 + sqrt(3) * 2 * np.abs(sin(argsin) / self.eta4)) \
            * exp(-sqrt(3) * 2 * np.abs(sin(argsin) / self.eta4))

        return K


class QPMatern52kernel():
    """ The quasi-periodic kernel based on the Matern 5/2 """
    def __init__(self, eta1, eta2, eta3, eta4):
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.eta4 = eta4

    def setpars(self, eta1=None, eta2=None, eta3=None, eta4=None):
        self.eta1 = eta1 if eta1 else self.eta1
        self.eta2 = eta2 if eta2 else self.eta2
        self.eta3 = eta3 if eta3 else self.eta3
        self.eta4 = eta4 if eta4 else self.eta4

    def __call__(self, x1, x2=None):
        if x1.ndim == 1:
            x1 = x1.reshape(-1, 1)

        if x2 is None:
            dists = squareform(pdist(x1, metric='euclidean'))
        else:
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            dists = cdist(x1, x2, metric='euclidean')

        argexp = dists / self.eta2
        argsin = np.pi * dists / self.eta3
        s = 2 * np.abs(sin(argsin))

        K = self.eta1**2 \
            * exp(-0.5 * argexp**2) \
            * (1 + sqrt(5) * s / self.eta4 + 5 * s**2 / (3 * self.eta4**2)) \
            * exp(-sqrt(5) * s / self.eta4)

        return K


class QPRQkernel():
    """ The quasi-periodic kernel based on the Rational Quadratic """
    def __init__(self, eta1, eta2, eta3, eta4, alpha):
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.eta4 = eta4
        self.alpha = alpha

    def setpars(self, eta1=None, eta2=None, eta3=None, eta4=None, alpha=None):
        self.eta1 = eta1 if eta1 else self.eta1
        self.eta2 = eta2 if eta2 else self.eta2
        self.eta3 = eta3 if eta3 else self.eta3
        self.eta4 = eta4 if eta4 else self.eta4
        self.alpha = alpha if alpha else self.alpha

    def __call__(self, x1, x2=None):
        if x1.ndim == 1:
            x1 = x1.reshape(-1, 1)

        if x2 is None:
            dists = squareform(pdist(x1, metric='euclidean'))
        else:
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            dists = cdist(x1, x2, metric='euclidean')

        argexp = dists / self.eta2
        s = np.abs(sin(np.pi * dists / self.eta3))

        K = self.eta1**2 * exp(-0.5 * argexp**2) \
                * (1 + 2*s**2/(self.alpha*self.eta4**2))**(-self.alpha)

        return K


class SqExpkernel():
    """ The quasi-periodic kernel """
    def __init__(self, eta1, eta2):
        self.eta1 = eta1
        self.eta2 = eta2

    def setpars(self, eta1=None, eta2=None):
        self.eta1 = eta1 if eta1 else self.eta1
        self.eta2 = eta2 if eta2 else self.eta2

    def __call__(self, x1, x2=None):
        if x1.ndim == 1:
            x1 = x1.reshape(-1, 1)

        if x2 is None:
            dists = squareform(pdist(x1, metric='euclidean'))
            argexp = dists / self.eta2
            K = self.eta1**2 * np.exp(-0.5 * argexp**2)
        else:
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            dists = cdist(x1, x2, metric='euclidean')
            argexp = dists / self.eta2
            K = self.eta1**2 * np.exp(-0.5 * argexp**2)

        return K


class GP():
    """ A simple Gaussian process class for regression problems """
    def __init__(self, kernel, t, yerr=None, white_noise=1.25e-12):
        """
        kernel : an instance of `QPkernel`
        t : the independent variable where the GP is "trained"
        (opt) yerr : array of observational uncertainties
        (opt) white_noise : value/array added to the diagonal of the kernel matrix
        """
        self.kernel = kernel
        self.t = t
        self.yerr = np.zeros_like(t) if yerr is None else yerr
        self.white_noise = white_noise
        self._C1 = -0.5 * t.size * np.log(2 * π)

    def _cov(self, xi, xj=None, add2diag=True):
        """ Calculate the kernel matrix evaluated at inputs xi (and xj) """
        K = self.kernel(xi, xj)
        if add2diag:
            K = K + jnp.diag(self.yerr**2)
            K = K + jnp.eye(self.t.size) * self.white_noise**2
        return K

    def log_likelihood(self, y):
        """
        The log marginal likelihood of observations y under the GP model
        """
        self.K = self._cov(self.t)
        self.L_ = cholesky(self.K, lower=True)
        self.alpha_ = cho_solve((self.L_, True), y)
        log_determinant = 2 * np.log(np.diag(self.L_)).sum()
        return self._C1 - 0.5 * log_determinant - 0.5 * y.dot(self.alpha_)

    def sample(self, t=None, size=1):
        """ Draw samples from the GP prior distribution;
        Output shape: (t.size, size) 
        """
        if t is None:
            K = self._cov(self.t)
        else:
            K = self._cov(t, add2diag=False)
        return multivariate_gaussian_samples(K, size).T

    def predict(self, y, t=None, return_std=False, return_cov=False):
        """
        Conditional predictive distribution of the GP model given observations
        y, evaluated at coordinates t.
        """
        if t is None:
            t = self.t

        self.K = self._cov(self.t)
        self.L_ = cholesky(self.K, lower=True)
        self.alpha_ = cho_solve((self.L_, True), y)
        K_trans = self.kernel(t, self.t)
        mean = K_trans.dot(self.alpha_)

        if return_cov:
            v = cho_solve((self.L_, True), K_trans.T)  # Line 5
            cov = self.kernel(t) - K_trans.dot(v)  # Line 6
            return mean, cov
        elif return_std:
            # compute inverse K_inv of K based on its Cholesky
            # decomposition L and its inverse L_inv
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            self._K_inv = L_inv.dot(L_inv.T)

            # Compute variance of predictive distribution
            var = np.diag(self.kernel(t))
            var = var - np.einsum("ij,ij->i",
                                  np.dot(K_trans, self._K_inv), K_trans)

            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            var_negative = var < 0
            if np.any(var_negative):
                var[var_negative] = 0.0
            return mean, np.sqrt(var)
        else:
            return mean

    def derivative(self, y, t=None, return_std=False, return_cov=False):
        """
        First derivative of the conditional predictive distribution of the GP
        model given observations y, evaluated at coordinates t.
        """
        if t is None:
            t = self.t

        self.K = self._cov(self.t)
        self.L_ = cholesky(self.K, lower=True)
        self.alpha_ = cho_solve((self.L_, True), y)

        deriv = self.kernel.gradient_wrt_x1(self.t, t).T @ self.alpha_
        return deriv

        K_trans = self.kernel(t, self.t)
        mean = K_trans.dot(self.alpha_)

        if return_cov:
            v = cho_solve((self.L_, True), K_trans.T)  # Line 5
            cov = self.kernel(t) - K_trans.dot(v)  # Line 6
            return mean, cov
        elif return_std:
            # compute inverse K_inv of K based on its Cholesky
            # decomposition L and its inverse L_inv
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            self._K_inv = L_inv.dot(L_inv.T)

            # Compute variance of predictive distribution
            var = np.diag(self.kernel(t))
            var = var - np.einsum("ij,ij->i",
                                  np.dot(K_trans, self._K_inv), K_trans)

            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            var_negative = var < 0
            if np.any(var_negative):
                var[var_negative] = 0.0
            return mean, np.sqrt(var)
        else:
            return mean

    def predict_with_hyperpars(self, results, sample, t=None, add_parts=True,
                               return_std=False):
        """ 
        Given the parameters in `sample`, return the GP predictive mean. If `t`
        is None, the prediction is done at the observed times and will contain
        other model components (systemic velocity, trend, instrument offsets) if
        `add_parts` is True. Otherwise, when `t` is given, the prediction is
        made at times `t` and *does not* contain these extra model components.
        """
        ind = results.indices['GPpars']
        GP_pars_sample = sample[ind]
        s = sample[0]

        self.kernel.setpars(*GP_pars_sample)
        self.white_noise = s

        y = results.y.copy()
        # put all data around 0
        y -= sample[-1]
        if results.multi:
            for i in range(1, results.n_instruments):
                y[results.obs == i] -= sample[results.indices['inst_offsets']][i-1]
        # and subtract the trend
        if results.trend:
            y -= sample[results.indices['trend']] * (results.t - results.tmiddle)

        if t is not None:
            pred = self.predict(y, t, return_cov=False, return_std=return_std)
            return pred

        mu = self.predict(y, results.t, return_cov=False,
                          return_std=return_std)

        if add_parts:
            # add the trend back
            mu += sample[results.indices['trend']] * (results.t - results.tmiddle)
            # add vsys and instrument offsets back
            mu += sample[-1]
            if results.multi:
                for i in range(1, results.n_instruments):
                    mu[results.obs == i] += sample[results.indices['inst_offsets']][i-1]

        return mu

    def sample_conditional(self, y, t, size=1):
        """ 
        Sample from the conditional predictive distribution of the GP model 
        given observations y, at coordinates t.
        """
        mu, cov = self.predict(y, t, return_cov=True)
        return multivariate_gaussian_samples(cov, size, mean=mu).T

    def sample_from_posterior(self, results, size=1):
        """ 
        Given the posterior sample in `results`, take one sample of the GP
        hyperparameters and the white noise, and return a sample from the GP
        prior given those parameters.
        """
        # choose a random sample
        i = np.random.choice(range(results.posterior_sample.shape[0]))
        sample = results.posterior_sample[i, :]
        ind = results.indices['GPpars']
        GPpars = sample[ind]
        s = sample[0]

        self.kernel.setpars(*GPpars)
        self.white_noise = s
        return self.sample(size=size)

    def sample_with_hyperpars(self, results, sample, size=1):
        """ 
        Given the value of the hyperparameters and the white noise in `sample`, return a 
        sample from the GP prior.
        """
        ind = results.indices['GPpars']
        GPpars = sample[ind]
        # s = sample[0]

        if results.model == 'RVFWHMmodel':
            η1RV, η1FWHM, η2RV, η2FWHM, η3RV, η3FWHM, η4RV, η4FWHM = \
                GPpars[results._GP_par_indices]

            if results.GPkernel == 'standard':
                results.GP1.kernel.pars = np.array([η1RV, η2RV, η3RV, η4RV])
                results.GP2.kernel.pars = np.array([η1FWHM, η2FWHM, η3FWHM, η4FWHM])
        else:
            results.GP.kernel.pars = GPpars

        return self.sample(size=size)

    def sample_conditional_from_posterior(self, results, t, size=1):
        """ 
        Given the posterior sample in `results`, take one sample of the GP
        hyperparameters and the white noise, and return a sample from the GP
        predictive given those parameters.
        """
        i = np.random.choice(range(results.posterior_sample.shape[0]))
        ind = results.indices['GPpars']
        sample = results.posterior_sample[i,:]
        GPpars = sample[ind]
        s = sample[0]

        self.kernel.setpars(*GPpars)
        self.white_noise = s

        # put all data around 0
        y = results.y.copy()
        y -= sample[-1]
        if results.multi:
            for i in range(1, results.n_instruments):
                y[results.obs == i] -= sample[results.indices['inst_offsets']]
        # and subtract the trend
        if results.trend:
            y -= sample[results.indices['trend']] * (t - results.tmiddle)

        mu, cov = self.predict(y, t, return_cov=True)
        return multivariate_gaussian_samples(cov, size, mean=mu).T

    def sample_conditional_with_hyperpars(self, results, sample, t, size=1):
        """ 
        Given the value of the hyperparameters and the white noise in `sample`,
        return a sample from the GP predictive.
        """
        ind = results.indices['GPpars']
        GPpars = sample[ind]
        s = sample[0]

        if results.model == 'RVFWHMmodel':
            η1RV, η1FWHM, η2RV, η2FWHM, η3RV, η3FWHM, η4RV, η4FWHM = \
                GPpars[results._GP_par_indices]

            if results.GPkernel == 'standard':
                results.GP1.kernel.pars = np.array([η1RV, η2RV, η3RV, η4RV])
                results.GP2.kernel.pars = np.array([η1FWHM, η2FWHM, η3FWHM, η4FWHM])
        else:
            results.GP.kernel.pars = GPpars

        if results.model == 'RVFWHMmodel':
            # which GP am I?
            if (results.e == self.yerr).all():
                print('1')
                y = results.y - results.eval_model(sample)[0]
            else:
                print('2')
                y = results.y2 - results.eval_model(sample)[1]
        else:
            y = results.y - results.eval_model(sample)

        mu, cov = self.predict(y, t, return_cov=True)
        return multivariate_gaussian_samples(cov, size, mean=mu).T


from tinygp import GaussianProcess
from tinygp.kernels import Matern32, ExpSquared, ExpSineSquared
from tinygp import transforms


class mixtureGP(GP):
    def __init__(self, kernels, X, dims, yerr=None, white_noise=1.25e-12):
        self.X = X
        self.wn = white_noise

    @partial(jit, static_argnums=(0,))
    def build_gp_predict(self, y, T, params, X, wn):
        qp = ExpSquared(params["scale1"]) \
              * ExpSineSquared(scale=params["period"], gamma=params["gamma"])
        kernel1 = params["amp1"] * transforms.Subspace(0, qp)
        kernel2 = params["amp2"] * transforms.Subspace(1, ExpSquared(scale=params["scale2"]))

        kernel = kernel1 + kernel2
        gp = GaussianProcess(kernel, X, diag=wn)
        return gp.predict(y, T)

    def predict(self, y, T, pars):
        params = {
            'amp1': pars[0],
            'scale1': pars[1],
            'period': pars[2],
            'gamma': pars[3],
            'amp2': pars[4],
            'scale2': pars[5],
        }
        mu = self.build_gp_predict(y, T, params, self.X, self.wn)
        return mu

    def predict_separate(self, y, T, pars):
        params = {
            'amp1': pars[0],
            'scale1': pars[1],
            'period': pars[2],
            'gamma': pars[3],
            'amp2': pars[4],
            'scale2': pars[5],
        }
        gp = self.build_gp(params)
        mu1 = gp.predict(y, T, kernel=gp.kernel.kernel1)
        mu2 = gp.predict(y, T, kernel=gp.kernel.kernel2)
        return mu1, mu2


# class QPkernel_celerite(terms.Term):
#     # This implements a quasi-periodic kernel (QPK) devised by Andrew Collier
#     # Cameron, which mimics a standard QP kernel with a roughness parameter 0.5,
#     # and has zero derivative at the origin: k(tau=0)=amp and k'(tau=0)=0
#     # The kernel defined in the celerite paper (Eq 56 in Foreman-Mackey et al. 2017)
#     # does not satisfy k'(tau=0)=0
#     #
#     # This new QPK has only 3 parameters, here called eta1, eta2, eta3
#     # corresponding to an amplitude, decay timescale and period.
#     """
#     docs
#     """
#     parameter_names = ("η1", "η2", "η3")

#     def get_real_coefficients(self, params):
#         # η1, η2, η3 = params
#         return np.zeros(0, dtype=np.float), np.zeros(0, dtype=np.float)

#     def get_complex_coefficients(self, params):
#         η1, η2, η3 = params
#         wbeat = 1.0 / η2
#         wrot = 2*np.pi/η3
#         amp = η1**2
#         c = wbeat
#         d = wrot
#         x = c/d
#         a = amp/2.
#         b = amp*x/2.
#         e = amp/8.
#         f = amp*x/4.
#         g = amp*(3./8. + 0.001)
#         return (
#             np.reshape([a, e, g], (3,)),
#             np.reshape([b, f, 0], (3,)),
#             np.reshape([c, c, c], (3,)),
#             np.reshape([d, 2*d, 0], (3,)),
#         )

#     @property
#     def J(self):
#         return 6

#     def setpars(self, eta1=None, eta2=None, eta3=None, eta4=None):
#         p = [eta1, eta2, eta3]
#         # if eta1: p.append(eta1)
#         # if eta2: p.append(eta2)
#         # if eta3: p.append(eta3)
#         self.set_parameter_vector(p)


# class GP_celerite():
#     """ A simple Gaussian process class for regression problems, based on celerite """
#     def __init__(self, kernel, t, yerr=None, white_noise=1e-10):
#         """
#         t : the independent variable where the GP is "trained"
#         (opt) yerr : array of observational uncertainties
#         (opt) white_noise : value/array added to the diagonal of the kernel matrix
#         """
#         self.kernel = kernel
#         self.t = t
#         self.yerr = yerr
#         self.white_noise = white_noise
#         self._GP = GPcel(self.kernel)
#         self._GP.compute(self.t, self.yerr)

#     def _cov(self, x1, x2=None, add2diag=True):
#         raise NotImplementedError

#     def log_likelihood(self, y):
#         """ The log marginal likelihood of observations y under the GP model """
#         raise NotImplementedError

#     def sample(self, t=None, size=1):
#         """ Draw samples from the GP prior distribution;
#         Output shape: (t.size, size)
#         """
#         raise NotImplementedError

#     def predict(self, y, t=None, return_std=False, return_cov=False):
#         """
#         Conditional predictive distribution of the GP model
#         given observations y, evaluated at coordinates t.
#         """
#         return self._GP.predict(y, t, return_cov=return_cov, return_var=return_std)


#     def predict_with_hyperpars(self, results, sample, t=None, add_parts=True):
#         """
#         Given the parameters in `sample`, return the GP predictive mean. If `t`
#         is None, the prediction is done at the observed times and will contain
#         other model components (systemic velocity, trend, instrument offsets) if
#         `add_parts` is True. Otherwise, when `t` is given, the prediction is
#         made at times `t` and *does not* contain these extra model components.
#         """
#         raise NotImplementedError

#     def sample_conditional(self, y, t, size=1):
#         """
#         Sample from the conditional predictive distribution of the GP model
#         given observations y, at coordinates t.
#         """
#         raise NotImplementedError


#     def sample_from_posterior(self, results, size=1):
#         """
#         Given the posterior sample in `results`, take one sample of the GP
#         hyperparameters and the white noise, and return a sample from the GP
#         prior given those parameters.
#         """
#         raise NotImplementedError

#     def sample_with_hyperpars(self, results, sample, size=1):
#         """
#         Given the value of the hyperparameters and the white noise in `sample`, return a
#         sample from the GP prior.
#         """
#         raise NotImplementedError


#     def sample_conditional_from_posterior(self, results, t, size=1):
#         """
#         Given the posterior sample in `results`, take one sample of the GP
#         hyperparameters and the white noise, and return a sample from the GP
#         predictive given those parameters.
#         """
#         raise NotImplementedError

#     def sample_conditional_with_hyperpars(self, results, sample, t, size=1):
#         """
#         Given the value of the hyperparameters and the white noise in `sample`,
#         return a sample from the GP predictive.
#         """
#         raise NotImplementedError


def multivariate_gaussian_samples(matrix, N, mean=None):
    """
    Generate samples from a multidimensional Gaussian with a given covariance.

    :param matrix: ``(k, k)``
        The covariance matrix.

    :param N:
        The number of samples to generate.

    :param mean: ``(k,)`` (optional)
        The mean of the Gaussian. Assumed to be zero if not given.

    :returns samples: ``(k,)`` or ``(N, k)``
        Samples from the given multivariate normal.

    """
    if mean is None:
        mean = np.zeros(len(matrix))
    samples = np.random.multivariate_normal(mean, matrix, N)
    if N == 1:
        return samples[0]
    return samples
