import sys
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.linalg import cholesky, cho_solve, solve_triangular

try:
    import celerite
    from celerite import GP as GPcel, terms
except ImportError:
    print('Please install celerite: https://celerite.readthedocs.io/en/stable/python/install')
    sys.exit(1)


class QPkernel():
    """ The quasi-periodic kernel """
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
            x1 = x1.reshape(-1,1)

        if x2 is None:
            dists = squareform(pdist(x1, metric='euclidean'))
            argexp = dists / self.eta2
            argsin = np.pi * dists / self.eta3
            K = self.eta1**2 * np.exp(-0.5 * argexp**2
                                      -2 * (np.sin(argsin) / self.eta4) ** 2
                                     )
        else:
            if x2.ndim == 1:
                x2 = x2.reshape(-1,1)
            dists = cdist(x1, x2, metric='euclidean')
            argexp = dists / self.eta2
            argsin = np.pi * dists / self.eta3
            K = self.eta1**2 * np.exp(-0.5 * argexp**2
                                      -2 * (np.sin(argsin) / self.eta4)**2
                                     )

        return K


class GP():
    """ A simple Gaussian process class for regression problems """
    def __init__(self, kernel, t, yerr=None, white_noise=1e-10):
        """ 
        kernel : an instance of `QPkernel`
        t : the independent variable where the GP is "trained"
        (opt) yerr : array of observational uncertainties
        (opt) white_noise : value/array added to the diagonal of the kernel matrix
        """
        assert isinstance(kernel, QPkernel), \
                                "kernel should be instance of QPkernel."
        self.kernel = kernel
        self.t = t
        self.yerr = yerr
        self.white_noise = white_noise

    def _cov(self, x1, x2=None, add2diag=True):
        """ Calculate the kernel matrix evaluated at inputs x1 (and x2) """
        K = self.kernel(x1, x2)
        if add2diag:
            if self.yerr is not None:
                K[np.diag_indices_from(K)] += self.yerr
            K[np.diag_indices_from(K)] += self.white_noise
        return K

    def log_likelihood(self, y):
        """ The log marginal likelihood of observations y under the GP model """
        pass

    def sample(self, t=None, size=1):
        """ Draw samples from the GP prior distribution; 
        Output shape: (t.size, size) 
        """
        if t is None: K = self._cov(self.t)
        else: K = self._cov(t, add2diag=False)
        return multivariate_gaussian_samples(K, size).T

    def predict(self, y, t=None, return_std=False, return_cov=False):
        """ 
        Conditional predictive distribution of the GP model 
        given observations y, evaluated at coordinates t.
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

        mu = self.predict(y, results.t, return_cov=False)

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
        GP_pars_sample = sample[ind]
        s = sample[0]

        self.kernel.setpars(*GP_pars_sample)
        self.white_noise = s
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
        GP_pars_sample = sample[ind]
        s = sample[0]

        self.kernel.setpars(*GP_pars_sample)
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


class QPkernel_celerite(terms.Term):
    # This implements a quasi-periodic kernel (QPK) devised by Andrew Collier
    # Cameron, which mimics a standard QP kernel with a roughness parameter 0.5,
    # and has zero derivative at the origin: k(tau=0)=amp and k'(tau=0)=0
    # The kernel defined in the celerite paper (Eq 56 in Foreman-Mackey et al. 2017)
    # does not satisfy k'(tau=0)=0
    #
    # This new QPK has only 3 parameters, here called eta1, eta2, eta3
    # corresponding to an amplitude, decay timescale and period.
    """
    docs
    """
    parameter_names = ("η1", "η2", "η3")

    def get_real_coefficients(self, params):
        # η1, η2, η3 = params
        return np.zeros(0, dtype=np.float), np.zeros(0, dtype=np.float)

    def get_complex_coefficients(self, params):
        η1, η2, η3 = params
        wbeat = 1.0 / η2
        wrot = 2*np.pi/η3
        amp = η1**2
        c = wbeat
        d = wrot
        x = c/d
        a = amp/2.
        b = amp*x/2.
        e = amp/8.
        f = amp*x/4.
        g = amp*(3./8. + 0.001)
        return (
            np.reshape([a, e, g], (3,)),
            np.reshape([b, f, 0], (3,)),
            np.reshape([c, c, c], (3,)),
            np.reshape([d, 2*d, 0], (3,)),
        )

    @property
    def J(self):
        return 6

    def setpars(self, eta1=None, eta2=None, eta3=None, eta4=None):
        p = [eta1, eta2, eta3]
        # if eta1: p.append(eta1)
        # if eta2: p.append(eta2)
        # if eta3: p.append(eta3)
        self.set_parameter_vector(p)


class GP_celerite():
    """ A simple Gaussian process class for regression problems, based on celerite """
    def __init__(self, kernel, t, yerr=None, white_noise=1e-10):
        """ 
        t : the independent variable where the GP is "trained"
        (opt) yerr : array of observational uncertainties
        (opt) white_noise : value/array added to the diagonal of the kernel matrix
        """
        self.kernel = kernel
        self.t = t
        self.yerr = yerr
        self.white_noise = white_noise
        self._GP = GPcel(self.kernel)
        self._GP.compute(self.t, self.yerr)

    def _cov(self, x1, x2=None, add2diag=True):
        raise NotImplementedError

    def log_likelihood(self, y):
        """ The log marginal likelihood of observations y under the GP model """
        raise NotImplementedError

    def sample(self, t=None, size=1):
        """ Draw samples from the GP prior distribution; 
        Output shape: (t.size, size) 
        """
        raise NotImplementedError

    def predict(self, y, t=None, return_std=False, return_cov=False):
        """ 
        Conditional predictive distribution of the GP model 
        given observations y, evaluated at coordinates t.
        """
        return self._GP.predict(y, t, return_cov=return_cov, return_var=return_std)


    def predict_with_hyperpars(self, results, sample, t=None, add_parts=True):
        """ 
        Given the parameters in `sample`, return the GP predictive mean. If `t`
        is None, the prediction is done at the observed times and will contain
        other model components (systemic velocity, trend, instrument offsets) if
        `add_parts` is True. Otherwise, when `t` is given, the prediction is
        made at times `t` and *does not* contain these extra model components.
        """
        raise NotImplementedError

    def sample_conditional(self, y, t, size=1):
        """ 
        Sample from the conditional predictive distribution of the GP model 
        given observations y, at coordinates t.
        """
        raise NotImplementedError


    def sample_from_posterior(self, results, size=1):
        """ 
        Given the posterior sample in `results`, take one sample of the GP
        hyperparameters and the white noise, and return a sample from the GP
        prior given those parameters.
        """
        raise NotImplementedError

    def sample_with_hyperpars(self, results, sample, size=1):
        """ 
        Given the value of the hyperparameters and the white noise in `sample`, return a 
        sample from the GP prior.
        """
        raise NotImplementedError


    def sample_conditional_from_posterior(self, results, t, size=1):
        """ 
        Given the posterior sample in `results`, take one sample of the GP
        hyperparameters and the white noise, and return a sample from the GP
        predictive given those parameters.
        """
        raise NotImplementedError

    def sample_conditional_with_hyperpars(self, results, sample, t, size=1):
        """ 
        Given the value of the hyperparameters and the white noise in `sample`,
        return a sample from the GP predictive.
        """
        raise NotImplementedError


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
