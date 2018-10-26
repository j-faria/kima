import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.linalg import cholesky, cho_solve, solve_triangular

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

    def _cov(self, x1, x2=None):
        """ Calculate the kernel matrix evaluated at inputs x1 (and x2) """
        K = self.kernel(x1, x2)
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
        else: K = self._cov(t)
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


    def sample_conditional(self, y, t, size=1):
        mu, cov = self.predict(y, t, return_cov=True)
        return multivariate_gaussian_samples(cov, size, mean=mu).T


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