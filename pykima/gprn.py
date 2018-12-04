#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:48:13 2018

@author: joaocamacho
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from copy import copy
#import matplotlib.pyplot as plt
#from scipy.stats import multivariate_normal


#because it will make my life easier down the line
pi, exp, sine, cosine, sqrt = np.pi, np.exp, np.sin, np.cos, np.sqrt


##### Kernel class #############################################################
class Kernel(object):
    """
        Definition the kernels for our GPRN, by default and  because it 
        simplifies my life, all kernels include a white noise term when we are
        working with the nodes making it the f hat in Wilson et al. (2012)
    """
    def __init__(self, *args):
        """
            Puts all kernel arguments in an array pars
        """
        self.pars = np.array(args)

    def __call__(self, r, t1 = None, t2=None):
        """
            r = t - t'
        """
        raise NotImplementedError

    def __repr__(self):
        """
            Representation of each kernel instance
        """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

##### Nodes
# Constant kernel
class Constant_node(Kernel):
    """
        This kernel returns its constant argument c with white noise
        Parameters:
            c = constant
            wn = white noise amplitude
    """
    def __init__(self, c, wn):
        super(Constant_node, self).__init__(c, wn)
        self.c = c
        self.c = wn

    def __call__(self, r):
        try:
            return self.c * np.ones_like(r) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.c * np.ones_like(r)

# Squared exponential kernel
class SquaredExponential_node(Kernel):
    """
        Squared Exponential kernel, also known as radial basis function or RBF 
    kernel in other works.
        Parameters:
            theta = amplitude
            ell = length-scale
            wn = white noise
    """
    def __init__(self, ell, wn):
        super(SquaredExponential_node, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try:
            return exp(-0.5 * r**2 / self.ell**2) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp(-0.5 * r**2 / self.ell**2)

# Periodic kernel 
class Periodic_node(Kernel):
    """
        Definition of the periodic kernel.
        Parameters:
            ell = lenght scale
            P = period
            wn = white noise
    """
    def __init__(self, ell, P, wn):
        super(Periodic_node, self).__init__(ell, P, wn)
        self.ell = ell
        self.P = P
        self.wn = wn

    def __call__(self, r):
        try:
            return exp( -2 * sine(pi*np.abs(r)/self.P)**2 / self.ell**2) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp( -2 * sine(pi*np.abs(r)/self.P)**2 / self.ell**2)

# Quasi-periodic kernel
class QuasiPeriodic_node(Kernel):
    """
        This kernel is the product between the exponential sine squared kernel 
    and the squared exponential kernel, commonly known as the quasi-periodic 
    kernel.
        Parameters:
            ell_e = evolutionary time scale
            P = kernel periodicity
            ell_p = length scale of the periodic component
            wn = white noise
    """
    def __init__(self, ell_e, P, ell_p, wn):
        super(QuasiPeriodic_node, self).__init__(ell_e, P, ell_p, wn)
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        try:
            return exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                       /self.ell_p**2 - r**2/(2*self.ell_e**2)) \
                       + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                       /self.ell_p**2 - r**2/(2*self.ell_e**2))

# Rational quadratic kernel 
class RationalQuadratic_node(Kernel):
    """
        Definition of the rational quadratic kernel.
        Parameters:
            alpha = weight of large and small scale variations
            ell = characteristic lenght scale to define the kernel "smoothness"
            wn = white noise amplitude
    """
    def __init__(self, alpha, ell, wn):
        super(RationalQuadratic_node, self).__init__(alpha, ell, wn)
        self.alpha = alpha
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try: 
            return 1 / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return 1 / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha

# Cosine kernel
class Cosine_node(Kernel):
    """
        Definition of the cosine kernel.
        Parameters:
            P = period
            wn = white noise amplitude
    """
    def __init__(self, P, wn):
        super(Cosine_node, self).__init__(P, wn)
        self.P = P
        self.wn = wn

    def __call__(self, r):
        try:
            return cosine(2*pi*np.abs(r) / self.P) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return cosine(2*pi*np.abs(r) / self.P)

# Exponential kernel
class Exponential_node(Kernel):
    """
        Definition of the exponential kernel. This kernel arises when 
    setting v=1/2 in the matern family of kernels
        Parameters:
            ell = characteristic lenght scale
            wn = white noise amplitude
    """
    def __init__(self, ell, wn):
        super(Exponential_node, self).__init__(ell, wn)
        self.ell = ell
        self.wn

    def __call__(self, r): 
        try:
            return exp(- np.abs(r)/self.ell) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp(- np.abs(r)/self.ell) 

# Matern 3/2 kernel
class Matern32_node(Kernel):
    """
        Definition of the Matern 3/2 kernel. This kernel arise when setting 
    v=3/2 in the matern family of kernels
        Parameters:
            ell = characteristic lenght scale
            wn = white noise amplitude
    """
    def __init__(self, ell, wn):
        super(Matern32_node, self).__init__(ell, wn)
        self.ell = ell

    def __call__(self, r):
        try:
            return (1.0 + np.sqrt(3.0)*np.abs(r)/self.ell) \
                        *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return (1.0 + np.sqrt(3.0)*np.abs(r)/self.ell) \
                        *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell)

# Matern 5/2 kernel
class Matern52_node(Kernel):
    """
        Definition of the Matern 5/2 kernel. This kernel arise when setting 
    v=5/2 in the matern family of kernels
        Parameters:
            ell = characteristic lenght scale  
            wn = white noise amplitude
    """
    def __init__(self, ell, wn):
        super(Matern52_node, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try:
            return (1.0 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                           *exp(-np.sqrt(5.0)*np.abs(r)/self.ell) \
                           + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return (1.0 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                           *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)

##### Weights
# Constant kernel
class Constant_weight(Kernel):
    """
        This kernel returns its constant argument c 
        Parameters:
            c = constant
    """
    def __init__(self, c):
        super(Constant_weight, self).__init__(c)
        self.c = c

    def __call__(self, r):
        return self.c * np.ones_like(r)

# Squared exponential kernel
class SquaredExponential_weight(Kernel):
    """
        Squared Exponential kernel, also known as radial basis function or RBF 
    kernel in other works.
        Parameters:
            weight = weight/amplitude of the kernel
            ell = length-scale
    """
    def __init__(self, weight, ell):
        super(SquaredExponential_weight, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell

    def __call__(self, r):
        return self.weight**2 * exp(-0.5 * r**2 / self.ell**2)

# Periodic kernel
class Periodic_weight(Kernel):
    """
        Definition of the periodic kernel.
        Parameters:
            weight = weight/amplitude of the kernel
            ell = lenght scale
            P = period
    """
    def __init__(self, weight, ell, P):
        super(Periodic_weight, self).__init__(weight, ell, P)
        self.weight = weight
        self.ell = ell
        self.P = P
        self.type = 'non-stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        return self.weight**2 * exp( -2 * sine(pi*np.abs(r)/self.P)**2 /self.ell**2)

# Quasi-periodic kernel
class QuasiPeriodic_weight(Kernel):
    """
        This kernel is the product between the exponential sine squared kernel 
    and the squared exponential kernel, commonly known as the quasi-periodic 
    kernel.
        Parameters:
            weight = weight/amplitude of the kernel
            ell_e = evolutionary time scale
            ell_p = length scale of the Periodic component
            P = kernel Periodicity
    """
    def __init__(self, weight, ell_e, P, ell_p):
        super(QuasiPeriodic_weight, self).__init__(weight, ell_e, P, ell_p)
        self.weight = weight
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call__(self, r):
        return self.weight**2 *exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                   /self.ell_p**2 - r**2/(2*self.ell_e**2))

# Rational quadratic kernel 
class RationalQuadratic_weight(Kernel):
    """
        Definition of the rational quadratic kernel.
        Parameters:
            weight = weight/amplitude of the kernel
            alpha = weight of large and small scale variations
            ell = characteristic lenght scale to define the kernel "smoothness"
    """
    def __init__(self, weight, alpha, ell):
        super(RationalQuadratic_weight, self).__init__(weight, alpha, ell)
        self.weight = weight
        self.alpha = alpha
        self.ell = ell

    def __call__(self, r):
        return self.weight**2 / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha

# Cosine kernel
class Cosine_weight(Kernel):
    """
        Definition of the cosine kernel.
        Parameters:
            weight = weight/amplitude of the kernel
            P = period
    """
    def __init__(self, weight, P):
        super(Cosine_weight, self).__init__(weight, P)
        self.weight = weight
        self.P = P

    def __call__(self, r):
        return self.weight**2 * cosine(2*pi*np.abs(r) / self.P)

# Exponential kernel
class Exponential_weight(Kernel):
    """
        Definition of the exponential kernel. This kernel arises when 
    setting v=1/2 in the matern family of kernels
        Parameters:
            weight = weight/amplitude of the kernel
            ell = characteristic lenght scale
    """
    def __init__(self, weight, ell):
        super(Exponential_weight, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r): 
        return self.weight**2 * exp(- np.abs(r)/self.ell)

# Matern 3/2 kernel 
class Matern32_weight(Kernel):
    """
        Definition of the Matern 3/2 kernel. This kernel arise when setting 
    v=3/2 in the matern family of kernels
        Parameters:
            weight = weight/amplitude of the kernel
            ell = characteristic lenght scale
    """
    def __init__(self, weight, ell):
        super(Matern32_weight, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        return self.weight**2 *(1.0 + np.sqrt(3.0)*np.abs(r)/self.ell) \
                    *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell)

# Matern 5/2 kernel
class Matern52_weight(Kernel):
    """
        Definition of the Matern 5/2 kernel. This kernel arise when setting 
    v=5/2 in the matern family of kernels
        Parameters:
            weight = weight/amplitude of the kernel
            ell = characteristic lenght scale  
    """
    def __init__(self, weight, ell):
        super(Matern52_weight, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell

    def __call__(self, r):
        return self.weight**2 * (1.0 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                                          *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)


#### Mean class ################################################################
from functools import wraps
def array_input(f):
    """
        decorator to provide the __call__ methods with an array
    """
    @wraps(f)
    def wrapped(self, t):
        t = np.atleast_1d(t)
        r = f(self, t)
        return r
    return wrapped


class MeanModel(object):
    _parsize = 0
    def __init__(self, *pars):
        #self.pars = list(pars)
        self.pars = np.array(pars)

    def __repr__(self):
        """ Representation of each instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

    @classmethod
    def initialize(cls):
        """ Initialize instance, setting all parameters to 0. """
        return cls( *([0.]*cls._parsize) )

    def __add__(self, b):
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)

class Sum(MeanModel):
    """
        Sum of two mean functions
    """
    def __init__(self, m1, m2):
        self.m1, self.m2 = m1, m2

    @property
    def _parsize(self):
        return self.m1._parsize + self.m2._parsize

    @property
    def pars(self):
        return self.m1.pars + self.m2.pars

    def initialize(self):
        return

    def __repr__(self):
        return "{0} + {1}".format(self.m1, self.m2)

    @array_input
    def __call__(self, t):
        return self.m1(t) + self.m2(t)

class Constant_mean(MeanModel):
    """ 
        A constant offset mean function
    """
    _parsize = 1
    def __init__(self, c):
        super(Constant_mean, self).__init__(c)

    @array_input
    def __call__(self, t):
        return np.full(t.shape, self.pars[0])

class Linear_mean(MeanModel):
    """ 
        A linear mean function
        m(t) = slope * t + intercept 
    """
    _parsize = 2
    def __init__(self, slope, intercept):
        super(Linear_mean, self).__init__(slope, intercept)

    @array_input
    def __call__(self, t):
        return self.pars[0] * t + self.pars[1]

class Parabola_mean(MeanModel):
    """ 
        A 2nd degree polynomial mean function
        m(t) = quad * t**2 + slope * t + intercept 
    """
    _parsize = 3
    def __init__(self, quad, slope, intercept):
        super(Parabola_mean, self).__init__(quad, slope, intercept)

    @array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)

class Cubic_mean(MeanModel):
    """ 
        A 3rd degree polynomial mean function
        m(t) = cub * t**3 + quad * t**2 + slope * t + intercept 
    """
    _parsize = 4
    def __init__(self, cub, quad, slope, intercept):
        super(Cubic_mean, self).__init__(cub, quad, slope, intercept)

    @array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)

class Sine_mean(MeanModel):
    """ 
        A sinusoidal mean function
        m(t) = amplitude * sine(ang_freq * t + phase)
    """
    _parsize = 3
    def __init__(self, amp, w, phi):
        super(Sine_mean, self).__init__(amp, w, phi)

    @array_input
    def __call__(self, t):
        return self.pars[0] * np.sin(self.pars[1]*t + self.pars[2])

class Keplerian_mean(MeanModel):
    """
        Keplerian function with T0
        tan[phi(t) / 2 ] = sqrt(1+e / 1-e) * tan[E(t) / 2] = true anomaly
        E(t) - e*sin[E(t)] = M(t) = eccentric anomaly
        M(t) = (2*pi*t/tau) + M0 = Mean anomaly
        P  = period in days
        e = eccentricity
        K = RV amplitude in m/s 
        w = longitude of the periastron
        phi = orbital phase

        RV = K[cos(w+v) + e*cos(w)] + sis_vel
    """
    _parsize = 5
    def __init__(self, P, K, e, w, phi):
        super(Keplerian_mean, self).__init__(P, K, e, w, phi)

    @array_input
    def __call__(self, t):
        P, K, e, w, phi = self.pars
        #mean anomaly
        T = t[0] - (P*phi)/(2.*np.pi)
        Mean_anom = [2*np.pi*(x1-T)/P  for x1 in t]
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0 = Mean_anom + e*np.sin(Mean_anom) + 0.5*(e**2)*np.sin(2*Mean_anom)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0 = E0 - e*np.sin(E0)
        niter=0
        while niter < 500:
            aux = Mean_anom - M0
            E1 = E0 + aux/(1 - e*np.cos(E0))
            M1 = E0 - e*np.sin(E0)
            niter += 1
            E0 = E1
            M0 = M1
        nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E0/2))
        RV = K*(e*np.cos(w)+np.cos(w+nu))
        return RV


##### GPRN class ###############################################################
class GPRN(object):
    """ 
        Class to create our Gaussian process regression network. See Wilson et
    al. (2012) for more information on this framework.
        Parameters:
            nodes = latent noide functions f(x), f hat in the article
            weight = latent weight funtion w(x), as far as I understood
                            this is the same kernels for all nodes except for 
                            the "amplitude" that varies from node to node
            weight_values = array with the weight w11, w12, etc... size needs to 
                        be equal to the number of nodes times the number of 
                        components (self.q * self.p)
            means = array of means functions being used, set it to None if a 
                    model doesn't use it
            time = time
            *args = the data, it should be given as data1, data1_error, etc...
    """ 
    def  __init__(self, nodes, weight, weight_values, means, time, *args):
        #node functions; f(x) in Wilson et al. (2012)
        self.nodes = np.array(nodes)
        #number of nodes being used; q in Wilson et al. (2012)
        self.q = len(self.nodes)
        #weight function; w(x) in Wilson et al. (2012)
        self.weight = weight
        #amplitudes of the weight function
        self.weight_values = np.array(weight_values)
        #mean functions
        self.means = np.array(means)
        #time
        self.time = time 
        #the data, it should be given as data1, data1_error, data2, ...
        self.args = args 
        #number of components of y(x); p in Wilson et al. (2012)
        self.p = int(len(self.args)/2)
        #total number of weights we will have
        self.qp =  self.q * self.p

        #to organize the data we now join everything
        self.tt = np.tile(time, self.p) #"extended" time
        self.y = [] 
        self.yerr = []
        for i,j  in enumerate(args):
            if i%2 == 0:
                self.y.append(j)
            else:
                self.yerr.append(j**2)
        self.y = np.array(self.y) #"extended" measurements
        self.yerr = np.array(self.yerr) #"extended" errors
        #check if the input was correct
        assert self.means.size == self.p, \
        'The numbers of means should be equal to the number of components'
        assert (i+1)/2 == self.p, \
        'Given data and number of components dont match'

    def _kernel_matrix(self, kernel, time = None):
        """
            Returns the covariance matrix created by evaluating a given kernel 
        at inputs time.
        """
        #if time is None we use the initial time of our complexGP
        r = time[:, None] - time[None, :] if time.any() else self.time[:, None] - self.time[None, :]
        K = kernel(r)
        return K

    def _predict_kernel_matrix(self, kernel, tstar):
        """
            To be used in GP prediction
        """
        r = tstar[:, None] - self.time[None, :]
        K = kernel(r)
        return K

    def _kernel_pars(self, kernel):
        """
            Returns a given kernel hyperparameters
        """
        return kernel.pars

    @property
    def mean_pars_size(self):
        return self._mean_pars_size

    @mean_pars_size.getter
    def mean_pars_size(self):
        self._mean_pars_size = 0
        for m in self.means:
            if m is None: self._mean_pars_size += 0
            else: self._mean_pars_size += m._parsize
        return self._mean_pars_size

    @property
    def mean_pars(self):
        return self._mean_pars

    @mean_pars.setter
    def mean_pars(self, pars):
        pars = list(pars)
        assert len(pars) == self.mean_pars_size
        self._mean_pars = copy(pars)
        for i, m in enumerate(self.means):
            if m is None: 
                continue
            j = 0
            for j in range(m._parsize):
                m.pars[j] = pars.pop(0)

    def _mean(self, means):
        """
            Returns the values of the mean functions
        """
        N = self.time.size
        m = np.zeros_like(self.tt)
        for i, meanfun in enumerate(means):
            if meanfun is None:
                continue
            else:
                m[i*N : (i+1)*N] = meanfun(self.time)
        return m

    def _covariance_matrix(self, nodes, weight, weight_values, time, position_p):
        """ 
            Creates a matrix for each dataset 
            Parameters:
                node = the node functions f(x) (f hat)
                weight = the weight funtion w(x)
                weight_values = array with the weights of w11, w12, etc... 
                time = time 
                position_p = position necessary to use the correct node
                                and weight
            Return:
                k_ii = block matrix in position ii
        """
        #block matrix starts empty
        k_ii = np.zeros((time.size, time.size))
        for i in range(1,self.q + 1):
            #hyperparameteres of the kernel of a given position
            nodePars = self._kernel_pars(nodes[i - 1])
            #all weight function will have the same parameters
            weightPars = weight.pars
            #except for the amplitude
            weightPars[0] =  weight_values[i-1 + self.q*(position_p-1)]
            #node and weight functions kernel
            w_xa = type(self.weight)(*weightPars)(time[:,None])
            f_hat = self._kernel_matrix(type(self.nodes[i - 1])(*nodePars),time)
            w_xw = type(self.weight)(*weightPars)(time[None,:])
            #now we add all the necessary stuff; eq. 4 of Wilson et al. (2012)
            k_ii = k_ii + (w_xa * f_hat * w_xw)
        return k_ii

    def predict_gp(self, nodes = None, weight = None, weight_values = None,
                   means = None, time = None, dataset = 1):
        """ 
            Conditional predictive distribution of the Gaussian process
            Parameters:
                time = values where the predictive distribution will be calculated
                nodes = the node functions f(x) (f hat)
                weight = the weight function w(x)
                weight_values = array with the weights of w11, w12, etc...
                means = list of means being used 
                dataset = 1,2,3,... accordingly to the data we are using, 
                        1 represents the first y(x), 2 the second y(x), etc...
            Returns:
                y_mean = mean vector
                y_std = standard deviation vector
                y_cov = covariance matrix
        """
        #Nodes
        nodes = nodes if nodes else self.nodes
        #Weights
        weight  = weight if weight else self.weight
        #Weight values
        weight_values = weight_values if weight_values else self.weight_values
        #means
        yy = np.concatenate(self.y)
        yy = yy - self._mean(means) if means else yy
        #Time
        time = time if time.any() else self.time

        new_y = np.array_split(yy, self.p)
        cov = self._covariance_matrix(nodes, weight, weight_values, 
                                      self.time, dataset)
        L1 = cho_factor(cov)
        sol = cho_solve(L1, new_y[dataset - 1])
        tshape = time[:, None] - self.time[None, :]

        k_ii = np.zeros((tshape.shape[0],tshape.shape[1]))
        for i in range(1,self.q + 1):
            #hyperparameteres of the kernel of a given position
            nodePars = self._kernel_pars(nodes[i - 1])
            #all weight function will have the same parameters
            weightPars = self._kernel_pars(weight)
            #except for the amplitude
            weightPars[0] =  weight_values[i-1 + self.q*(dataset - 1)]
            #node and weight functions kernel
            w_xa = type(self.weight)(*weightPars)(time[:,None])
            f_hat = self._predict_kernel_matrix(type(self.nodes[i - 1])(*nodePars), time)
            w_xw = type(self.weight)(*weightPars)(self.time[None,:])
            #now we add all the necessary stuff; eq. 4 of Wilson et al. (2012)
            k_ii = k_ii + (w_xa * f_hat * w_xw)

        Kstar = k_ii
        Kstarstar = self._covariance_matrix(nodes, weight, weight_values, time, 
                                            dataset)

        y_mean = np.dot(Kstar, sol) #mean
        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_std, y_cov

#    def sample(self, kernel, time):
#        """ 
#            Returns samples from the kernel
#            Parameters:
#                kernel = covariance funtion
#                time = time array
#            Returns:
#                Sample of K 
#        """
#        mean = np.zeros_like(time)
#        cov = self.compute_matrix(kernel, time)
#        norm = multivariate_normal(mean, cov, allow_singular=True)
#        return norm.rvs()




