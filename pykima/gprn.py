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

#because it will make my life easier down the line
pi, exp, sine, cosine, sqrt = np.pi, np.exp, np.sin, np.cos, np.sqrt


##### node class ###############################################################
class Node(object):
    """
        Definition the node functions (kernels) of our GPRN, by default and 
    because it simplifies my life, all kernels include a white noise term
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

# Constant kernel
class Constant_node(Node):
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
class SquaredExponential_node(Node):
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
class Periodic_node(Node):
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
class QuasiPeriodic_node(Node):
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
class RationalQuadratic_node(Node):
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
class Cosine_node(Node):
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
class Exponential_node(Node):
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
class Matern32_node(Node):
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
class Matern52_node(Node):
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


##### weight class #############################################################
class Weight(object):
    """
        Definition the weight functions (kernels) of our GPRN.
    """
    def __init__(self, *args):
        """
            Puts all kernel arguments in an array pars
        """
        self.pars = np.array(args)

    def __call__(self, r):
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

# Constant kernel
class Constant_weight(Weight):
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
class SquaredExponential_weight(Weight):
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
class Periodic_weight(Weight):
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
class QuasiPeriodic_weight(Weight):
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
class RationalQuadratic_weight(Weight):
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
class Cosine_weight(Weight):
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
class Exponential_weight(Weight):
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
class Matern32_weight(Weight):
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
class Matern52_weight(Weight):
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


#### mean class ################################################################

#   TO
#   BE
#   IMPLEMENTED
#   ...


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





