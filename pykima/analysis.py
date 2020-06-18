import numpy as np
from scipy.stats import norm, t as T

from .utils import get_planet_mass, get_planet_semimajor_axis

def most_probable_np(results):
    """ 
    Return the value of Np with the highest posterior probability.
    Arguments:
        results: a KimaResults instance. 
    """
    Nplanets = results.posterior_sample[:, results.index_component].astype(int)
    values, counts = np.unique(Nplanets, return_counts=True)
    return values[counts.argmax()]


def passes_threshold_np(results, threshold=150):
    """ 
    Return the value of Np supported by the data, considering a posterior ratio
    threshold (default 150).
    Arguments:
        results: a KimaResults instance. 
        threshold: posterior ratio threshold, default 150.
    """
    Nplanets = results.posterior_sample[:, results.index_component].astype(int)
    values, counts = np.unique(Nplanets, return_counts=True)
    ratios = counts[1:] / counts[:-1]
    if np.any(ratios > threshold):
        i = np.argmax(np.where(ratios > 150)[0]) + 1
        return values[i]
    else:
        return values[0]



def planet_parameters(results, star_mass=1.0, sample=None, printit=True):
    if sample is None:
        sample = results.maximum_likelihood_sample(printit=printit)

    mass_errs = False 
    if isinstance(star_mass, (tuple, list)):
        mass_errs = True

    format_tuples = lambda data: '{:10.5f} \pm {:.5f}'.format(data[0], data[1]) if mass_errs else '{:12.5f}'.format(data)

    if printit:
        print()
        final_string = star_mass if not mass_errs else '{} \pm {}'.format(star_mass[0], star_mass[1])
        print('Calculating planet masses with Mstar = {} Msun'.format(final_string))


    indices = results.indices
    mc = results.max_components

    # planet parameters
    pars = sample[indices['planets']].copy()
    # number of planets in this sample
    nplanets = (pars[:mc] != 0).sum()

    if printit:
        extra_padding = 12*' ' if mass_errs else ''
        print(20* ' ' + ('%12s' % 'Mp [Mearth]') + extra_padding + ('%12s' % 'Mp [Mjup]'))

    masses = []
    for j in range(int(nplanets)):
        P = pars[j + 0 * mc]
        if P == 0.0:
            continue
        K = pars[j + 1 * mc]
        # phi = pars[j + 2 * mc]
        # t0 = t[0] - (P * phi) / (2. * np.pi)
        ecc = pars[j + 3 * mc]
        # w = pars[j + 4 * mc]
        m_mjup, m_mearth = get_planet_mass(P, K, ecc, star_mass=star_mass)

        if printit:
            s = 18 * ' '
            s += format_tuples(m_mearth)
            s += format_tuples(m_mjup)
            masses.append(m_mearth)
            print(s)
    return np.array(masses)


def find_outliers(results, sample, threshold=10, verbose=False):
    """ 
    Estimate which observations are outliers, for a model with a Student t
    likelihood. This function first calculates the residuals given the
    parameters in `sample`. Then it computes the relative probability of each
    residual point given a Student-t (Td) and a Gaussian (Gd) likelihoods. If
    the probability Td is larger than Gd (by a factor of `threshold`), the point
    is flagged as an outlier. The function returns an "outlier mask".
    """
    res = results
    # the model must have studentt = true
    if not res.studentT:
        raise ValueError('studentt option is off, cannot estimate outliers')
    
    # calculate residuals for this sample
    resid = res.y - res.model(sample)
    
    # this sample's jitter and degrees of freedom
    J = sample[res.indices['jitter']]
    nu = sample[res.indices['nu']]
    
    # probabilities within the Gaussian and Student-t likelihoods
    Gd = norm(loc=0, scale=np.hypot(res.e, J)).pdf(resid)
    Td = T(df=nu, loc=0, scale=np.hypot(res.e, J)).pdf(resid)
    
    # if Td/Gd > threshold, the point is an outlier, in the sense that it is
    # more likely within the Student-t likelihood than it would have been within
    # the Gaussian likelihood
    ratio = Td / Gd
    outlier = ratio > threshold
    if verbose:
        print(f'Found {outlier.sum()} outliers')

    return outlier


def column_dynamic_ranges(results):
    """ Return the range of each column in the posterior file """
    return results.posterior_sample.ptp(axis=0)

def columns_with_dynamic_range(results):
    """ Return the columns in the posterior file which vary """
    dr = column_dynamic_ranges(results)
    return np.nonzero(dr)[0]