from collections import namedtuple

import numpy as np
from scipy.stats import norm, t as T, binned_statistic

from .utils import get_planet_mass, get_planet_semimajor_axis, mjup2mearth


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


def sort_planet_samples(res, planet_samples):
    # here we sort the planet_samples array by the orbital period
    # this is a bit difficult because the organization of the array is
    # P1 P2 K1 K2 ....
    samples = np.empty_like(planet_samples)
    n = res.max_components * res.n_dimensions
    mc = res.max_components
    p = planet_samples[:, :mc]
    ind_sort_P = np.arange(np.shape(p)[0])[:, np.newaxis], np.argsort(p)
    for i, j in zip(range(0, n, mc), range(mc, n + mc, mc)):
        samples[:, i:j] = planet_samples[:, i:j][ind_sort_P]
    return samples


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


def detection_limits(results, star_mass=1.0, Np=None, bins=200, plot=True,
                     sorted_samples=True, return_mask=False):
    """ 
    Calculate detection limits using samples with more than `Np` planets. By 
    default, this function uses the value of `Np` which passes the posterior
    probability threshold.

    Arguments
    ---------
    star_mass : float or tuple
        Stellar mass and optionally its uncertainty [in solar masses].
    Np : int
        Consider only posterior samples with more than `Np` planets.
    bins : int
        Number of bins at which to calculate the detection limits. The period 
        ranges from the minimum to the maximum orbital period in the posterior.
    plot : bool
        Whether to plot the detection limits
    sorted_samples : bool
        undoc
    return_mask: bool
        undoc
    
    Returns
    -------
    P, K, E, M : ndarray
        Orbital periods, semi-amplitudes, eccentricities, and planet masses used
        in the calculation of the detection limits. These correspond to all the
        posterior samples with more than `Np` planets
    s : DLresult, namedtuple
        Detection limits result, with attributes `max` and `bins`. The `max` 
        array is in units of Earth masses, `bins` is in days.
    """
    res = results
    if Np is None:
        Np = passes_threshold_np(res)
    print(f'Using samples with Np > {Np}')

    mask = res.posterior_sample[:, res.index_component] > Np
    pars = res.posterior_sample[mask, res.indices['planets']]

    if sorted_samples:
        pars = sort_planet_samples(res, pars)

    mc = res.max_components
    periods = slice(0 * mc, 1 * mc)
    amplitudes = slice(1 * mc, 2 * mc)
    eccentricities = slice(3 * mc, 4 * mc)

    P = pars[:, periods]
    K = pars[:, amplitudes]
    E = pars[:, eccentricities]

    inds = np.nonzero(P)

    P = P[inds]
    K = K[inds]
    E = E[inds]
    M = get_planet_mass(P, K, E, star_mass=star_mass, full_output=True)[2]

    if P.max() / P.min() > 100:
        bins = 10**np.linspace(np.log10(P.min()), np.log10(P.max()), bins)
    else:
        bins = np.linspace(P.min(), P.max(), bins)

    # bins_start = bins[:-1]# - np.ediff1d(bins)/2
    # bins_end = bins[1:]# + np.ediff1d(bins)/2
    # bins_start = np.append(bins_start, bins_end[-1])
    # bins_end = np.append(bins_end, P.max())

    DLresult = namedtuple('DLresult', ['max', 'bins'])
    s = binned_statistic(P, M, statistic='max', bins=bins)
    s = DLresult(max=s.statistic * mjup2mearth,
                 bins=s.bin_edges[:-1] + np.ediff1d(s.bin_edges) / 2)

    # s99 = binned_statistic(P, M, statistic=lambda x: np.percentile(x, 99),
    #                        bins=bins)
    # s99 = DLresult(max=s99.statistic,
    #                bins=s99.bin_edges[:-1] + np.ediff1d(s99.bin_edges) / 2)

    if plot:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1, constrained_layout=True)
        if isinstance(star_mass, tuple):
            star_mass = star_mass[0]

        sP = np.sort(P)
        one_ms = 4.919e-3 * star_mass**(2. / 3) * sP**(1. / 3) * 1
        kw = dict(color='C0', alpha=1, zorder=3)
        ax.loglog(sP, 5 * one_ms * mjup2mearth, ls=':', **kw)
        ax.loglog(sP, 3 * one_ms * mjup2mearth, ls='--', **kw)
        ax.loglog(sP, one_ms * mjup2mearth, ls='-', **kw)

        ax.loglog(P, M * mjup2mearth, 'k.', ms=2, alpha=0.2, zorder=-1)
        ax.loglog(s.bins, s.max, color='C3')
        # ax.hlines(s.max, bins_start, bins_end, lw=2)
        # ax.loglog(s99.bins, s99.max * mjup2mearth)

        lege = [f'{i} m/s' for i in (5, 3, 1)] 
        lege += ['posterior samples', 'binned maximum']

        ax.legend(lege, ncol=2, frameon=False)
        ax.set(ylim=(0.5, None))
        ax.set(xlabel='Orbital period [days]', ylabel='Planet mass [M$_\odot$]')

    if return_mask:
        return P, K, E, M, s, mask
    else:
        return P, K, E, M, s



def column_dynamic_ranges(results):
    """ Return the range of each column in the posterior file """
    return results.posterior_sample.ptp(axis=0)


def columns_with_dynamic_range(results):
    """ Return the columns in the posterior file which vary """
    dr = column_dynamic_ranges(results)
    return np.nonzero(dr)[0]
