from collections import namedtuple

import numpy as np
from scipy.stats import norm, t as T, binned_statistic
from scipy.cluster.vq import kmeans2

from .utils import (get_planet_mass, get_planet_semimajor_axis,
                    get_planet_mass_and_semimajor_axis, mjup2mearth)


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
    if isinstance(J, float):
        pass
    elif J.shape[0] > 1:
        # one jitter per instrument
        J = J[(res.obs - 1).astype(int)]

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


def detection_limits(results,
                     star_mass=1.0,
                     Np=None,
                     bins=200,
                     plot=True,
                     semi_amplitude=False,
                     semi_major_axis=False,
                     logX=True,
                     sorted_samples=True,
                     return_mask=False,
                     remove_nan=True,
                     add_jitter=False,
                     show_prior=False,
                     show_eccentricity=False,
                     smooth=False,
                     smooth_degree=3):
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
    plot : bool (default True)
        Whether to plot the detection limits
    semi_amplitude : bool (default False)
        Show the detection limits for semi-amplitude, instead of planet mass
    semi_major_axis : bool (default False)
        Show semi-major axis in the x axis, instead of orbital period
    logX : bool (default True)
        Should X-axis bins be logarithmic ?
    sorted_samples : bool
        undoc
    return_mask: bool
        undoc
    remove_nan : bool
        remove bins with no counts (no posterior samples)
    add_jitter ...
    show_prior ...
    show_prior : bool (default False)
        If true, color the posterior samples according to orbital eccentricity
    smooth : bool (default False)
        Smooth the binned maximum with a polynomial
    smooth_degree : int (default 3)
        Degree of the polynomial used for smoothing

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

    if res.verbose:
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
    # J = res.extra_sigma[mask]

    # if add_jitter:
    #     # K = np.hypot(K, res.extra_sigma[mask])
    #     K += res.extra_sigma[mask]

    M, A = get_planet_mass_and_semimajor_axis(P, K, E, star_mass=star_mass,
                                              full_output=True)
    M = M[2]
    A = A[2]

    # Mjit = get_planet_mass(P, np.hypot(K, J), E, star_mass=star_mass,
    #                        full_output=True)
    # Mjit = Mjit[2]

    if semi_major_axis:
        start, end = A.min(), A.max()
    else:
        start, end = P.min(), P.max()

    if logX:
        bins = 10**np.linspace(np.log10(start), np.log10(end), bins)

    # bins_start = bins[:-1]# - np.ediff1d(bins)/2
    # bins_end = bins[1:]# + np.ediff1d(bins)/2
    # bins_start = np.append(bins_start, bins_end[-1])
    # bins_end = np.append(bins_end, P.max())

    DLresult = namedtuple('DLresult', ['max', 'bins'])

    if semi_amplitude:
        if semi_major_axis:
            s = binned_statistic(A, K, statistic='max', bins=bins)
            # sj = binned_statistic(A,
            #                       np.hypot(K, J),
            #                       statistic='max',
            #                       bins=bins)
        else:
            s = binned_statistic(P, K, statistic='max', bins=bins)
            # sj = binned_statistic(P,
            #                       np.hypot(K, J),
            #                       statistic='max',
            #                       bins=bins)
    else:
        if semi_major_axis:
            s = binned_statistic(A,
                                 M * mjup2mearth,
                                 statistic='max',
                                 bins=bins)
            # sj = binned_statistic(A,
            #                       Mjit * mjup2mearth,
            #                       statistic='max',
            #                       bins=bins)
        else:
            s = binned_statistic(P,
                                 M * mjup2mearth,
                                 statistic='max',
                                 bins=bins)
            # sj = binned_statistic(P,
            #                       Mjit * mjup2mearth,
            #                       statistic='max',
            #                       bins=bins)

    if remove_nan:
        w = ~np.isnan(s.statistic)
    else:
        w = np.full(s.statistic.size, True)

    bins = s.bin_edges[:-1] + np.ediff1d(s.bin_edges) / 2

    s = DLresult(max=s.statistic[w], bins=bins[w])
    # sj = DLresult(max=sj.statistic[w], bins=bins[w])

    # s99 = binned_statistic(P, M, statistic=lambda x: np.percentile(x, 99),
    #                        bins=bins)
    # s99 = DLresult(max=s99.statistic,
    #                bins=s99.bin_edges[:-1] + np.ediff1d(s99.bin_edges) / 2)

    if plot:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1, constrained_layout=True)
        if isinstance(star_mass, tuple):
            star_mass = star_mass[0]

        if show_eccentricity:
            plotf = ax.scatter
        else:
            plotf = ax.plot

        lege = []
        kw_points = dict(color='k', ms=2, alpha=0.3, zorder=-1)
        if semi_amplitude:  # K [m/s] in the y axis
            if semi_major_axis:
                plotf(A, K, '.', **kw_points)
            else:
                plotf(P, K, '.', **kw_points)

        else:  # M [M⊕] in the y axis
            if semi_major_axis:
                sX = np.sort(A)
            else:
                sX = np.sort(P)

            one_ms = 4.919e-3 * star_mass**(2. / 3) * sX**(1. / 3) * 1
            kw_lines = dict(color='C0', alpha=1, zorder=3)
            ls = (':', '--', '-')
            for i, f in enumerate((5, 3, 1)):
                ax.loglog(sX, f * one_ms * mjup2mearth, ls=ls[i], **kw_lines)
                lege.append(f'{f} m/s')

            if semi_major_axis:
                plotf(A, M * mjup2mearth, '.', **kw_points)
            else:
                plotf(P, M * mjup2mearth, '.', **kw_points)

        ax.set_xscale('log')
        ax.set_yscale('log')

        if show_eccentricity:
            ax.colorbar()

        if smooth:
            p = np.polyfit(np.log10(s.bins), np.log10(s.max), smooth_degree)
            ax.loglog(s.bins, 10**np.polyval(p, np.log10(s.bins)), color='m')
        else:
            ax.loglog(s.bins, s.max, '-o', ms=3, color='m')

        if not semi_major_axis:
            kwargs = dict(ls='--', lw=2, alpha=0.5, zorder=-1, color='C5')
            ax.axvline(res.t.ptp(), 0.5, 1, **kwargs)

        # ax.hlines(s.max, bins_start, bins_end, lw=2)
        # ax.loglog(s99.bins, s99.max * mjup2mearth)

        lege += ['posterior samples', 'binned maximum', 'time span']
        if smooth:
            lege[-2] = '(smoothed) ' + lege[-2]

        if add_jitter:
            ax.fill_between(s.bins, s.max, s.max + sj.max,
                            color='r', alpha=0.1, lw=0)

        ax.legend(lege, ncol=2, frameon=False)
        # ax.set(ylim=(0.5, None))

        if semi_amplitude:
            Ly = r'Semi-amplitude [m/s]'
        else:
            Ly = r'Planet mass [M$_\oplus$]'

        if semi_major_axis:
            Lx = 'Semi-major axis [AU]'
        else:
            Lx = 'Orbital period [days]'

        ax.set(xlabel=Lx, ylabel=Ly)

        try:
            ax.set_title(res.star)
        except AttributeError:
            pass

    if return_mask:
        return P, K, E, M, s, mask
    else:
        return P, K, E, M, s


def parameter_clusters_kmeans(results, n_clusters=None, include_ecc=True):
    res = results

    if include_ecc:
        data = np.c_[res.T, res.A, res.E]
    else:
        data = np.c_[res.T, res.A]

    if n_clusters is None:
        k = res.max_components
    else:
        k = n_clusters

    centroids, labels = kmeans2(data, k, minit='++')
    return centroids, labels


def column_dynamic_ranges(results):
    """ Return the range of each column in the posterior file """
    return results.posterior_sample.ptp(axis=0)


def columns_with_dynamic_range(results):
    """ Return the columns in the posterior file which vary """
    dr = column_dynamic_ranges(results)
    return np.nonzero(dr)[0]
