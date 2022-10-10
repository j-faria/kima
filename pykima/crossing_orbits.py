# -*- coding: utf-8 -*-
# original author Matthew Standing mxs1263@student.bham.ac.uk
# https://github.com/j-faria/kima/pull/57

import os
import numpy as np

from .results import KimaResults

# from astropy.constants
G = 6.67408e-11  # m3/(kg s2)
Msun = 1.9884754153381438e+30  # kg
Rsun = 695700000.0  # m
RHOsun = 1.4098263  # g/cm3 = 3*Msun / 4*pi*Rsun**3


def rem_crossing_orbits(results=None):
    """
    Removes crossing orbits from the posterior samples. This function assumes
    that the posterior samples are "sorted" so that the first planet in the 
    system is the most "likely". Then, crossing orbit checks are done starting
    from the final entry in a row (system) end ending in the first entry.

    Parameters:
        results_path       : Str
                             Path to 'posterior_sample.txt' Kima output file
        no_p               : Float
                             Max number of planets set for Kima to explore
        no_files           : Float
                             Number of data files input into Kima

    Returns
    -------
    mod_results : KimaResults
        Modified `KimaResults` instance with parameters of crossing orbits 
        replaced with NaN, an unmodified `posterior_sample_original` array, and 
        the `removed_crossing` attribute set to True.
    """

    if results is None:
        if not os.path.exists('posterior_sample.txt'):
            raise FileNotFoundError('Cannot find file "posterior_sample.txt"')
        results = KimaResults('')

    res = results # for shorts

    samples = res.posterior_sample[:, res.index_component+1:-2].copy()
    samples[samples==0] = np.nan

    if res.log_period:
        samples[:,:res.max_components] = np.exp(samples[:,:res.max_components])
    
    P = samples[:, :res.max_components] # all periods
    E = samples[:, 3*res.max_components:4*res.max_components] # all eccentricities

    # calculate periastrons
    r_peri = ((P**2)**(1/3)) * (1 - E)  #days^2/3
    # calculate apoastrons
    r_apast = ((P**2)**(1/3)) * (1 + E)  #days^2/3

    total = np.count_nonzero(~np.isnan(r_peri))
    count = 0
    with np.errstate(invalid='ignore'):
        for j in range(r_peri.shape[0]):
            for i in range(r_peri.shape[1] - 1, -1, -1):
                #Check if crosses interior periastrons
                if (r_apast[j, i] > r_peri[j, np.where(r_peri[j, i] < r_peri[j, :i])]).any():
                    r_peri[j, i] = np.nan
                    r_apast[j, i] = np.nan
                    count += 1
                #Check if crosses exterior apastrons
                if (r_apast[j, i] > r_apast[j, np.where(r_peri[j, i] < r_apast[j, :i])]).any():
                    r_peri[j, i] = np.nan
                    r_apast[j, i] = np.nan
                    count += 1
                #Check if a lower indexed orbit crosses
                if np.logical_and((r_apast[j, i] < r_apast[j, :i]), (r_peri[j, i] > r_peri[j, :i])).any():
                    r_peri[j, i] = np.nan
                    r_apast[j, i] = np.nan
                    count += 1

    print("Number of crossing orbits removed = %d (out of %d)" % (count, total))

    crossing = np.isnan(r_peri)

    mc = res.max_components
    for i in range(res.n_dimensions):
        samples[:,i*mc:(i+1)*mc][crossing] = np.nan


    if res.log_period:
        samples[:,:res.max_components] = np.log(samples[:,:res.max_components])

    res.posterior_sample_original = res.posterior_sample.copy()
    res.posterior_sample[:, res.index_component+1:-2] = samples
    
    res.removed_crossing = True
    res.get_marginals()

    return res


def rem_roche(results=None, RHOplanet=7.8, Mstar=Msun, Rstar=Rsun,
              RHOstar=RHOsun):
    """
    Remove orbits that cross the Roche lobe of the star. By default, assume a
    solid iron planet orbiting a Sun-like star. Will save a new version of
    posterior_sample.txt (remroche_posterior_sample.txt) in results_path given

    Parameters
    ----------
    results : KimaResults, optional
        An instance of `KimaResults`. If None, try to get it from a file with
        posterior samples in the current directory.
    RHOplanet : float, optional
        Density of planet used in roche limit equation. Default is 7.8 g/cm^3 
        corresponding to a solid iron planet.
    Mstar : float, optional
        Stellar mass used in roche limit equation. Default is mass of the Sun.
    Rstar : float, optional
        Stellar radius used in roche limit equation. Default is radius of the Sun.
    RHOstar : float, optional
        Stellar density used in roche limit equation. Default is mean solar 
        density, approximately 1.41 g/cm^3.

    Returns
    -------
    mod_results : KimaResults
        Modified `KimaResults` instance with parameters of Roche crossing orbits 
        replaced with NaN and `removed_roche_crossing` attribute set to True.
    """


    if results is None:
        if not os.path.exists('posterior_sample.txt'):
            raise FileNotFoundError('Cannot find file "posterior_sample.txt"')
        results = KimaResults('')

    P = np.exp(results.T) if results.log_period else results.T
    E = results.E

    r_peri = ((P**2.)**(1. / 3.)) * (1 - E)  #days^2/3
    #Calculate apastrons
    r_apast = ((P**2.)**(1. / 3.)) * (1 + E)  #days^2/3

    #Calculating roche radius
    roche_lim = Rstar * 2.44 * (RHOstar / RHOplanet)**(1. / 3.)  #metres

    roche_P = np.sqrt(((4. * np.pi**2.) * roche_lim**3.) / (G * Mstar))
    roche_P = roche_P / (60. * 60. * 24.)  #convert into days
    roche_a = (roche_P**2.)**(1. / 3.)  #days^2/3

    #Remove any orbits closer than the roche limit
    roche_crossing = np.where(r_peri < roche_a)[0]
    count = roche_crossing.size

    results.T[roche_crossing] = np.nan

    results.removed_roche_crossing = True
    print("Number of roche crossing orbits removed =", count)

    return results
    # #Update results dataframe with removed values
    # results[periods] = allP
    # results[eccentricities] = alle
    # results[phis] = allphi
    # results[semiamps] = allK
    # results[omegas] = allomega

    # #Save as new results .txt file
    # new_header = ' '.join(header)
    # np.savetxt(r'{0}/remroche_posterior_sample.txt'.format(results_path),\
    #            results.values, header=new_header)

    # return results, allP, alle, allphi, allK, allomega

