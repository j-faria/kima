#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:10:02 2019

@author: Matthew Standing

email: mxs1263@student.bham.ac.uk
"""

import numpy as np
import astropy.constants as c
import pandas as pd

def rem_crossing_orbits(results_path, no_p, no_files):
    """
    Removes any crossing orbits proposed by kima in the posterior contained in posterior_sample.txt
    ASSUMES that posterior_sample.txt is sorted so that the first planet in the system is most
    likely and so on, until the final planet which is the least likely to be true.
    Therefore orbit cross checks from the final entry in a row (system) to the first.
    Will save a new version of posterior_sample.txt (remcross_posterior_sample.txt) in results_path given

    Parameters:
        results_path       : Str
                             Path to 'posterior_sample.txt' Kima output file
        no_p               : Float
                             Max number of planets set for Kima to explore
        no_files           : Float
                             Number of data files input into Kima

    returns:
        results: Dataframe object with removed orbits as NaN's
        allP, alle, allphi, allK, allomega: Arrays of values for all remaining
                                            proposed planetary systems
    """
    def _create_headers(hdr, n):
        return [hdr + str(i+1) for i in range(n)]

    periods = _create_headers('P', no_p)
    eccentricities = _create_headers('e', no_p)
    semiamps = _create_headers('K', no_p)
    phis = _create_headers('Phi', no_p)
    omegas = _create_headers('w', no_p)

    #Create header for panda dataframe
    header = _create_headers('Jitter', no_files) + _create_headers('Offset', no_files-1) + \
             ["ndim", "maxNp", "No_p"] + periods + semiamps + phis + eccentricities + omegas \
             + ["Staleness", "v_sys"]

    #Reading in results file
    results = pd.read_csv('{0}/posterior_sample.txt'.format(results_path), sep=" ",
                          header=None, skiprows=2, names=header)

    ###############################################################################
    #Create arrays of parameters
    allP = np.array(results[periods].values.tolist()) #periods
    alle = np.array(results[eccentricities].values.tolist()) #eccentricities
    allK = np.array(results[semiamps].values.tolist())
    allphi = np.array(results[phis].values.tolist())
    allomega = np.array(results[omegas].values.tolist())

    #Set M=1 and 4pi**2 = 1 as they act as constants in the comparison
    #Calculate periastrons
    r_peri = ((allP ** 2.) ** (1./3.)) * (1-alle) #days^2/3
    #Calculate apastrons
    r_apast = ((allP ** 2.) ** (1./3.)) * (1+alle) #days^2/3

    count = 0
    with np.errstate(invalid='ignore'): #To surpress runtime warnings from any NaN's
        for j in range(r_peri.shape[0]):
            for i in range(r_peri.shape[1]-1, -1, -1):
                #Check if crosses interior periastrons
                if (r_apast[j, i] > r_peri[j, np.where(r_peri[j, i] < r_peri[j, :i])]).any():
                    r_peri[j, i] = 0
                    r_apast[j, i] = 0
                    count += 1
                #Check if crosses exterior apastrons
                if (r_apast[j, i] > r_apast[j, np.where(r_peri[j, i] < r_apast[j, :i])]).any():
                    r_peri[j, i] = 0
                    r_apast[j, i] = 0
                    count += 1
                #Check if a lower indexed orbit crosses
                if np.logical_and((r_apast[j, i] < r_apast[j, :i]), (r_peri[j, i] > r_peri[j, :i])).any():
                    r_peri[j, i] = 0
                    r_apast[j, i] = 0
                    count += 1

    #Converting planetary orbital values which cross to be NaN's
    allP = np.where(r_peri == 0, np.nan, allP)
    alle = np.where(r_peri == 0, np.nan, alle)
    allphi = np.where(r_peri == 0, np.nan, allphi)
    allK = np.where(r_peri == 0, np.nan, allK)
    allomega = np.where(r_peri == 0, np.nan, allomega)

    print("Number of crossing orbits removed =", count)

    #Update results dataframe with removed values
    results[periods] = allP
    results[eccentricities] = alle
    results[phis] = allphi
    results[semiamps] = allK
    results[omegas] = allomega

    #Save as new results .txt file
    new_header = ' '.join(header)
    np.savetxt(r'{0}/remcrossing_posterior_sample.txt'.format(results_path),\
               results.values, header=new_header)

    return results, allP, alle, allphi, allK, allomega



def rem_roche(results_path, no_p, no_files, planet_dens=7.8, star_M=c.M_sun.value, star_dens=1.41, R_s=c.R_sun.value):
    """
    Removes any orbits that cross the roche lobe calculate using given parameters proposed
    by kima in the posterior contained in posterior_sample.txt.
    If no parameters are given it assumes a solid iron planet orbiting a sun-like star.
    Will save a new version of posterior_sample.txt (remroche_posterior_sample.txt) in results_path given

    Parameters:
        results_path       : Str
                             Path to 'posterior_sample.txt' Kima output file
        no_p               : Float
                             Max number of planets set for Kima to explore
        no_files           : Float
                             Number of data files input into Kima

        planet_dens        : Float -optional
                             Density of planet used in roche lim equation, set to 7.8g/cm^3
                             (solid iron planet)
        star_M             : Float -optional
                             Mass of star used in roche lim equation, set to mass of sun
        star_dens          : Float -optional
                             Density of star used in roche lim equation, set to 1.41g/cm^3
                             (density of sun)
        R_s                : Float -optional
                             Radius of star used in roche lim equation, set to radius of sun

    returns:
        results: Dataframe object with removed orbits as NaN's
        allP, alle, allphi, allK, allomega: Arrays of values for all remaining
                                            proposed planetary systems

    """

    def _create_headers(hdr, n):
        return [hdr + str(i+1) for i in range(n)]

    periods = _create_headers('P', no_p)
    eccentricities = _create_headers('e', no_p)
    semiamps = _create_headers('K', no_p)
    phis = _create_headers('Phi', no_p)
    omegas = _create_headers('w', no_p)

    header = _create_headers('Jitter', no_files) + _create_headers('Offset', no_files-1) + \
             ["ndim", "maxNp", "No_p"] + periods + semiamps + phis + eccentricities + omegas \
             + ["Staleness", "v_sys"]

    results = pd.read_csv('{0}/posterior_sample.txt'.format(results_path), sep=" ",
                          header=None, skiprows=2, names=header)

    ###############################################################################
    allP = np.array(results[periods].values.tolist()) #periods
    alle = np.array(results[eccentricities].values.tolist()) #eccentricities
    allK = np.array(results[semiamps].values.tolist())
    allphi = np.array(results[phis].values.tolist())
    allomega = np.array(results[omegas].values.tolist())

    r_peri = ((allP ** 2.) ** (1./3.)) * (1-alle) #days^2/3
    #Calculate apastrons
    r_apast = ((allP ** 2.) ** (1./3.)) * (1+alle) #days^2/3

    #Calculating roche radius
    roche_lim = R_s * 2.44*(star_dens/planet_dens)**(1./3.) #metres

    roche_P = np.sqrt(((4. * np.pi ** 2.) * roche_lim ** 3.) / (c.G.value * star_M))
    roche_P = roche_P / (60. * 60. * 24.) #convert into days
    roche_a = (roche_P ** 2.) ** (1./3.) #days^2/3

    count = 0

    #Remove any orbits closer than the roche limit
    with np.errstate(invalid='ignore'): #To surpress runtime warnings from any NaN's
        for j in range(r_peri.shape[0]):
            for i in range(r_peri.shape[1]-1, -1, -1):
                if r_peri[j, i] < roche_a:
                    r_peri[j, i] = 0
                    r_apast[j, i] = 0
                    count += 1

    #Converting planetary orbital values which cross to be NaN's
    allP = np.where(r_peri == 0, np.nan, allP)
    alle = np.where(r_peri == 0, np.nan, alle)
    allphi = np.where(r_peri == 0, np.nan, allphi)
    allK = np.where(r_peri == 0, np.nan, allK)
    allomega = np.where(r_peri == 0, np.nan, allomega)

    print("Number of roche crossing orbits removed =", count)

    #Update results dataframe with removed values
    results[periods] = allP
    results[eccentricities] = alle
    results[phis] = allphi
    results[semiamps] = allK
    results[omegas] = allomega

    #Save as new results .txt file
    new_header = ' '.join(header)
    np.savetxt(r'{0}/remroche_posterior_sample.txt'.format(results_path),\
               results.values, header=new_header)

    return results, allP, alle, allphi, allK, allomega


def rem_timespan(results_path, no_p, no_files, timespan, multiple=4):
    """
    Removes any proposed orbits with periods greater than a specified multiple of the data timespan
    If not specified multiple set to 4
    Will save a new version of posterior_sample.txt (remtimespan_posterior_sample.txt) in results_path given

    Parameters:
        results_path       : Str
                             Path to 'posterior_sample.txt' Kima output file
        no_p               : Float
                             Max number of planets set for Kima to explore
        no_files           : Float
                             Number of data files input into Kima
        timespan           : Float
                             Timespan of your data
        multiple           : Float Optional
                             Multiple of timespan to exclude, set to 4
                             e.g. Any proposed planetary periods and corresponding parameters with
                             timespans > 4 x timespan will be converted to NaN values

    returns:
        results: Dataframe object with removed orbits as NaN's

    """

    def _create_headers(hdr, n):
        return [hdr + str(i+1) for i in range(n)]

    periods = _create_headers('P', no_p)
    eccentricities = _create_headers('e', no_p)
    semiamps = _create_headers('K', no_p)
    phis = _create_headers('Phi', no_p)
    omegas = _create_headers('w', no_p)

    header = _create_headers('Jitter', no_files) + _create_headers('Offset', no_files-1) + \
             ["ndim", "maxNp", "No_p"] + periods + semiamps + phis + eccentricities + omegas \
             + ["Staleness", "v_sys"]

    results = pd.read_csv('{0}/posterior_sample.txt'.format(results_path), sep=" ",
                          header=None, skiprows=2, names=header)

    ###############################################################################
    unphysical_period = multiple*timespan

    a = np.array(results[periods].values.tolist())
    results[periods] = np.where(a > unphysical_period, np.nan, results[periods]).tolist()
    results[eccentricities] = np.where(a > unphysical_period, np.nan, results[eccentricities]).tolist()
    results[phis] = np.where(a > unphysical_period, np.nan, results[phis]).tolist()
    results[semiamps] = np.where(a > unphysical_period, np.nan, results[semiamps]).tolist()
    results[omegas] = np.where(a > unphysical_period, np.nan, results[omegas]).tolist()

    #Convert all 0 values to NaN's
    results = results.replace(0, np.nan)
    results["Staleness"] = results["Staleness"].replace(np.nan, 0) #Undo for staleness

    #Save as new results .txt file
    new_header = ' '.join(header)
    np.savetxt(r'{0}/remtimespan_posterior_sample.txt'.format(results_path),\
               results.values, header=new_header)

    return results