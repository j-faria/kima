__all__ = ['keplerian', 'true_anomaly', 'ecc_anomaly']

import numpy as np
pi = np.pi

def keplerian(time, p, k, ecc, omega, omdot, t0, vsys):
    vel = np.zeros_like(time)
    p, k, ecc, omega, omdot, t0 = np.atleast_1d(p, k, ecc, omega, omdot, t0)

    with np.errstate(divide='raise'):
        for i in range(p.size):
            p[i] = Period_change(p[i], omdot[i])
            M = 2.*pi * (time-t0[i]) / p[i]
            E = ecc_anomaly(M, ecc[i])
            nu = true_anomaly(E, ecc[i])
            wdot = omdot[i]*(4.8481*(0.000001)/365.25) #convert from arcsec per year to radians per day
            vel += k[i] * (np.cos(omega[i]+nu+wdot*(time-t0[i])) + ecc[i]*np.cos(omega[i]+wdot*(time-t0[i])))
        vel += vsys
    return vel

def true_anomaly(E, e):
    return 2. * np.arctan(np.sqrt((1. + e) / (1. - e)) * np.tan(E / 2.))


def ecc_anomaly(M, e):
    M = np.atleast_1d(M)
    E0 = M; E = M
    for _ in range(200):
        g = E0 - e * np.sin(E0) - M
        gp = 1.0 - e * np.cos(E0)
        E = E0 - g / gp

        # check for convergence
        if (np.linalg.norm(E - E0, ord=1) <= 1.234e-10):
            return E
        # keep going
        E0 = E

    # no convergence, return the best estimate
    return E
    
def Period_change(P2,wdot):
    '''
    Convert the observational period to the anomalistic period, change due to the 
    apsidal precession which we assume is spread evenly/circular orbit
    '''
    return P2*(1+wdot/(3600*180*365*2)*P2)