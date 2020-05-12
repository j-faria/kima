import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as Tstudent, norm

from pykima.keplerian import keplerian

# reproducible
np.random.seed(43)

vsys = 11.1  # (m/s)
P = 32.1  # (days)
K = 4.6  # (m/s)
ecc = 0.1
w = 0.3


def create_d1():
    N = 68
    # from 01/03/2010 to 24/01/2019, because why not
    t = np.sort(np.random.uniform(55256, 58507, N))

    rv = np.full_like(t, vsys)

    Tp = t[0] + 10
    planet = keplerian(t, P, K, ecc, w, Tp, 0.0)
    rv += planet

    srv = np.random.uniform(0.6, 0.9, t.size)
    jit = 0.3

    err = norm(0, np.hypot(np.median(srv), jit)).rvs(t.size)
    # err = Tstudent(df=2.1, loc=0, scale=np.hypot(np.median(srv), jit)).rvs(t.size)
    rv += err

    # outliers!!!
    rv[10] += 16.1
    rv[15] += 20
    rv[56] -= 13

    header = 'bjd\tvrad\tsvrad\n---\t----\t-----'
    np.savetxt('d1.txt', np.c_[t, rv, srv],
               header=header, comments='', fmt='%6.3f')

    return t, rv, srv, (P, K, ecc, w, Tp, vsys, jit)


def plot_dataset():
    t, rv, srv, pars = create_d1()

    tt = np.linspace(t[0], t[-1], 5000)
    pl = keplerian(tt, *pars[:6])

    _, ax = plt.subplots(1, 1)
    ax.set(xlabel='Time [days]', ylabel='RV [m/s]')

    ax.errorbar(t, rv, srv, fmt='o')
    ax.plot(tt, pl, 'k', zorder=-1)

    plt.show()