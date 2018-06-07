import pytest
import numpy as np
import numpy.testing as npt

def test_import():
    import pykima


## pykima.keplerian

def test_true_ecc_anomaly():
    from pykima.keplerian import true_anomaly, ecc_anomaly
    
    # true anomaly f
    npt.assert_allclose(true_anomaly(0., 0.), 0.0)
    npt.assert_allclose(true_anomaly(np.pi, 0.), np.pi)

    # eccentric anomaly E
    npt.assert_allclose(ecc_anomaly(0., 0.), 0.)
    npt.assert_allclose(ecc_anomaly(np.pi, 0.), np.pi)
    npt.assert_allclose(ecc_anomaly(2*np.pi, 0.), 2*np.pi)
    npt.assert_allclose(ecc_anomaly(0., 0.2), 0.)
    # checked with http://orbitsimulator.com/sheela/kepler.htm
    E = np.rad2deg(ecc_anomaly(np.deg2rad(80), 0.2))
    npt.assert_allclose(E, 91.45545886486815, rtol=1e-8)

def test_keplerian():
    from pykima.keplerian import keplerian
    # keplerian(time, p, k, ecc, omega, t0, vsys)

    times1 = [0.,]
    times2 = [0., 1., 2.]

    # before we were testing getting a NaN result when P=0
    # now it should raise a FloatingPointError
    with pytest.warns(RuntimeWarning):
        with pytest.raises(FloatingPointError, message="Expecting FloatingPointError"):
            keplerian(times1, 0., 0., 0., 0., 0., 0.)
            # assert np.isnan(result)
            keplerian(times2, 0., 0., 0., 0., 0., 0.)
            # assert np.all(np.isnan(result))

    npt.assert_allclose(keplerian(times1, 100., 1., 0., 0., 0., 0.), 1.)
    npt.assert_allclose(keplerian(times2, 1., 1., 0., 0., 0., 0.), 1.)


## pykima.display

def test_percentiles():
    from pykima.display import percentile68_ranges, percentile68_ranges_latex

    a = np.random.uniform(size=int(1e6))
    b = np.random.randn(int(1e6))

    ra = percentile68_ranges(a)
    rb = percentile68_ranges(b)

    npt.assert_allclose(ra[0], np.median(a))
    npt.assert_allclose(ra[0]+ra[1], np.percentile(a, 84))
    npt.assert_allclose(ra[0]-ra[2], np.percentile(a, 16))
    npt.assert_allclose(ra[0], 0.5, rtol=1e-2)

    npt.assert_allclose(rb[0], np.median(b))
    npt.assert_allclose(rb[1], 0.9944578832097531677397, rtol=1e-2)
    npt.assert_allclose(rb[2], 0.9944578832097531677397, rtol=1e-2)
    npt.assert_allclose(rb[0], 0.0, atol=1e-2)

    a = np.linspace(0, 10, 100)
    npt.assert_allclose(percentile68_ranges(a), [5.0, 3.4, 3.4])
    assert percentile68_ranges_latex(a) == '$5.00 ^{+3.40} _{-3.40}$'


def test_planet_mass():
    from pykima.display import get_planet_mass
    # get_planet_mass(P, K, e, star_mass=1.0, full_output=False, verbose=False)
    npt.assert_allclose(get_planet_mass(0., 0., 0.), (0., 0.))
    npt.assert_allclose(get_planet_mass(np.random.rand(), 0., np.random.rand()), 
                        (0., 0.))


def test_KimaResults():
    pass




## pykima.dnest4

def test_logsumexp():
    from pykima.classic import logsumexp
    from scipy.special import logsumexp as s_logsumexp

    a = np.random.rand(10)
    npt.assert_allclose(logsumexp(a), s_logsumexp(a))


## pykima.showresults

# def test_showresults():
    # pass
