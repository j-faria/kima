import sys, os
import re
import math
import datetime as dt
import configparser

import numpy as np
from scipy import stats

import urepr
from loguniform import LogUniform, ModifiedLogUniform
from kumaraswamy import kumaraswamy

# CONSTANTS
mjup2mearth = 317.8284065946748  # 1 Mjup in Mearth

template_setup = """
[kima]
    GP: false
    GP_kernel: 0
    MA: false
    hyperpriors: false
    trend: false
    degree: 0
    multi_instrument: false
    known_object: false
    n_known_object: 0
    studentt: false
    indicator_correlations: false
    indicators:

    file: filename.txt
    units: kms
    skip: 0
    multi: false
    files:
    M0_epoch: 0.0

[priors.general]
[priors.planets]
"""


def need_model_setup(exception):
    print()
    print("[FATAL] Couldn't find the file kima_model_setup.txt")
    print("Probably didn't include a call to save_setup() in the")
    print("RVModel constructor (this is the recommended solution).")
    print("As a workaround, create a file called `kima_model_setup.txt`,")
    print("and add to it (after editting!) the following options:")
    print(template_setup)

    sys.tracebacklimit = 0
    raise exception


def read_model_setup(filename='kima_model_setup.txt'):
    setup = configparser.ConfigParser()
    setup.optionxform = str

    try:
        open('kima_model_setup.txt')
    except IOError as exc:
        need_model_setup(exc)

    setup.read(filename)

    # if sys.version_info < (3, 0):
    #     setup = setup._sections
    #     # because we cheated, we need to cheat a bit more...
    #     setup['kima']['GP'] = setup['kima'].pop('gp')

    return setup


def read_datafile(datafile, skip):
    """
    Read data from `datafile` for multiple instruments.
    Can be str, in which case the 4th column is assumed to contain an integer
    identifier of the instrument.
    Or list, in which case each element will be one different filename
    containing three columns each.
    """
    if isinstance(datafile, list):
        data = np.empty((0, 3))
        obs = np.empty((0, ))
        for i, df in enumerate(datafile):
            d = np.loadtxt(df, usecols=(0, 1, 2), skiprows=skip, ndmin=2)
            data = np.append(data, d, axis=0)
            obs = np.append(obs, (i + 1) * np.ones((d.shape[0])))
        return data, obs
    else:
        data = np.loadtxt(datafile, usecols=(0, 1, 2), skiprows=skip)
        obs = np.loadtxt(datafile, usecols=(3, ), skiprows=skip, dtype=int)
        uobs = np.unique(obs)
        if uobs.min() > 0:
            uobs -= uobs.min()

        return data, obs


def read_datafile_mo(datafile, skip):
    """
    Read data from `datafile` for multiple instruments and multi outputs.
    Can be str, in which case the 4th column is assumed to contain an integer
    identifier of the instrument.
    Or list, in which case each element will be one different filename
    containing three columns each.
    """
    if isinstance(datafile, list):
        data = np.empty((0, 5))
        obs = np.empty((0, ))
        for i, df in enumerate(datafile):
            d = np.loadtxt(df, usecols=range(5), skiprows=skip, ndmin=2)
            data = np.append(data, d, axis=0)
            obs = np.append(obs, (i + 1) * np.ones((d.shape[0])))
        return data, obs
    else:
        raise NotImplementedError
        # data = np.loadtxt(datafile, usecols=range(5), skiprows=skip)
        # obs = np.loadtxt(datafile, usecols=(3, ), skiprows=skip, dtype=int)
        # uobs = np.unique(obs)
        # if uobs.min() > 0:
        #     uobs -= uobs.min()
        # return data


def show_tips():
    """ Show a few tips on how to use kima """
    tips = (
        "Press Ctrl+C in any of kima's plots to copy the figure.",
        "Run 'kima-showresults all' to plot every figure.",
        "Use the 'kima-template' script to create a new bare-bones directory.")
    if np.random.rand() < 0.2:  # only sometimes, otherwise it's annoying :)
        tip = np.random.choice(tips)
        print('[kima TIP] ' + tip)


def rms(array):
    """ Root mean square of array """
    return np.sqrt(np.sum(array**2) / array.size)


def wrms(array, weights):
    """ Weighted root mean square of array, given weights """
    mu = np.average(array, weights=weights)
    return np.sqrt(np.sum(weights * (array - mu)**2) / np.sum(weights))


def apply_argsort(arr1, arr2, axis=-1):
    """
    Apply arr1.argsort() on arr2, along `axis`.
    """
    # check matching shapes
    assert arr1.shape == arr2.shape, "Shapes don't match!"

    i = list(np.ogrid[[slice(x) for x in arr1.shape]])
    i[axis] = arr1.argsort(axis)
    return arr2[i]


def percentile68_ranges(a, min=None, max=None):
    if min is None and max is None:
        mask = np.ones_like(a, dtype=bool)
    elif min is None:
        mask = a < max
    elif max is None:
        mask = a > min
    else:
        mask = (a > min) & (a < max)
    lp, median, up = np.percentile(a[mask], [16, 50, 84])
    return (median, up - median, median - lp)


def percentile68_ranges_latex(a, min=None, max=None):
    median, plus, minus = percentile68_ranges(a, min, max)
    return '$' + urepr.core.uformatul(median, plus, minus, 'L') + '$'


def percentile_ranges(a, percentile=68, min=None, max=None):
    if min is None and max is None:
        mask = np.ones_like(a, dtype=bool)
    elif min is None:
        mask = a < max
    elif max is None:
        mask = a > min
    else:
        mask = (a > min) & (a < max)
    half = percentile / 2
    lp, median, up = np.percentile(a[mask], [50 - half, 50, 50 + half])
    return (median, up - median, median - lp)


def percentile_ranges_latex(a, percentile, min=None, max=None):
    median, plus, minus = percentile_ranges(a, percentile, min, max)
    return '$' + urepr.core.uformatul(median, plus, minus, 'L') + '$'


def clipped_mean(arr, min, max):
    """ Mean of `arr` between `min` and `max` """
    mask = (arr > min) & (arr < max)
    return np.mean(arr[mask])


def clipped_std(arr, min, max):
    """ std of `arr` between `min` and `max` """
    mask = (arr > min) & (arr < max)
    return np.std(arr[mask])


def get_planet_mass(P, K, e, star_mass=1.0, full_output=False, verbose=False):
    """
    Calculate the planet (minimum) mass Msini given orbital period `P`,
    semi-amplitude `K`, eccentricity `e`, and stellar mass. If star_mass is a
    tuple with (estimate, uncertainty), this (Gaussian) uncertainty will be
    taken into account in the calculation.

    Units:
        P [days]
        K [m/s]
        e []
        star_mass [Msun]
    Returns:
        if P is float:
            if star_mass is float:
                Msini [Mjup], Msini [Mearth]
            if star_mass is tuple:
                (Msini, error_Msini) [Mjup], (Msini, error_Msini) [Mearth]
        if P is array:
            if full_output: mean Msini [Mjup], std Msini [Mjup], Msini [Mjup] (array)
            else: mean Msini [Mjup], std Msini [Mjup], mean Msini [Mearth], std Msini [Mearth]
    """
    if verbose: print('Using star mass = %s solar mass' % star_mass)

    try:
        P = float(P)
        # calculate for one value of the orbital period
        # then K, e, and star_mass should also be floats
        assert isinstance(K, float) and isinstance(e, float)
        uncertainty_star_mass = False
        if isinstance(star_mass, tuple) or isinstance(star_mass, list):
            star_mass = np.random.normal(star_mass[0], star_mass[1], 5000)
            uncertainty_star_mass = True

        m_mj = 4.919e-3 * star_mass**(2. / 3) * P**(1. / 3) * K * np.sqrt(1 -
                                                                          e**2)
        m_me = m_mj * mjup2mearth
        if uncertainty_star_mass:
            return (m_mj.mean(), m_mj.std()), (m_me.mean(), m_me.std())
        else:
            return m_mj, m_me

    except TypeError:
        # calculate for an array of periods
        if isinstance(star_mass, tuple) or isinstance(star_mass, list):
            # include (Gaussian) uncertainty on the stellar mass
            star_mass = np.random.normal(star_mass[0], star_mass[1], P.size)

        m_mj = 4.919e-3 * star_mass**(2. / 3) * P**(1. / 3) * K * np.sqrt(1 -
                                                                          e**2)
        m_me = m_mj * mjup2mearth

        if full_output:
            return m_mj.mean(), m_mj.std(), m_mj
        else:
            return (m_mj.mean(), m_mj.std(), m_me.mean(), m_me.std())


def get_planet_mass_latex(P, K, e, star_mass=1.0, earth=False, **kargs):
    out = get_planet_mass(P, K, e, star_mass, full_output=True, verbose=False)

    if isinstance(P, float):
        if earth:
            return '$%f$' % out[1]
        else:
            return '$%f$' % out[0]
    else:
        if earth:
            return percentile68_ranges_latex(out[2] * mjup2mearth)
        else:
            return percentile68_ranges_latex(out[2])


def get_planet_semimajor_axis(P, K, star_mass=1.0, full_output=False,
                              verbose=False):
    """
    Calculate the semi-major axis of the planet's orbit given
    orbital period `P`, semi-amplitude `K`, and stellar mass.
    Units:
        P [days]
        K [m/s]
        star_mass [Msun]
    Returns:
        if P is float: a [AU]
        if P is array:
            if full_output: mean a [AU], std a [AU], a [AU] (array)
            else: mean a [AU], std a [AU]
    """
    if verbose: print('Using star mass = %s solar mass' % star_mass)

    # gravitational constant G in AU**3 / (Msun * day**2), to the power of 1/3
    f = 0.0666378476025686

    if isinstance(P, float):
        # calculate for one value of the orbital period
        # then K and star_mass should also be floats
        assert isinstance(K, float)
        assert isinstance(star_mass, float)
        a = f * star_mass**(1. / 3) * (P / (2 * np.pi))**(2. / 3)
        return a  # in AU
    else:
        if isinstance(star_mass, tuple) or isinstance(star_mass, list):
            star_mass = star_mass[0] + star_mass[1] * np.random.randn(P.size)
        a = f * star_mass**(1. / 3) * (P / (2 * np.pi))**(2. / 3)

        if full_output:
            return a.mean(), a.std(), a
        else:
            return a.mean(), a.std()


def get_planet_semimajor_axis_latex(P, K, star_mass=1.0, earth=False, **kargs):
    out = get_planet_semimajor_axis(P, K, star_mass, full_output=True,
                                    verbose=False)
    if isinstance(P, float):
        return '$%f$' % out
    else:
        return '$%f$' % out[0]


def get_planet_mass_and_semimajor_axis(P, K, e, star_mass=1.0,
                                       full_output=False, verbose=False):
    """
    Calculate the planet (minimum) mass Msini and the semi-major axis given
    orbital period `P`, semi-amplitude `K`, eccentricity `e`, and stellar mass.
    If star_mass is a tuple with (estimate, uncertainty), this (Gaussian)
    uncertainty will be taken into account in the calculation.

    Units:
        P [days]
        K [m/s]
        e []
        star_mass [Msun]
    Returns:
        (M, A) where
            M is the output of get_planet_mass
            A is the output of get_planet_semimajor_axis
    """
    # this is just a convenience function for calling
    # get_planet_mass and get_planet_semimajor_axis

    if verbose:
        print('Using star mass = %s solar mass' % star_mass)

    mass = get_planet_mass(P, K, e, star_mass, full_output, verbose=False)
    a = get_planet_semimajor_axis(P, K, star_mass, full_output, verbose=False)
    return mass, a


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def _prior_to_dist():
    """ Convert a prior name to a prior object """
    d = {
        'Uniform': stats.uniform,
        'LogUniform': stats.reciprocal,
        'ModifiedLogUniform': ModifiedLogUniform,
        'Gaussian': stats.norm,
        'TruncatedGaussian': stats.truncnorm,
        'Exponential': stats.expon,
        'Kumaraswamy': kumaraswamy,
        'Laplace': stats.laplace,
        'Cauchy': stats.cauchy,
    }
    return d


def _get_prior_parts(prior):
    if not isinstance(prior, str):
        raise ValueError('`prior` should be a string, got', prior)

    try:
        inparens = re.search(r'\((.*?)\)', prior).group(1)
    except AttributeError:
        raise ValueError('Cannot decode "%s", seems badly formed' % prior)
    try:
        truncs = re.search(r'\[(.*?)\]', prior).group(1)
    except AttributeError:
        truncs = ''

    name = prior[:prior.find('(')]

    return inparens, truncs, name


def find_prior_limits(prior):
    """
    Find lower and upper limits of a prior from the kima_model_setup.txt file.
    """
    inparens, truncs, name = _get_prior_parts(prior)
    # print(inparens, truncs, name)

    if 'Truncated' in name:
        return tuple(float(v) for v in truncs.split(','))

    if name == 'ModifiedLogUniform':
        return (0.0, float(inparens.split(';')[1]))

    if name == 'Uniform':
        v1, v2 = inparens.split(';')
        return (float(v1), float(v2) - float(v1))

    if name == 'LogUniform':
        return tuple(float(v) for v in inparens.split(';'))

    if name == 'Gaussian':
        return (-np.inf, np.inf)

    if name == 'Kumaraswamy':
        return (0.0, 1.0)


def find_prior_parameters(prior):
    """ Find parameters of a prior from the kima_model_setup.txt file. """
    inparens, _, name = _get_prior_parts(prior)

    truncated = False
    if 'Truncated' in name:
        inparens = ';'.join(inparens.split(';')[:-1])
        name = name.replace('Truncated', '')
        truncated = True

    twopars = ('LogUniform', 'ModifiedLogUniform', 'Gaussian', 'Kumaraswamy',
               'Cauchy')

    if name in twopars:
        r = [float(v) for v in inparens.split(';')]
    elif name == 'Uniform':
        v1, v2 = inparens.split(';')
        r = [float(v1), float(v2) - float(v1)]
    elif name == 'Exponential':
        r = [
            float(inparens),
        ]
    else:
        r = [
            np.nan,
        ]

    if truncated:
        r.append('T')

    return tuple(r)


def get_prior(prior):
    """ Return a scipt.stats-like prior from a kima_model_setup.txt prior """
    _, _, name = _get_prior_parts(prior)
    pars = find_prior_parameters(prior)

    if 'T' in pars:
        a, b = find_prior_limits(prior)
        loc, scale, _ = pars
        a, b = (a - loc) / scale, (b - loc) / scale

    try:
        d = _prior_to_dist()
        if 'T' in pars:
            return d[name](a=a, b=b, loc=loc, scale=scale)
        else:
            return d[name](*pars)
    except KeyError:
        return None


def hyperprior_samples(size):
    pi, log, rng = np.pi, np.log, np.random
    U = rng.rand(size)
    muP = 5.901 + np.tan(pi * (0.97 * rng.rand(size) - 0.485))
    wP = 0.1 + 2.9 * rng.rand(size)
    P = np.where(U < 0.5, muP + wP * log(2. * U), muP - wP * log(2. - 2. * U))
    return np.exp(P)


def get_star_name(data_file):
    """ Find star name (usually works for approx. standard filenames) """
    bn = os.path.basename(data_file)
    try:
        pattern = '|'.join([
            'HD\d+', 'HIP\d+', 'HR\d+', 'BD-\d+', 'CD-\d+', 'NGC\d+No\d+',
            'GJ\d+', 'Gl\d+'
        ])
        return re.findall(pattern, bn)[0]

    except IndexError:  # didn't find correct name
        if bn.endswith('_harps.rdb'): return bn[:-10]

    return 'unknown'


def get_instrument_name(data_file):
    """ Find instrument name """
    bn = os.path.basename(data_file)
    try:
        pattern = '|'.join([
            'ESPRESSO',
            'HARPS[^\W_]*[\d+]*',
            'HIRES',
            'APF',
            'CORALIE',
            'HJS',
            'ELODIE',
            'KECK',
            'HET',
            'LICK',
        ])
        return re.findall(pattern, bn, re.IGNORECASE)[0]

    except IndexError:
        try:
            return re.findall(pattern, bn.upper())[0]
        except IndexError:  # didn't find
            return data_file


# covert dates to/from Julian days and MJD
# originally from https://gist.github.com/jiffyclub/1294443
# author: Matt Davis (http://github.com/jiffyclub)


def mjd_to_jd(mjd):
    """ Convert Modified Julian Day to Julian Day.

    Parameters
    ----------
    mjd : float
        Modified Julian Day

    Returns
    -------
    jd : float
        Julian Day
    """
    return mjd + 2400000.5


def jd_to_mjd(jd):
    """ Convert Julian Day to Modified Julian Day

    Parameters
    ----------
    jd : float
        Julian Day

    Returns
    -------
    mjd : float
        Modified Julian Day
    """
    return jd - 2400000.5


def date_to_jd(year, month, day):
    """ Convert a date (year, month, day) to Julian Day.

    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.

    Parameters
    ----------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.
    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.
    day : float
        Day, may contain fractional part.

    Returns
    -------
    jd : float
        Julian Day

    Examples
    --------
    Convert 6 a.m., February 17, 1985 to Julian Day
    >>> date_to_jd(1985, 2, 17.25)
    2446113.75
    """
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month

    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or (year == 1582 and month < 10)
            or (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)

    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)

    D = math.trunc(30.6001 * (monthp + 1))

    jd = B + C + D + day + 1720994.5

    return jd


def jd_to_date(jd):
    """ Convert Julian Day to date.

    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.

    Parameters
    ----------
    jd : float
        Julian Day

    Returns
    -------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.
    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.
    day : float
        Day, may contain fractional part.

    Examples
    --------
    Convert Julian Day 2446113.75 to year, month, and day.
    >>> jd_to_date(2446113.75)
    (1985, 2, 17.25)
    """
    jd = jd + 0.5

    F, I = math.modf(jd)
    I = int(I)

    A = math.trunc((I - 1867216.25) / 36524.25)

    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I

    C = B + 1524

    D = math.trunc((C - 122.1) / 365.25)

    E = math.trunc(365.25 * D)

    G = math.trunc((C - E) / 30.6001)

    day = C - E + F - math.trunc(30.6001 * G)

    if G < 13.5:
        month = G - 1
    else:
        month = G - 13

    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715

    return year, month, day


def hmsm_to_days(hour=0, min=0, sec=0, micro=0):
    """
    Convert hours, minutes, seconds, and microseconds to fractional days.

    Parameters
    ----------
    hour : int, optional
        Hour number. Defaults to 0.
    min : int, optional
        Minute number. Defaults to 0.
    sec : int, optional
        Second number. Defaults to 0.
    micro : int, optional
        Microsecond number. Defaults to 0.

    Returns
    -------
    days : float
        Fractional days.

    Examples
    --------
    >>> hmsm_to_days(hour=6)
    0.25
    """
    days = sec + (micro / 1.e6)

    days = min + (days / 60.)

    days = hour + (days / 60.)

    return days / 24.


def days_to_hmsm(days):
    """ Convert fractional days to hours, minutes, seconds, and microseconds.
    Precision beyond microseconds is rounded to the nearest microsecond.

    Parameters
    ----------
    days : float
        A fractional number of days. Must be less than 1.

    Returns
    -------
    hour : int
        Hour number.
    min : int
        Minute number.
    sec : int
        Second number.
    micro : int
        Microsecond number.

    Raises
    ------
    ValueError
        If `days` is >= 1.

    Examples
    --------
    >>> days_to_hmsm(0.1)
    (2, 24, 0, 0)
    """
    hours = days * 24.
    hours, hour = math.modf(hours)

    mins = hours * 60.
    mins, min = math.modf(mins)

    secs = mins * 60.
    secs, sec = math.modf(secs)

    micro = round(secs * 1.e6)

    return int(hour), int(min), int(sec), int(micro)


def datetime_to_jd(date):
    """ Convert a `datetime.datetime` object to Julian Day.

    Parameters
    ----------
    date : `datetime.datetime` instance

    Returns
    -------
    jd : float
        Julian day.

    Examples
    --------
    >>> d = datetime.datetime(1985, 2, 17, 6)
    >>> d
    datetime.datetime(1985, 2, 17, 6, 0)
    >>> jdutil.datetime_to_jd(d)
    2446113.75
    """
    days = date.day + hmsm_to_days(date.hour, date.minute, date.second,
                                   date.microsecond)

    return date_to_jd(date.year, date.month, days)
