"""
This module defines the `KimaResults` class to hold results from a run.
"""

import os
import pickle
from typing import List, Union
import zipfile
import time
import tempfile
import inspect
from string import ascii_lowercase
from dataclasses import dataclass, field

from .pykepler import keplerian as kepleriancpp
from .keplerian import keplerian as keplerianpy
from .GP import (GP, RBFkernel, QPkernel, QPCkernel, PERkernel, QPpCkernel,
                 mixtureGP)

from .analysis import get_planet_mass_and_semimajor_axis
from .utils import (read_datafile, read_datafile_rvfwhm, read_model_setup,
                    get_star_name, mjup2mearth, get_prior, get_instrument_name,
                    _show_kima_setup, read_big_file, wrms)

from . import display

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, randint as discrete_uniform
try:  # only available in scipy 1.1.0
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

pathjoin = os.path.join
colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]


def keplerian(*args, **kwargs):
    return np.array(kepleriancpp(*args, **kwargs))


def _read_priors(setup):
    priors = list(setup['priors.general'].values())
    prior_names = list(setup['priors.general'].keys())
    try:
        for section in ('priors.planets', 'priors.hyperpriors', 'priors.GP'):
            try:
                priors += list(setup[section].values())
                prior_names += list(setup[section].keys())
            except KeyError:
                continue

        priors += list(setup['priors.known_object'].values())
        prior_names += ['KO_' + k for k in setup['priors.known_object'].keys()]
    except KeyError:
        pass

    priors = {
        n: v
        for n, v in zip(prior_names, [get_prior(p) for p in priors])
    }

    return priors


@dataclass
class data_holder:
    """ A simple class to hold the datasets used in kima

    Attributes:
        t (ndarray): The observation times
        y (ndarray): The observed radial velocities
        e (ndarray): The radial velocity uncertainties
        obs (ndarray): Identifier for the instrument of each observation
        N (int): Total number of observations
    """
    t: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    e: np.ndarray = field(init=False)
    obs: np.ndarray = field(init=False)
    N: int = field(init=False)


@dataclass
class posterior_holder:
    """ A simple class to hold the posterior samples

    Attributes:
        P (ndarray): The orbital period(s)
        K (ndarray): The semi-amplitude(s)
        e (ndarray): The orbital eccentricities(s)
        ω (ndarray): The argument(s) of pericenter
        φ (ndarray): The mean anomaly(ies) at the epoch
    """
    P: np.ndarray = field(init=False, repr=False)
    K: np.ndarray = field(init=False, repr=False)
    e: np.ndarray = field(init=False, repr=False)
    ω: np.ndarray = field(init=False, repr=False)
    φ: np.ndarray = field(init=False, repr=False)


class KimaResults:
    r""" A class to hold, analyse, and display the results from kima

    Attributes:
        model (str):
            The type of kima model
        priors (dict):
            A dictionary with the priors used in the model
        ESS (int):
            Effective sample size
        evidence (float):
            The log-evidence ($\ln Z$) of the model
        information (float):
            The Kulback-Leibler divergence between prior and posterior

        data (data_holder): The data
        posteriors (posterior_holder): The marginal posterior samples
    """

    data: data_holder
    posteriors: posterior_holder

    model: str
    priors: dict
    GPmodel: bool

    evidence: float
    information: float
    ESS: int

    _debug = False

    def __init__(self, options, save_plots=False, return_figs=True,
                 verbose=False):

        self.options = options
        self.save_plots = save_plots
        self.return_figs = return_figs
        self.verbose = verbose

        self.removed_crossing = False
        self.removed_roche_crossing = False

        self.setup = setup = read_model_setup()
        try:
            self.model = setup['kima']['model']
        except KeyError:
            self.model = 'RVmodel'

        if self._debug:
            print('model:', self.model)

        try:
            self.fix = setup['kima']['fix'] == 'true'
        except KeyError:
            self.fix = True
        self.npmax = int(setup['kima']['npmax'])

        # read the priors
        self.priors = _read_priors(setup)
        if self._debug:
            print('finished reading priors')

        # and the data
        self._read_data()
        if self._debug:
            print('finished reading data')

        self.posterior_sample = np.atleast_2d(
            read_big_file('posterior_sample.txt'))

        try:
            self.posterior_lnlike = np.atleast_2d(
                read_big_file('posterior_sample_info.txt'))
            self.lnlike_available = True
        except IOError:
            self.lnlike_available = False
            print('Could not find file "posterior_sample_info.txt", '
                  'log-likelihoods will not be available.')

        try:
            t1 = time.time()
            self.sample = np.atleast_2d(read_big_file('sample.txt'))
            t2 = time.time()
            self.sample_info = np.atleast_2d(read_big_file('sample_info.txt'))
            with open('sample.txt', 'r') as fs:
                header = fs.readline()
                header = header.replace('#', '').replace('  ', ' ').strip()
                self.parameters = [p for p in header.split(' ') if p != '']

            # different sizes can happen when running the model and sample_info
            # was updated while reading sample.txt
            if self.sample.shape[0] != self.sample_info.shape[0]:
                minimum = min(self.sample.shape[0], self.sample_info.shape[0])
                self.sample = self.sample[:minimum]
                self.sample_info = self.sample_info[:minimum]
        except IOError:
            self.sample = None
            self.sample_info = None

        if self._debug:
            print('finished reading sample file', end=' ')
            print(f'(took {t2 - t1:.1f} seconds)')

        self.posteriors = posterior_holder()
        self.indices = {}
        self.total_parameters = 0

        self._current_column = 0
        # read jitters
        self._read_jitters()
        # read limb-darkening coefficients
        if self.model == 'TRANSITmodel':
            self._read_limb_dark()
        # find trend in the compiled model and read it
        self._read_trend()
        # multiple instruments? read offsets
        self._read_multiple_instruments()
        # activity indicator correlations?
        self._read_actind_correlations()
        # find GP in the compiled model
        self._read_GP()
        # find MA in the compiled model
        self._read_MA()
        # find KO in the compiled model
        self._read_KO()

        self._read_components()

        # staleness (ignored)
        self._current_column += 1

        self._read_studentt()

        if self.model == 'RVFWHMmodel':
            self.C2 = self.posterior_sample[:, self._current_column]
            self.indices['C2'] = self._current_column
            self._current_column += 1

        self.vsys = self.posterior_sample[:, -1]
        self.indices['vsys'] = -1

        # build the marginal posteriors for planet parameters
        self.get_marginals()

        # make the plots, if requested
        self.make_plots(options, self.save_plots)

    def __repr__(self):
        return f'KimaResults(lnZ={self.evidence:.1f}, ESS={self.ESS})'

    def _read_data(self):
        setup = self.setup
        section = 'data' if 'data' in setup else 'kima'

        try:
            self.multi = setup[section]['multi'] == 'true'
        except KeyError:
            self.multi = False

        if self.model == 'HierarchicalRVmodel':
            self.multi = True

        if self.multi:
            if setup[section]['files'] == '':
                # multi is true but in only one file
                data_file = setup[section]['file']
                self.multi_onefile = True
            else:
                data_file = setup[section]['files'].split(',')[:-1]
                self.multi_onefile = False
                # raise NotImplementedError('TO DO')
        else:
            data_file = setup[section]['file']

        if self.verbose:
            print('Loading data file %s' % data_file)
        self.data_file = data_file

        self.data_skip = int(setup[section]['skip'])
        self.units = setup[section]['units']
        self.M0_epoch = float(setup[section]['M0_epoch'])

        if self.multi:
            if self.model == 'RVFWHMmodel':
                data, obs = read_datafile_rvfwhm(self.data_file,
                                                 self.data_skip)
            else:
                data, obs = read_datafile(self.data_file, self.data_skip)

            # make sure the times are sorted when coming from multiple
            # instruments
            ind = data[:, 0].argsort()
            data = data[ind]
            obs = obs[ind]
            self.n_instruments = np.unique(obs).size
            if self.model == 'RVFWHMmodel':
                self.n_jitters = 2 * self.n_instruments
            elif self.model == 'HierarchicalRVmodel':
                self.n_instruments = 1
                self.n_jitters = 1
            else:
                self.n_jitters = self.n_instruments

        else:
            if self.model == 'RVFWHMmodel':
                cols = range(5)
                self.n_jitters = 2
            else:
                cols = range(3)
                self.n_jitters = 1
            self.n_instruments = 1

            data = np.loadtxt(self.data_file, skiprows=self.data_skip,
                              usecols=cols)
            obs = np.ones_like(data[:, 0], dtype=int)
            self._extra_data = np.loadtxt(self.data_file,
                                          skiprows=self.data_skip)

        # to m/s
        if self.units == 'kms':
            data[:, 1] *= 1e3
            data[:, 2] *= 1e3
            if self.model == 'RVFWHMmodel':
                data[:, 3] *= 1e3
                data[:, 4] *= 1e3

        # arbitrary units?
        if 'arb' in self.units:
            self.arbitrary_units = True
        else:
            self.arbitrary_units = False

        self.data = data_holder()
        self.data.t = data[:, 0].copy()
        self.data.y = data[:, 1].copy()
        self.data.e = data[:, 2].copy()
        self.data.obs = obs.copy()
        self.data.N = self.data.t.size

        if self.model == 'RVFWHMmodel':
            self.data.y2 = data[:, 3].copy()
            self.data.e2 = data[:, 4].copy()

        self.tmiddle = self.data.t.min() + 0.5 * self.data.t.ptp()

    def _read_jitters(self):
        i1, i2 = self._current_column, self._current_column + self.n_jitters
        self.posteriors.jitter = self.posterior_sample[:, i1:i2]
        self.jitter = self.posterior_sample[:, i1:i2]
        self._current_column += self.n_jitters
        self.indices['jitter_start'] = i1
        self.indices['jitter_end'] = i2
        self.indices['jitter'] = slice(i1, i2)
        if self._debug:
            print('finished reading jitters')

    def _read_limb_dark(self):
        i1, i2 = self._current_column, self._current_column + 2
        self.u = self.posterior_sample[:, i1:i2]
        self._current_column += 2
        self.indices['u_start'] = i1
        self.indices['u_end'] = i2
        self.indices['u'] = slice(i1, i2)
        if self._debug:
            print('finished reading limb darkening')

    def _read_trend(self):
        self.trend = self.setup['kima']['trend'] == 'true'
        self.trend_degree = int(self.setup['kima']['degree'])

        if self.trend:
            n_trend = self.trend_degree
            i1 = self._current_column
            i2 = self._current_column + n_trend
            self.trendpars = self.posterior_sample[:, i1:i2]
            self._current_column += n_trend
            self.indices['trend'] = slice(i1, i2)
        else:
            n_trend = 0
        self.total_parameters += n_trend

        if self._debug:
            print('finished reading trend, trend =', self.trend)

    def _read_multiple_instruments(self):
        if self.multi:
            # there are n instruments and n-1 offsets per output
            if self.model == 'RVFWHMmodel':
                n_inst_offsets = 2 * (self.n_instruments - 1)
            else:
                n_inst_offsets = self.n_instruments - 1

            istart = self._current_column
            iend = istart + n_inst_offsets
            ind = np.s_[istart:iend]
            self.inst_offsets = self.posterior_sample[:, ind]
            self._current_column += n_inst_offsets
            self.indices['inst_offsets_start'] = istart
            self.indices['inst_offsets_end'] = iend
            self.indices['inst_offsets'] = slice(istart, iend)
        else:
            n_inst_offsets = 0
        self.total_parameters += n_inst_offsets

        if self._debug:
            print('finished reading multiple instruments')

    def _read_actind_correlations(self):
        setup = self.setup
        try:
            self.indcorrel = setup['kima']['indicator_correlations'] == 'true'
        except KeyError:
            self.indcorrel = False

        if self.indcorrel:
            self.activity_indicators = setup['kima']['indicators'].split(',')
            n_act_ind = len(self.activity_indicators)
            istart = self._current_column
            iend = istart + n_act_ind
            ind = np.s_[istart:iend]
            self.betas = self.posterior_sample[:, ind]
            self._current_column += n_act_ind
            self.indices['betas_start'] = istart
            self.indices['betas_end'] = iend
            self.indices['betas'] = slice(istart, iend)
        else:
            n_act_ind = 0
        self.total_parameters += n_act_ind

    def _read_components(self):
        # how many parameters per component
        self.n_dimensions = int(self.posterior_sample[0, self._current_column])
        self._current_column += 1

        # maximum number of components
        self.max_components = self.npmax
        self._current_column += 1

        # find hyperpriors in the compiled model
        self.hyperpriors = self.setup['kima']['hyperpriors'] == 'true'

        # number of hyperparameters (muP, wP, muK)
        if self.hyperpriors:
            n_dist_print = 3
            istart = self._current_column
            iend = istart + n_dist_print
            self._current_column += n_dist_print
            self.indices['hyperpriors'] = slice(istart, iend)
        elif self.model == 'BDmodel':
            n_dist_print = 2
            istart = self._current_column
            iend = istart + n_dist_print
            self._current_column += n_dist_print
            self.indices['hyperpriors'] = slice(istart, iend)
        else:
            n_dist_print = 0

        # if hyperpriors, then the period is sampled in log
        self.log_period = self.hyperpriors

        # the column with the number of planets in each sample
        self.index_component = self._current_column
        # # try to correct fix and npmax
        uni = np.unique(self.posterior_sample[:, self.index_component])
        if uni.size > 1:
            self.fix = False
        self.npmax = int(uni.max())

        if not self.fix:
            self.priors['np_prior'] = discrete_uniform(0, self.npmax + 1)

        self.indices['np'] = self.index_component
        self._current_column += 1

        # indices of the planet parameters
        n_planet_pars = self.max_components * self.n_dimensions
        istart = self._current_column
        iend = istart + n_planet_pars
        self._current_column += n_planet_pars
        self.indices['planets'] = slice(istart, iend)
        for j, p in zip(range(self.n_dimensions),
                        ('P', 'K', 'φ', 'e', 'ω', 'λ')):
            iend = istart + self.max_components
            self.indices[f'planets.{p}'] = slice(istart, iend)
            istart += self.max_components

    def _read_studentt(self):
        # student-t likelihood?
        try:
            self.studentT = self.setup['kima']['studentt'] == 'true'
        except KeyError:
            self.studentT = False

        if self.studentT:
            self.nu = self.posterior_sample[:, self._current_column]
            self.indices['nu'] = self._current_column
            self._current_column += 1
            self.total_parameters += 1

    @property
    def _GP_par_indices(self):
        """
        indices for specific GP hyperparameters:
        eta1_RV, eta1_FWHM, eta2_RV, eta2_FWHM, eta3_RV, eta3_FWHM, eta4_RV, eta4_FWHM
        """
        if self.model == 'RVFWHMmodel':
            i = [0, 1]
            _i = 2
            num = self.n_hyperparameters - self._n_shared_hyperparameters
            for j in range(2, 2 + num + 1):
                eta = f'share_eta{j}'
                if getattr(self, eta):
                    i += [_i, _i]
                    _i += 1
                else:
                    i += [_i, _i + 1]
                    _i += 1
        else:
            i = [0, 1, 2, 3]

        return i

    def _read_GP(self):
        try:
            self.GPmodel = self.setup['kima']['GP'] == 'true'
            if self.GPmodel:
                if 'kernel' in self.setup['kima']:
                    self.GPkernel = self.setup['kima']['kernel']
                else:
                    self.GPkernel = self.setup['kima']['GP_kernel']
        except KeyError:
            self.GPmodel = False

        if self.GPmodel:
            if self.model in ('GPmodel', 'GPmodel_systematics'):
                try:
                    n_hyperparameters = {
                        'standard': 4,
                        'periodic': 3,
                        'qpc': 5,
                        'RBF': 2,
                    }
                    n_hyperparameters = n_hyperparameters[self.GPkernel]
                except KeyError:
                    raise ValueError(
                        f'GP kernel = {self.GPkernel} not recognized')

                if self.model == 'GPmodel_systematics':
                    n_hyperparameters += 2

                self.n_hyperparameters = n_hyperparameters

            elif self.model == 'RVFWHMmodel':
                _n_shared = 0
                for i in range(2, 7):
                    setattr(self, f'share_eta{i}',
                            self.setup['kima'][f'share_eta{i}'] == 'true')
                    if getattr(self, f'share_eta{i}'):
                        _n_shared += 1

                n_hyperparameters = 2  # at least 2 x eta1
                n_hyperparameters += 1 if self.share_eta2 else 2
                n_hyperparameters += 1 if self.share_eta3 else 2
                n_hyperparameters += 1 if self.share_eta4 else 2
                if self.GPkernel == 'qpc':
                    n_hyperparameters += 1 if self.share_eta5 else 2
                if self.GPkernel == 'qp_plus_cos':
                    n_hyperparameters += 1 if self.share_eta5 else 2
                    n_hyperparameters += 1 if self.share_eta6 else 2
                self.n_hyperparameters = n_hyperparameters
                self._n_shared_hyperparameters = _n_shared

            istart = self._current_column
            iend = istart + n_hyperparameters
            self.etas = self.posterior_sample[:, istart:iend]

            # if self.model == 'RVmodel':
            #     for i in range(n_hyperparameters):
            #         name = 'eta' + str(i + 1)
            #         ind = istart + i
            #         setattr(self, name, self.posterior_sample[:, ind])

            self._current_column += n_hyperparameters
            self.indices['GPpars_start'] = istart
            self.indices['GPpars_end'] = iend
            self.indices['GPpars'] = slice(istart, iend)

            t, e = self.data.t, self.data.e
            kernels = {
                'standard': QPkernel(1, 1, 1, 1),
                'periodic': PERkernel(1, 1, 1),
                'qpc': QPCkernel(1, 1, 1, 1, 1),
                'RBF': RBFkernel(1, 1),
                'qp_plus_cos': QPpCkernel(1, 1, 1, 1, 1, 1),
            }

            if self.model == 'RVFWHMmodel':
                self.GP1 = GP(kernels[self.GPkernel], t, e, white_noise=0.0)
                self.GP2 = GP(kernels[self.GPkernel], t, self.data.e2,
                              white_noise=0.0)
            elif self.model == 'GPmodel_systematics':
                X = np.c_[self.data.t, self._extra_data[:, 3]]
                self.GP = mixtureGP([], X, None, e)
            else:
                self.GP = GP(kernels[self.GPkernel], t, e, white_noise=0.0)


        else:
            n_hyperparameters = 0

    def _read_MA(self):
        # find MA in the compiled model
        try:
            self.MAmodel = self.setup['kima']['MA'] == 'true'
        except KeyError:
            self.MAmodel = False

        if self.MAmodel:
            n_MAparameters = 2
            istart = self._current_column
            iend = istart + n_MAparameters
            self.MA = self.posterior_sample[:, istart:iend]
            self._current_column += n_MAparameters
        else:
            n_MAparameters = 0
        self.total_parameters += n_MAparameters

    def _read_KO(self):
        try:
            self.KO = self.setup['kima']['known_object'] == 'true'
            self.nKO = int(self.setup['kima']['n_known_object'])
        except KeyError:
            self.KO = False
            self.nKO = 0

        if self.KO:
            if self.model in ('BINARIESmodel', 'TRANSITmodel'):
                n_KOparameters = 6 * self.nKO
            else:
                n_KOparameters = 5 * self.nKO
            start = self._current_column
            koinds = slice(start, start + n_KOparameters)
            self.KOpars = self.posterior_sample[:, koinds]
            self._current_column += n_KOparameters
            self.indices['KOpars'] = koinds
        else:
            n_KOparameters = 0
        self.total_parameters += n_KOparameters

    @property
    def _mc(self):
        """ Maximum number of Keplerians in the model """
        return self.max_components

    @property
    def _nd(self):
        """ Number of parameters per Keplerian """
        return self.n_dimensions

    @property
    def parameter_priors(self):
        """ A list of priors which can be indexed using self.indices """
        n = self.posterior_sample.shape[1]
        priors = np.full(n, None)

        if self.model == 'RVmodel':
            priors[self.indices['jitter']] = self.priors['Jprior']

        elif self.model == 'RVFWHMmodel':
            for i in range(self.n_instruments):
                priors[i] = self.priors['Jprior']
            for i in range(self.n_instruments, 2 * self.n_instruments):
                priors[i] = self.priors['J2prior']

        if self.trend:
            names = ('slope_prior', 'quadr_prior', 'cubic_prior')
            trend_priors = [self.priors[n] for n in names if n in self.priors]
            priors[self.indices['trend']] = trend_priors

        if self.multi:
            no = self.n_instruments - 1
            if self.model == 'RVmodel':
                prior = self.priors['offsets_prior']
                priors[self.indices['inst_offsets']] = np.array(no * [prior])
            elif self.model == 'RVFWHMmodel':
                prior1 = self.priors['offsets_prior']
                prior2 = self.priors['offsets2_prior']
                offset_priors = no * [prior1] + no * [prior2]
                priors[self.indices['inst_offsets']] = np.array(offset_priors)

        if self.GPmodel:
            if self.model == 'RVmodel':
                priors[self.indices['GPpars']] = [
                    self.priors[f'eta{i}_prior'] for i in range(5)
                ]
            elif self.model == 'RVFWHMmodel':
                i = self.indices['GPpars_start']
                priors[i] = self.priors['eta1_1_prior']
                i += 1
                priors[i] = self.priors['eta1_2_prior']
                i += 1
                if self.share_eta2:
                    priors[i] = self.priors['eta2_1_prior']
                    i += 1
                else:
                    priors[i] = self.priors['eta2_1_prior']
                    priors[i + 1] = self.priors['eta2_1_prior']
                    i += 2
                #
                if self.share_eta3:
                    priors[i] = self.priors['eta3_1_prior']
                    i += 1
                else:
                    priors[i] = self.priors['eta3_1_prior']
                    priors[i + 1] = self.priors['eta3_1_prior']
                    i += 2
                #
                if self.share_eta4:
                    priors[i] = self.priors['eta4_1_prior']
                    i += 1
                else:
                    priors[i] = self.priors['eta4_1_prior']
                    priors[i + 1] = self.priors['eta4_1_prior']
                    i += 2

        if self.fix:
            from .utils import Fixed
            priors[self.indices['np']] = Fixed(self.npmax)
        else:
            try:
                priors[self.indices['np']] = self.priors['np_prior']
            except KeyError:
                priors[self.indices['np']] = discrete_uniform(
                    0, self.npmax + 1)

        if self.max_components > 0:
            planet_priors = []
            for i in range(self.max_components):
                planet_priors.append(self.priors['Pprior'])
            for i in range(self.max_components):
                planet_priors.append(self.priors['Kprior'])
            for i in range(self.max_components):
                planet_priors.append(self.priors['phiprior'])
            for i in range(self.max_components):
                planet_priors.append(self.priors['eprior'])
            for i in range(self.max_components):
                planet_priors.append(self.priors['wprior'])
            priors[self.indices['planets']] = planet_priors

        try:
            priors[self.indices['vsys']] = self.priors['Cprior']
        except KeyError:
            priors[self.indices['vsys']] = self.priors['Vprior']
            priors[self.indices['C2']] = self.priors['C2prior']

        return priors

    @property
    def _parameter_priors_full(self):
        from .utils import ZeroDist
        parameter_priors = self.parameter_priors
        for i, p in enumerate(parameter_priors):
            if p is None:
                parameter_priors[i] = ZeroDist()
        return parameter_priors

    @classmethod
    def load(cls, filename: str = None, diagnostic: bool = False, **kwargs):
        """
        Load a KimaResults object from the current directory, a pickle file, or
        a zip file.

        Args:
            filename (str, optional):
                If given, load the model from this file. Can be a zip or pickle
                file. Defaults to None.
            diagnostic (bool, optional):
                Whether to plot the DNest4 diagnotics. Defaults to False.
            **kwargs: Extra keyword arguments passed to `showresults`

        Returns:
            res (KimaResults): An object holding the results
        """
        from .utils import chdir

        if filename is None:
            from .showresults import showresults
            return showresults(force_return=True, **kwargs)

        try:
            if filename.endswith('.zip'):
                zf = zipfile.ZipFile(filename, 'r')
                names = zf.namelist()
                needs = ('sample.txt', 'levels.txt', 'sample_info.txt',
                         'kima_model_setup.txt')
                wants = ('posterior_sample.txt', 'posterior_sample_info.txt')

                for need in needs:
                    if need not in names:
                        raise ValueError('%s does not contain a "%s" file' %
                                         (filename, need))

                with tempfile.TemporaryDirectory() as dirpath:
                    for need in needs:
                        zf.extract(need, path=dirpath)

                    for want in wants:
                        try:
                            zf.extract(need, path=dirpath)
                        except FileNotFoundError:
                            pass

                    try:
                        zf.extract('evidence', path=dirpath)
                        zf.extract('information', path=dirpath)
                    except Exception:
                        pass

                    with chdir(dirpath):
                        setup = read_model_setup()

                        section = 'data' if 'data' in setup else 'kima'
                        try:
                            multi = setup[section]['multi'] == 'true'
                        except KeyError:
                            multi = False

                        if multi:
                            datafiles = setup[section]['files'].split(',')
                            datafiles = list(filter(None, datafiles))
                        else:
                            datafiles = np.atleast_1d(setup['data']['file'])

                        datafiles = list(map(os.path.basename, datafiles))
                        for df in datafiles:
                            zf.extract(df)

                        if os.path.exists(wants[0]):
                            res = cls('')
                            res.evidence = float(open('evidence').read())
                            res.information = float(open('information').read())
                            res.ESS = res.posterior_sample.shape[0]
                        else:
                            from .showresults import showresults
                            res = showresults(verbose=False)

                        # from .classic import postprocess
                        # postprocess()


                return res

            else:
                try:
                    with open(filename, 'rb') as f:
                        return pickle.load(f)
                except UnicodeDecodeError:
                    with open(filename, 'rb') as f:
                        return pickle.load(f, encoding='latin1')

        except Exception:
            # print('Unable to load data from ', filename, ':', e)
            raise

    def show_kima_setup(self):
        return _show_kima_setup()

    def save_pickle(self, filename: str, verbose=True):
        """ Pickle this KimaResults object into a file.

        Args:
            filename (str): The name of the file where to save the model
            verbose (bool, optional): Print a message. Defaults to True.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=2)
        if verbose:
            print('Wrote to file "%s"' % f.name)

    def save_zip(self, filename: str, verbose=True):
        """ Save this KimaResults object and the text files into a zip.

        Args:
            filename (str): The name of the file where to save the model
            verbose (bool, optional): Print a message. Defaults to True.
        """
        if not filename.endswith('.zip'):
            filename = filename + '.zip'

        zf = zipfile.ZipFile(filename, 'w', compression=zipfile.ZIP_DEFLATED)
        tosave = ('sample.txt', 'sample_info.txt', 'levels.txt',
                  'sampler_state.txt', 'posterior_sample.txt',
                  'posterior_sample_info.txt')
        for f in tosave:
            zf.write(f)

        text = open('kima_model_setup.txt').read()
        for f in self.data_file:
            text = text.replace(f, os.path.basename(f))
        zf.writestr('kima_model_setup.txt', text)

        try:
            zf.writestr('evidence', str(self.evidence))
            zf.writestr('information', str(self.information))
        except AttributeError:
            pass

        for f in np.atleast_1d(self.data_file):
            zf.write(f, arcname=os.path.basename(f))

        zf.close()
        if verbose:
            print('Wrote to file "%s"' % zf.filename)

    def get_marginals(self):
        """
        Get the marginal posteriors from the posterior_sample matrix.
        They go into self.T, self.A, self.E, etc
        """
        max_components = self.max_components
        index_component = self.index_component

        # periods
        i1 = 0 * max_components + index_component + 1
        i2 = 0 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.posteriors.P = self.posterior_sample[:, s]
        self.T = self.posteriors.P

        # amplitudes
        i1 = 1 * max_components + index_component + 1
        i2 = 1 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.posteriors.K = self.posterior_sample[:, s]
        self.A = self.posterior_sample[:, s]

        # phases
        i1 = 2 * max_components + index_component + 1
        i2 = 2 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.posteriors.φ = self.posterior_sample[:, s]
        self.phi = self.posterior_sample[:, s]

        # eccentricities
        i1 = 3 * max_components + index_component + 1
        i2 = 3 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.posteriors.e = self.posterior_sample[:, s]
        self.E = self.posterior_sample[:, s]

        # omegas
        i1 = 4 * max_components + index_component + 1
        i2 = 4 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.posteriors.ω = self.posterior_sample[:, s]
        self.Omega = self.posterior_sample[:, s]

        if self.model == 'BDmodel':
            i1 = 5 * max_components + index_component + 1
            i2 = 5 * max_components + index_component + max_components + 1
            s = np.s_[i1:i2]
            self.posteriors.λ = self.posterior_sample[:, s]
            self.Lambda = self.posterior_sample[:, s]


        # times of periastron
        self.posteriors.Tp = (self.T * self.phi) / (2 * np.pi) + self.M0_epoch
        self.Tp = (self.T * self.phi) / (2. * np.pi) + self.M0_epoch

        # times of inferior conjunction (transit, if the planet transits)
        f = np.pi / 2 - self.Omega
        ee = 2 * np.arctan(
            np.tan(f / 2) * np.sqrt((1 - self.E) / (1 + self.E)))
        Tc = self.Tp + self.T / (2 * np.pi) * (ee - self.E * np.sin(ee))
        self.posteriors.Tc = Tc


        # which = self.T != 0
        # self.T = self.T[which].flatten()
        # self.A = self.A[which].flatten()
        # self.E = self.E[which].flatten()
        # self.phi = self.phi[which].flatten()
        # self.Omega = self.Omega[which].flatten()
        # self.Tp = self.Tp[which].flatten()

    def get_medians(self):
        """ return the median values of all the parameters """
        if self.posterior_sample.shape[0] % 2 == 0:
            print(
                'Median is not a solution because number of samples is even!!')

        self.medians = np.median(self.posterior_sample, axis=0)
        self.means = np.mean(self.posterior_sample, axis=0)
        return self.medians, self.means

    def _select_posterior_samples(self, Np=None, mask=None):
        if mask is None:
            mask = np.ones(self.ESS, dtype=bool)

        if Np is None:
            return self.posterior_sample[mask].copy()
        else:
            mask_Np = self.posterior_sample[:, self.index_component] == Np
            return self.posterior_sample[mask & mask_Np].copy()

    def log_prior(self, sample):
        logp = []
        for p, v in zip(self.parameter_priors, sample):
            if p is None:
                # continue
                logp.append(0.0)
            else:
                try:
                    logp.append(p.logpdf(v))
                except AttributeError:
                    logp.append(p.logpmf(v))

        _np = int(sample[self.indices['np']])
        st = self.indices['planets'].start
        k = 0
        for j in range(self._nd):
            for i in range(_np, self._mc):
                logp.pop(st + i + 3 * j - k)
                k += 1
        # return logp
        return np.sum(logp)

    def log_posterior(self, sample, separate=False):
        logp = self.log_prior(sample)
        index = (self.posterior_sample == sample).sum(axis=1).argmax()
        logl = self.posterior_lnlike[index, 1]
        if separate:
            return logp + logl, logl, logp
        return logp + logl

    def map_sample(self, Np=None, mask=None, printit=True, cache=True):
        from tqdm import tqdm

        if cache and hasattr(self, '_map_sample'):
            map_sample = self._map_sample
        else:
            samples = self._select_posterior_samples(Np, mask)
            logpost = []
            for sample in tqdm(samples):
                logpost.append(self.log_posterior(sample))
            logpost = np.array(logpost)
            ind = logpost.argmax()
            self._map_sample = map_sample = samples[ind]

        logpost, loglike, logprior = self.log_posterior(
            map_sample, separate=True)

        if printit:
            print('Sample with the highest posterior value')
            print(f'(logLike = {loglike:.2f}, logPrior = {logprior:.2f},',
                  end=' ')
            print(f'logPost = {logpost:.2f})')

            if Np is not None:
                print(f'from samples with {Np} Keplerians only')

            msg = '-> might not be representative '\
                  'of the full posterior distribution\n'
            print(msg)

            self.print_sample(map_sample)

        return map_sample

    def maximum_likelihood_sample(self, from_posterior=False, Np=None,
                                  printit=True, mask=None):
        """
        Get the maximum likelihood sample. By default, this is the highest
        likelihood sample found by DNest4. If `from_posterior` is True, this
        returns instead the highest likelihood sample *from those that represent
        the posterior*. The latter may change, due to random choices, between
        different calls to "showresults". If `Np` is given, select only samples
        with that number of planets.
        """
        if self.sample_info is None and not self.lnlike_available:
            print('log-likelihoods are not available! '
                  'maximum_likelihood_sample() doing nothing...')
            return

        if from_posterior:
            if mask is None:
                mask = np.ones(self.ESS, dtype=bool)

            if Np is None:
                ind = np.argmax(self.posterior_lnlike[:, 1])
                maxlike = self.posterior_lnlike[ind, 1]
                pars = self.posterior_sample[ind]
            else:
                mask = self.posterior_sample[:, self.index_component] == Np
                ind = np.argmax(self.posterior_lnlike[mask, 1])
                maxlike = self.posterior_lnlike[mask][ind, 1]
                pars = self.posterior_sample[mask][ind]
        else:
            if mask is None:
                mask = np.ones(self.sample.shape[0], dtype=bool)

            if Np is None:
                ind = np.argmax(self.sample_info[mask, 1])
                maxlike = self.sample_info[mask][ind, 1]
                pars = self.sample[mask][ind]
            else:
                mask = self.sample[:, self.index_component] == Np
                ind = np.argmax(self.sample_info[mask, 1])
                maxlike = self.sample_info[mask][ind, 1]
                pars = self.sample[mask][ind]

        if printit:
            if from_posterior:
                print('Posterior sample with the highest likelihood value',
                      end=' ')
            else:
                print('Sample with the highest likelihood value', end=' ')

            print('(logL = {:.2f})'.format(maxlike))

            if Np is not None:
                print('from samples with %d Keplerians only' % Np)

            msg = '-> might not be representative '\
                  'of the full posterior distribution\n'
            print(msg)

            self.print_sample(pars)

        return pars

    def median_sample(self, Np=None, printit=True):
        """
        Get the median posterior sample. If `Np` is given, select only from
        posterior samples with that number of planets.
        """

        if Np is None:
            pars = np.median(self.posterior_sample, axis=0)
        else:
            mask = self.posterior_sample[:, self.index_component] == Np
            pars = np.median(self.posterior_sample[mask, :], axis=0)

        if printit:
            print('Median posterior sample')
            if Np is not None:
                print('from samples with %d Keplerians only' % Np)
            print(
                '-> might not be representative of the full posterior distribution\n'
            )

            self.print_sample(pars)

        return pars

    def print_sample(self, p, star_mass=1.0, show_a=False, show_m=False,
                     mass_units='mjup', show_Tp=False, squeeze=False):

        if show_a or show_m:
            print('considering stellar mass:', star_mass)
            uncertainty_star_mass = False
            if isinstance(star_mass, tuple) or isinstance(star_mass, list):
                uncertainty_star_mass = True

        if self.multi:
            instruments = self.instruments
            instruments = [os.path.splitext(inst)[0] for inst in instruments]

        print('jitter:')
        if squeeze:
            if self.model == 'RVFWHMmodel':
                inst = instruments + instruments
                data = self.n_instruments * ['RV'
                                             ] + self.n_instruments * ['FWHM']
            else:
                inst = instruments
                data = self.n_instruments * ['']

            for i, jit in enumerate(p[self.indices['jitter']]):
                print(f'  {data[i]:5s} ({inst[i]}): {jit:.2f} m/s')
        else:
            if self.model == 'RVFWHMmodel':
                print(f'{"RV":>10s}', end=': ')
                print(p[self.indices['jitter']][:self.n_instruments])
                print(f'{"FWHM":>10s}', end=': ')
                print(p[self.indices['jitter']][self.n_instruments:])
            else:
                print(p[self.indices['jitter']])

        npl = int(p[self.index_component])
        if npl > 0:
            print('number of planets: ', npl)
            print('orbital parameters: ', end='')

            pars = ['P', 'K', 'M0', 'e', 'ω']
            if self.model == 'BINARIESmodel':
                pars.append('ŵ')

            n = self.n_dimensions

            if squeeze:
                print('\n' + 10 * ' ', end='')
                for i in range(npl):
                    print('%-10s' % ascii_lowercase[1 + i], end='')
                print()
                for par in pars:
                    par = 'φ' if par == 'M0' else par
                    print(f'  {par:2s}', ':    ', end='')
                    try:
                        sample_pars = p[self.indices[f'planets.{par}']]
                        for v in sample_pars:
                            print('%-10f' % v, end='')
                    except KeyError:
                        pass
                    print()
            else:
                if show_a:
                    pars.append('a')
                    n += 1
                if show_m:
                    pars.append('Mp')
                    n += 1

                print((n * ' {:>10s} ').format(*pars))

                for i in range(0, npl):
                    formatter = {'all': lambda v: f'{v:11.5f}'}
                    with np.printoptions(formatter=formatter, linewidth=1000):
                        planet_pars = p[
                            self.indices['planets']][i::self.max_components]
                        P, K, M0, ecc, *ω = planet_pars

                        if show_a or show_m:
                            (m, _), a = get_planet_mass_and_semimajor_axis(
                                P, K, ecc, star_mass)

                            if uncertainty_star_mass:
                                m = m[0]
                                a = a[0]

                        if show_a:
                            planet_pars = np.append(planet_pars, a)
                        if show_m:
                            if mass_units != 'mjup':
                                if mass_units.lower() == 'mearth':
                                    m *= mjup2mearth
                            planet_pars = np.append(planet_pars, m)

                        s = str(planet_pars)
                        s = s.replace('[', '').replace(']', '')
                    s = s.rjust(20 + len(s))
                    print(s)

        if self.KO:
            print('number of known objects: ', self.nKO)
            print('orbital parameters: ', end='')

            pars = ['P', 'K', 'M0', 'e', 'ω']
            if self.model == 'BINARIESmodel':
                pars.append('ŵ')

            print((self.n_dimensions * ' {:>10s} ').format(*pars))

            for i in range(0, self.nKO):
                formatter = {'all': lambda v: f'{v:11.5f}'}
                with np.printoptions(formatter=formatter):
                    s = str(p[self.indices['KOpars']][i::self.nKO])
                    s = s.replace('[', '').replace(']', '')
                s = s.rjust(20 + len(s))
                print(s)

        if self.GPmodel:
            print('GP parameters: ', end='')
            if self.model == 'RVFWHMmodel':
                pars = ('η1 RV', 'η1 FWHM', 'η2', 'η3', 'η4')
            else:
                pars = ('η1', 'η2', 'η3', 'η4')

            if squeeze:
                print()
                values = p[self.indices['GPpars']]
                for par, v in zip(pars, values):
                    print(f'  {par:8s}:', v)
            else:
                print((len(pars) * ' {:>10s} ').format(*pars))
                formatter = {'all': lambda v: f'{v:11.5f}'}
                with np.printoptions(formatter=formatter):
                    s = str(p[self.indices['GPpars']])
                    s = s.replace('[', '').replace(']', '')
                s = s.rjust(15 + len(s))
                print(s)

            # , *p[self.indices['GPpars']])
            # if self.GPkernel in (0, 2, 3):
            #     eta1, eta2, eta3, eta4 = pars[self.indices['GPpars']]
            #     print('GP parameters: ', eta1, eta2, eta3, eta4)
            # elif self.GPkernel == 1:
            #     eta1, eta2, eta3 = pars[self.indices['GPpars']]
            #     print('GP parameters: ', eta1, eta2, eta3)

        if self.trend:
            names = ('slope', 'quadr', 'cubic')
            units = ['m/s/yr', 'm/s/yr²', 'm/s/yr³']
            trend = p[self.indices['trend']].copy()
            # transfrom from /day to /yr
            trend *= 365.25**np.arange(1, self.trend_degree + 1)

            for name, unit, trend_par in zip(names, units, trend):
                print(name + ':', '%-8.5f' % trend_par, unit)

        if self.multi:
            ni = self.n_instruments - 1
            print('instrument offsets: ', end=' ')
            # print('(relative to %s) ' % self.data_file[-1])
            print('(relative to %s) ' % instruments[-1])
            s = 20 * ' '
            s += (ni * ' {:20s} ').format(*instruments)
            print(s)

            i = self.indices['inst_offsets']
            s = 20 * ' '
            s += (ni * ' {:<20.3f} ').format(*p[i])
            print(s)

        print('vsys: ', p[-1])

    def _sort_planets_by_amplitude(self, sample, decreasing=True):
        new_sample = sample.copy()
        ind = np.argsort(new_sample[self.indices['planets.K']])
        if decreasing:
            ind = ind[::-1]
        nd = self.n_dimensions
        mc = self.max_components
        pars = new_sample[self.indices['planets']]
        for i in range(nd):
            pars[i * mc:(i + 1) * mc] = pars[i * mc:(i + 1) * mc][ind]
        return new_sample

    def _get_tt(self, N=1000, over=0.1):
        """
        Create array for model prediction plots. This simply returns N
        linearly-spaced times from t[0]-over*Tspan to t[-1]+over*Tspan.
        """
        start = self.data.t.min() - over * self.data.t.ptp()
        end = self.data.t.max() + over * self.data.t.ptp()
        return np.linspace(start, end, N)

    def _get_ttGP(self, N=1000, over=0.1):
        """ Create array of times for GP prediction plots. """
        kde = gaussian_kde(self.data.t)
        ttGP = kde.resample(N - self.data.N).reshape(-1)
        # constrain ttGP within observed times, to not waste
        ttGP = (ttGP + self.data.t[0]) % self.data.t.ptp() + self.data.t[0]
        # add the observed times as well
        ttGP = np.r_[ttGP, self.data.t]
        ttGP.sort()  # in-place
        return ttGP

    def eval_model(self, sample, t = None,
                   include_planets=True, include_known_object=True,
                   include_trend=True, single_planet: int = None,
                   except_planet: Union[int, List] = None):
        """
        Evaluate the deterministic part of the model at one posterior `sample`.
        If `t` is None, use the observed times. Instrument offsets are only
        added if `t` is None, but the systemic velocity is always added.
        To evaluate at all posterior samples, consider using
            np.apply_along_axis(self.eval_model, 1, self.posterior_sample)

        Note: this function does *not* evaluate the GP component of the model.

        Arguments:
            sample (array): One posterior sample, with shape (npar,)
            t (array):
                Times at which to evaluate the model, or None to use observed
                times
            include_planets (bool):
                Whether to include the contribution from the planets
            include_known_object (bool):
                Whether to include the contribution from the known object
                planets
            include_trend (bool):
                Whether to include the contribution from the trend
            single_planet (int):
                Index of a single planet to *include* in the model, starting at
                1. Use positive values (1, 2, ...) for the Np planets and
                negative values (-1, -2, ...) for the known object planets.
            except_planet (Union[int, List]):
                Index (or list of indices) of a single planet to *exclude* from
                the model, starting at 1. Use positive values (1, 2, ...) for
                the Np planets and negative values (-1, -2, ...) for the known
                object planets.
        """
        if sample.shape[0] != self.posterior_sample.shape[1]:
            n1 = sample.shape[0]
            n2 = self.posterior_sample.shape[1]
            msg = '`sample` has wrong dimensions, expected %d got %d' % (n2,
                                                                         n1)
            raise ValueError(msg)

        data_t = False
        if t is None or t is self.data.t:
            t = self.data.t.copy()
            data_t = True

        if self.model == 'RVFWHMmodel':
            v = np.zeros((2, t.size))
        else:
            v = np.zeros_like(t)

        if include_planets:
            if single_planet and except_planet:
                raise ValueError("'single_planet' and 'except_planet' "
                                 "cannot be used together")

            # except_planet should be a list to exclude more than one planet
            if except_planet is not None:
                if isinstance(except_planet, int):
                    except_planet = [except_planet]

            # known_object ?
            if self.KO and include_known_object:
                pars = sample[self.indices['KOpars']].copy()
                for j in range(self.nKO):
                    if single_planet is not None:
                        if j + 1 != -single_planet:
                            continue
                    if except_planet is not None:
                        if j + 1 in except_planet:
                            continue

                    P = pars[j + 0 * self.nKO]
                    K = pars[j + 1 * self.nKO]
                    phi = pars[j + 2 * self.nKO]
                    # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
                    ecc = pars[j + 3 * self.nKO]
                    w = pars[j + 4 * self.nKO]
                    if self.model == 'BINARIESmodel':
                        wdot = pars[j + 5 * self.nKO]
                        tp = self.M0_epoch - (P * phi) / (2 * np.pi)
                        v += keplerianpy(t, P, K, ecc, w, wdot, tp, 0.0)
                    else:
                        v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

            # get the planet parameters for this sample
            pars = sample[self.indices['planets']].copy()

            # how many planets in this sample?
            # nplanets = pars.size / self.n_dimensions
            nplanets = (pars[:self.max_components] != 0).sum()

            # add the Keplerians for each of the planets
            for j in range(int(nplanets)):

                if single_planet is not None:
                    if j + 1 != single_planet:
                        continue
                if except_planet is not None:
                    if j + 1 in except_planet:
                        continue

                P = pars[j + 0 * self.max_components]
                if P == 0.0:
                    continue
                K = pars[j + 1 * self.max_components]
                phi = pars[j + 2 * self.max_components]
                # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
                ecc = pars[j + 3 * self.max_components]
                w = pars[j + 4 * self.max_components]
                # print(P, K, ecc, w, phi, self.M0_epoch)
                if self.model == 'RVFWHMmodel':
                    v[0, :] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                else:
                    v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

        # systemic velocity (and C2) for this sample
        if self.model == 'RVFWHMmodel':
            C = np.c_[sample[self.indices['vsys']], sample[self.indices['C2']]]
            v += C.reshape(-1, 1)
        else:
            v += sample[self.indices['vsys']]

        # if evaluating at the same times as the data, add instrument offsets
        # otherwise, don't
        if self.multi and data_t:  # and len(self.data_file) > 1:
            offsets = sample[self.indices['inst_offsets']]
            ii = self.data.obs.astype(int) - 1

            if self.model == 'RVFWHMmodel':
                ni = self.n_instruments
                offsets = np.pad(offsets.reshape(-1, ni - 1), ((0, 0), (0, 1)))
                v += np.take(offsets, ii, axis=1)
            else:
                offsets = np.pad(offsets, (0, 1))
                v += np.take(offsets, ii)

        # add the trend, if present
        if include_trend and self.trend:
            trend_par = sample[self.indices['trend']]
            # polyval wants coefficients in reverse order, and vsys was already
            # added so the last coefficient is 0
            trend_par = np.r_[trend_par[::-1], 0.0]
            if self.model == 'RVFWHMmodel':
                v[0, :] += np.polyval(trend_par, t - self.tmiddle)
            else:
                v += np.polyval(trend_par, t - self.tmiddle)

        return v

    def planet_model(self, sample: np.ndarray, t: np.ndarray = None,
                     include_known_object=True, single_planet: int = None,
                     except_planet: Union[int, List] = None):
        """
        Evaluate the planet part of the model at one posterior `sample`. If `t`
        is None, use the observed times. To evaluate at all posterior samples,
        consider using

            np.apply_along_axis(self.planet_model, 1, self.posterior_sample)

        Note:
            this function does *not* evaluate the GP component of the model
            nor the systemic velocity and instrument offsets.

        Arguments:
            sample (ndarray):
                One posterior sample, with shape (npar,)
            t (ndarray):
                Times at which to evaluate the model, or None to use observed
                times
            include_known_object (bool):
                Whether to include the contribution from the known object
                planets
            single_planet (int):
                Index of a single planet to *include* in the model, starting at
                1. Use positive values (1, 2, ...) for the Np planets and
                negative values (-1, -2, ...) for the known object planets.
            except_planet (Union[int, List]):
                Index (or list of indices) of a single planet to *exclude* from
                the model, starting at 1. Use positive values (1, 2, ...) for
                the Np planets and negative values (-1, -2, ...) for the known
                object planets.
        """
        if sample.shape[0] != self.posterior_sample.shape[1]:
            n1 = sample.shape[0]
            n2 = self.posterior_sample.shape[1]
            msg = '`sample` has wrong dimensions, expected %d got %d' % (n2,
                                                                         n1)
            raise ValueError(msg)

        if t is None or t is self.data.t:
            t = self.data.t.copy()

        if self.model == 'RVFWHMmodel':
            v = np.zeros((2, t.size))
        else:
            v = np.zeros_like(t)

        if single_planet and except_planet:
            raise ValueError("'single_planet' and 'except_planet' "
                             "cannot be used together")

        # except_planet should be a list to exclude more than one planet
        if except_planet is not None:
            except_planet = np.atleast_1d(except_planet)

        # known_object ?
        if self.KO and include_known_object:
            pars = sample[self.indices['KOpars']].copy()
            for j in range(self.nKO):
                if single_planet is not None:
                    if j + 1 != -single_planet:
                        continue
                if except_planet is not None:
                    if j + 1 in -except_planet:
                        continue

                P = pars[j + 0 * self.nKO]
                K = pars[j + 1 * self.nKO]
                phi = pars[j + 2 * self.nKO]
                # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
                ecc = pars[j + 3 * self.nKO]
                w = pars[j + 4 * self.nKO]
                v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

        # get the planet parameters for this sample
        pars = sample[self.indices['planets']].copy()

        # how many planets in this sample?
        # nplanets = pars.size / self.n_dimensions
        nplanets = (pars[:self.max_components] != 0).sum()

        # add the Keplerians for each of the planets
        for j in range(int(nplanets)):

            if single_planet is not None:
                if j + 1 != single_planet:
                    continue
            if except_planet is not None:
                if j + 1 in except_planet:
                    continue

            P = pars[j + 0 * self.max_components]
            if P == 0.0:
                continue
            K = pars[j + 1 * self.max_components]
            phi = pars[j + 2 * self.max_components]
            # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
            ecc = pars[j + 3 * self.max_components]
            w = pars[j + 4 * self.max_components]
            if self.model == 'RVFWHMmodel':
                v[0, :] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
            else:
                v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

        return v

    def stochastic_model(self, sample, t=None, return_std=False,
                         derivative=False, **kwargs):
        """
        Evaluate the stochastic part of the model (GP) at one posterior sample.
        If `t` is None, use the observed times. Instrument offsets are only
        added if `t` is None, but the systemic velocity is always added.
        To evaluate at all posterior samples, consider using

            np.apply_along_axis(self.stochastic_model, 1, self.posterior_sample)

        Arguments:
            sample (ndarray):
                One posterior sample, with shape (npar,)
            t (ndarray, optional):
                Times at which to evaluate the model, or None to use observed
                times
            return_std (bool, optional):
                Whether to return the standard deviation of the predictive.
                Default is False.
            derivative (bool, optional):
                Return the first time derivative of the GP prediction instead
        """

        if sample.shape[0] != self.posterior_sample.shape[1]:
            n1 = sample.shape[0]
            n2 = self.posterior_sample.shape[1]
            msg = '`sample` has wrong dimensions, '\
                  'should be %d got %d' % (n2, n1)
            raise ValueError(msg)

        if t is None or t is self.data.t:
            t = self.data.t.copy()

        if not self.GPmodel:
            return np.zeros_like(t)

        if self.model == 'RVFWHMmodel':
            D = np.vstack((self.data.y, self.data.y2))
            r = D - self.eval_model(sample)
            GPpars = sample[self.indices['GPpars']]

            if self.GPkernel == 'standard':
                (η1RV, η1FWHM, η2RV, η2FWHM, η3RV, η3FWHM, η4RV,
                 η4FWHM) = GPpars[self._GP_par_indices]
                self.GP1.kernel.pars = np.array([η1RV, η2RV, η3RV, η4RV])
                self.GP2.kernel.pars = np.array(
                    [η1FWHM, η2FWHM, η3FWHM, η4FWHM])

            elif self.GPkernel == 'qpc':
                (η1RV, η1FWHM, η2RV, η2FWHM, η3RV, η3FWHM, η4RV, η4FWHM, η5RV,
                 η5FWHM) = GPpars[self._GP_par_indices]

                self.GP1.kernel.pars = np.array([η1RV, η2RV, η3RV, η4RV, η5RV])
                self.GP2.kernel.pars = np.array(
                    [η1FWHM, η2FWHM, η3FWHM, η4FWHM, η5FWHM])

            elif self.GPkernel == 'qp_plus_cos':
                (η1RV, η1FWHM, η2RV, η2FWHM, η3RV, η3FWHM, η4RV, η4FWHM, η5RV,
                 η5FWHM, η6RV, η6FWHM) = GPpars[self._GP_par_indices]

                self.GP1.kernel.pars = np.array(
                    [η1RV, η2RV, η3RV, η4RV, η5RV, η6RV])
                self.GP2.kernel.pars = np.array(
                    [η1FWHM, η2FWHM, η3FWHM, η4FWHM, η5FWHM, η6FWHM])


            if derivative:
                out0 = self.GP1.derivative(r[0], t, return_std=return_std)
                out1 = self.GP2.derivative(r[1], t, return_std=return_std)
            else:
                out0 = self.GP1.predict(r[0], t, return_std=return_std)
                out1 = self.GP2.predict(r[1], t, return_std=return_std)

            if return_std:
                return (
                    np.vstack([out0[0], out1[0]]),
                    np.vstack([out0[1], out1[1]])
                )
            else:
                return np.vstack([out0, out1])

        else:
            r = self.data.y - self.eval_model(sample)
            if self.model == 'GPmodel_systematics':
                x = self._extra_data[:, 3]
                X = np.c_[t, interp1d(self.data.t, x, bounds_error=False)(t)]
                GPpars = sample[self.indices['GPpars']]
                mu = self.GP.predict(r, X, GPpars)
                # print(GPpars)
                # self.GP.kernel.pars = GPpars
                return mu
            else:
                GPpars = sample[self.indices['GPpars']]
                self.GP.kernel.pars = GPpars
                return self.GP.predict(r, t, return_std=return_std)

    def full_model(self, sample: np.ndarray, t: np.ndarray = None, **kwargs):
        """
        Evaluate the full model at one posterior sample, including the GP. If
        `t` is `None`, use the observed times. Instrument offsets are only added
        if `t` is `None`, but the systemic velocity is always added. To evaluate
        at all posterior samples, consider using
        
            np.apply_along_axis(self.full_model, 1, self.posterior_sample)

        Arguments:
            sample (ndarray): One posterior sample, with shape (npar,)
            t (ndarray, optional):
                Times at which to evaluate the model, or `None` to use observed
                times
            **kwargs: Keyword arguments passed directly to `eval_model`
        """
        deterministic = self.eval_model(sample, t, **kwargs)
        stochastic = self.stochastic_model(sample, t)
        return deterministic + stochastic

    def burst_model(self, sample, t=None, v=None):
        """
        For models with multiple instruments, this function "bursts" the
        computed RV into `n_instruments` individual arrays. This is mostly
        useful for plotting the RV model together with the original data.

        Arguments:
            sample (ndarray): One posterior sample, with shape (npar,)
            t (ndarray): Times at which to evaluate the model
            v (ndarray): Pre-computed RVs. If `None`, calls `self.eval_model`
        """
        if v is None:
            v = self.eval_model(sample, t)
        if t is None:
            t = self.data.t.copy()

        if not self.multi:
            # print('not multi_instrument, burst_model adds nothing')
            return v

        ni = self.n_instruments
        offsets = sample[self.indices['inst_offsets']]
        if self.model == 'RVFWHMmodel':
            offsets = np.pad(offsets.reshape(-1, ni - 1), ((0, 0), (0, 1)))
        else:
            offsets = np.pad(offsets, (0, 1))

        if self._time_overlaps[0]:
            v = np.tile(v, (self.n_instruments, 1))
            if self.model == 'RVFWHMmodel':
                offsets = np.insert(offsets[0],
                                    np.arange(1, offsets.shape[1] + 1),
                                    offsets[1])
                v = (v.T + offsets).T
                # this constrains the RV to the times of each instrument
                for i in range(self.n_instruments):
                    obst = self.data.t[self.data.obs == i + 1]
                    # RV
                    v[2 * i, t < obst.min()] = np.nan
                    v[2 * i, t > obst.max()] = np.nan
                    # FWHM
                    v[2 * i + 1, t < obst.min()] = np.nan
                    v[2 * i + 1, t > obst.max()] = np.nan
            else:
                v = (v.T + offsets).T
                # this constrains the RV to the times of each instrument
                for i in range(self.n_instruments):
                    obst = self.data.t[self.data.obs == i + 1]
                if i == 0:
                    v[i, t < obst.min()] = np.nan
                if i < self.n_instruments - 1:
                    v[i, t > obst.max()] = np.nan

        else:
            time_bins = np.sort(np.r_[t[0], self._offset_times])
            ii = np.digitize(t, time_bins) - 1

            #! HACK!
            obs_is_sorted = np.all(np.diff(self.data.obs) >= 0)
            if not obs_is_sorted:
                ii = -ii.max() * (ii - ii.max())
            #! end HACK!

            if self.model == 'RVFWHMmodel':
                v += np.take(offsets, ii, axis=1)
            else:
                v += np.take(offsets, ii)

        return v

    def residuals(self, sample, full=False):
        if self.model == 'RVFWHMmodel':
            D = np.vstack([self.data.y, self.data.y2])
        else:
            D = self.data.y

        if full:
            return D - self.full_model(sample)
        else:
            return D - self.eval_model(sample)

    def residual_rms(self, sample, weighted=True, printit=True):
        r = self.residuals(sample, full=True)
        if self.model == 'RVFWHMmodel':
            r = r[0]

        vals = []
        val = wrms(r, weights=1 / self.e**2)
        if printit:
            print(f'full: {val:.3f} m/s')
        vals.append(val)

        if self.multi:
            for inst, o in zip(self.instruments, np.unique(self.data.obs)):
                val = wrms(r[self.data.obs == o],
                           weights=1 / self.e[self.data.obs == o]**2)
                if printit:
                    print(f'{inst}: {val:.3f} m/s')
                vals.append(val)

        return np.array(vals)

    def from_prior(self, n=1):
        """ Generate `n` samples from the priors for all parameters. """
        prior_samples = []
        for i in range(n):
            prior = []
            for p in self.parameter_priors:
                if p is None:
                    prior.append(None)
                else:
                    prior.append(p.rvs())
            prior_samples.append(np.array(prior))
        return np.array(prior_samples)

    def simulate_from_sample(self, sample, times, add_noise=True, errors=True,
                             append_to_file=False):
        y = self.full_model(sample, times)
        e = np.zeros_like(y)

        if add_noise:
            if self.model == 'RVFWHMmodel':
                n1 = np.random.normal(0, self.e.mean(), times.size)
                n2 = np.random.normal(0, self.e2.mean(), times.size)
                y += np.c_[n1, n2].T
            elif self.model == 'RVmodel':
                n = np.random.normal(0, self.e.mean(), times.size)
                y += n

        if errors:
            if self.model == 'RVFWHMmodel':
                er1 = np.random.uniform(self.e.min(), self.e.max(), times.size)
                er2 = np.random.uniform(self.e2.min(), self.e2.max(),
                                        times.size)
                e += np.c_[er1, er2].T

            elif self.model == 'RVmodel':
                er = np.random.uniform(self.e.min(), self.e.max(), times.size)
                e += er

        if append_to_file:
            last_file = self.data_file[-1]
            name, ext = os.path.splitext(last_file)
            n = times.size
            file = f'{name}_+{n}sim{ext}'
            print(file)

            with open(file, 'w') as out:
                out.writelines(open(last_file).readlines())
                if self.model == 'RVFWHMmodel':
                    kw = dict(delimiter='\t', fmt=['%.5f'] + 4 * ['%.9f'])
                    np.savetxt(out, np.c_[times, y[0], e[0], y[1], e[1]], **kw)
                elif self.model == 'RVmodel':
                    kw = dict(delimiter='\t', fmt=['%.5f'] + 2 * ['%.9f'])
                    np.savetxt(out, np.c_[times, y, e], **kw)

        if errors:
            return y, e
        else:
            return y

    @property
    def star(self):
        if self.multi:
            return get_star_name(self.data_file[0])
        else:
            return get_star_name(self.data_file)

    @property
    def instruments(self):
        if self.multi:
            if self.multi_onefile:
                return ['inst %d' % i for i in np.unique(self.data.obs)]
            else:
                return list(map(get_instrument_name, self.data_file))
        else:
            return get_instrument_name(self.data_file)

    @property
    def Np(self):
        return self.posterior_sample[:, self.index_component]

    @property
    def ratios(self):
        bins = np.arange(self.max_components + 2)
        n, _ = np.histogram(self.Np, bins=bins)
        n = n.astype(np.float)
        n[n == 0] = np.nan
        r = n.flat[1:] / n.flat[:-1]
        r[np.isnan(r)] = np.inf
        return r

    @property
    def _error_ratios(self):
        # self if a KimaResults instance
        from scipy.stats import multinomial
        bins = np.arange(self.max_components + 2)
        n, _ = np.histogram(self.Np, bins=bins)
        prob = n / self.ESS
        r = multinomial(self.ESS, prob).rvs(10000)
        r = r.astype(np.float)
        r[r == 0] = np.nan
        return (r[:, 1:] / r[:, :-1]).std(axis=0)

    @property
    def _time_overlaps(self):
        # check for overlaps in the time from different instruments
        if not self.multi:
            raise ValueError('Model is not multi_instrument')

        def minmax(x):
            return x.min(), x.max()

        # are the instrument identifiers all sorted?
        # st = np.lexsort(np.vstack([self.t, self.data.obs]))
        obs_is_sorted = np.all(np.diff(self.data.obs) >= 0)

        # if not, which ones are not sorted?
        if not obs_is_sorted:
            which_not_sorted = np.unique(
                self.data.obs[1:][np.diff(self.data.obs) < 0])

        overlap = []
        for i in range(1, self.n_instruments):
            t1min, t1max = minmax(self.data.t[self.data.obs == i])
            t2min, t2max = minmax(self.data.t[self.data.obs == i + 1])
            # if the instrument IDs are sorted or these two instruments
            # (i and i+1) are not the ones not-sorted
            if obs_is_sorted or i not in which_not_sorted:
                if t2min < t1max:
                    overlap.append((i, i + 1))
            # otherwise the check is different
            else:
                if t1min < t2max:
                    overlap.append((i, i + 1))

        return len(overlap) > 0, overlap

    @property
    def _offset_times(self):
        if not self.multi:
            raise ValueError('Model is not multi_instrument, no offset times')

        # check for overlaps
        has_overlaps, overlap = self._time_overlaps
        if has_overlaps:
            _o = []
            m = np.full_like(self.data.obs, True, dtype=bool)
            for ov in overlap:
                _o.append(self.data.t[self.data.obs == ov[0]].max())
                _o.append(self.data.t[self.data.obs == ov[1]].min())
                m &= self.data.obs != ov[0]

            _1 = self.data.t[m][np.ediff1d(self.data.obs[m], 0, None) != 0]
            _2 = self.data.t[m][np.ediff1d(self.data.obs[m], None, 0) != 0]
            return np.sort(np.r_[_o, np.mean((_1, _2), axis=0)])

        # otherwise it's much easier
        else:
            _1 = self.data.t[np.ediff1d(self.data.obs, 0, None) != 0]
            _2 = self.data.t[np.ediff1d(self.data.obs, None, 0) != 0]
            return np.mean((_1, _2), axis=0)

    @property
    def data_properties(self):
        t = self.data.t
        prop = {
            'time span': (t.ptp(), 'days', True),
            'mean time gap': (np.ediff1d(t).mean(), 'days', True),
            'median time gap': (np.median(np.ediff1d(t)), 'days', True),
            'shortest time gap': (np.ediff1d(t).min(), 'days', True),
            'longest time gap': (np.ediff1d(t).max(), 'days', True),
        }
        width = max(list(map(len, prop.keys()))) + 2
        for k, v in prop.items():
            print(f'{k:{width}s}: {v[0]:10.6f}  {v[1]}', end=' ')
            if v[2]:
                print(f'({1/v[0]:10.6f} {v[1]}⁻¹)')

    @property
    def eta1(self):
        if self.GPmodel:
            return self.posterior_sample[:, self.indices['GPpars']][:, 0]
        return None

    @property
    def eta2(self):
        if self.GPmodel:
            i = 2 if self.model == 'RVFWHMmodel' else 1
            return self.posterior_sample[:, self.indices['GPpars']][:, i]
        return None

    @property
    def eta3(self):
        if self.GPmodel:
            i = 3 if self.model == 'RVFWHMmodel' else 2
            return self.posterior_sample[:, self.indices['GPpars']][:, i]
        return None

    @property
    def eta4(self):
        if self.GPmodel:
            i = 4 if self.model == 'RVFWHMmodel' else 3
            return self.posterior_sample[:, self.indices['GPpars']][:, i]
        return None

    # most of the following methods just dispatch to display
    make_plots = display.make_plots

    #
    phase_plot = display.phase_plot

    #
    make_plot1 = display.make_plot1
    plot1 = display.make_plot1
    plot_posterior_np = display.make_plot1

    #
    make_plot2 = display.make_plot2
    plot2 = display.make_plot2

    #
    make_plot3 = display.make_plot3
    plot3 = display.make_plot3

    #
    make_plot4 = display.make_plot4
    plot4 = display.make_plot4

    #
    make_plot5 = display.make_plot5
    plot5 = display.make_plot5

    def get_sorted_planet_samples(self, full=True):
        # all posterior samples for the planet parameters
        # this array is nsamples x (n_dimensions*max_components)
        # that is, nsamples x 5, nsamples x 10, for 1 and 2 planets for example
        if full:
            samples = self.posterior_sample.copy()
        else:
            samples = self.posterior_sample[:, self.indices['planets']].copy()

        if self.max_components == 0:
            return samples

        # here we sort the samples array by the orbital period
        # this is a bit difficult because the organization of the array is
        # P1 P2 K1 K2 ....
        sorted_samples = samples.copy()
        n = self.max_components * self.n_dimensions
        mc = self.max_components
        p = samples[:, self.indices['planets.P']]
        ind_sort_P = np.arange(p.shape[0])[:, np.newaxis], np.argsort(p)

        for par in ('P', 'K', 'φ', 'e', 'ω'):
            sorted_samples[:, self.indices[f'planets.{par}']] = \
                samples[:, self.indices[f'planets.{par}']][ind_sort_P]

        return sorted_samples

    def _apply_cuts_period(self, pmin=None, pmax=None, return_mask=False):
        """ apply cuts in orbital period """
        if pmin is None and pmax is None:
            if return_mask:
                return np.ones(self.ESS, dtype=bool)
            else:
                return self.posterior_sample

        periods = self.posterior_sample[:, self.indices['planets.P']]

        if pmin is None:
            mask_min = np.ones(self.ESS, dtype=bool)
        else:
            mask_min = np.logical_and.reduce((periods > pmin).T)
        if pmax is None:
            mask_max = np.ones(self.ESS, dtype=bool)
        else:
            mask_max = np.logical_and.reduce((periods < pmax).T)

        if return_mask:
            return mask_min & mask_max
        else:
            return self.posterior_sample[mask_min & mask_max]

    #
    def plot_random_samples(self, ncurves=50, samples=None, over=0.1,
                            ntt=5000, pmin=None, pmax=None, show_vsys=False,
                            isolate_known_object=True, full_plot=False,
                            ignore_outliers=False, **kwargs):
        import inspect
        args = inspect.getcallargs(self.plot_random_samples)
        args['res'] = args['self']
        if self.model == 'RVFWHMmodel':
            return display.plot_random_samples_rvfwhm(**args)
        else:
            return display.plot_random_samples(**args)
        # print(locals)
        # plot_random_samples = _fun[model]
    # plot6 = display.plot_random_samples

    #
    hist_vsys = display.hist_vsys
    hist_jitter = display.hist_jitter
    hist_correlations = display.hist_correlations
    hist_trend = display.hist_trend
    hist_MA = display.hist_MA
    hist_nu = display.hist_nu
