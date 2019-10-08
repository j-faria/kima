from __future__ import print_function

import re, os, sys
pathjoin = os.path.join
from copy import copy
from string import ascii_lowercase

from math import ceil
import pickle
import zipfile
import tempfile
try:
    import configparser
except ImportError:
    # Python 2
    import ConfigParser as configparser

from .keplerian import keplerian
from .GP import GP, QPkernel
from .utils import need_model_setup, get_planet_mass, get_planet_semimajor_axis,\
                   percentile68_ranges, percentile68_ranges_latex,\
                   read_datafile, lighten_color, wrms, get_prior, \
                   hyperprior_samples, get_star_name, get_instrument_name

from .analysis import passes_threshold_np

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.stats import gaussian_kde
try: # only available in scipy 1.1.0
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None
from scipy.stats._continuous_distns import reciprocal_gen
import corner

try:
    from astroML.plotting import hist_tools
    hist_tools_available = True
except ImportError:
    hist_tools_available = False

colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]


class KimaResults(object):
    """ A class to hold, analyse, and display the results from kima """

    def __init__(self, options, data_file=None, save_plots=False,
                 verbose=False, fiber_offset=None, hyperpriors=None,
                 trend=None, GPmodel=None,
                 posterior_samples_file='posterior_sample.txt'):

        self.options = options
        debug = False  # 'debug' in options
        self.save_plots = save_plots
        self.verbose = verbose

        self.removed_crossing = False
        self.removed_roche_crossing = False

        pwd = os.getcwd()
        path_to_this_file = os.path.abspath(__file__)
        top_level = os.path.dirname(os.path.dirname(path_to_this_file))

        if debug:
            print()
            print('running on:', pwd)
            print('top_level:', top_level)
            print()

        setup = configparser.ConfigParser()
        setup.optionxform = str

        try:
            open('kima_model_setup.txt')
        except IOError as exc:
            need_model_setup(exc)

        setup.read('kima_model_setup.txt')

        if sys.version_info < (3, 0):
            setup = setup._sections
            # because we cheated, we need to cheat a bit more...
            setup['kima']['obs_after_HARPS_fibers'] = setup['kima'].pop(
                'obs_after_harps_fibers')
            setup['kima']['GP'] = setup['kima'].pop('gp')

        self.setup = setup

        # read the priors
        priors = list(setup['priors.general'].values())
        prior_names = list(setup['priors.general'].keys())
        for section in ('priors.planets', 'priors.hyperpriors', 'priors.GP'):
            try:
                priors += list(setup[section].values())
                prior_names += list(setup[section].keys())
            except KeyError:
                continue

        try:
            priors += list(setup['priors.known_object'].values())
            prior_names += [
                'KO_' + k for k in setup['priors.known_object'].keys()
            ]
        except KeyError:
            pass

        self.priors = {
            n: v
            for n, v in zip(prior_names, [get_prior(p) for p in priors])
        }

        if data_file is None:
            self.multi = setup['kima']['multi'] == 'true'
            if self.multi:
                if setup['kima']['files'] == '':
                    # multi is true but in only one file
                    data_file = setup['kima']['file']
                else:
                    data_file = setup['kima']['files'].split(',')[:-1]
                    # raise NotImplementedError('TO DO')
            else:
                data_file = setup['kima']['file']

        self.data_skip = int(setup['kima']['skip'])
        self.units = setup['kima']['units']

        if verbose:
            print('Loading data file %s' % data_file)
        self.data_file = data_file

        self.data_skip = int(setup['kima']['skip'])
        self.units = setup['kima']['units']

        if debug:
            print('--- skipping first %d rows of data file' % self.data_skip)

        if self.multi:
            self.data, self.obs = read_datafile(self.data_file, self.data_skip)
            # make sure the times are sorted when coming from multiple instruments
            ind = self.data[:, 0].argsort()
            self.data = self.data[ind]
            self.obs = self.obs[ind]
            self.n_instruments = np.unique(self.obs).size
            self.n_jitters = self.n_instruments
        else:
            self.data = np.loadtxt(self.data_file, skiprows=self.data_skip,
                                   usecols=(0, 1, 2))
            self.n_jitters = 1

        # to m/s
        if self.units == 'kms':
            self.data[:, 1] *= 1e3
            self.data[:, 2] *= 1e3

        self.t = self.data[:, 0].copy()
        self.y = self.data[:, 1].copy()
        self.e = self.data[:, 2].copy()
        self.tmiddle = self.data[:, 0].min() + 0.5 * self.data[:, 0].ptp()

        self.posterior_sample = np.atleast_2d(
            np.loadtxt(posterior_samples_file))

        try:
            self.posterior_lnlike = np.atleast_2d(
                np.loadtxt('posterior_sample_info.txt'))
            self.lnlike_available = True
        except IOError:
            self.lnlike_available = False
            print('Could not find file "posterior_sample_info.txt", '\
                  'log-likelihoods will not be available.')

        try:
            self.sample = np.loadtxt('sample.txt')
        except IOError:
            self.sample = None

        self.indices = {}

        start_parameters = 0
        if self.multi:
            i1, i2 = start_parameters, start_parameters + self.n_jitters
            self.extra_sigma = self.posterior_sample[:, i1:i2]
            start_parameters += self.n_jitters - 1
            self.indices['jitter_start'] = i1
            self.indices['jitter_end'] = i2
            self.indices['jitter'] = slice(i1, i2)
        else:
            self.extra_sigma = self.posterior_sample[:, start_parameters]
            self.indices['jitter'] = start_parameters

        # find trend in the compiled model
        if trend is None:
            self.trend = setup['kima']['trend'] == 'true'
        else:
            self.trend = trend

        if debug: print('trend:', self.trend)

        if self.trend:
            n_trend = 1
            i1 = start_parameters + 1
            i2 = start_parameters + n_trend + 1
            self.trendpars = self.posterior_sample[:, i1:i2]
            self.indices['trend'] = i1
        else:
            n_trend = 0

        # find fiber offset in the compiled model
        if fiber_offset is None:
            self.fiber_offset = setup['kima'][
                'obs_after_HARPS_fibers'] == 'true'
        else:
            self.fiber_offset = fiber_offset

        if debug: print('obs_after_fibers:', self.fiber_offset)

        if self.fiber_offset:
            n_offsets = 1
            offset_index = start_parameters + n_offsets + n_trend
            self.offset = self.posterior_sample[:, offset_index]
            self.indices['fiber_offset'] = offset_index
        else:
            n_offsets = 0

        # multiple instruments ??
        if self.multi:
            # there are n instruments and n-1 offsets
            n_inst_offsets = self.n_instruments - 1
            istart = start_parameters + n_offsets + n_trend + 1
            iend = istart + n_inst_offsets
            ind = np.s_[istart:iend]
            self.inst_offsets = self.posterior_sample[:, ind]
            self.indices['inst_offsets_start'] = istart
            self.indices['inst_offsets_end'] = iend
            self.indices['inst_offsets'] = slice(istart, iend)
        else:
            n_inst_offsets = 0

        # activity indicator correlations?
        self.indcorrel = setup['kima']['indicator_correlations'] == 'true'
        if self.indcorrel:
            self.activity_indicators = setup['kima']['indicators'].split(',')
            n_act_ind = len(self.activity_indicators)
            istart = start_parameters + n_offsets + n_trend + n_inst_offsets + 1
            iend = istart + n_act_ind
            ind = np.s_[istart:iend]
            self.betas = self.posterior_sample[:, ind]
            self.indices['betas_start'] = istart
            self.indices['betas_end'] = iend
            self.indices['betas'] = slice(istart, iend)
        else:
            n_act_ind = 0

        # find GP in the compiled model
        if GPmodel is None:
            self.GPmodel = setup['kima']['GP'] == 'true'
        else:
            self.GPmodel = GPmodel

        if debug:
            print('GP model:', self.GPmodel)

        if self.GPmodel:
            n_hyperparameters = 4
            start_hyperpars = start_parameters + n_trend + n_offsets + n_inst_offsets + 1
            self.etas = self.posterior_sample[:, start_hyperpars:
                                              start_hyperpars +
                                              n_hyperparameters]

            for i in range(n_hyperparameters):
                name = 'eta' + str(i + 1)
                ind = start_hyperpars + i
                setattr(self, name, self.posterior_sample[:, ind])

            self.GP = GP(
                QPkernel(1, 1, 1, 1), self.data[:, 0], self.data[:, 2],
                white_noise=0.)

            self.indices['GPpars_start'] = start_hyperpars
            self.indices['GPpars_end'] = start_hyperpars + n_hyperparameters
            self.indices['GPpars'] = slice(start_hyperpars,
                                           start_hyperpars + n_hyperparameters)
        else:
            n_hyperparameters = 0

        # find MA in the compiled model
        self.MAmodel = setup['kima']['MA'] == 'true'

        if debug:
            print('MA model:', self.MAmodel)

        if self.MAmodel:
            n_MAparameters = 2
            start_hyperpars = start_parameters + n_trend + n_offsets + n_inst_offsets + n_hyperparameters + 1
            self.MA = self.posterior_sample[:, start_hyperpars:
                                            start_hyperpars + n_MAparameters]
        else:
            n_MAparameters = 0

        # find KO in the compiled model
        try:
            self.KO = setup['kima']['known_object'] == 'true'
        except KeyError:
            self.KO = False

        if self.KO:
            n_KOparameters = 5
            start = start_parameters + n_trend + n_offsets + n_inst_offsets + n_hyperparameters + n_MAparameters + 1
            koinds = slice(start, start + n_KOparameters)
            self.KOpars = self.posterior_sample[:, koinds]
            self.indices['KOpars'] = koinds
        else:
            n_KOparameters = 0


        start_objects_print = start_parameters + n_offsets + n_inst_offsets + \
                              n_trend + n_act_ind + n_hyperparameters + \
                              n_MAparameters + n_KOparameters + 1

        # how many parameters per component
        self.n_dimensions = int(self.posterior_sample[0, start_objects_print])
        # maximum number of components
        self.max_components = int(
            self.posterior_sample[0, start_objects_print + 1])

        # find hyperpriors in the compiled model
        if hyperpriors is None:
            self.hyperpriors = setup['kima']['hyperpriors'] == 'true'
        else:
            self.hyperpriors = hyperpriors

        # number of hyperparameters (muP, wP, muK)
        n_dist_print = 3 if self.hyperpriors else 0
        # if hyperpriors, then the period is sampled in log
        self.log_period = self.hyperpriors

        # the column with the number of planets in each sample
        self.index_component = start_objects_print + 1 + n_dist_print + 1
        self.indices['np'] = self.index_component

        # indices of the planet parameters
        self.indices['planets'] = slice(self.index_component + 1, -2)

        # build the marginal posteriors for planet parameters
        self.get_marginals()

        # make the plots, if requested
        self.make_plots(options, self.save_plots)

    @classmethod
    def load(cls, filename, diagnostic=False):
        """Load a KimaResults object from a pickle or .zip file."""
        try:
            if filename.endswith('.zip'):
                zf = zipfile.ZipFile(filename, 'r')
                names = zf.namelist()
                needs = ('sample.txt', 'levels.txt', 'sample_info.txt',
                         'posterior_sample.txt', 'posterior_sample_info.txt',
                         'kima_model_setup.txt')
                for need in needs:
                    if need not in names:
                        raise ValueError('%s does not contain a "%s" file' %
                                         (filename, need))

                with tempfile.TemporaryDirectory() as dirpath:
                    for need in needs:
                        zf.extract(need, path=dirpath)
                    try:
                        zf.extract('evidence', path=dirpath)
                        zf.extract('information', path=dirpath)
                    except Exception:
                        pass

                    pwd = os.getcwd()
                    os.chdir(dirpath)

                    setup = configparser.ConfigParser()
                    setup.optionxform = str
                    setup.read('kima_model_setup.txt')

                    if setup['kima']['multi'] == 'true':
                        datafiles = setup['kima']['files'].split(',')
                        datafiles = list(filter(None, datafiles))  # remove ''
                    else:
                        datafiles = np.atleast_1d(setup['kima']['file'])

                    datafiles = list(map(os.path.basename, datafiles))
                    for df in datafiles:
                        zf.extract(df)

                    if diagnostic:
                        from .classic import postprocess
                        postprocess()

                    res = cls('')
                    res.evidence = float(open('evidence').read())
                    res.information = float(open('information').read())

                    os.chdir(pwd)

                return res

            else:
                try:
                    with open(filename, 'rb') as f:
                        return pickle.load(f)
                except UnicodeDecodeError:
                    with open(filename, 'rb') as f:
                        return pickle.load(f, encoding='latin1')

        except Exception as e:
            # print('Unable to load data from ', filename, ':', e)
            raise

    def save_pickle(self, filename, verbose=True):
        """Pickle this KimaResults object into a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=2)
        if verbose:
            print('Wrote to file "%s"' % f.name)

    def save_zip(self, filename, verbose=True):
        """Save this KimaResults object and the text files into a zip."""
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

    def make_plots(self, options, save_plots=False):
        self.save_plots = save_plots
        if options == 'all':  # can be 'all' if called from the interpreter
            options = ('1', '2', '3', '4', '5', '6p', '7', '8')

        allowed_options = {
            # keys are allowed options (strings)
            # values are lists
            #   first item in the list can be a callable, a tuple of callables
            #   or a str
            #       if a callable, it is called
            #       if a tuple of callables, they are all called without arguments
            #       if a str, it is exec'd in globals(), locals()
            #   second item in the list is a dictionary
            #       each entry is an argument with which the callable is called

            '1': [self.make_plot1, {}],
            '2': [self.make_plot2, {'show_prior':True}],
            '3': [self.make_plot3, {}],
            '4': [self.make_plot4, {}],
            '5': [self.make_plot5, {}],
            '6': [
                self.plot_random_planets,
                {
                    'show_vsys': True,
                    'show_trend': True
                }
            ],
            '6p': [
                'self.plot_random_planets(show_vsys=True, show_trend=True);'\
                'self.phase_plot(self.maximum_likelihood_sample(Np=passes_threshold_np(self)))',
                {}
            ],
            '7': [(self.hist_offset, self.hist_vsys, self.hist_extra_sigma,
                   self.hist_trend, self.hist_correlations), {}],
            '8': [self.hist_MA, {}],
        }

        for item in allowed_options.items():
            if item[0] in options:
                methods = item[1][0]
                kwargs = item[1][1]
                if isinstance(methods, tuple):
                    [m() for m in methods]
                elif isinstance(methods, str):
                    exec(methods)
                else:
                    methods(**kwargs)

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
        self.T = self.posterior_sample[:, s]
        self.Tall = np.copy(self.T)

        # amplitudes
        i1 = 1 * max_components + index_component + 1
        i2 = 1 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.A = self.posterior_sample[:, s]
        self.Aall = np.copy(self.A)

        # phases
        i1 = 2 * max_components + index_component + 1
        i2 = 2 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.phi = self.posterior_sample[:, s]
        self.phiall = np.copy(self.phi)

        # eccentricities
        i1 = 3 * max_components + index_component + 1
        i2 = 3 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.E = self.posterior_sample[:, s]
        self.Eall = np.copy(self.E)

        # omegas
        i1 = 4 * max_components + index_component + 1
        i2 = 4 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.Omega = self.posterior_sample[:, s]
        self.Omegaall = np.copy(self.Omega)

        # times of periastron
        self.T0 = self.data[0, 0] - (self.T * self.phi) / (2. * np.pi)
        self.T0all = np.copy(self.T0)

        which = self.T != 0
        self.T = self.T[which].flatten()
        self.A = self.A[which].flatten()
        self.E = self.E[which].flatten()
        self.phi = self.phi[which].flatten()
        self.Omega = self.Omega[which].flatten()
        self.T0 = self.T0[which].flatten()

    def get_medians(self):
        """ return the median values of all the parameters """
        if self.posterior_sample.shape[0] % 2 == 0:
            print(
                'Median is not a solution because number of samples is even!!')

        self.medians = np.median(self.posterior_sample, axis=0)
        self.means = np.mean(self.posterior_sample, axis=0)
        return self.medians, self.means

    def maximum_likelihood_sample(self, Np=None, printit=True):
        """ 
        Get the posterior sample with the highest log likelihood. If `Np` is 
        given, select only from posterior samples with that number of planets.
        """
        if not self.lnlike_available:
            print('log-likelihoods are not available! '\
                  'maximum_likelihood_sample() doing nothing...')
            return

        if Np is None:
            ind = np.argmax(self.posterior_lnlike[:, 1])
            maxlike = self.posterior_lnlike[ind, 1]
            pars = self.posterior_sample[ind]
        else:
            mask = self.posterior_sample[:, self.index_component] == Np
            ind = np.argmax(self.posterior_lnlike[mask, 1])
            maxlike = self.posterior_lnlike[mask][ind, 1]
            pars = self.posterior_sample[mask][ind]

        if printit:
            print('Posterior sample with the highest likelihood value '\
                  '({:.2f})'.format(maxlike))
            if Np is not None:
                print('from samples with %d Keplerians only' % Np)
            print(
                '-> might not be representative of the full posterior distribution\n'
            )

            print('extra_sigma: ', pars[0])
            npl = int(pars[self.index_component])
            if npl > 0:
                print('number of planets: ', npl)
                print('orbital parameters: ', end='')
                # s = 20 * ' '
                s = (self.n_dimensions * ' {:>10s} ').format(
                    'P', 'K', 'phi', 'e', 'lam')
                print(s)
                # print()
                for i in range(0, npl):
                    s = (self.n_dimensions *
                         ' {:10.5f} ').format(*pars[self.index_component + 1 +
                                                    i:-2:self.max_components])
                    # if i>0:
                    s = 20 * ' ' + s
                    print(s)

            if self.GPmodel:
                print('GP parameters: ', self.eta1[ind], self.eta2[ind],
                      self.eta3[ind], self.eta4[ind])
            if self.trend:
                print('slope: ', self.trendpars[ind][0])
            if self.multi:
                ni = self.n_instruments - 1
                print('instrument offsets: ', end=' ')
                print('(relative to %s) ' % self.data_file[-1])
                s = 20 * ' '
                s += (ni * ' {:20s} ').format(*self.data_file)
                print(s)

                i = self.indices['inst_offsets']
                s = 20 * ' '
                s += (
                    ni * ' {:<20.3f} ').format(*self.posterior_sample[ind][i])
                print(s)

            print('vsys: ', pars[-1])

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

            print('extra_sigma: ', pars[0])
            npl = int(pars[self.index_component])
            if npl > 0:
                print('number of planets: ', npl)
                print('orbital parameters: ', end='')
                # s = 20 * ' '
                s = (self.n_dimensions * ' {:>10s} ').format(
                    'P', 'K', 'phi', 'e', 'lam')
                print(s)
                # print()
                for i in range(0, npl):
                    s = (self.n_dimensions *
                         ' {:10.5f} ').format(*pars[self.index_component + 1 +
                                                    i:-2:self.max_components])
                    # if i>0:
                    s = 20 * ' ' + s
                    print(s)

            if self.GPmodel:
                print('GP parameters: ', pars[self.indices['GPpars']])
            if self.trend:
                print('slope: ', pars[self.indices['trend']])

            print('vsys: ', pars[-1])

        return pars

    def model(self, sample, t=None):
        """ 
        Evaluate the complete model at one posterior sample. If `t` is None, use
        the data times. Instrument offsets are only added to the model if `t` is
        None, but the systemic velocity is always added. This function does
        *not* evaluate the GP component of the model.
        """

        if sample.shape[0] != self.posterior_sample.shape[1]:
            n1 = sample.shape[0]
            n2 = self.posterior_sample.shape[1]
            raise ValueError(
                '`sample` has wrong dimensions, should be %d got %d' % (n2,
                                                                        n1))

        data_t = False
        if t is None:
            t = self.data[:, 0].copy()
            data_t = True

        v = np.zeros_like(t)
        if self.GPmodel:
            v_at_t = np.zeros_like(t)

        # get the planet parameters for this sample
        pars = sample[self.indices['planets']].copy()

        # how many planets in this sample?
        # nplanets = pars.size / self.n_dimensions
        nplanets = (pars[:self.max_components] != 0).sum()

        # add the Keplerians for each of the planets
        for j in range(int(nplanets)):
            P = pars[j + 0 * self.max_components]
            if P == 0.0:
                continue
            K = pars[j + 1 * self.max_components]
            phi = pars[j + 2 * self.max_components]
            t0 = t[0] - (P * phi) / (2. * np.pi)
            ecc = pars[j + 3 * self.max_components]
            w = pars[j + 4 * self.max_components]
            v += keplerian(t, P, K, ecc, w, t0, 0.)

            if self.GPmodel:
                v_at_t += keplerian(t, P, K, ecc, w, t0, 0.)

        # systemic velocity for this sample
        vsys = sample[-1]
        v += vsys
        if self.GPmodel:
            v_at_t += vsys

        # if evaluating at the same times as the data, add instrument offsets
        if data_t and self.multi and len(self.data_file) > 1:
            offsets = sample[self.indices['inst_offsets']]
            number_offsets = offsets.size
            for j in range(number_offsets + 1):
                if j == number_offsets:
                    of = 0.
                else:
                    of = offsets[j]
                instrument_mask = self.obs == j + 1
                v[instrument_mask] += of

        # add the trend, if present
        if self.trend:
            v += sample[self.indices['trend']] * (t - self.tmiddle)
            if self.GPmodel:
                v_at_t += sample[self.indices['trend']] * (t - self.tmiddle)

        # the GP prediction
        # if self.GPmodel:
        #     self.GP.kernel.setpars(self.eta1[i], self.eta2[i], self.eta3[i],
        #                             self.eta4[i])
        #     mu = self.GP.predict(y - v_at_t, ttGP, return_std=False)

        return v

    def residuals(self, sample):
        return self.y - self.model(sample)

    def phase_plot(self, sample):
        """ Plot the phase curves given the solution in `sample` """
        # this is probably the most complicated function in the whole file!!

        if self.max_components == 0:
            print('Model has no planets! phase_plot() doing nothing...')
            return

        # make copies to not change attributes
        t, y, e = self.t.copy(), self.y.copy(), self.e.copy()
        if t[0] > 24e5:
            t -= 24e5

        def kima_pars_to_keplerian_pars(p):
            # transforms kima planet pars (P,K,phi,ecc,w)
            # to pykima.keplerian.keplerian pars (P,K,ecc,w,T0,vsys=0)
            assert p.size == self.n_dimensions
            P = p[0]
            phi = p[2]
            t0 = t[0] - (P * phi) / (2. * np.pi)
            return np.array([P, p[1], p[3], p[4], t0, 0.0])

        mc = self.max_components

        # get the planet parameters for this sample
        pars = sample[self.indices['planets']].copy()

        # sort by decreasing amplitude (arbitrary)
        ind = np.argsort(pars[1 * mc:2 * mc])[::-1]
        for i in range(self.n_dimensions):
            pars[i * mc:(i + 1) * mc] = pars[i * mc:(i + 1) * mc][ind]

        # (function to) get parameters for individual planets
        this_planet_pars = lambda i: pars[i::self.max_components]
        parsi = lambda i: kima_pars_to_keplerian_pars(this_planet_pars(i))

        # extract periods, phases and calculate times of periastron
        P = pars[0 * mc:1 * mc]
        phi = pars[2 * mc:3 * mc]
        T0 = t[0] - (P * phi) / (2. * np.pi)

        # how many planets in this sample?
        # nplanets = int(pars.size / self.n_dimensions) <-- this is wrong!
        nplanets = (pars[:self.max_components] != 0).sum()
        planetis = list(range(nplanets))

        if nplanets == 0:
            print('Sample has no planets! phase_plot() doing nothing...')
            return

        # get the model for this sample
        # (this adds in the instrument offsets and the systemic velocity)
        v = self.model(sample)

        # put all data around zero
        y -= sample[-1]  # subtract this sample's systemic velocity
        if self.multi:
            # subtract each instrument's offset
            for i in range(self.n_instruments - 1):
                of = sample[self.indices['inst_offsets']][i]
                y[self.obs == i + 1] -= sample[self.indices['inst_offsets']][i]
        if self.trend:  # and subtract the trend
            y -= sample[self.indices['trend']] * (t - self.tmiddle)

        ekwargs = {
            'fmt': 'o',
            'mec': 'none',
            'ms': 5,
            'capsize': 0,
            'elinewidth': 0.8,
        }

        # very complicated logic just to make the figure the right size
        fs = (max(6.4, 6.4 + 1 * (nplanets - 2)),
              max(4.8, 4.8 + 1 * (nplanets - 3)))
        fig = plt.figure('phase plot', constrained_layout=True, figsize=fs)
        nrows = {
            1: 2,
            2: 2,
            3: 2,
            4: 3,
            5: 3,
            6: 3,
            7: 4,
            8: 4,
            9: 4,
            10: 4
        }[nplanets]
        ncols = nplanets if nplanets <= 3 else 3
        gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                               height_ratios=[2] * (nrows - 1) + [1])
        gs_indices = {i: (i // 3, i % 3) for i in range(50)}

        # for each planet in this sample
        for i, letter in zip(range(nplanets), ascii_lowercase[1:]):
            ax = fig.add_subplot(gs[gs_indices[i]])

            p = P[i]
            t0 = T0[i]

            ## plot the keplerian curve in phase (3 times)
            phase = np.linspace(0, 1, 200)
            tt = phase * p + t0
            vv = keplerian(tt, *parsi(i))
            for j in (-1, 0, 1):
                alpha = 0.3 if j in (-1, 1) else 1
                ax.plot(
                    np.sort(phase) + j, vv[np.argsort(phase)], 'k',
                    alpha=alpha)

            ## the other planets which are not the ith
            other = copy(planetis)
            other.remove(i)

            ## subtract the other planets from the data and plot it (the data)
            if self.multi:
                for k in range(1, self.n_instruments + 1):
                    m = self.obs == k
                    phase = ((t[m] - t0) / p) % 1.0
                    other_planet_v = np.array(
                        [keplerian(t[m], *parsi(i)) for i in other])
                    other_planet_v = other_planet_v.sum(axis=0)
                    yy = y[m].copy()
                    yy -= other_planet_v
                    ee = e[m].copy()

                    # one color for each instrument
                    color = ax._get_lines.prop_cycler.__next__()['color']

                    for j in (-1, 0, 1):
                        alpha = 0.3 if j in (-1, 1) else 1
                        label = self.data_file[k - 1] if j == 0 else None
                        ax.errorbar(
                            np.sort(phase) + j, yy[np.argsort(phase)],
                            ee[np.argsort(phase)], color=color, alpha=alpha, 
                            **ekwargs)

            else:
                phase = ((t - t0) / p) % 1.0
                other_planet_v = np.array(
                    [keplerian(t, *parsi(i)) for i in other])
                other_planet_v = other_planet_v.sum(axis=0)
                yy = y.copy()
                yy -= other_planet_v

                color = ax._get_lines.prop_cycler.__next__()['color']

                for j in (-1, 0, 1):
                    alpha = 0.3 if j in (-1, 1) else 1
                    ax.errorbar(
                        np.sort(phase) + j, yy[np.argsort(phase)],
                        e[np.argsort(phase)], color=color, alpha=alpha, 
                        **ekwargs)

            ax.set_xlim(-0.2, 1.2)
            ax.set(xlabel="phase", ylabel="radial velocity [m/s]")
            ax.set_title('%s' % letter, loc='left')
            ax.set_title('P=%.2f days' % p, loc='right')

        ax = fig.add_subplot(gs[-1, :])
        if self.multi:
            for k in range(1, self.n_instruments + 1):
                m = self.obs == k
                # label = self.data_file[k - 1]
                ax.errorbar(t[m], self.y[m] - v[m], e[m], **ekwargs)
        else:
            ax.errorbar(t, self.y - v, e, **ekwargs)

        # ax.legend()
        ax.axhline(y=0, ls='--', alpha=0.5, color='k')
        ax.set_ylim(np.tile(np.abs(ax.get_ylim()).max(), 2) * [-1, 1])
        ax.set(xlabel='Time [days]', ylabel='residuals [m/s]')
        ax.set_title('rms=%.2f m/s' % wrms(self.y - v, 1 / e**2), loc='right')

        if self.save_plots:
            filename = 'kima-showresults-fig6.1.png'
            print('saving in', filename)
            fig.savefig(filename)

    @property
    def ratios(self):
        bins = np.arange(self.max_components + 2)
        nplanets = self.posterior_sample[:, self.index_component]
        n, _ = np.histogram(nplanets, bins=bins)
        return n.flat[1:] / n.flat[:-1]

    def make_plot1(self):
        """ Plot the histogram of the posterior for Np """
        fig, ax = plt.subplots(1, 1)
        # n, _, _ = plt.hist(self.posterior_sample[:, self.index_component], 100)

        bins = np.arange(self.max_components + 2)
        nplanets = self.posterior_sample[:, self.index_component]
        n, _ = np.histogram(nplanets, bins=bins)
        ax.bar(bins[:-1], n, zorder=2)

        if self.removed_crossing:
            ic = self.index_component
            nn = (~np.isnan(self.posterior_sample[:, ic + 1:ic + 11])).sum(
                axis=1)
            nn, _ = np.histogram(nn, bins=bins)
            ax.bar(bins[:-1], nn, color='r', alpha=0.2, zorder=2)
            ax.legend(['all posterior samples', 'crossing orbits removed'])
        else:
            pt_Np = passes_threshold_np(self)
            ax.bar(pt_Np, n[pt_Np], color='C3', zorder=2)
            # top = np.mean(ax.get_ylim())
            # ax.arrow(pt_Np, top, 0, -.4*top, lw=2, head_length=1, fc='k', ec='k')

        xlim = (-0.5, self.max_components + 0.5)
        xticks = np.arange(self.max_components + 1)
        ax.set(xlabel='Number of Planets',
               ylabel='Number of Posterior Samples', xlim=xlim, xticks=xticks,
               title='Posterior distribution for $N_p$')

        nn = n[np.nonzero(n)]
        print('Np probability ratios: ', nn.flat[1:] / nn.flat[:-1])

        if self.save_plots:
            filename = 'kima-showresults-fig1.png'
            print('saving in', filename)
            fig.savefig(filename)

    def make_plot2(self, nbins=100, bins=None, plims=None, logx=True,
                   kde=False, kde_bw=None, show_peaks=False, show_prior=False):
        """ 
        Plot the histogram (or the kde) of the posterior for orbital period P.
        Optionally provide the number of histogram bins, the bins themselves,
        the limits in orbital period, or the kde bandwidth. If both kde and
        show_peaks are true, the routine attempts to plot the most prominent
        peaks in the posterior.
        """

        if self.max_components == 0:
            print('Model has no planets! make_plot2() doing nothing...')
            return

        if self.log_period:
            T = np.exp(self.T)
            # print('exponentiating period!')
        else:
            T = self.T

        fig, ax = plt.subplots(1, 1)

        kwargs = {'ls': '--', 'lw': 2, 'alpha': 0.5, 'zorder': -1}
        # mark 1 year
        year = 365.25
        ax.axvline(x=year, color='r', label='1 year', **kwargs)
        # ax.axvline(x=year/2., ls='--', color='r', lw=3, alpha=0.6)
        # plt.axvline(x=year/3., ls='--', color='r', lw=3, alpha=0.6)

        # mark the timespan of the data
        ax.axvline(x=self.t.ptp(), color='b', label='timespan', **kwargs)

        if kde:
            NN = 3000
            kdef = gaussian_kde(T, bw_method=kde_bw)
            if plims is None:
                if logx:
                    xx = np.logspace(np.log10(T.min()), np.log10(T.max()), NN)
                    y = kdef(xx)
                    ax.semilogx(xx, y)
                else:
                    xx = np.linspace(T.min(), T.max(), NN)
                    y = kdef(xx)
                    ax.plot(xx, y)
            else:
                a, b = plims
                if logx:
                    xx = np.logspace(np.log10(a), np.log10(b), NN)
                    y = kdef(xx)
                    ax.semilogx(xx, y)
                else:
                    xx = np.linspace(a, b, NN)
                    y = kdef(xx)
                    ax.plot(xx, y)

            # show the limits of the kde evaluation
            ax.vlines([xx.min(), xx.max()], ymin=0, ymax=0.03, color='k',
                      transform=ax.get_xaxis_transform())

            if show_prior:
                prior = get_prior(self.setup['priors.planets']['Pprior'])
                if isinstance(prior.dist, reciprocal_gen):
                    # show pdf per logarithmic interval
                    ax.plot(xx, xx * prior.pdf(xx), 'k', label='prior')
                else:
                    ax.plot(xx, prior.pdf(xx), 'k', label='prior')

            if show_peaks and find_peaks:
                peaks, _ = find_peaks(y, prominence=0.1)
                for peak in peaks:
                    s = r'P$\simeq$%.2f' % xx[peak]
                    ax.text(xx[peak], y[peak], s, ha='left')

        else:
            if bins is None:
                if plims is None:
                    # by default, 100 bins in log between 0.1 and 1e7
                    bins = 10**np.linspace(
                        np.log10(1e-1), np.log10(1e7), nbins)
                else:
                    bins = 10**np.linspace(
                        np.log10(plims[0]), np.log10(plims[1]), nbins)

            ax.hist(T, bins=bins, alpha=0.8)

            if show_prior and T.size > 100:
                kwargs = {
                    'bins': bins,
                    'alpha': 0.15,
                    'color': 'k',
                    'zorder': -1,
                    'label': 'prior'
                }

                if self.hyperpriors:
                    P = hyperprior_samples(T.size)
                    ax.hist(P, **kwargs)
                else:
                    prior = get_prior(self.setup['priors.planets']['Pprior'])
                    ax.hist(prior.rvs(T.size), **kwargs)

        ax.legend()
        ax.set(xscale='log' if logx else 'linear', xlabel=r'(Period/days)',
               ylabel='KDE density' if kde else 'Number of Posterior Samples',
               title='Posterior distribution for the orbital period(s)')
        ax.set_ylim(bottom=0)

        if plims is not None:
            ax.set_xlim(plims)

        if self.save_plots:
            filename = 'kima-showresults-fig2.png'
            print('saving in', filename)
            fig.savefig(filename)

    def make_plot3(self, points=True):
        """
        Plot the 2d histograms of the posteriors for semi-amplitude 
        and orbital period and eccentricity and orbital period.
        If `points` is True, plot each posterior sample, else plot hexbins
        """

        if self.max_components == 0:
            print('Model has no planets! make_plot3() doing nothing...')
            return

        if self.log_period:
            T = np.exp(self.T)
            # print('exponentiating period!')
        else:
            T = self.T

        A, E = self.A, self.E

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # the y scale in loglog looks bad if the semi-amplitude doesn't have
        # high dynamic range; the threshold of 30 is arbitrary
        Khdr_threshold = 30

        if points:
            if A.ptp() > Khdr_threshold:
                ax1.loglog(T, A, '.', markersize=2, zorder=2)
            else:
                ax1.semilogx(T, A, '.', markersize=2, zorder=2)

            ax2.semilogx(T, E, '.', markersize=2, zorder=2)

        else:
            if A.ptp() > 30:
                ax1.hexbin(T, A, gridsize=50, bins='log', xscale='log',
                           yscale='log', cmap=plt.get_cmap('afmhot_r'))
            else:
                ax1.hexbin(T, A, gridsize=50, bins='log', xscale='log',
                           yscale='linear', cmap=plt.get_cmap('afmhot_r'))

            ax2.hexbin(T, E, gridsize=50, bins='log', xscale='log',
                       cmap=plt.get_cmap('afmhot_r'))

        if self.removed_crossing:
            if points:
                mc, ic = self.max_components, self.index_component

                i1, i2 = 0 * mc + ic + 1, 0 * mc + ic + mc + 1
                T = self.posterior_sample_original[:, i1:i2]
                if self.log_period:
                    T = np.exp(T)

                i1, i2 = 1 * mc + ic + 1, 1 * mc + ic + mc + 1
                A = self.posterior_sample_original[:, i1:i2]

                i1, i2 = 3 * mc + ic + 1, 3 * mc + ic + mc + 1
                E = self.posterior_sample_original[:, i1:i2]

                if A.ptp() > Khdr_threshold:
                    ax1.loglog(T, A, '.', markersize=1, alpha=0.05, color='r',
                               zorder=1)
                else:
                    ax1.semilogx(T, A, '.', markersize=1, alpha=0.05,
                                 color='r', zorder=1)

                ax2.semilogx(T, E, '.', markersize=1, alpha=0.05, color='r',
                             zorder=1)

        ax1.set(ylabel='Semi-amplitude [m/s]',
                title='Joint posterior semi-amplitude $-$ orbital period')
        ax2.set(ylabel='Eccentricity', xlabel='Period [days]',
                title='Joint posterior eccentricity $-$ orbital period',
                ylim=[0, 1], xlim=[0.1, 1e7])

        if self.save_plots:
            filename = 'kima-showresults-fig3.png'
            print('saving in', filename)
            fig.savefig(filename)

    def make_plot4(self, Np=None, ranges=None):
        """ 
        Plot histograms for the GP hyperparameters. If Np is not None, highlight
        the samples with Np Keplerians. 
        """
        if not self.GPmodel:
            print('Model does not have GP! make_plot4() doing nothing...')
            return

        available_etas = [v for v in dir(self) if v.startswith('eta')][:-1]
        labels = ['eta1', 'eta2', 'eta3', 'eta4']
        if ranges is None:
            ranges = 4 * [None]
        print(ranges)

        if Np is not None:
            m = self.posterior_sample[:, self.index_component] == Np

        fig, axes = plt.subplots(2, int(len(available_etas) / 2))
        for i, eta in enumerate(available_etas):
            ax = np.ravel(axes)[i]
            ax.hist(getattr(self, eta), bins=40, range=ranges[i])

            if Np is not None:
                ax.hist(
                    getattr(self, eta)[m], bins=40, histtype='step', alpha=0.5,
                    label='$N_p$=%d samples' % Np, range=ranges[i])
                ax.legend()

            ax.set(xlabel=labels[i], ylabel='posterior samples')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if self.save_plots:
            filename = 'kima-showresults-fig4.png'
            print('saving in', filename)
            fig.savefig(filename)

    def make_plot5(self, show=True, ranges=None):
        """ Corner plot for the GP hyperparameters """

        if not self.GPmodel:
            print('Model does not have GP! make_plot5() doing nothing...')
            return

        # these are the limits of the default prior for eta3
        self.pmin = 10.
        self.pmax = 40.
        # but we try to accomodate if the prior is changed
        if self.eta3.min() < self.pmin:
            self.pmin = np.floor(self.eta3.min())
        if self.eta3.max() > self.pmax:
            self.pmax = np.ceil(self.eta3.max())

        # available_etas = [v for v in dir(self) if v.startswith('eta')]
        available_etas = ['eta1', 'eta2', 'eta3', 'eta4']
        labels = [r'$s$'] * self.n_jitters
        labels += [
            r'$\eta_%d$' % (i + 1) for i, _ in enumerate(available_etas)
        ]
        units = ['m/s'] * self.n_jitters + ['m/s', 'days', 'days', None]
        xlabels = []
        for label, unit in zip(labels, units):
            xlabels.append(label +
                           ' (%s)' % unit if unit is not None else label)

        ### all Np together
        self.post_samples = np.c_[self.extra_sigma, self.etas]
        # if self.multi:
        #     variables = list(self.extra_sigma.T)
        # else:
        #     variables = [self.extra_sigma]

        # for eta in available_etas:
        #     variables.append(getattr(self, eta))

        # self.post_samples = np.vstack(variables).T
        # ranges = [1.]*(len(available_etas) + self.extra_sigma.shape[1])

        if ranges is None:
            ranges = [1.] * self.post_samples.shape[1]
        # ranges[3] = (self.pmin, self.pmax)

        c = corner.corner
        try:
            self.corner1 = c(
                self.post_samples,
                labels=xlabels,
                show_titles=True,
                plot_contours=False,
                plot_datapoints=True,
                plot_density=False,
                # fill_contours=True, smooth=True,
                # contourf_kwargs={'cmap':plt.get_cmap('afmhot'), 'colors':None},
                hexbin_kwargs={
                    'cmap': plt.get_cmap('afmhot_r'),
                    'bins': 'log'
                },
                hist_kwargs={'density': True},
                range=ranges,
                data_kwargs={'alpha': 1},
            )
        except AssertionError as exc:
            print('AssertionError from corner in make_plot5()', end='')
            if "I don't believe" in str(exc):
                print(', you probably need to get more posterior samples')
            return

        self.corner1.suptitle(
            'Joint and marginal posteriors for GP hyperparameters')

        if show:
            self.corner1.tight_layout(rect=[0, 0.03, 1, 0.95])

        if self.save_plots:
            filename = 'kima-showresults-fig5.png'
            print('saving in', filename)
            self.corner1.savefig(filename)

    def get_sorted_planet_samples(self):
        # all posterior samples for the planet parameters
        # this array is nsamples x (n_dimensions*max_components)
        # that is, nsamples x 5, nsamples x 10, for 1 and 2 planets for example
        try:
            self.planet_samples
        except AttributeError:
            self.planet_samples = \
                    self.posterior_sample[:, self.index_component+1:-2].copy()

        if self.max_components == 0:
            return self.planet_samples

        # here we sort the planet_samples array by the orbital period
        # this is a bit difficult because the organization of the array is
        # P1 P2 K1 K2 ....
        samples = np.empty_like(self.planet_samples)
        n = self.max_components * self.n_dimensions
        mc = self.max_components
        p = self.planet_samples[:, :mc]
        ind_sort_P = np.arange(np.shape(p)[0])[:, np.newaxis], np.argsort(p)
        for i, j in zip(range(0, n, mc), range(mc, n + mc, mc)):
            samples[:, i:j] = self.planet_samples[:, i:j][ind_sort_P]

        return samples

    def apply_cuts_period(self, samples, pmin=None, pmax=None,
                          return_mask=False):
        """ apply cuts in orbital period """
        too_low_periods = np.zeros_like(samples[:, 0], dtype=bool)
        too_high_periods = np.zeros_like(samples[:, 0], dtype=bool)

        if pmin is not None:
            too_low_periods = samples[:, 0] < pmin
            samples = samples[~too_low_periods, :]

        if pmax is not None:
            too_high_periods = samples[:, 1] > pmax
            samples = samples[~too_high_periods, :]

        if return_mask:
            mask = ~too_low_periods & ~too_high_periods
            return samples, mask
        else:
            return samples

    def corner_planet_parameters(self, pmin=None, pmax=None):
        """ Corner plot of the posterior samples for the planet parameters """

        labels = [r'$P$', r'$K$', r'$\phi$', 'ecc', 'va']

        samples = self.get_sorted_planet_samples()
        samples = self.apply_cuts_period(samples, pmin, pmax)

        # samples is still nsamples x (n_dimensions*max_components)
        # let's separate each planets' parameters
        data = []
        for i in range(self.max_components):
            data.append(samples[:, i::self.max_components])

        # separate bins for each parameter
        bins = []
        for planetp in data:
            if hist_tools_available:
                bw = hist_tools.freedman_bin_width
                # bw = hist_tools.knuth_bin_width
                this_planet_bins = []
                for sample in planetp.T:
                    this_planet_bins.append(
                        bw(sample, return_bins=True)[1].size)
                bins.append(this_planet_bins)
            else:
                bins.append(None)

        # set the parameter ranges to include everythinh
        def r(x, over=0.2):
            return x.min() - over * x.ptp(), x.max() + over * x.ptp()

        ranges = []
        for i in range(self.n_dimensions):
            i1, i2 = self.max_components * i, self.max_components * (i + 1)
            ranges.append(r(samples[:, i1:i2]))

        #
        c = corner.corner
        fig = None
        colors = plt.rcParams["axes.prop_cycle"]

        for i, (datum, colorcycle) in enumerate(zip(data, colors)):
            fig = c(
                datum,
                fig=fig,
                labels=labels,
                show_titles=len(data) == 1,
                plot_contours=False,
                plot_datapoints=True,
                plot_density=False,
                bins=bins[i],
                range=ranges,
                color=colorcycle['color'],
                # fill_contours=True, smooth=True,
                # contourf_kwargs={'cmap':plt.get_cmap('afmhot'), 'colors':None},
                #hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
                hist_kwargs={'normed': True},
                # range=[1., 1., (0, 2*np.pi), (0., 1.), (0, 2*np.pi)],
                data_kwargs={
                    'alpha': 1,
                    'ms': 3,
                    'color': colorcycle['color']
                },
            )

        plt.show()

    def corner_known_object(self, fig=None):
        """ Corner plot of the posterior samples for the known object parameters """
        if not self.KO:
            print(
                'Model has no known object! corner_known_object() doing nothing...'
            )
            return

        labels = [r'$P$', r'$K$', r'$\phi$', 'ecc', 'w']
        corner.corner(self.KOpars, fig=fig, labels=labels,
                      hist_kwars={'normed': True})


    def plot_random_planets(self, ncurves=20, over=0.1, pmin=None, pmax=None,
                            show_vsys=False, show_trend=False):
        """
        Display the RV data together with curves from the posterior predictive.
        A total of `ncurves` random samples are chosen, and the Keplerian 
        curves are calculated covering 100 + `over`% of the data timespan.
        If the model has a GP component, the prediction is calculated using the
        GP hyperparameters for each of the random samples.
        """
        colors = [cc['color'] for cc in plt.rcParams["axes.prop_cycle"]]

        samples = self.get_sorted_planet_samples()
        if self.max_components > 0:
            samples, mask = \
                self.apply_cuts_period(samples, pmin, pmax, return_mask=True)
        else:
            mask = np.ones(samples.shape[0], dtype=bool)

        t = self.data[:, 0].copy()
        if t[0] > 24e5:
            t -= 24e5

        tt = np.linspace(t.min() - over * t.ptp(),
                         t.max() + over * t.ptp(), 5000 + int(100 * over))

        if self.GPmodel:
            # let's be more reasonable for the number of GP prediction points
            ## OLD: linearly spaced points (lots of useless points within gaps)
            # ttGP = np.linspace(t[0], t[-1], 1000 + t.size*3)
            ## NEW: have more points near where there is data
            kde = gaussian_kde(t)
            ttGP = kde.resample(25000 + t.size * 3).reshape(-1)
            # constrain ttGP within observed times, to not waste (this could go...)
            ttGP = (ttGP + t[0]) % t.ptp() + t[0]
            ttGP = np.r_[ttGP, t]
            ttGP.sort()  # in-place

            if t.size > 100:
                ncurves = 10

        y = self.data[:, 1].copy()
        yerr = self.data[:, 2].copy()

        ncurves = min(ncurves, samples.shape[0])

        # select random `ncurves` indices
        # from the (sorted, period-cut) posterior samples
        ii = np.random.randint(samples.shape[0], size=ncurves)

        fig, ax = plt.subplots(1, 1)
        ax.set_title('Posterior samples in RV data space')

        ## plot the Keplerian curves
        for i in ii:
            v = np.zeros_like(tt)
            v_at_t = np.zeros_like(t)
            if self.GPmodel:
                v_at_ttGP = np.zeros_like(ttGP)

            # known object, if set
            if self.KO:
                pars = self.KOpars[i]
                P = pars[0]
                K = pars[1]
                phi = pars[2]
                t0 = t[0] - (P * phi) / (2. * np.pi)
                ecc = pars[3]
                w = pars[4]
                v += keplerian(tt, P, K, ecc, w, t0, 0.)
                v_at_t += keplerian(t, P, K, ecc, w, t0, 0.)

            # get the planet parameters for the current (ith) sample
            pars = samples[i, :].copy()
            # how many planets in this sample?
            nplanets = pars.size / self.n_dimensions
            # add the Keplerians for each of the planets
            for j in range(int(nplanets)):
                P = pars[j + 0 * self.max_components]
                if P == 0.0 or np.isnan(P):
                    continue
                K = pars[j + 1 * self.max_components]
                phi = pars[j + 2 * self.max_components]
                t0 = t[0] - (P * phi) / (2. * np.pi)
                ecc = pars[j + 3 * self.max_components]
                w = pars[j + 4 * self.max_components]
                v += keplerian(tt, P, K, ecc, w, t0, 0.)
                v_at_t += keplerian(t, P, K, ecc, w, t0, 0.)
                if self.GPmodel:
                    v_at_ttGP += keplerian(ttGP, P, K, ecc, w, t0, 0.)

            # systemic velocity for the current (ith) sample
            vsys = self.posterior_sample[mask][i, -1]
            v += vsys
            v_at_t += vsys
            if self.GPmodel:
                v_at_ttGP += vsys

            # add the trend, if present
            if self.trend:
                v += self.trendpars[i] * (tt - self.tmiddle)
                v_at_t += self.trendpars[i] * (t - self.tmiddle)
                if self.GPmodel:
                    v_at_ttGP += self.trendpars[i] * (ttGP - self.tmiddle)
                if show_trend:
                    ax.plot(tt, vsys + self.trendpars[i] * (tt - self.tmiddle),
                            alpha=0.2, color='m', ls=':')

            # plot the MA "prediction"
            # if self.MAmodel:
            #     vMA = v_at_t.copy()
            #     dt = np.ediff1d(self.t)
            #     sigmaMA, tauMA = self.MA.mean(axis=0)
            #     vMA[1:] += sigmaMA * np.exp(np.abs(dt) / tauMA) * (self.y[:-1] - v_at_t[:-1])
            #     vMA[np.abs(vMA) > 1e6] = np.nan
            #     ax.plot(t, vMA, 'o-', alpha=1, color='m')

            # v only has the Keplerian components, not the GP predictions
            # ax.plot(tt, v, alpha=0.2, color='k')

            # add the instrument offsets, if present
            if self.multi and len(self.data_file) > 1:
                number_offsets = self.inst_offsets.shape[1]
                for j in range(number_offsets + 1):
                    if j == number_offsets:
                        # the last dataset defines the systemic velocity,
                        # so the offset is zero
                        of = 0.
                    else:
                        of = self.inst_offsets[i, j]

                    instrument_mask = self.obs == j + 1
                    start = t[instrument_mask].min()
                    end = t[instrument_mask].max()
                    ptp = t[instrument_mask].ptp()
                    time_mask = (tt > start - over * ptp) & (tt <
                                                             end + over * ptp)

                    v_i = v.copy()
                    v_i[time_mask] += of
                    ax.plot(tt[time_mask], v_i[time_mask], alpha=0.1,
                            color=lighten_color(colors[j], 1.5))

                    time_mask = (t >= start) & (t <= end)
                    v_at_t[time_mask] += of
                    if self.GPmodel:
                        time_mask = (ttGP >= start) & (ttGP <= end)
                        v_at_ttGP[time_mask] += of
            else:
                ax.plot(tt, v, alpha=0.1, color='k')

            # plot the GP prediction
            if self.GPmodel:
                self.GP.kernel.setpars(self.eta1[i], self.eta2[i],
                                       self.eta3[i], self.eta4[i])
                mu = self.GP.predict(y - v_at_t, ttGP, return_std=False)
                ax.plot(ttGP, mu + v_at_ttGP, alpha=0.2, color='plum')

            if show_vsys:
                ax.plot(t, vsys * np.ones_like(t), alpha=0.2, color='r',
                        ls='--')
                if self.multi:
                    for j in range(self.inst_offsets.shape[1]):
                        instrument_mask = self.obs == j + 1
                        start = t[instrument_mask].min()
                        end = t[instrument_mask].max()

                        of = self.inst_offsets[i, j]

                        ax.hlines(vsys + of, xmin=start, xmax=end, alpha=0.2,
                                  color=colors[j])

        ## we could also choose to plot the GP prediction using the median of
        ## the hyperparameters posterior distributions
        # if self.GPmodel:
        #     # set the GP parameters to the median (or mean?) of their posteriors
        #     eta1, eta2, eta3, eta4 = np.median(self.etas, axis=0)
        #     # eta1, eta2, eta3, eta4 = np.mean(self.etas, axis=0)
        #     self.GP.kernel.setpars(eta1, eta2, eta3, eta4)
        #
        #     # set the orbital parameters to the median of their posteriors
        #     P,K,phi,ecc,w = np.median(samples, axis=0)
        #     t0 = t[0] - (P*phi)/(2.*np.pi)
        #     mu_orbital = keplerian(t, P, K, ecc, w, t0, 0.)
        #
        #     # calculate the mean and std prediction from the GP model
        #     mu, std = self.GP.predict(y - mu_orbital, ttGP, return_std=True)
        #     mu_orbital = keplerian(ttGP, P, K, ecc, w, t0, 0.)
        #
        #     # 2-sigma region around the predictive mean
        #     ax.fill_between(ttGP, y1=mu+mu_orbital-2*std, y2=mu+mu_orbital+2*std,
        #                     alpha=0.3, color='m')

        ## plot the data
        if self.fiber_offset:
            mask = t < 57170
            if self.multi:
                for j in range(self.inst_offsets.shape[1] + 1):
                    m = self.obs == j + 1
                    ax.errorbar(t[m & mask], y[m & mask], yerr[m & mask],
                                fmt='o', color=colors[j])
            else:
                ax.errorbar(t[mask], y[mask], yerr[mask], fmt='o')

            yshift = np.vstack([y[~mask], y[~mask] - self.offset.mean()])
            for i, ti in enumerate(t[~mask]):
                ax.errorbar(ti, yshift[0, i], fmt='o', color='m', alpha=0.2)
                ax.errorbar(ti, yshift[1, i], yerr[~mask][i], fmt='o',
                            color='r')
        else:
            if self.multi:
                for j in range(self.inst_offsets.shape[1] + 1):
                    m = self.obs == j + 1
                    ax.errorbar(t[m], y[m], yerr[m], fmt='o', color=colors[j],
                                label=self.data_file[j])
                ax.legend()
            else:
                ax.errorbar(t, y, yerr, fmt='o')

        ax.set(xlabel='Time [days]', ylabel='RV [m/s]')
        plt.tight_layout()

        if self.save_plots:
            filename = 'kima-showresults-fig6.png'
            print('saving in', filename)
            fig.savefig(filename)

    def hist_offset(self):
        """ Plot the histogram of the posterior for the fiber offset """
        if not self.fiber_offset:
            print('Model has no fiber offset! hist_offset() doing nothing...')
            return

        units = ' (m/s)'  # if self.units == 'ms' else ' (km/s)'
        estimate = percentile68_ranges_latex(self.offset) + units

        fig, ax = plt.subplots(1, 1)
        ax.hist(self.offset)
        title = 'Posterior distribution for fiber offset \n %s' % estimate
        ax.set(xlabel='fiber offset (m/s)', ylabel='posterior samples',
               title=title)

        if self.save_plots:
            filename = 'kima-showresults-fig7.1.png'
            print('saving in', filename)
            fig.savefig(filename)

    def hist_vsys(self, show_offsets=True):
        """ Plot the histogram of the posterior for the systemic velocity """
        vsys = self.posterior_sample[:, -1]
        units = ' (m/s)'  # if self.units == 'ms' else ' (km/s)'
        estimate = percentile68_ranges_latex(vsys) + units

        fig, ax = plt.subplots(1, 1)
        ax.hist(vsys)
        title = 'Posterior distribution for $v_{\\rm sys}$ \n %s' % estimate
        ax.set(xlabel='vsys' + units, ylabel='posterior samples', title=title)

        if self.save_plots:
            filename = 'kima-showresults-fig7.2.png'
            print('saving in', filename)
            fig.savefig(filename)

        if show_offsets and self.multi:
            n_inst_offsets = self.inst_offsets.shape[1]
            fig, axs = plt.subplots(1, n_inst_offsets, sharey=True,
                                    figsize=(n_inst_offsets * 3,
                                             5), squeeze=True)
            if n_inst_offsets == 1:
                axs = [
                    axs,
                ]

            for i in range(n_inst_offsets):
                a = self.inst_offsets[:, i]
                estimate = percentile68_ranges_latex(a) + units
                axs[i].hist(a)
                axs[i].set(xlabel='offset %d' % (i + 1), title=estimate,
                           ylabel='posterior samples')

            title = 'Posterior distribution(s) for instrument offset(s)'
            fig.suptitle(title)

            if self.save_plots:
                filename = 'kima-showresults-fig7.2.1.png'
                print('saving in', filename)
                fig.savefig(filename)

    def hist_extra_sigma(self):
        """ Plot the histogram of the posterior for the additional white noise """
        units = ' (m/s)'  # if self.units == 'ms' else ' (km/s)'

        if self.multi:  # there are n_instruments jitters
            fig, axs = plt.subplots(1, self.n_instruments, sharey=True,
                                    figsize=(self.n_instruments * 3,
                                             5), squeeze=True)
            for i, jit in enumerate(self.extra_sigma.T):
                estimate = percentile68_ranges_latex(jit) + units
                axs[i].hist(jit)
                axs[i].set(xlabel='jitter %d' % (i + 1), title=estimate,
                           ylabel='posterior samples')

            title = 'Posterior distribution(s) for extra white noise(s)'
            fig.suptitle(title)

        else:
            estimate = percentile68_ranges_latex(self.extra_sigma) + units
            fig, ax = plt.subplots(1, 1)
            ax.hist(self.extra_sigma)
            title = 'Posterior distribution for extra white noise $s$ \n %s' % estimate
            ax.set(xlabel='extra sigma (m/s)', ylabel='posterior samples',
                   title=title)

        if self.save_plots:
            filename = 'kima-showresults-fig7.3.png'
            print('saving in', filename)
            fig.savefig(filename)

    def hist_correlations(self):
        """ Plot the histogram of the posterior for the activity correlations """
        if not self.indcorrel:
            print(
                'Model has no activity correlations! hist_correlations() doing nothing...'
            )
            return

        # units = ' (m/s)' if self.units=='ms' else ' (km/s)'
        # estimate = percentile68_ranges_latex(self.offset) + units

        n = len(self.activity_indicators)
        fig, axs = plt.subplots(n, 1, constrained_layout=True)

        for i, ax in enumerate(np.ravel(axs)):
            estimate = percentile68_ranges_latex(self.betas[:, i])
            estimate = '$c_{%s}$ = %s' % (self.activity_indicators[i],
                                          estimate)
            ax.hist(self.betas[:, i], label=estimate)
            ax.set(ylabel='posterior samples',
                   xlabel='$c_{%s}$' % self.activity_indicators[i])
            leg = ax.legend(frameon=False)
            leg.legendHandles[0].set_visible(False)

        title = 'Posterior distribution for activity correlations'
        fig.suptitle(title)

        if self.save_plots:
            filename = 'kima-showresults-fig7.4.png'
            print('saving in', filename)
            fig.savefig(filename)

    def hist_trend(self, per_year=True):
        """ 
        Plot the histogram of the posterior for the slope of a linear trend
        """
        if not self.trend:
            print('Model has no trend! hist_trend() doing nothing...')
            return

        units = ' (m/s/yr)'  # if self.units=='ms' else ' (km/s)'

        trend = self.trendpars.copy()

        if per_year:  # transfrom from m/s/day to m/s/yr
            trend *= 365.25

        estimate = percentile68_ranges_latex(trend) + units

        fig, ax = plt.subplots(1, 1)
        ax.hist(trend.ravel())
        title = 'Posterior distribution for slope \n %s' % estimate
        ax.set(xlabel='slope' + units, ylabel='posterior samples', title=title)

        if self.save_plots:
            filename = 'kima-showresults-fig7.5.png'
            print('saving in', filename)
            fig.savefig(filename)

    def hist_MA(self):
        """ Plot the histogram of the posterior for the MA parameters """
        if not self.MAmodel:
            print('Model has no MA! hist_MA() doing nothing...')
            return

        # units = ' (m/s/day)' # if self.units=='ms' else ' (km/s)'
        # estimate = percentile68_ranges_latex(self.trendpars) + units

        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
        ax1.hist(self.MA[:, 0])
        ax2.hist(self.MA[:, 1])
        title = 'Posterior distribution for MA parameters'
        fig.suptitle(title)
        ax1.set(xlabel=r'$\sigma$ MA [m/s]', ylabel='posterior samples')
        ax2.set(xlabel=r'$\tau$ MA [days]', ylabel='posterior samples')

        if self.save_plots:
            filename = 'kima-showresults-fig7.6.png'
            print('saving in', filename)
            fig.savefig(filename)
