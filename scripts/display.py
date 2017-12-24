import sys
import re
import os
pathjoin = os.path.join

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

try:
    from fast_histogram import histogram1d, histogram2d
    fast_histogram_available = True
except ImportError:
    fast_histogram_available = False

try:
    from astroML.plotting import hist_tools
    hist_tools_available = True
except ImportError:
    hist_tools_available = False

from keplerian import keplerian

import corner

colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
mjup2mearth = 317.8284065946748


def apply_argsort(arr1, arr2, axis=-1):
    """
    Apply arr1.argsort() on arr2, along `axis`.
    """
    # check matching shapes
    assert arr1.shape == arr2.shape, "Shapes don't match!"

    i = list(np.ogrid[[slice(x) for x in arr1.shape]])
    i[axis] = arr1.argsort(axis)
    return arr2[i]

def percentile68_ranges(a):
    lp, median, up = np.percentile(a, [16, 50, 84])
    return (median, up-median, median-lp)

def percentile68_ranges_latex(a):
    lp, median, up = np.percentile(a, [16, 50, 84])
    return r'$%.2f ^{+%.2f} _{-%.2f}$' % (median, up-median, median-lp)

# def get_aliases(Preal):
#     fs = np.array([0.0027381631, 1.0, 1.0027]) #, 0.018472000025212765])
#     return np.array([abs(1 / (1./Preal + i*fs)) for i in range(-2, 2)]).T


def get_planet_mass(P, K, e, star_mass=1.0, full_output=False, verbose=False):
    if verbose: print('Using star mass = %s solar mass' % star_mass)
    # print 3.5e-2 * K * (P/365.)**(1./3) * star_mass**(2./3)
    # print 9.077e-3 * star_mass**(2./3) * (P/(2*np.pi))**(1./3) * K * np.sqrt(1-e**2)

    if isinstance(P, float):
        assert isinstance(star_mass, float)
        m_mj = 4.919e-3 * star_mass**(2./3) * P**(1./3) * K * np.sqrt(1-e**2)
        m_me = m_mj * mjup2mearth
        return m_mj, m_me
    else:
      if isinstance(star_mass, tuple) or isinstance(star_mass, list):
        star_mass = star_mass[0] + star_mass[1]*np.random.randn(P.size)
      m_mj = 4.919e-3 * star_mass**(2./3) * P**(1./3) * K * np.sqrt(1-e**2)
      m_me = m_mj * mjup2mearth
      
      if full_output:
        # return map(percentile68_ranges, [m_mj, m_me]), m_mj
        return m_mj.mean(), m_mj.std(), m_mj
      else:
        # return map(percentile68_ranges, [m_mj, m_me])
        return (m_mj.mean(), m_mj.std(), m_me.mean(), m_me.std())


class DisplayResults(object):
    def __init__(self, options, data_file=None, 
                 fiber_offset=None, hyperpriors=None,
                 posterior_samples_file='posterior_sample.txt'):

        self.options = options
        debug = 'debug' in self.options

        pwd = os.getcwd()
        path_to_this_file = os.path.abspath(__file__)
        top_level = os.path.dirname(os.path.dirname(path_to_this_file))

        if debug:
            print()
            print('running on:', pwd)
            print('top_level:', top_level)
            print()

        def get_skip(line):
            load_args = re.findall(r'\((.*?)\)', line, re.DOTALL)[1]
            load_args = load_args.split(',')
            if len(load_args) == 3:
                # use gave 'skip' option
                return int(load_args[2])
            else:
                # default is skip=2
                return 2

        # find datafile in the compiled model
        self.data_skip = 2 # by default
        if data_file is None:
            try:
                # either in an example directory
                with open(pathjoin(pwd, 'kima_setup.cpp')) as f:
                    for line in f.readlines():
                        if 'datafile = ' in line and '/*' not in line: 
                            data_file = re.findall('"(.*?)"', line, re.DOTALL)[0]

                        if 'get_instance().load' in line:
                            self.data_skip = get_skip(line)

            except IOError:
                # or in the main kima directory
                with open(pathjoin(top_level, 'src', 'main.cpp')) as f:
                    for line in f.readlines():
                        if 'get_instance().load' in line and '/*' not in line:
                            break
                self.data_skip = get_skip(line)
                data_file = re.findall('"(.*?)"', line, re.DOTALL)[0]
                data_file = pathjoin(top_level, data_file)

        print('Loading data file %s' % data_file)
        self.data_file = data_file
        if debug:
            print('--- skipping first %d rows of data file' % self.data_skip)

        self.data = np.loadtxt(self.data_file, 
                               skiprows=self.data_skip, usecols=(0,1,2))

        # to m/s
        self.data[:, 1] *= 1e3
        self.data[:, 2] *= 1e3

        self.posterior_sample = np.atleast_2d(np.loadtxt(posterior_samples_file))
        try:
            self.sample = np.loadtxt('sample.txt')
        except IOError:
            self.sample = None

        try:
            spl = os.path.splitext(posterior_samples_file)
            lnlikes_filename = '_lnlikelihoods'.join(spl)

            self.posterior_sample_lnlikes = np.atleast_2d(np.loadtxt(lnlikes_filename))
            self.max_likelihood_index = np.argmax(self.posterior_sample_lnlikes)
        
        except IOError:
            print('Sample likelihoods not available!!! This is bad!')


        start_parameters = 0
        self.extra_sigma = self.posterior_sample[:, start_parameters]

        # find trend in the compiled model
        try:
            with open(pathjoin(pwd, 'kima_setup.cpp')) as f:
                self.trend = 'bool trend = true' in f.read()
        except IOError:
            with open(pathjoin(top_level, 'src', 'main.cpp')) as f:
                self.trend = 'bool trend = true' in f.read()
        
        if debug: 
            print('trend:', self.trend)


        if self.trend:
            n_trend = 1
            i1 = start_parameters + 1
            i2 = start_parameters + n_trend + 1
            self.trendpars = self.posterior_sample[:, i1:i2]
        else:
            n_trend = 0



        # find fiber offset in the compiled model
        if fiber_offset is None:
            try:
                with open(pathjoin(pwd, 'kima_setup.cpp')) as f:
                    self.fiber_offset = \
                        'bool obs_after_HARPS_fibers = true' in f.read()
            except IOError:
                with open(pathjoin(top_level, 'src', 'main.cpp')) as f:
                    self.fiber_offset = \
                        'bool obs_after_HARPS_fibers = true' in f.read()
        else:
            self.fiber_offset = fiber_offset

        if debug: 
            print('obs_after_fibers:', self.fiber_offset)

        if self.fiber_offset:
            n_offsets = 1
            offset_index = start_parameters+n_hyperparameters+n_offsets
            self.offset = self.posterior_sample[:, offset_index]
        else:
            n_offsets = 0


        # find GP in the compiled model
        try:
            with open(pathjoin(pwd, 'kima_setup.cpp')) as f:
                self.GPmodel = 'bool GP = true' in f.read()
        except IOError:
            with open(pathjoin(top_level, 'src', 'main.cpp')) as f:
                self.GPmodel = 'bool GP = true' in f.read()
        
        if debug:
            print('GP model:', self.GPmodel)

        if self.GPmodel:
            n_hyperparameters = 4
            for i in range(n_hyperparameters):
                name = 'eta' + str(i+1)
                ind = start_parameters + n_trend + n_offsets + 1 + i
                setattr(self, name, self.posterior_sample[:, ind])
        else:
            n_hyperparameters = 0



        start_objects_print = start_parameters + n_offsets + \
                              n_trend + n_hyperparameters + 1
        # how many parameters per component
        self.n_dimensions = int(self.posterior_sample[0, start_objects_print])
        # maximum number of components
        self.max_components = int(self.posterior_sample[0, start_objects_print+1])

        # find hyperpriors in the compiled model
        if hyperpriors is None:
            try:
                with open(pathjoin(pwd, 'kima_setup.cpp')) as f:
                    self.hyperpriors = \
                        'bool hyperpriors = true' in f.read()
            except IOError:
                with open(pathjoin(top_level, 'src', 'main.cpp')) as f:
                    self.hyperpriors = \
                        'bool hyperpriors = true' in f.read()
        else:
            self.hyperpriors = hyperpriors
        
        # number of hyperparameters (muP, wP, muK)
        n_dist_print = 3 if self.hyperpriors else 0
        # if hyperpriors, then the period is sampled in log
        self.log_period = self.hyperpriors

        # the column with the number of planets in each sample
        self.index_component = start_objects_print + 1 + n_dist_print + 1

        # build the marginal posteriors for planet parameters
        self.get_marginals()

        if '1' in options:
            self.make_plot1()
        if '2' in options:
            self.make_plot2()
        if '3' in options:
            self.make_plot3()
        if '4' in options:
            self.make_plot4()
        if '5' in options:
            self.make_plot5()

        if '10' in options:
            self.corner_planet_parameters()


    def get_marginals(self):
        """ 
        Get the marginal posteriors from the posterior_sample matrix.
        They go into self.T, self.A, self.E, etc
        """

        max_components = self.max_components
        index_component = self.index_component

        # periods
        i1 = 0*max_components + index_component + 1
        i2 = 0*max_components + index_component + max_components + 1
        s = np.s_[i1 : i2]
        self.T = self.posterior_sample[:,s]
        self.Tall = np.copy(self.T)

        # amplitudes
        i1 = 1*max_components + index_component + 1
        i2 = 1*max_components + index_component + max_components + 1
        s = np.s_[i1 : i2]
        self.A = self.posterior_sample[:,s]
        self.Aall = np.copy(self.A)

        # phases
        i1 = 2*max_components + index_component + 1
        i2 = 2*max_components + index_component + max_components + 1
        s = np.s_[i1 : i2]
        self.phi = self.posterior_sample[:,s]
        self.phiall = np.copy(self.phi)

        # eccentricities
        i1 = 3*max_components + index_component + 1
        i2 = 3*max_components + index_component + max_components + 1
        s = np.s_[i1 : i2]
        self.E = self.posterior_sample[:,s]
        self.Eall = np.copy(self.E)

        # omegas
        i1 = 4*max_components + index_component + 1
        i2 = 4*max_components + index_component + max_components + 1
        s = np.s_[i1 : i2]
        self.Omega = self.posterior_sample[:,s]
        self.Omegaall = np.copy(self.Omega)

        # times of periastron
        self.T0 = self.data[0,0] - (self.T*self.phi)/(2.*np.pi)
        self.T0all = np.copy(self.T0)


        which = self.T != 0
        self.T = self.T[which].flatten()
        self.A = self.A[which].flatten()
        self.E = self.E[which].flatten()


    def get_medians(self):
        """ return the median values of all the parameters """
        if self.posterior_sample.shape[0] % 2 == 0:
            print('Median is not a solution because number of samples is even!!')

        self.medians = np.median(self.posterior_sample, axis=0)
        self.means = np.mean(self.posterior_sample, axis=0)
        return self.medians, self.means


    def get_posterior_statistics(self, N=None):
        """ print the maximum likelihood estimate of the parameters and the posterior median """
        N = 2
        if N is None:
            i = self.posterior_sample[:, -1].argmax()
            pars = self.posterior_sample[i, :]
        else:
            mask = self.posterior_sample[:, self.index_component]==N
            self.mask = mask
            i = self.posterior_sample[mask, -1].argmax()
            pars = self.posterior_sample[mask][i, :]

        print('maximum likelihood ')
        print(pars[:5])
        print(pars[pars != 0])

        sort_periods = False
        if sort_periods:
            # sort the periods (this works quite well with 2 planets...)
            periods = np.exp(self.Tall)
            amplitudes = self.Aall
            eccentricities = self.Eall
            sorted_periods = apply_argsort(periods, periods, axis=1)
            sorted_amplitudes = apply_argsort(periods, amplitudes, axis=1)
            sorted_eccentricities = apply_argsort(periods, eccentricities, axis=1)

            P1, P2 = sorted_periods.T
            K1, K2 = sorted_amplitudes.T
            e1, e2 = sorted_eccentricities.T
            assert P1.shape == P2.shape

        if N == 2:
            periods = np.exp(self.Tall[mask,:2])
            amplitudes = self.Aall[mask, :2]
            eccentricities = self.Eall[mask, :2]

            sorted_periods = apply_argsort(periods, periods, axis=1)
            sorted_amplitudes = apply_argsort(periods, amplitudes, axis=1)
            sorted_eccentricities = apply_argsort(periods, eccentricities, axis=1)

            P1, P2 = sorted_periods.T
            K1, K2 = sorted_amplitudes.T
            e1, e2 = sorted_eccentricities.T
        else:
            pass

        print()
        print('medians:')
        print()

        a = '$%7.5f\,^{+\,%7.5f}_{-\,%7.5f}$' % percentile68_ranges(P1)
        b = ' & $%4.3f$' % P1.std()
        print('%-40s' % a, b)

        a, b = '$%3.2f\,^{+\,%3.2f}_{-\,%3.2f}$' % percentile68_ranges(K1), ' & $%4.3f$' % K1.std()
        print('%-40s' % a, b)
        
        a, b = '$%4.3f\,^{+\,%4.3f}_{-\,%4.3f}$' % percentile68_ranges(e1), ' & $%4.3f$' % e1.std()
        print('%-40s' % a, b)

        a, b = '$%7.5f\,^{+\,%7.5f}_{-\,%7.5f}$' % percentile68_ranges(P2), ' & $%4.3f$' % P2.std()
        print('%-40s' % a, b)

        a, b = '$%3.2f\,^{+\,%3.2f}_{-\,%3.2f}$' % percentile68_ranges(K2), ' & $%4.3f$' % K2.std()
        print('%-40s' % a, b)

        a, b = '$%4.3f\,^{+\,%4.3f}_{-\,%4.3f}$' % percentile68_ranges(e2), ' & $%4.3f$' % e2.std()
        print('%-40s' % a, b)



        ############################################################

        mjup2mearth  = 317.828
        star_mass = 0.913


        m_mj = 4.919e-3 * star_mass**(2./3) * P1**(1./3) * K1 * np.sqrt(1-e1**2)
        m_me = m_mj * mjup2mearth
        # a = ((system.star_mass + m_me*mearth2msun)/(m_me*mearth2msun)) * sqrt(1.-ecc**2) * K * (P*mean_sidereal_day/(2*np.pi)) / au2m

        print('b - $%4.2f\,^{+\,%4.2f}_{-\,%4.2f}$ [MEarth]' % percentile68_ranges(m_me))
        # print '%8s %11.4f +- %7.4f [AU]' % ('a', a.n, a.s)



        m_mj = 4.919e-3 * star_mass**(2./3) * P2**(1./3) * K2 * np.sqrt(1-e2**2)
        m_me = m_mj * mjup2mearth
        # a = ((system.star_mass + m_me*mearth2msun)/(m_me*mearth2msun)) * sqrt(1.-ecc**2) * K * (P*mean_sidereal_day/(2*np.pi)) / au2m

        print('c - $%4.2f\,^{+\,%4.2f}_{-\,%4.2f}$ [MEarth]' % percentile68_ranges(m_me))
        # print '%8s %11.4f +- %7.4f [AU]' % ('a', a.n, a.s)



    def make_plot1(self):
        """ Plot the histogram of the posterior for Np """
        plt.figure()
        n, bins, _ = plt.hist(self.posterior_sample[:, self.index_component], 100)
        plt.xlabel('Number of Planets')
        plt.ylabel('Number of Posterior Samples')
        plt.xlim([-0.5, self.max_components+.5])

        nn = n[np.nonzero(n)]
        print('probability ratios: ', nn.flat[1:] / nn.flat[:-1])

        plt.show()


    def make_plot2(self, bins=None):
        """ 
        Plot the histogram of the posterior for orbital period P.
        Optionally provide the histogram bins.
        """

        if self.max_components == 0:
            print('Current model does not have planets, doing nothing...')
            return

        if self.log_period:
            T = np.exp(self.T)
            print('exponentiating period!')
        else:
            T = self.T
        
        plt.figure()

        # mark 1 year and 0.5 year
        year = 365.25
        plt.axvline(x=year, ls='--', color='r', lw=3, alpha=0.6)
        plt.axvline(x=year/2., ls='--', color='r', lw=3, alpha=0.6)
        # plt.axvline(x=year/3., ls='--', color='r', lw=3, alpha=0.6)

        # mark the timespan of the data
        plt.axvline(x=self.data[:,0].ptp(), ls='--', color='b', lw=4, alpha=0.5)

        # by default, 100 bins in log between 0.1 and 1e7
        if bins is None:
            bins = 10 ** np.linspace(np.log10(1e-1), np.log10(1e7), 100)

        plt.hist(T, bins=bins, alpha=0.5)

        plt.xscale("log")
        plt.xlabel(r'(Period/days)')
        plt.ylabel('Number of Posterior Samples')
        plt.show()


    def make_plot3(self, points=True):
        """
        Plot the 2d histograms of the posteriors for semi-amplitude 
        and orbital period and eccentricity and orbital period.
        If `points` is True, plot each posterior sample, else plot hexbins
        """
        if 'hexbin' in self.options:
            points = False

        if self.log_period:
            T = np.exp(self.T)
            print('exponentiating period!')
        else:
            T = self.T
        A, E = self.A, self.E


        fig, axs = plt.subplots(2, 1, sharex=True)
        ax1, ax2 = axs

        if points:
            ax1.loglog(T, A, '.', markersize=1)
        else:
            ax1.hexbin(T, A, gridsize=50, 
                       bins='log', xscale='log', yscale='log',
                       cmap=plt.get_cmap('afmhot_r'))


        if points:
            ax2.semilogx(T, E, 'b.', markersize=2)
        else:
            ax2.hexbin(T, E, gridsize=50, 
                       bins='log', xscale='log',
                       cmap=plt.get_cmap('afmhot_r'))
        
        ax1.set_ylabel(r'Semi-amplitude (m/s)')
        ax2.set_ylabel('Eccentricity')
        ax2.set_xlabel(r'(Period/days)')
        ax2.set_ylim(0, 1)
        ax2.set_xlim([0.1, 1e7])

        plt.show()



    def make_plot31(self, star_mass=1.0, trend=0., points=True):
        from astropy import units as u

        T = np.exp(self.T) if log_period else self.T
        A, E = self.A, self.E

        T = T * u.day
        A = A * u.meter/u.second

        Mmj = get_planet_mass(T.value, A.value, E, star_mass=star_mass, full_output=True)
        Mmj = Mmj[2] * u.jupiterMass
        # aliases1 = get_aliases(0.85359165)
        # aliases2 = get_aliases(3.691)
        # aliases3 = get_aliases(9.03580275887)

        fig = plt.figure()

        ax1 = fig.add_subplot(2,1,1)
        #plot(truth[1009:1009 + int(truth[1008])]/log(10.), log10(truth[1018:1018 + int(truth[1008])]), 'ro', markersize=7)
        #hold(True)
        if points:
            # ax1.loglog(T, Mmj, '.', markersize=1)
            ax1.scatter(T, Mmj, c=E, s=5)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
        else:
            ax1.hexbin(T[::100], A[::100], gridsize=50, bins='log', xscale='log', yscale='log',
                       cmap=plt.get_cmap('afmhot_r'))
        # data = np.vstack([np.exp(T[::100]), A[::100]]).T
        # sns.jointplot(x=np.exp(T[::100]), y=A[::100], kind="hex", color="k");
        # sns.kdeplot(data=np.exp(T[::100]), data2=A[::100], bw=[0.1, 1], shade=True, ax=ax1)

        ax1.set_ylabel(r'Mass (Mjup)')

        # mu = self.posterior_sample[:, self.index_component-1]
        # ax = fig.add_subplot(2,2,2, sharey=ax1)
        # ax.hist(mu, bins=30, orientation="horizontal")
        # ax.set_ylim(ax1.get_ylim())
        # ax.set_yscale('log')

        ax2 = fig.add_subplot(2,1,2, sharex=ax1)
        #plot(truth[1009:1009 + int(truth[1008])]/log(10.), truth[1038:1038 + int(truth[1008])], 'ro', markersize=7)
        #hold(True)
        if points:
            ax2.semilogx(T, E, 'b.', markersize=2)
        else:
            ax2.hexbin(T[::100], E[::100], gridsize=50, bins='log', xscale='log',
                       cmap=plt.get_cmap('afmhot_r'))
        # ax.axvline(x=0.85359165, color='r')
        # ax.axvline(x=3.691, color='r')
        # ax.vlines([3.9359312722691815, 3.8939651463077287, 3.8528844910117557, 3.8126615742655399, 3.7732698100471724, 3.7346836998277957, 3.6968787775300003, 3.6598315577957377, 3.6235194873338745, 3.5879208991356037, 3.5530149693624242, 3.5187816767264146], ymin=0, ymax=1)
        # ax.vlines(aliases1, ymin=0, ymax=1, color='g', alpha=0.4)
        # ax.vlines(aliases2, ymin=0, ymax=1, color='y', alpha=0.4)
        # ax.vlines(aliases3, ymin=0, ymax=1, color='c', alpha=0.4)
        
        ax2.set_xlim([10, 1e7])
        ax2.set_xlabel(r'(Period/days)')
        ax2.set_ylabel('Eccentricity')

        for ax in (ax1,ax2):
            ax.axvline(x=self.data[:,0].ptp(), ymin=0, ymax=1, color='k')
            ax.axvline(x=2*self.data[:,0].ptp(), ymin=0, ymax=1, color='k')

        # minimum mass from Feng et al 2015
        Dt = self.data[:,0].ptp() / 365.25
        Mmin = 0.0164 * (Dt)**(4/3.) * \
               (np.abs(trend)) * \
               (star_mass)**(2/3.)
        print(Mmin * u.jupiterMass)
        ax1.axhline(y=Mmin, xmin=0, xmax=1, color='k')

        Mmin = lambda P: (1/28.4) * P**(1/3.) * \
                         (star_mass)**(2/3.) * \
                         (np.abs(trend)*Dt/2.)
        print(Mmin(Dt))
        PP = np.linspace(2*Dt*365.25, 100*Dt*365.25)
        ax1.plot(PP, Mmin(PP/365.25), 'g')
        # ax1.axhline(y=Mmin, xmin=0, xmax=1, color='k')

        plt.show()



    def make_plot4(self):
        """ Plot histograms for the GP hyperparameters """
        if not self.GPmodel:
            print('Current model does not have GP, doing nothing...')
            return

        available_etas = [v for v in dir(self) if v.startswith('eta')]
        
        fig, axes = plt.subplots(2, len(available_etas)/2)
        for i, eta in enumerate(available_etas):
            ax = np.ravel(axes)[i]
            ax.hist(getattr(self, eta), bins=40)
            ax.set_xlabel(eta)
        plt.tight_layout()
        plt.show()


    def make_plot5(self, show=True, save=False):
        """ Corner plot for the GP hyperparameters """

        if not self.GPmodel:
            print('Current model does not have GP, doing nothing...')
            return

        self.pmin = 10.
        self.pmax = 40.

        available_etas = [v for v in dir(self) if v.startswith('eta')]
        labels = ['$s$'] + ['$\eta_%d$' % (i+1) for i in range(len(available_etas))]

        ### color code by number of planets
        # self.corner1 = None
        # for N in range(6)[::-1]:
        #     mask = self.posterior_sample[:, self.index_component] == N
        #     if mask.any():
        #         self.post_samples = np.vstack((self.extra_sigma, self.eta1, self.eta2, self.eta3, self.eta4)).T
        #         self.post_samples = self.post_samples[mask, :]
        #         # self.post_samples = np.vstack((self.extra_sigma, self.eta1, self.eta2, self.eta3, self.eta4, self.eta5)).T
        #         print self.post_samples.shape
        #         # print (self.pmin, self.pmax)
        #         # labels = ['$\sigma_{extra}$', '$\eta_1$', '$\eta_2$', '$\eta_3$', '$\eta_4$', '$\eta_5$']
                
        #         self.corner1 = corner.corner(self.post_samples, fig=self.corner1, labels=labels, show_titles=True,
        #                                      plot_contours=False, plot_datapoints=True, plot_density=False,
        #                                      # fill_contours=True, smooth=True,
        #                                      # contourf_kwargs={'cmap':plt.get_cmap('afmhot'), 'colors':None},
        #                                      hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
        #                                      hist_kwargs={'normed':True, 'color':colors[N]},
        #                                      range=[1., 1., 1., (self.pmin, self.pmax), 1],
        #                                      shared_axis=True, data_kwargs={'alpha':1, 'color':colors[N]},
        #                                      )

        #         ax = self.corner1.axes[3]
        #         ax.plot([2,2.1], color=colors[N], lw=3)
        #     else:
        #         print 'Skipping N=%d, no posterior samples...' % N
        # ax.legend([r'$N_p=%d$'%N for N in range(6)[::-1]])


        ### all Np together
        variables = [self.extra_sigma]
        for eta in available_etas:
            variables.append(getattr(self, eta))

        self.post_samples = np.vstack(variables).T
        # print self.post_samples.shape

        ranges = [1.]*(len(available_etas)+1)
        ranges[3] = (self.pmin, self.pmax)

        c = corner.corner        
        self.corner1 = c(self.post_samples, labels=labels, show_titles=True,
                         plot_contours=False, plot_datapoints=True, plot_density=False,
                         # fill_contours=True, smooth=True,
                         # contourf_kwargs={'cmap':plt.get_cmap('afmhot'), 'colors':None},
                         hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
                         hist_kwargs={'normed':True},
                         range=ranges, shared_axis=True, data_kwargs={'alpha':1},
                         )

        if show:
            plt.show()
        
        if save:
            self.corner1.savefig(save)


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
        ind_sort_P = np.arange(np.shape(p)[0])[:,np.newaxis], np.argsort(p)
        for i,j in zip(range(0, n, mc), range(mc, n+mc, mc)):
            samples[:,i:j] = self.planet_samples[:,i:j][ind_sort_P]

        return samples

    def apply_cuts_period(self, samples, pmin=None, pmax=None, return_mask=False):
        """ apply cuts in orbital period """
        too_low_periods = np.zeros_like(samples[:,0], dtype=bool)
        too_high_periods = np.zeros_like(samples[:,0], dtype=bool)

        if pmin is not None:
            too_low_periods = samples[:,0] < pmin
            samples = samples[~too_low_periods, :]
            
        if pmax is not None:
            too_high_periods = samples[:,1] > pmax
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
                    this_planet_bins.append(bw(sample, return_bins=True)[1].size)
                bins.append(this_planet_bins)
            else:
                bins.append(None)


        # set the parameter ranges to include everythinh
        def r(x, over=0.2):
            return x.min() - over*x.ptp(), x.max() + over*x.ptp()

        ranges = []
        for i in range(self.n_dimensions):
            i1, i2 = self.max_components*i, self.max_components*(i+1)
            ranges.append( r(samples[:, i1:i2]) )

        # 
        c = corner.corner
        fig = None
        colors = plt.rcParams["axes.prop_cycle"]

        for i, (datum, colorcycle) in enumerate(zip(data, colors)):
            fig = c(datum, fig=fig, labels=labels, show_titles=len(data)==1,
                    plot_contours=False, plot_datapoints=True, plot_density=False,
                    bins=bins[i], range=ranges, color=colorcycle['color'],
                    # fill_contours=True, smooth=True,
                    # contourf_kwargs={'cmap':plt.get_cmap('afmhot'), 'colors':None},
                    #hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
                    hist_kwargs={'normed':True},
                    # range=[1., 1., (0, 2*np.pi), (0., 1.), (0, 2*np.pi)],
                    data_kwargs={'alpha':1, 'ms':3, 'color':colorcycle['color']},
                    )

        plt.show()


    def plot_random_planets(self, ncurves=50, over=0.1, 
                            pmin=None, pmax=None, show_vsys=False):
        """
        Display the RV data together with curves from the posterior predictive.
        A total of `ncurves` random samples are chosen,
        and the Keplerian curves are calculated covering 100 + `over`%
        of the timespan of the data.
        """
        samples = self.get_sorted_planet_samples()
        samples, mask = \
            self.apply_cuts_period(samples, pmin, pmax, return_mask=True)
        # print mask

        t = self.data[:,0].copy()
        tt = np.linspace(t[0]-over*t.ptp(), t[-1]+over*t.ptp(), 
                         10000+int(100*over))

        y = self.data[:,1].copy()
        yerr = self.data[:,2].copy()

        # select random `ncurves` indices 
        # from the (sorted, period-cut) posterior samples
        ii = np.random.randint(samples.shape[0], size=ncurves)

        fig, ax = plt.subplots(1,1)

        ## plot the Keplerian curves
        for i in ii:
            v = np.zeros_like(tt)
            pars = samples[i, :].copy()
            nplanets = pars.size / self.n_dimensions
            for j in range(nplanets):
                P = pars[j + 0*self.max_components]
                K = pars[j + 1*self.max_components]
                phi = pars[j + 2*self.max_components]
                t0 = t[0] - (P*phi)/(2.*np.pi)
                ecc = pars[j + 3*self.max_components]
                w = pars[j + 4*self.max_components]
                print(P)
                v += keplerian(tt, P, K, ecc, w, t0, 0.)

            vsys = self.posterior_sample[mask][i, -1]
            v += vsys
            ax.plot(tt, v, alpha=0.2, color='k')
            if show_vsys:
                ax.plot(t, vsys*np.ones_like(t), alpha=0.2, color='r', ls='--')


        ## plot the data
        if self.fiber_offset:
            mask = t < 57170
            ax.errorbar(t[mask], y[mask], yerr[mask], fmt='o')
            yshift = np.vstack([y[~mask], y[~mask]-self.offset.mean()])
            for i, ti in enumerate(t[~mask]):
                ax.errorbar(ti, yshift[0,i], fmt='o', color='m', alpha=0.3)
                ax.errorbar(ti, yshift[1,i], yerr[~mask][i], fmt='o', color='r')
        else:
            ax.errorbar(t, y, yerr, fmt='o')

        ax.set(xlabel='Time [days]', ylabel='RV [m/s]')
        plt.show()


    def hist_offset(self):
        """ Plot the histogram of the posterior for the fiber offset """
        if not self.fiber_offset:
            print('Model has no fiber offset! doing nothing...')
            return

        fig, ax = plt.subplots(1,1)
        if hist_tools_available:
            bw = hist_tools.freedman_bin_width
            _, bins = bw(self.offset, return_bins=True)
        else:
            bins = None

        ax.hist(self.offset, bins=bins)
        plt.show()



if __name__ == '__main__':
    print('Arguments: ', sys.argv[1:])
    options = ' '.join(sys.argv[1:])

    res = DisplayResults(options)
    globals()['res'] = res

