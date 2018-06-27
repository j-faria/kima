import re, os, sys
pathjoin = os.path.join

try:
    import configparser
except ImportError:
    # Python 2
    import ConfigParser as configparser

from .keplerian import keplerian
from .utils import need_model_setup, get_planet_mass, get_planet_semimajor_axis,\
                   percentile68_ranges, percentile68_ranges_latex

import matplotlib.pyplot as plt
import numpy as np
import corner

try:
    from astroML.plotting import hist_tools
    hist_tools_available = True
except ImportError:
    hist_tools_available = False

colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

class KimaResults(object):
    def __init__(self, options, data_file=None, 
                 fiber_offset=None, hyperpriors=None, trend=None, GPmodel=None,
                 posterior_samples_file='posterior_sample.txt'):

        self.options = options
        debug = False # 'debug' in options

        pwd = os.getcwd()
        path_to_this_file = os.path.abspath(__file__)
        top_level = os.path.dirname(os.path.dirname(path_to_this_file))

        if debug:
            print()
            print('running on:', pwd)
            print('top_level:', top_level)
            print()

        setup = configparser.ConfigParser()
        s = setup.read('kima_model_setup.txt')
        if len(s) == 0:
            need_model_setup()
            sys.exit(0)

        if data_file is None:
            data_file = setup['kima']['file']
        print('Loading data file %s' % data_file)
        self.data_file = data_file

        self.data_skip = int(setup['kima']['skip'])
        self.units = setup['kima']['units']
        
        if debug:
            print('--- skipping first %d rows of data file' % self.data_skip)

        self.data = np.loadtxt(self.data_file, 
                               skiprows=self.data_skip, usecols=(0,1,2))

        # to m/s
        if self.units == 'kms':
            self.data[:, 1] *= 1e3
            self.data[:, 2] *= 1e3

        self.posterior_sample = np.atleast_2d(np.loadtxt(posterior_samples_file))
        try:
            self.sample = np.loadtxt('sample.txt')
        except IOError:
            self.sample = None


        start_parameters = 0
        self.extra_sigma = self.posterior_sample[:, start_parameters]

        # find trend in the compiled model
        if trend is None:
            self.trend = setup['kima']['trend'] == 'true'
        else:
            self.trend = trend

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
            self.fiber_offset = setup['kima']['obs_after_HARPS_fibers'] == 'true'
        else:
            self.fiber_offset = fiber_offset

        if debug: 
            print('obs_after_fibers:', self.fiber_offset)

        if self.fiber_offset:
            n_offsets = 1
            offset_index = start_parameters+n_offsets+n_trend
            self.offset = self.posterior_sample[:, offset_index]
        else:
            n_offsets = 0


        # find GP in the compiled model
        if GPmodel is None:
            self.GPmodel = setup['kima']['GP'] == 'true'
        else:
            self.GPmodel = GPmodel

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
            self.hyperpriors = setup['kima']['hyperpriors'] == 'true'
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

        allowed_options = {'1': [self.make_plot1, {}],
                           '2': [self.make_plot2, {}],
                           '3': [self.make_plot3, {}],
                           '4': [self.make_plot4, {}],
                           '5': [self.make_plot5, {}],
                           '6': [self.plot_random_planets, 
                                    {'show_vsys':True, 'show_trend':True}],
                           '7': [(self.hist_offset,
                                  self.hist_vsys,
                                  self.hist_extra_sigma), {}],
                          }

        for item in allowed_options.items():
            if item[0] in options:
                methods = item[1][0]
                kwargs = item[1][1]
                if isinstance(methods, tuple):
                    [m() for m in methods]
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
        _, ax = plt.subplots(1,1)
        # n, _, _ = plt.hist(self.posterior_sample[:, self.index_component], 100)
        
        bins = np.arange(self.max_components+2)
        nplanets = self.posterior_sample[:, self.index_component]
        n, _ = np.histogram(nplanets, bins=bins)
        ax.bar(bins[:-1], n)

        ax.set(xlabel='Number of Planets',
               ylabel='Number of Posterior Samples',
               xlim=[-0.5, self.max_components+.5],
               xticks=np.arange(self.max_components+1),
               title='Posterior distribution for $N_p$'
              )

        nn = n[np.nonzero(n)]
        print('Np probability ratios: ', nn.flat[1:] / nn.flat[:-1])


    def make_plot2(self, bins=None):
        """ 
        Plot the histogram of the posterior for orbital period P.
        Optionally provide the histogram bins.
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

        # mark 1 year and 0.5 year
        year = 365.25
        ax.axvline(x=year, ls='--', color='r', lw=3, alpha=0.6)
        # ax.axvline(x=year/2., ls='--', color='r', lw=3, alpha=0.6)
        # plt.axvline(x=year/3., ls='--', color='r', lw=3, alpha=0.6)

        # mark the timespan of the data
        ax.axvline(x=self.data[:,0].ptp(), ls='--', color='b', lw=3, alpha=0.5)

        # by default, 100 bins in log between 0.1 and 1e7
        if bins is None:
            bins = 10 ** np.linspace(np.log10(1e-1), np.log10(1e7), 100)

        ax.hist(T, bins=bins, alpha=0.5)

        ax.legend(['1 year', 'timespan'])
        ax.set(xscale="log",
               xlabel=r'(Period/days)',
               ylabel='Number of Posterior Samples',
               title='Posterior distribution for the orbital period(s)')
        # plt.show()


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


        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        if points:
            ax1.loglog(T, A, '.', markersize=2)
        else:
            ax1.hexbin(T, A, gridsize=50, 
                       bins='log', xscale='log', yscale='log',
                       cmap=plt.get_cmap('afmhot_r'))


        if points:
            ax2.semilogx(T, E, '.', markersize=2)
        else:
            ax2.hexbin(T, E, gridsize=50, bins='log', xscale='log',
                       cmap=plt.get_cmap('afmhot_r'))
        
        ax1.set(ylabel='Semi-amplitude [m/s]',
                title='Joint posterior semi-amplitude $-$ orbital period')
        ax2.set(ylabel='Eccentricity',
                xlabel='Period [days]',
                title='Joint posterior eccentricity $-$ orbital period',
                ylim=[0, 1],
                xlim=[0.1, 1e7])


    def make_plot4(self):
        """ Plot histograms for the GP hyperparameters """
        if not self.GPmodel:
            print('Model does not have GP! make_plot4() doing nothing...')
            return

        available_etas = [v for v in dir(self) if v.startswith('eta')]
        labels = [r'$\eta_%d$' % (i+1) for i,_ in enumerate(available_etas)]
        units = ['m/s', 'days', 'days', None]
        xlabels = []
        for label, unit in zip(labels, units):
            xlabels.append(label + ' (%s)' % unit 
                                if unit is not None else label)

        fig, axes = plt.subplots(2, len(available_etas)//2)
        fig.suptitle('Posterior distributions for GP hyperparameters')

        for i, eta in enumerate(available_etas):
            ax = np.ravel(axes)[i]
            ax.hist(getattr(self, eta), bins=40)
            ax.set(xlabel=xlabels[i], ylabel='posterior samples')
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])


    def make_plot5(self, show=True, save=False):
        """ Corner plot for the GP hyperparameters """

        if not self.GPmodel:
            print('Model does not have GP! make_plot5() doing nothing...')
            return

        self.pmin = 10.
        self.pmax = 40.

        available_etas = [v for v in dir(self) if v.startswith('eta')]
        labels = [r'$s$'] + [r'$\eta_%d$' % (i+1) for i,_ in enumerate(available_etas)]
        units = ['m/s', 'm/s', 'days', 'days', None]
        xlabels = []
        for label, unit in zip(labels, units):
            xlabels.append(label + ' (%s)' % unit 
                                if unit is not None else label)


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

        ranges = [1.]*(len(available_etas)+1)
        ranges[3] = (self.pmin, self.pmax)

        c = corner.corner        
        self.corner1 = c(self.post_samples, labels=xlabels, show_titles=True,
                         plot_contours=False, plot_datapoints=True, plot_density=False,
                         # fill_contours=True, smooth=True,
                         # contourf_kwargs={'cmap':plt.get_cmap('afmhot'), 'colors':None},
                         hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
                         hist_kwargs={'normed':True}, 
                         range=ranges, data_kwargs={'alpha':1},
                         )

        self.corner1.suptitle('Joint and marginal posteriors for GP hyperparameters')

        if show:
            self.corner1.tight_layout(rect=[0, 0.03, 1, 0.95])
        
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


    def plot_random_planets(self, ncurves=50, over=0.1, pmin=None, pmax=None, 
                            show_vsys=False, show_trend=False):
        """
        Display the RV data together with curves from the posterior predictive.
        A total of `ncurves` random samples are chosen,
        and the Keplerian curves are calculated covering 100 + `over`%
        of the timespan of the data.
        """
        samples = self.get_sorted_planet_samples()
        if self.max_components > 0:
            samples, mask = \
                self.apply_cuts_period(samples, pmin, pmax, return_mask=True)
        else:
            mask = np.ones(samples.shape[0], dtype=bool)

        t = self.data[:,0].copy()
        tt = np.linspace(t[0]-over*t.ptp(), t[-1]+over*t.ptp(), 
                         10000+int(100*over))

        y = self.data[:,1].copy()
        yerr = self.data[:,2].copy()

        # select random `ncurves` indices 
        # from the (sorted, period-cut) posterior samples
        ii = np.random.randint(samples.shape[0], size=ncurves)

        _, ax = plt.subplots(1,1)

        ## plot the Keplerian curves
        for i in ii:
            v = np.zeros_like(tt)
            pars = samples[i, :].copy()
            nplanets = pars.size / self.n_dimensions
            for j in range(int(nplanets)):
                P = pars[j + 0*self.max_components]
                if P==0.0:
                    continue
                K = pars[j + 1*self.max_components]
                phi = pars[j + 2*self.max_components]
                t0 = t[0] - (P*phi)/(2.*np.pi)
                ecc = pars[j + 3*self.max_components]
                w = pars[j + 4*self.max_components]
                # print(P)
                v += keplerian(tt, P, K, ecc, w, t0, 0.)

            vsys = self.posterior_sample[mask][i, -1]
            v += vsys
            if self.trend:
                v += self.trendpars[i]*(tt - t[0])
                if show_trend:
                    ax.plot(tt, vsys+self.trendpars[i]*(tt - t[0]), 
                            alpha=0.2, color='m', ls=':')

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
        plt.tight_layout()
        # plt.show()


    def hist_offset(self):
        """ Plot the histogram of the posterior for the fiber offset """
        if not self.fiber_offset:
            print('Model has no fiber offset! hist_offset() doing nothing...')
            return

        units = ' (m/s)' if self.units=='ms' else ' (km/s)'
        estimate = percentile68_ranges_latex(self.offset) + units

        _, ax = plt.subplots(1,1)
        ax.hist(self.offset)
        title = 'Posterior distribution for fiber offset \n %s' % estimate
        ax.set(xlabel='fiber offset (m/s)', ylabel='posterior samples',
               title=title)


    def hist_vsys(self):
        """ Plot the histogram of the posterior for the systemic velocity """
        vsys = self.posterior_sample[:,-1]
        units = ' (m/s)' if self.units=='ms' else ' (km/s)'
        estimate = percentile68_ranges_latex(vsys) + units

        _, ax = plt.subplots(1,1)
        ax.hist(vsys)
        title = 'Posterior distribution for $v_{\\rm sys}$ \n %s' % estimate
        ax.set(xlabel='vsys' + units, ylabel='posterior samples',
               title=title)


    def hist_extra_sigma(self):
        """ Plot the histogram of the posterior for the additional white noise """
        units = ' (m/s)' if self.units=='ms' else ' (km/s)'
        estimate = percentile68_ranges_latex(self.extra_sigma) + units

        _, ax = plt.subplots(1,1)
        ax.hist(self.extra_sigma)
        title = 'Posterior distribution for extra white noise $s$ \n %s' % estimate
        ax.set(xlabel='extra sigma (m/s)', ylabel='posterior samples',
               title=title)