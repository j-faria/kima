import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
from scipy.stats import gaussian_kde
import sys
import re
import os
import george
from george import kernels

sys.path.append('/home/joao/Work/OPEN')
from OPEN.classes import params as params_paper
# from OPEN.classes import MyFormatter
colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

sys.path.append('/home/joao/Work/corner/build/lib')
import corner
reload(corner)
import corner_analytic
reload(corner_analytic)

plt.rc("font", size=14, family="serif", serif="Computer Sans")
plt.rc("text", usetex=True)

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


def get_aliases(Preal):
    fs = np.array([0.0027381631, 1.0, 1.0027]) #, 0.018472000025212765])
    return np.array([abs(1 / (1./Preal + i*fs)) for i in range(-6, 6)]).T


class DisplayResults(object):
    def __init__(self, options, data_file=None, posterior_samples_file='posterior_sample.txt'):
        self.options = options

        if data_file is None:
            path_to_this_file = os.path.abspath(__file__)
            top_level = os.path.dirname(os.path.dirname(path_to_this_file))
            print top_level
            with open(os.path.join(top_level, 'src', 'main.cpp')) as f:
                c = f.readlines()
            for line in c:
                if 'loadnew' in line and '/*' not in line: l = line
            data_file = re.findall('"(.*?)"', l, re.DOTALL)[0]

        print 'Loading data file %s' % data_file
        data_file = os.path.join(top_level, data_file)

        # self.data = np.loadtxt('1planet_plus_gp.rv')
        # self.data = np.loadtxt('HD41248_harps_mean_corr.rdb')
        # self.data = np.loadtxt('BT1.txt')
        self.data = np.loadtxt(data_file, skiprows=2)
        mean_vrad = self.data[:, 1].mean()
        self.data[:, 1] = (self.data[:, 1] - mean_vrad)*1e3 + mean_vrad
        self.data[:, 2] *= 1e3

        # self.truth = np.loadtxt('fake_data_like_nuoph.truth')
        # posterior_samples_file = 'resultsCorot7/upto10/posterior_sample.txt'
        self.posterior_sample = np.atleast_2d(np.loadtxt(posterior_samples_file))

        start_parameters = 0
        # (nsamples x 1000)
        self.signals = self.posterior_sample[:, :start_parameters]

        self.extra_sigma = self.posterior_sample[:, start_parameters]

        n_hyperparameters = 4
        for i in range(n_hyperparameters):
            name = 'eta' + str(i+1)
            setattr(self, name, self.posterior_sample[:, start_parameters+1+i])

        # if n_hyperparameters == 4:
        #     self.eta1, self.eta2, self.eta3, self.eta4 = self.posterior_sample[:, start_parameters+1:start_parameters+5].T
        # elif n_hyperparameters == 5:
        #     self.eta1, self.eta2, self.eta3, self.eta4, self.eta5 = self.posterior_sample[:, start_parameters+1:start_parameters+6].T
        # else:
        #     self.nu = self.posterior_sample[:, start_parameters+n_hyperparameters].T
        # self.eta1, self.eta2, self.eta3, self.eta4, self.eta5 = self.posterior_sample[:, start_parameters+1:start_parameters+6].T

        n_offsets = 0
        self.offset = self.posterior_sample[:, start_parameters+n_hyperparameters+n_offsets]

        start_objects_print = start_parameters + n_offsets + n_hyperparameters + 1
        # how many parameters per component
        self.n_dimensions = int(self.posterior_sample[0, start_objects_print])
        # maximum number of components
        self.max_components = int(self.posterior_sample[0, start_objects_print+1])

        n_dist_print = 3

        self.index_component = start_objects_print + 1 + n_dist_print + 1

        self.get_marginals()
        if '8' in options:
            if 'cut' in options:
                i = options.index('cut')
                cut = float(options[i+1])
                options.pop(i)
                options.pop(i)
            else:
                cut = None
            self.make_plot8(cut=cut)
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
        if '6' in options:
            self.make_plot6()
        if '7' in options:
            self.make_plot7()
        if 'prior' in options:
            self.make_plot_priors()
        if '9' in options:
            self.make_plot9()
        if '10' in options:
            self.plot_all_planet_params()
        


    def make_plot1(self):
        plt.figure()
        n, bins, _ = plt.hist(self.posterior_sample[:, self.index_component], 100)
        plt.xlabel('Number of Planets')
        plt.ylabel('Number of Posterior Samples')
        plt.xlim([-0.5, self.max_components+.5])

        nn = n[np.nonzero(n)]
        print 'probability ratios: ', nn.flat[1:] / nn.flat[:-1]

        plt.show()


    def get_marginals(self):
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

        which = self.T != 0
        self.T = self.T[which].flatten()
        self.A = self.A[which].flatten()
        self.E = self.E[which].flatten()
        # Trim
        #s = sort(T)
        #left, middle, right = s[0.25*len(s)], s[0.5*len(s)], s[0.75*len(s)]
        #iqr = right - left
        #s = s[logical_and(s > middle - 5*iqr, s < middle + 5*iqr)]

    def get_medians(self):
        """ return the median values of all the parameters """
        if self.posterior_sample.shape[0] % 2 == 0:
            print 'Median is not a solution because number of samples is even!!'

        self.medians = np.median(self.posterior_sample, axis=0)
        self.means = np.mean(self.posterior_sample, axis=0)
        return self.medians, self.means

    def period_sort(self):
        """ rearrange the parameter table such that the periods are sorted """
        periods = np.exp(self.Tall)

        # replace non-existing planets (log(P)=0, P=1) by a negative value so they don't mess things up
        np.place(periods, periods==1., -99)

        self.posterior_sample_unsorted = np.copy(self.posterior_sample)


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

        print 'maximum likelihood '
        print pars[:5]
        print pars[pars != 0]

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
            periods = np.exp(res.Tall[mask,:2])
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

        print 
        print 'medians:'
        print

        a = '$%7.5f\,^{+\,%7.5f}_{-\,%7.5f}$' % percentile68_ranges(P1)
        b = ' & $%4.3f$' % P1.std()
        print '%-40s' % a, b

        a, b = '$%3.2f\,^{+\,%3.2f}_{-\,%3.2f}$' % percentile68_ranges(K1), ' & $%4.3f$' % K1.std()
        print '%-40s' % a, b
        
        a, b = '$%4.3f\,^{+\,%4.3f}_{-\,%4.3f}$' % percentile68_ranges(e1), ' & $%4.3f$' % e1.std()
        print '%-40s' % a, b

        a, b = '$%7.5f\,^{+\,%7.5f}_{-\,%7.5f}$' % percentile68_ranges(P2), ' & $%4.3f$' % P2.std()
        print '%-40s' % a, b

        a, b = '$%3.2f\,^{+\,%3.2f}_{-\,%3.2f}$' % percentile68_ranges(K2), ' & $%4.3f$' % K2.std()
        print '%-40s' % a, b

        a, b = '$%4.3f\,^{+\,%4.3f}_{-\,%4.3f}$' % percentile68_ranges(e2), ' & $%4.3f$' % e2.std()
        print '%-40s' % a, b



        ############################################################

        mjup2mearth  = 317.828
        star_mass = 0.913


        m_mj = 4.919e-3 * star_mass**(2./3) * P1**(1./3) * K1 * np.sqrt(1-e1**2)
        m_me = m_mj * mjup2mearth
        # a = ((system.star_mass + m_me*mearth2msun)/(m_me*mearth2msun)) * sqrt(1.-ecc**2) * K * (P*mean_sidereal_day/(2*np.pi)) / au2m

        print 'b - $%4.2f\,^{+\,%4.2f}_{-\,%4.2f}$ [MEarth]' % percentile68_ranges(m_me)
        # print '%8s %11.4f +- %7.4f [AU]' % ('a', a.n, a.s)



        m_mj = 4.919e-3 * star_mass**(2./3) * P2**(1./3) * K2 * np.sqrt(1-e2**2)
        m_me = m_mj * mjup2mearth
        # a = ((system.star_mass + m_me*mearth2msun)/(m_me*mearth2msun)) * sqrt(1.-ecc**2) * K * (P*mean_sidereal_day/(2*np.pi)) / au2m

        print 'c - $%4.2f\,^{+\,%4.2f}_{-\,%4.2f}$ [MEarth]' % percentile68_ranges(m_me)
        # print '%8s %11.4f +- %7.4f [AU]' % ('a', a.n, a.s)





    def make_plot2(self):
        T = self.T
        plt.figure()
        plt.hist(np.exp(T), bins=np.logspace(min(T), max(T), base=np.e, num=1000), alpha=0.5)
        plt.xlabel(r'(Period/days)')
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("symlog")
        #for i in xrange(1009, 1009 + int(truth[1008])):
        #  axvline(truth[i]/log(10.), color='r')
        plt.ylabel('Number of Posterior Samples')
        plt.show()

    def make_plot3(self, paper=False, points=True):

        T, A, E = self.T, self.A, self.E

        # aliases1 = get_aliases(0.85359165)
        # aliases2 = get_aliases(3.691)
        # aliases3 = get_aliases(9.03580275887)

        if paper:
            with plt.rc_context(params_paper):
                figwidth = 3.543311946  # in inches = \hsize = 256.0748pt
                figheight = 0.95 * figwidth

                fig = plt.figure(figsize=(figwidth, figheight))
                # fig.subplots_adjust(hspace=0.3, left=0.14, right=0.95, top=0.95)
                ax1 = fig.add_subplot(2,1,1)
                ax1.hexbin(np.exp(T[::100]), A[::100], gridsize=100, bins='log', xscale='log', yscale='log',
                           cmap=plt.get_cmap('afmhot_r'))
                ax1.set_ylabel(r'Semi-amplitude [$\ms$]')
                ax1.set_xlim([0.1, 1000])
                ax1.set_xticklabels([])
                ax1.set_yticklabels(['', '', '0.01', '0.1', '1', '10'])

                ax2 = fig.add_subplot(2,1,2)
                ax2.hexbin(np.exp(T[::100]), E[::100], gridsize=100, bins='log', xscale='log',
                           cmap=plt.get_cmap('afmhot_r'))

                ax2.set_xlim([0.1, 1000])
                print ax2.get_xticklabels()
                ax2.set_xticklabels(['', '0.1', '1', '10', '100', '1000'])
                ax2.set_xlabel(r'Period [days]')
                ax2.set_ylabel('Eccentricity')

                fig.tight_layout()
                return
                # fig.savefig('/home/joao/phd/RJGP_paper_HD41248/figures/jointplot.pdf')


        else:
            fig = plt.figure()

            ax1 = fig.add_subplot(2,1,1)
            #plot(truth[1009:1009 + int(truth[1008])]/log(10.), log10(truth[1018:1018 + int(truth[1008])]), 'ro', markersize=7)
            #hold(True)
            if points:
                ax1.loglog(np.exp(T), A, '.', markersize=1)
                ax1.set_xscale('log')
                ax1.set_yscale('log')
            else:
                ax1.hexbin(np.exp(T[::100]), A[::100], gridsize=50, bins='log', xscale='log', yscale='log',
                           cmap=plt.get_cmap('afmhot_r'))
            # data = np.vstack([np.exp(T[::100]), A[::100]]).T
            # sns.jointplot(x=np.exp(T[::100]), y=A[::100], kind="hex", color="k");
            # sns.kdeplot(data=np.exp(T[::100]), data2=A[::100], bw=[0.1, 1], shade=True, ax=ax1)

            ax1.set_ylabel(r'Amplitude (m/s)')

            # mu = self.posterior_sample[:, self.index_component-1]
            # ax = fig.add_subplot(2,2,2, sharey=ax1)
            # ax.hist(mu, bins=30, orientation="horizontal")
            # ax.set_ylim(ax1.get_ylim())
            # ax.set_yscale('log')

            ax2 = fig.add_subplot(2,1,2, sharex=ax1)
            #plot(truth[1009:1009 + int(truth[1008])]/log(10.), truth[1038:1038 + int(truth[1008])], 'ro', markersize=7)
            #hold(True)
            if points:
                ax2.semilogx(np.exp(T), E, 'b.', markersize=2)
            else:
                ax2.hexbin(np.exp(T[::100]), E[::100], gridsize=50, bins='log', xscale='log',
                           cmap=plt.get_cmap('afmhot_r'))
            # ax.axvline(x=0.85359165, color='r')
            # ax.axvline(x=3.691, color='r')
            # ax.vlines([3.9359312722691815, 3.8939651463077287, 3.8528844910117557, 3.8126615742655399, 3.7732698100471724, 3.7346836998277957, 3.6968787775300003, 3.6598315577957377, 3.6235194873338745, 3.5879208991356037, 3.5530149693624242, 3.5187816767264146], ymin=0, ymax=1)
            # ax.vlines(aliases1, ymin=0, ymax=1, color='g', alpha=0.4)
            # ax.vlines(aliases2, ymin=0, ymax=1, color='y', alpha=0.4)
            # ax.vlines(aliases3, ymin=0, ymax=1, color='c', alpha=0.4)
            
            ax2.set_xlim([0.1, 1000])
            ax2.set_xlabel(r'(Period/days)')
            ax2.set_ylabel('Eccentricity')

            plt.show()


    def make_plot4(self):
        plt.figure()
        available_etas = [v for v in dir(self) if v.startswith('eta')]

        for i, eta in enumerate(available_etas):
            plt.subplot(2, 3, i+1)
            plt.hist(getattr(self, eta), bins=40)
            plt.xlabel(eta)
        plt.show()



    # # data[:,0] -= data[:,0].min()
    # t = np.linspace(data[:,0].min(), data[:,0].max(), 1000)
    # c = np.random.choice(, size=10, replace=False)

    # fig = figure()
    # ax = fig.add_subplot(111)
    # ax.errorbar(data[:,0], data[:,1], fmt='b.', yerr=data[:,2])
    # data_ylimits = ax.get_ylim()
    # # plot random posterior sample signals
    # ax.plot(t, signals[c, :].T, alpha=0.4)
    # ax.set_ylim(data_ylimits)
    # # ax.errorbar(data[:,0], data[:,1], fmt='b.', yerr=data[:,2])

    def make_plot5(self, show=True, save=False):
        # self.periods = np.exp(self.Tall[:,0])
        # self.periods[self.periods == 1.] = -99
        # self.periods = np.ma.masked_invalid(self.periods)
        self.pmin = 20. #self.periods.mean() - 2*self.periods.std()
        self.pmax = 100. #self.periods.mean() + 2*self.periods.std()

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
        # self.post_samples = np.vstack((self.extra_sigma, self.eta1, self.eta2, self.eta3, self.eta4)).T
        variables = [self.extra_sigma]
        for eta in available_etas:
            variables.append(getattr(self, eta))

        self.post_samples = np.vstack(variables).T
        print self.post_samples.shape

        ranges = [1.]*(len(available_etas)+1)
        ranges[3] = (self.pmin, self.pmax)
        # print (self.pmin, self.pmax)
        # labels = ['$\sigma_{extra}$', '$\eta_1$', '$\eta_2$', '$\eta_3$', '$\eta_4$', '$\eta_5$']
        
        self.corner1 = corner.corner(self.post_samples, labels=labels, show_titles=True,
                                     plot_contours=False, plot_datapoints=True, plot_density=False,
                                     # fill_contours=True, smooth=True,
                                     # contourf_kwargs={'cmap':plt.get_cmap('afmhot'), 'colors':None},
                                     hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
                                     hist_kwargs={'normed':True},
                                     range=ranges,
                                     shared_axis=True, data_kwargs={'alpha':1},
                                     )



        if show:
            plt.show()
        
        if save:
            self.corner1.savefig(save)



    def make_plot6(self, plot_samples=False, show=True, N=0):
        """ 
        plot data with maximum likelihood solution and optionally
        random posterior samples
        """
        l = Lookup(cache=True)
        data = self.data
        t = self.data[:,0]
        tt = np.linspace(t[0], t[-1], 3000)

        kde = gaussian_kde(t)
        ttt = kde.resample(50000)

        tt = np.append(t, tt)
        tt = np.append(t, ttt)
        tt.sort()

        y = self.data[:,1]
        yerr = self.data[:,2]

        ic = self.index_component

        fig = plt.figure()
        ax = fig.add_subplot(311)
        ax.errorbar(data[:,0], data[:,1], fmt='ro', yerr=data[:,2])

        if plot_samples:
            # array with posterior sample indices that have at least one planet
            hasplanets = self.posterior_sample[:, self.index_component] > 0.
            # choose 10 random samples with planets
            ch = np.random.choice(np.where(hasplanets==True)[0], size=1, replace=False)

            for i in ch:
                pars = self.posterior_sample[i, :]
                velt = np.zeros_like(t)
                veltt = np.zeros_like(tt)

                extra_sigma = pars[0]
                eta1 = pars[1]
                eta2 = pars[2]
                eta3 = pars[3]
                eta4 = pars[4]
                # print 'GP pars: ', extra_sigma, eta1, eta2, eta3, eta4
                background = pars[-2]

                self.kernel = eta1 * kernels.ExpSquaredKernel(eta2) * kernels.ExpSine2Kernel(eta4, eta3)
                self.gp = george.GP(self.kernel, mean=background)
                self.gp.compute(t, yerr)

                print int(pars[ic])
                for j in range(int(pars[ic])):
                    T = np.exp(pars[ic+1+j])
                    A = pars[ic+3+j]
                    phi = pars[ic+5+j]
                    v0 = np.sqrt(1 - pars[ic+7+j])
                    viewing_angle = pars[ic+9+j]

                    
                    # print '- planet pars: ', T, A, phi, v0, viewing_angle
                    arg = 2.*np.pi*t/T + phi
                    velt += A * l.evaluate(arg, v0, viewing_angle)

                    arg = 2.*np.pi*tt/T + phi
                    veltt += A * l.evaluate(arg, v0, viewing_angle)

                
                mu = self.gp.predict(y-velt, tt, mean_only=True)
                mu += veltt

                ax.plot(tt, mu, 'k-', alpha=0.2)

        ##################################
        # the maximum likelihood solution:
        if N is None:
            i = self.posterior_sample[:, -1].argmax()
            pars = self.posterior_sample[i, :]
        else:
            mask = self.posterior_sample[:, self.index_component]==N
            i = self.posterior_sample[mask, -1].argmax()
            pars = self.posterior_sample[mask][i, :]
        
        ##################################
        # the median solution restricted:
        # pars = np.median(self.posterior_sample[mask], axis=0)
        print pars

        # print pars
        velt = np.zeros_like(t)
        veltt = np.zeros_like(tt)

        extra_sigma = pars[0]
        eta1 = pars[1]
        eta2 = pars[2]
        eta3 = pars[3]
        eta4 = pars[4]
        print 'GP pars: ', extra_sigma, eta1, eta2, eta3, eta4
        background = pars[-2]

        self.kernel = eta1 * kernels.ExpSquaredKernel(eta2) * kernels.ExpSine2Kernel(eta4, eta3)
        self.gp = george.GP(self.kernel, mean=background)
        self.gp.compute(t, yerr)


        velt_individual = np.zeros((t.size, int(pars[ic])))
        veltt_individual = np.zeros((tt.size, int(pars[ic])))

        nplanets = self.max_components
        ii = range(1, 5*nplanets, nplanets)
        for j in range(int(pars[ic])):
            T = np.exp(pars[ic+ii[0]+j])
            A = pars[ic+ii[1]+j]
            phi = pars[ic+ii[2]+j]
            ecc = pars[ic+ii[3]+j]
            v0 = np.sqrt(1 - ecc)
            viewing_angle = pars[ic+ii[4]+j]
            
            print '- planet pars: ', T, A, phi, ecc, viewing_angle
            arg = 2.*np.pi*t/T + phi
            rv = A * l.evaluate(arg, v0, viewing_angle)
            velt_individual[:,j] = rv
            velt += rv

            arg = 2.*np.pi*tt/T + phi
            rv = A * l.evaluate(arg, v0, viewing_angle)
            veltt_individual[:,j] = rv
            veltt += rv

        mu = self.gp.predict(y-velt, tt, mean_only=True)
        mut = self.gp.predict(y-velt, t, mean_only=True)
        mu += veltt
        mut_maxlike = mut

        ax.plot(tt, mu, 'k-', lw=1.5, alpha=0.9, label='MLE')       
        ax.legend(frameon=False, loc='best')

        print ''

        ax = fig.add_subplot(312, sharex=ax)
        ax.errorbar(data[:,0], data[:,1], fmt='ro', yerr=data[:,2])


        ##################################
        # the median solution:
        self.get_medians()
        pars = self.medians
        velt = np.zeros_like(t)
        veltt = np.zeros_like(tt)

        extra_sigma = pars[0]
        eta1 = pars[1]
        eta2 = pars[2]
        eta3 = pars[3]
        eta4 = pars[4]
        print 'GP pars: ', extra_sigma, eta1, eta2, eta3, eta4
        background = pars[-2]

        self.kernel = eta1 * kernels.ExpSquaredKernel(eta2) * kernels.ExpSine2Kernel(eta4, eta3)
        self.gp = george.GP(self.kernel, mean=background)
        self.gp.compute(t, yerr)


        velt_individual = np.zeros((t.size, int(pars[ic])))
        veltt_individual = np.zeros((tt.size, int(pars[ic])))

        nplanets = self.max_components
        ii = range(1, 5*nplanets, nplanets)
        for j in range(int(pars[ic])):
            T = np.exp(pars[ic+ii[0]+j])
            A = pars[ic+ii[1]+j]
            phi = pars[ic+ii[2]+j]
            ecc = pars[ic+ii[3]+j]
            v0 = np.sqrt(1 - ecc)
            viewing_angle = pars[ic+ii[4]+j]
            
            print '- planet pars: ', T, A, phi, ecc, viewing_angle
            arg = 2.*np.pi*t/T + phi

            rv = A * l.evaluate(arg, v0, viewing_angle)
            velt_individual[:,j] = rv
            velt += rv


            arg = 2.*np.pi*tt/T + phi
            rv = A * l.evaluate(arg, v0, viewing_angle)
            veltt_individual[:,j] = rv
            veltt += rv
            # ax.plot(tt, rv, 'k:', lw=1, alpha=1)


        mu = self.gp.predict(y-velt, tt, mean_only=True)
        mut = self.gp.predict(y-velt, t, mean_only=True)
        # mu, cov = self.gp.predict(y-velt, tt)
        # std = np.sqrt(np.diag(cov))
        
        mu += veltt
        # std += veltt

        ax.plot(tt, mu, 'k-', lw=1.5, alpha=0.9, label='median')
        # ax.fill_between(tt, y1=mu-std, y2=mu+std, alpha=0.1)

        # ax.plot(tt, np.zeros_like(tt), 'o')

        ax.legend(frameon=False, loc='best')


        ax = fig.add_subplot(313, sharex=ax)
        ax.plot(t, y - mut_maxlike, 'ro-', alpha=0.9)
        ax.axhline(y=0)

        if show: plt.show()

        return tt, mu, mut, velt_individual, veltt_individual


    def make_plot7(self):
        """ plot data with random posterior samples """
        from OPEN.ext.keplerian import keplerian
        data = self.data
        t = self.data[:,0]

        ttt = np.linspace(t.min(), t.max(), 10000)

        kde = gaussian_kde(t)
        tt = kde.resample(500)
        tt = np.append(t, tt)
        tt.sort()

        y = self.data[:,1]
        yerr = self.data[:,2]

        print 'Data has', y.size, 'points'

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.errorbar(data[:,0], data[:,1], fmt='b.', yerr=data[:,2])


        # array with posterior sample indices that have at least one planet
        # hasplanets = self.posterior_sample[:, self.index_component] > 0.


        def exp2_kernel(tau, dt):
            return np.exp(-0.5 * dt ** 2 / tau)

        def expsine2_kernel(gamma, period, dt):
            return np.exp(- 2. * (np.sin(np.pi * dt / period) / gamma) ** 2)

        def linear_kernel(dt):
            return dt

        def kernel(params, dt):
            # amp, tau, gamma, period = np.exp(params)
            eta1, eta2, eta3, eta4, eta5 = params
            K = eta1*eta1 * exp2_kernel(eta2, dt) * expsine2_kernel(eta4, eta3, dt) + eta5 * linear_kernel(dt)
            return K


        available_etas = [v for v in dir(self) if v.startswith('eta')]
        netas = len(available_etas)

        # choose 10 random posterior samples
        ch = np.random.choice(self.posterior_sample.shape[0], size=50, replace=False)
        # ch = [25]*10

        for i in ch:
            pars = self.posterior_sample[i, :]
            
            nplanets = int(pars[self.index_component])
            print nplanets, 'planets'
            if nplanets != 1:
                continue

            # vel = np.zeros_like(tt)
            # print pars

            # pp = pars
            # pp[5] *= 10
            # K = kernel(pp[1:6], tt[None, :] - tt[:, None])
            # K[np.diag_indices_from(K)] += 1e-8
            # yy = np.random.multivariate_normal(pars[-2] + np.zeros_like(tt), K)

            extra_sigma = pars[0]
            if netas == 4:
                eta1 = pars[1]
                eta2 = pars[2]
                eta3 = pars[3]
                eta4 = pars[4]
                print 'GP pars: ', extra_sigma, eta1, eta2, eta3, eta4
                self.kernel = eta1 * kernels.ExpSquaredKernel(eta2) * kernels.ExpSine2Kernel(eta4, eta3)
            elif netas == 5:
                eta1 = pars[1]
                eta2 = pars[2]
                eta3 = pars[3]
                eta4 = pars[4]
                eta5 = pars[5]
                print 'GP pars: ', extra_sigma, eta1, eta2, eta3, eta4, eta5
                self.kernel = eta1 * kernels.ExpSquaredKernel(eta2) * kernels.ExpSine2Kernel(eta4, eta3) + eta5 * kernels.DotProductKernel()
            elif netas == 2:
                eta1 = pars[1]
                eta2 = pars[2]
                print 'GP pars: ', extra_sigma, eta1, eta2
                self.kernel = eta1 * kernels.ExpSquaredKernel(eta2)


            self.gp = george.GP(self.kernel, mean=pars[-1])
            self.gp.compute(t, yerr)

            gpmean = self.gp.predict(y, tt, mean_only=True)
            # vel = self.gp.sample(tt)
            # ax.plot(tt, gpmean, 'k-', alpha=0.2)
            # ax.plot(tt, yy, 'g')


            v = np.zeros_like(ttt)
            velt = np.zeros_like(t)

            for j in range(nplanets):
                planet_pars = pars[self.index_component+j+1 : -2 : self.max_components]
                P = np.exp(planet_pars[0])
                K = planet_pars[1]#*1e3
                phi = planet_pars[2]
                t0 = t[0] - (P*phi)/(2.*np.pi)
                ecc = planet_pars[3]
                w = planet_pars[4]
                print 'planet pars:', '\t', '  '.join([str(par) for par in [P, K, ecc, phi, t0]])
                # vsys = pars[-1]
                v1 = keplerian(ttt, P, K, ecc, w, t0, 0.)
                v += v1
                velt += keplerian(t, P, K, ecc, w, t0, 0.) 
            ax.plot(ttt, v, alpha=0.6, color='g')

            mu = self.gp.predict(y-velt, ttt, mean_only=True)
            # mut = self.gp.predict(y-velt, t, mean_only=True)
            # # mu, cov = self.gp.predict(y-velt, tt)
            # # std = np.sqrt(np.diag(cov))
            
            mu += v
            # # std += veltt

            ax.plot(ttt, mu, 'k-', lw=1.5, alpha=0.05)


        # self.get_medians()
        # pars = self.medians
        # extra_sigma, eta1, eta2, eta3, eta4 = pars[:5]
        # self.kernel = eta1 * kernels.ExpSquaredKernel(eta2) * kernels.ExpSine2Kernel(eta4, eta3) #+ kernels.WhiteKernel(extra_sigma)
        # self.gp = george.GP(self.kernel, mean=pars[-2])
        # self.gp.compute(t, yerr)
        # mu, cov = self.gp.predict(y, tt)
        # std = np.sqrt(np.diag(cov))

        # ax.plot(tt, mu, 'g-', lw=1.5)
        # ax.fill_between(tt, y1=mu-std, y2=mu+std, alpha=0.1)

        # print 'sample 1...'
        # print pars
        # ax.plot(tt, self.gp.sample_conditional(y, tt), 'k-', lw=.5, alpha=0.2)
        # # print 'sample 2...'
        # # ax.plot(tt, self.gp.sample_conditional(y, tt), 'k-', lw=.5, alpha=0.2)

        # ax = fig.add_subplot(212, sharex=ax)
        # dmGP = data[:,1] - self.gp.predict(y, t, mean_only=True)
        # ax.errorbar(t, dmGP, fmt='b-.', yerr=data[:,2])

        # np.savetxt('data_minus_GP.txt', zip(t, dmGP, data[:,2]), header='jdb\tvrad\tsvrad\n---\t----\t-----')

        plt.show()

    def make_plot_priors(self):

        self.make_plot5(show=False)

        fs = np.array(['t', 'uniform', 'loguniform', 'loguniform', 'uniform', 'loguniform'])
        labels = ['P', '$\sigma_{extra}$', '$\eta_1$', '$\eta_2$', '$\eta_3$', '$\eta_4$']
        pars =  [1, [0, 2],  None,             None,             [10, 30], None]
        kpars = [{}, {}, {'a':0.1,'b':10}, {'a':10,'b':100}, {},       {'a':0.1,'b':10}]

        fig = corner_analytic.corner(fs, dist_args=pars, dist_kwargs=kpars, 
                                     labels=labels, shared_axis=True, only_diag=True,
                                     fig=self.corner1, )
        plt.show()

    def make_plot8(self, show=True, cut=None):

        nplanets = self.max_components

        if cut is None:
            P = np.exp(self.Tall)
            print P.shape
            K = self.Aall


            self.pmin = 22. #self.periods.mean() - 2*self.periods.std()
            self.pmax = 29. #self.periods.mean() + 2*self.periods.std()

            self.post_samples = np.hstack((P, K))
            print self.post_samples.shape
            # print (self.pmin, self.pmax)
            labels = ['$P_%d$' % i for i in range(1, P.shape[1]+1)]
            labels += ['$K_%d$' % i for i in range(1, K.shape[1]+1)]
            r = [1.] * self.post_samples.shape[1]
            # r[0] = (0, 100)
            # r[0] = (3.6, 3.8)
            # r[1] = (0.84, 0.86)
        else:
            P1_mask = np.exp(self.Tall) < cut
            P2_mask = np.exp(self.Tall) > cut

            P1 = np.exp(self.Tall)[P1_mask]
            P2 = np.exp(self.Tall)[P2_mask]

            # # P1 = np.exp(self.Tall[:,0])
            # # P2 = np.exp(self.Tall[:,1])

            # print P1.shape, P2.shape

            K1 = self.Aall[P1_mask]
            K2 = self.Aall[P2_mask]

            # K1 = self.Aall[:,0]
            # K2 = self.Aall[:,1]

            self.post_samples = np.vstack((P1, P2, K1, K2)).T
            labels = ['$P_%d$' % i for i in range(1, 3)]
            labels += ['$K_%d$' % i for i in range(1, 3)]
            r = [1.] * 4
            r[0] = (0.845, 0.86)

        self.corner1 = corner.corner(self.post_samples, labels=labels, show_titles=True,
                                     truths=(0.85359165, 0.85359165, None, None, None, None, None),
                                     plot_contours=False, plot_datapoints=False, plot_density=True,
                                     hist_kwargs={'normed':True},
                                     range=r, title_fmt=".5f",
                                     shared_axis=True, data_kwargs={'alpha':0.5})

        if show:
            plt.show()      

    def make_plot9(self, show=True):

        nplanets = self.max_components

        P = np.exp(self.Tall)
        P = P.flatten()

        K = self.Aall
        K = K.flatten()


        self.post_samples = np.vstack((P, K)).T
        print self.post_samples.shape
        # print (self.pmin, self.pmax)
        labels = ['P', 'K']
        r = [1.] * self.post_samples.shape[1]
        r[0] = (0.1, 30)
        # r[1] = (0.84, 0.86)
        
        self.corner1 = corner.corner(self.post_samples, labels=labels, show_titles=True,
                                     # truths=(23.1, 56.8, 1.95, 2.3),
                                     plot_contours=False, plot_datapoints=True, plot_density=False,
                                     hist_kwargs={'normed':True},
                                     range=r,
                                     shared_axis=True, data_kwargs={'alpha':0.5})

        if show:
            plt.show()      


    def plot_all_planet_params(self, planet=None):
        labels = ['$P$', '$K$', '$\phi$', 'ecc', 'va']

        nsamples = self.posterior_sample.shape[0]
        self.post_samples = np.zeros((self.max_components*nsamples, self.n_dimensions))

        k = 0
        for j in range(self.max_components):
            for i in range(nsamples):
                planet_index = self.index_component + 1
                self.post_samples[k, :] = self.posterior_sample[i, planet_index+j:-2:self.max_components]
                if self.post_samples[k, 0] == 0.:
                    self.post_samples[k, :] = np.nan
                else:
                    self.post_samples[k, 0] = np.exp(self.post_samples[k, 0])
                k += 1
            # self.post_samples[:, self.n_dimensions*i] = np.exp(self.post_samples[:, self.n_dimensions*i])
        self.post_samples = self.post_samples[~np.isnan(self.post_samples).any(axis=1)]

        if planet is None:
            corner.corner(self.post_samples, labels=labels, show_titles=False,
                                     plot_contours=False, plot_datapoints=True, plot_density=False,
    #                                  # fill_contours=True, smooth=True,
    #                                  # contourf_kwargs={'cmap':plt.get_cmap('afmhot'), 'colors':None},
    #                                  hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
    #                                  hist_kwargs={'normed':True},
                                     range=[1., 1., (0, 2*np.pi), (0., 1.), (0, 2*np.pi)],
                                     shared_axis=True, data_kwargs={'alpha':1, 'ms':3},
                                     )

        plt.show()


    def plot_random_planets(self):
        from OPEN.ext.keplerian import keplerian

        nsamples = self.posterior_sample.shape[0]
        self.post_samples = np.zeros((self.max_components*nsamples, self.n_dimensions))

        k = 0
        for j in range(self.max_components):
            for i in range(nsamples):
                planet_index = self.index_component + 1
                self.post_samples[k, :] = self.posterior_sample[i, planet_index+j:-2:self.max_components]
                if self.post_samples[k, 0] == 0.:
                    self.post_samples[k, :] = np.nan
                else:
                    self.post_samples[k, 0] = np.exp(self.post_samples[k, 0])
                k += 1
            # self.post_samples[:, self.n_dimensions*i] = np.exp(self.post_samples[:, self.n_dimensions*i])
        self.post_samples = self.post_samples[~np.isnan(self.post_samples).any(axis=1)]


        t = self.data[:,0]
        tt = np.linspace(t[0], t[-1], 1000)

        y = self.data[:,1]
        yerr = self.data[:,2]


        ii = np.random.randint(self.post_samples.shape[0], size=30)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.errorbar(t, y, yerr, fmt='o')

        for i in ii:
            pars = self.post_samples[i, :]
            P = pars[0]
            K = pars[1]
            phi = pars[2]
            t0 = t[0] - (P*phi)/(2.*np.pi)
            ecc = pars[3]
            w = pars[4]
            vsys = self.posterior_sample[i, -1]
            v = keplerian(tt, P, K, ecc, w, t0, vsys)
            ax.plot(tt, v, alpha=0.1, color='k')

        plt.show()

# sys.exit(0)

# saveFrames = False # For making movies
# if saveFrames:
#   os.system('rm Frames/*.png')

# ion()
# for i in xrange(0, posterior_sample.shape[0]):
#   hold(False)
#   errorbar(data[:,0], data[:,1], fmt='b.', yerr=data[:,2])
#   hold(True)
#   plot(t, posterior_sample[i, 0:1000], 'r')
#   xlim([-0.05*data[:,0].max(), 1.05*data[:,0].max()])
#   ylim([-1.5*max(abs(data[:,1])), 1.5*max(abs(data[:,1]))])
#   #axhline(0., color='k')
#   xlabel('Time (days)', fontsize=16)
#   ylabel('Radial Velocity (m/s)', fontsize=16)
#   draw()
#   if saveFrames:
#     savefig('Frames/' + '%0.4d'%(i+1) + '.png', bbox_inches='tight')
#     print('Frames/' + '%0.4d'%(i+1) + '.png')


# ioff()
# show()
# print __name__

if __name__ == '__main__':
    print 'Arguments: ', sys.argv
    try:
        options = sys.argv
    except IndexError:
        options = ''

    res = DisplayResults(options)
    globals()['res'] = res


elif __name__ == 'display':
    pass
