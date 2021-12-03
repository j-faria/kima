from string import ascii_lowercase
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde
from scipy.stats._continuous_distns import reciprocal_gen
from scipy.signal import find_peaks
from corner import corner

from .analysis import passes_threshold_np, find_outliers
from .GP import KERNEL
from .utils import (get_prior, hyperprior_samples, percentile68_ranges_latex,
                    wrms)

from .keplerian import keplerian
# from .results import KimaResults


def make_plots(res, options, save_plots=False):
    res.save_plots = save_plots
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
        '1': [res.make_plot1, {}],
        '2': [res.make_plot2, {'show_prior': True}],
        '3': [res.make_plot3, {}],
        '4': [res.make_plot4, {}],
        '5': [res.make_plot5, {}],
        '6': [
            res.plot_random_planets,
            {
                'show_vsys': True,
                'show_trend': True
            }
        ],
        '6p': [
            'res.plot_random_planets(show_vsys=True, show_trend=True);'\
            'res.phase_plot(res.maximum_likelihood_sample(Np=passes_threshold_np(res)))',
            {}
        ],
        '7': [
            (res.hist_vsys,
             res.hist_extra_sigma,
             res.hist_trend,
             res.hist_correlations
             ), {}],
        '8': [res.hist_MA, {}],
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


def make_plot1(res):
    """ Plot the histogram of the posterior for Np """
    fig, ax = plt.subplots(1, 1)

    bins = np.arange(res.max_components + 2)
    nplanets = res.posterior_sample[:, res.index_component]
    n, _ = np.histogram(nplanets, bins=bins)
    ax.bar(bins[:-1], n, zorder=2)

    if res.removed_crossing:
        ic = res.index_component
        nn = (~np.isnan(res.posterior_sample[:, ic + 1:ic + 11])).sum(axis=1)
        nn, _ = np.histogram(nn, bins=bins)
        ax.bar(bins[:-1], nn, color='r', alpha=0.2, zorder=2)
        ax.legend(['all posterior samples', 'crossing orbits removed'])
    else:
        pt_Np = passes_threshold_np(res)
        ax.bar(pt_Np, n[pt_Np], color='C3', zorder=2)
        # top = np.mean(ax.get_ylim())
        # ax.arrow(pt_Np, top, 0, -.4*top, lw=2, head_length=1, fc='k', ec='k')

    xlim = (-0.5, res.max_components + 0.5)
    xticks = np.arange(res.max_components + 1)
    ax.set(xlabel='Number of Planets', ylabel='Number of Posterior Samples',
           xlim=xlim, xticks=xticks, title='Posterior distribution for $N_p$')

    nn = n[np.nonzero(n)]
    print('Np probability ratios: ', nn.flat[1:] / nn.flat[:-1])

    if res.save_plots:
        filename = 'kima-showresults-fig1.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def make_plot2(res, nbins=100, bins=None, plims=None, logx=True, density=False,
               kde=False, kde_bw=None, show_peaks=False, show_prior=False):
    """
    Plot the histogram (or the kde) of the posterior for the orbital period(s).
    Optionally provide the number of histogram bins, the bins themselves, the
    limits in orbital period, or the kde bandwidth. If both `kde` and
    `show_peaks` are true, the routine attempts to plot the most prominent
    peaks in the posterior.
    """

    if res.max_components == 0:
        print('Model has no planets! make_plot2() doing nothing...')
        return

    if res.log_period:
        T = np.exp(res.T)
    else:
        T = res.T

    fig, ax = plt.subplots(1, 1)

    kwargs = {'ls': '--', 'lw': 2, 'alpha': 0.5, 'zorder': -1}
    # mark 1 year
    year = 365.25
    ax.axvline(x=year, color='r', label='1 year', **kwargs)
    # ax.axvline(x=year/2., ls='--', color='r', lw=3, alpha=0.6)
    # plt.axvline(x=year/3., ls='--', color='r', lw=3, alpha=0.6)

    # mark the timespan of the data
    ax.axvline(x=res.t.ptp(), color='b', label='timespan', **kwargs)

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
            prior = get_prior(res.setup['priors.planets']['Pprior'])
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
                # prior = res.priors['Pprior']
                # if isinstance(prior.dist, uniform_gen):
                #     a, b = prior.interval(1)
                #     if logx:
                #         bins = 10**np.linspace(np.log10(a), np.log10(b), nbins)
                #     else:
                #         bins = np.linspace(a, b, nbins)
                # else:
                # default, 100 bins in log between 0.1 and 1e7
                bins = 10**np.linspace(np.log10(1e-1), np.log10(1e7), nbins)
            else:
                bins = 10**np.linspace(np.log10(plims[0]), np.log10(plims[1]),
                                       nbins)

        ax.hist(T, bins=bins, alpha=0.8, density=density)

        if show_prior and T.size > 100:
            kwargs = {
                'bins': bins,
                'alpha': 0.15,
                'color': 'k',
                'zorder': -1,
                'label': 'prior',
                'density': density,
            }

            if res.hyperpriors:
                P = hyperprior_samples(T.size)
                ax.hist(P, **kwargs)
            else:
                try:
                    prior = get_prior(res.setup['priors.planets']['Pprior'])
                    ax.hist(prior.rvs(T.size), **kwargs)
                except (KeyError, AttributeError):
                    pass

    ax.legend()
    ax.set(xscale='log' if logx else 'linear', xlabel=r'(Period/days)',
           ylabel='KDE density' if kde else 'Number of Posterior Samples',
           title='Posterior distribution for the orbital period(s)')
    ax.set_ylim(bottom=0)

    if plims is not None:
        ax.set_xlim(plims)

    if res.save_plots:
        filename = 'kima-showresults-fig2.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def make_plot3(res, points=True, gridsize=50):
    """
    Plot the 2d histograms of the posteriors for semi-amplitude and orbital
    period and eccentricity and orbital period. If `points` is True, plot
    each posterior sample, else plot hexbins
    """

    if res.max_components == 0:
        print('Model has no planets! make_plot3() doing nothing...')
        return
    if res.T.size == 0:
        print(
            'None of the posterior samples have planets! make_plot3() doing nothing...'
        )
        return

    if res.log_period:
        T = np.exp(res.T)
        # print('exponentiating period!')
    else:
        T = res.T

    A, E = res.A, res.E

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # the y scale in loglog looks bad if the semi-amplitude doesn't have
    # high dynamic range; the threshold of 30 is arbitrary
    Khdr_threshold = 30

    if points:
        if A.size > 1 and A.ptp() > Khdr_threshold:
            ax1.loglog(T, A, '.', markersize=2, zorder=2)
        else:
            ax1.semilogx(T, A, '.', markersize=2, zorder=2)

        ax2.semilogx(T, E, '.', markersize=2, zorder=2)

    else:
        if A.size > 1 and A.ptp() > 30:
            ax1.hexbin(T, A, gridsize=gridsize, bins='log', xscale='log',
                       yscale='log', cmap=plt.get_cmap('afmhot_r'))
        else:
            ax1.hexbin(T, A, gridsize=gridsize, bins='log', xscale='log',
                       yscale='linear', cmap=plt.get_cmap('afmhot_r'))

        ax2.hexbin(T, E, gridsize=gridsize, bins='log', xscale='log',
                   cmap=plt.get_cmap('afmhot_r'))

    if res.removed_crossing:
        if points:
            mc, ic = res.max_components, res.index_component

            i1, i2 = 0 * mc + ic + 1, 0 * mc + ic + mc + 1
            T = res.posterior_sample_original[:, i1:i2]
            if res.log_period:
                T = np.exp(T)

            i1, i2 = 1 * mc + ic + 1, 1 * mc + ic + mc + 1
            A = res.posterior_sample_original[:, i1:i2]

            i1, i2 = 3 * mc + ic + 1, 3 * mc + ic + mc + 1
            E = res.posterior_sample_original[:, i1:i2]

            if A.ptp() > Khdr_threshold:
                ax1.loglog(T, A, '.', markersize=1, alpha=0.05, color='r',
                           zorder=1)
            else:
                ax1.semilogx(T, A, '.', markersize=1, alpha=0.05, color='r',
                             zorder=1)

            ax2.semilogx(T, E, '.', markersize=1, alpha=0.05, color='r',
                         zorder=1)

    ax1.set(ylabel='Semi-amplitude [m/s]',
            title='Joint posterior semi-amplitude $-$ orbital period')
    ax2.set(ylabel='Eccentricity', xlabel='Period [days]',
            title='Joint posterior eccentricity $-$ orbital period',
            ylim=[0, 1], xlim=[0.1, 1e7])

    try:
        ax2.set(xlim=res.priors['Pprior'].support())
    except (AttributeError, KeyError, ValueError):
        pass

    if res.save_plots:
        filename = 'kima-showresults-fig3.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def make_plot4(res, Np=None, ranges=None):
    """
    Plot histograms for the GP hyperparameters. If Np is not None, highlight
    the samples with Np Keplerians. 
    """
    if not res.GPmodel:
        print('Model does not have GP! make_plot4() doing nothing...')
        return

    available_etas = [v for v in dir(res) if v.startswith('eta')][:-1]
    labels = available_etas
    if ranges is None:
        ranges = len(labels) * [None]

    if Np is not None:
        m = res.posterior_sample[:, res.index_component] == Np

    fig, axes = plt.subplots(2, int(np.ceil(len(available_etas) / 2)))

    if res.GPkernel is KERNEL.celerite:
        axes[-1, -1].axis('off')

    for i, eta in enumerate(available_etas):
        ax = np.ravel(axes)[i]
        ax.hist(getattr(res, eta), bins=40, range=ranges[i])

        if Np is not None:
            ax.hist(
                getattr(res, eta)[m], bins=40, histtype='step', alpha=0.5,
                label='$N_p$=%d samples' % Np, range=ranges[i])
            ax.legend()

        ax.set(xlabel=labels[i], ylabel='posterior samples')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if res.save_plots:
        filename = 'kima-showresults-fig4.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def make_plot5(res, show=True, ranges=None):
    """ Corner plot for the GP hyperparameters """

    if not res.GPmodel:
        print('Model does not have GP! make_plot5() doing nothing...')
        return

    # these are the limits of the default prior for eta3
    res.pmin = 10.
    res.pmax = 40.
    # but we try to accomodate if the prior is changed
    if res.eta3.min() < res.pmin:
        res.pmin = np.floor(res.eta3.min())
    if res.eta3.max() > res.pmax:
        res.pmax = np.ceil(res.eta3.max())

    # available_etas = [v for v in dir(res) if v.startswith('eta')]
    available_etas = ['eta1', 'eta2', 'eta3', 'eta4']
    labels = [r'$s$'] * res.n_jitters
    labels += [r'$\eta_%d$' % (i + 1) for i, _ in enumerate(available_etas)]
    units = ['m/s'] * res.n_jitters + ['m/s', 'days', 'days', None]
    xlabels = []
    if res.multi:
        for i in range(res.n_jitters):
            label = r'%s$_{{\rm %s}}}$' % (labels[i], res.instruments[i])
            xlabels.append(label)
    else:
        xlabels.append(labels[0])

    for label, unit in zip(labels[res.n_jitters:], units[res.n_jitters:]):
        xlabels.append(label)
        #    ' (%s)' % unit if unit is not None else label)

    # all Np together
    res.post_samples = np.c_[res.extra_sigma, res.etas]
    # if res.multi:
    #     variables = list(res.extra_sigma.T)
    # else:
    #     variables = [res.extra_sigma]

    # for eta in available_etas:
    #     variables.append(getattr(res, eta))

    # res.post_samples = np.vstack(variables).T
    # ranges = [1.]*(len(available_etas) + res.extra_sigma.shape[1])

    if ranges is None:
        ranges = [1.] * res.post_samples.shape[1]
    # ranges[3] = (res.pmin, res.pmax)

    try:
        res.corner1 = corner(
            res.post_samples,
            labels=xlabels,
            show_titles=True,
            plot_contours=False,
            plot_datapoints=False,
            plot_density=True,
            # fill_contours=True,
            smooth=True,
            contourf_kwargs={
                'cmap': plt.get_cmap('afmhot'),
                'colors': None
            },
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

    res.corner1.suptitle(
        'Joint and marginal posteriors for GP hyperparameters')

    if show:
        res.corner1.tight_layout(rect=[0, 0.03, 1, 0.95])

    if res.save_plots:
        filename = 'kima-showresults-fig5.png'
        print('saving in', filename)
        res.corner1.savefig(filename)

    if res.return_figs:
        return res.corner1


def hist_vsys(res, show_offsets=True, specific=None, show_prior=False,
              **kwargs):
    """
    Plot the histogram of the posterior for the systemic velocity and for
    the between-instrument offsets (if `show_offsets` is True and the model
    has multiple instruments). If `specific` is not None, it should be a
    tuple with the name of the datafiles for two instruments (matching
    `res.data_file`). In that case, this function works out the RV offset
    between the `specific[0]` and `specific[1]` instruments.
    """
    figures = []

    vsys = res.posterior_sample[:, -1]

    if res.arbitrary_units:
        units = ' (arbitrary)'
    else:
        units = ' (m/s)'  # if res.units == 'ms' else ' (km/s)'

    estimate = percentile68_ranges_latex(vsys) + units

    fig, ax = plt.subplots(1, 1)
    figures.append(fig)

    ax.hist(vsys, **kwargs)

    title = 'Posterior distribution for $v_{\\rm sys}$ \n %s' % estimate
    if kwargs.get('density', False):
        ylabel = 'posterior'
    else:
        ylabel = 'posterior samples'
    ax.set(xlabel='vsys' + units, ylabel=ylabel, title=title)

    if show_prior:
        try:
            low, upp = res.priors['Cprior'].interval(1)
            # if np.isinf(low) or np.isinf(upp):
            #     xx = np.linspace(vsys.min(), vsys.max(), 500)
            # else:
            #     xx = np.linspace(0.999*low, (1/0.999)*upp, 500)
            d = kwargs.get('density', False)
            ax.hist(res.priors['Cprior'].rvs(res.ESS), density=d, alpha=0.15,
                    color='k', zorder=-1)
            ax.legend(['posterior', 'prior'])

        except Exception as e:
            print(str(e))

    if res.save_plots:
        filename = 'kima-showresults-fig7.2.png'
        print('saving in', filename)
        fig.savefig(filename)

    if show_offsets and res.multi:
        n_inst_offsets = res.inst_offsets.shape[1]
        fig, axs = plt.subplots(1, n_inst_offsets, sharey=True,
                                figsize=(n_inst_offsets * 3, 5), squeeze=True,
                                constrained_layout=True)
        figures.append(fig)
        if n_inst_offsets == 1:
            axs = [
                axs,
            ]

        for i in range(n_inst_offsets):
            # wrt = get_instrument_name(res.data_file[-1])
            # this = get_instrument_name(res.data_file[i])
            wrt = res.instruments[-1]
            this = res.instruments[i]
            label = 'offset\n%s rel. to %s' % (this, wrt)
            a = res.inst_offsets[:, i]
            estimate = percentile68_ranges_latex(a) + units
            axs[i].hist(a)

            axs[i].set(xlabel=label, title=estimate,
                       ylabel='posterior samples')

        title = 'Posterior distribution(s) for instrument offset(s)'
        fig.suptitle(title)

        if res.save_plots:
            filename = 'kima-showresults-fig7.2.1.png'
            print('saving in', filename)
            fig.savefig(filename)

        if specific is not None:
            assert isinstance(specific, tuple), '`specific` should be a tuple'
            assert len(specific) == 2, '`specific` should have size 2'
            assert specific[
                0] in res.data_file, 'first element is not in res.data_file'
            assert specific[
                1] in res.data_file, 'second element is not in res.data_file'

            # all RV offsets are with respect to the last data file
            if res.data_file[-1] in specific:
                i = specific.index(res.data_file[-1])
                # toggle: if i is 0 it becomes 1, if it's 1 it becomes 0
                i ^= 1
                # wrt = get_instrument_name(res.data_file[-1])
                # this = get_instrument_name(specific[i])
                wrt = res.instruments[-1]
                this = res.instruments[i]
                label = 'offset\n%s rel. to %s' % (this, wrt)
                offset = res.inst_offsets[:, res.data_file.index(specific[i])]
                estimate = percentile68_ranges_latex(offset) + units
                fig, ax = plt.subplots(1, 1, constrained_layout=True)
                ax.hist(offset)
                ax.set(xlabel=label, title=estimate,
                       ylabel='posterior samples')
            else:
                # wrt = get_instrument_name(specific[1])
                # this = get_instrument_name(specific[0])
                wrt = res.instruments[specific[1]]
                this = res.instruments[specific[0]]
                label = 'offset\n%s rel. to %s' % (this, wrt)
                of1 = res.inst_offsets[:, res.data_file.index(specific[0])]
                of2 = res.inst_offsets[:, res.data_file.index(specific[1])]
                estimate = percentile68_ranges_latex(of1 - of2) + units
                fig, ax = plt.subplots(1, 1, constrained_layout=True)
                ax.hist(of1 - of2)
                ax.set(xlabel=label, title=estimate,
                       ylabel='posterior samples')

    else:
        figures.append(None)

    if res.return_figs:
        return figures


def hist_extra_sigma(res, show_prior=False, **kwargs):
    """
    Plot the histogram of the posterior for the additional white noise
    """
    if res.arbitrary_units:
        units = ' (arbitrary)'
    else:
        units = ' (m/s)'  # if res.units == 'ms' else ' (km/s)'

    if res.multi:  # there are n_instruments jitters
        # lambda substrs
        fig, axs = plt.subplots(1, res.n_instruments, sharey=True,
                                figsize=(res.n_instruments * 3, 5),
                                squeeze=True)
        for i, jit in enumerate(res.extra_sigma.T):
            # inst = get_instrument_name(res.data_file[i])
            inst = res.instruments[i]
            estimate = percentile68_ranges_latex(jit) + units
            axs[i].hist(jit)
            axs[i].set(xlabel='%s' % inst, title=estimate,
                       ylabel='posterior samples')

        title = 'Posterior distribution(s) for extra white noise(s)'
        fig.suptitle(title)

        #! missing show_prior

    else:
        estimate = percentile68_ranges_latex(res.extra_sigma)
        estimate += units

        fig, ax = plt.subplots(1, 1)
        ax.hist(res.extra_sigma, **kwargs)
        title = 'Posterior distribution for extra white noise $s$ \n %s' % estimate
        if kwargs.get('density', False):
            ylabel = 'posterior'
        else:
            ylabel = 'posterior samples'
        ax.set(xlabel='extra sigma' + units, ylabel=ylabel, title=title)

        if show_prior:
            try:
                # low, upp = res.priors['Jprior'].interval(1)
                d = kwargs.get('density', False)
                ax.hist(res.priors['Jprior'].rvs(res.ESS), density=d,
                        alpha=0.15, color='k', zorder=-1)
                ax.legend(['posterior', 'prior'])

            except Exception as e:
                print(str(e))

    if res.save_plots:
        filename = 'kima-showresults-fig7.3.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def hist_correlations(res):
    """ Plot the histogram of the posterior for the activity correlations """
    if not res.indcorrel:
        msg = 'Model has no activity correlations! '\
              'hist_correlations() doing nothing...'
        print(msg)
        return

    # units = ' (m/s)' if res.units=='ms' else ' (km/s)'
    # estimate = percentile68_ranges_latex(res.offset) + units

    n = len(res.activity_indicators)
    fig, axs = plt.subplots(n, 1, constrained_layout=True)

    for i, ax in enumerate(np.ravel(axs)):
        estimate = percentile68_ranges_latex(res.betas[:, i])
        estimate = '$c_{%s}$ = %s' % (res.activity_indicators[i], estimate)
        ax.hist(res.betas[:, i], label=estimate)
        ax.set(ylabel='posterior samples',
               xlabel='$c_{%s}$' % res.activity_indicators[i])
        leg = ax.legend(frameon=False)
        leg.legendHandles[0].set_visible(False)

    title = 'Posterior distribution for activity correlations'
    fig.suptitle(title)

    if res.save_plots:
        filename = 'kima-showresults-fig7.4.png'
        print('saving in', filename)
        fig.savefig(filename)


def hist_trend(res, per_year=True, show_prior=False):
    """
    Plot the histogram of the posterior for the coefficients of the trend
    """
    if not res.trend:
        print('Model has no trend! hist_trend() doing nothing...')
        return

    deg = res.trend_degree
    names = ['slope', 'quadr', 'cubic']
    if res.arbitrary_units:
        units = [' (/yr)', ' (/yr²)', ' (/yr³)']
    else:
        units = [' (m/s/yr)', ' (m/s/yr²)', ' (m/s/yr³)']

    trend = res.trendpars.copy()

    if per_year:  # transfrom from /day to /yr
        trend *= 365.25**np.arange(1, res.trend_degree + 1)

    fig, ax = plt.subplots(deg, 1, constrained_layout=True, squeeze=False)
    fig.suptitle('Posterior distribution for trend coefficients')
    for i in range(deg):
        estimate = percentile68_ranges_latex(trend[:, i]) + units[i]

        ax[i, 0].hist(trend[:, i].ravel(), label='posterior')
        if show_prior:
            prior = res.priors[names[i] + '_prior']
            f = 365.25**(i + 1) if per_year else 1.0
            ax[i, 0].hist(
                prior.rvs(res.ESS) * f, alpha=0.15, color='k', zorder=-1,
                label='prior')

        ax[i, 0].set(xlabel=f'{names[i]}{units[i]}',
                     title=f'{names[i]} = {estimate}')

        if show_prior:
            ax[i, 0].legend()

    fig.set_constrained_layout_pads(w_pad=0.3)
    fig.text(0.01, 0.5, 'posterior samples', rotation=90, va='center')

    if res.save_plots:
        filename = 'kima-showresults-fig7.5.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def hist_MA(res):
    """ Plot the histogram of the posterior for the MA parameters """
    if not res.MAmodel:
        print('Model has no MA! hist_MA() doing nothing...')
        return

    # units = ' (m/s/day)' # if res.units=='ms' else ' (km/s)'
    # estimate = percentile68_ranges_latex(res.trendpars) + units

    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
    ax1.hist(res.MA[:, 0])
    ax2.hist(res.MA[:, 1])
    title = 'Posterior distribution for MA parameters'
    fig.suptitle(title)
    ax1.set(xlabel=r'$\sigma$ MA [m/s]', ylabel='posterior samples')
    ax2.set(xlabel=r'$\tau$ MA [days]', ylabel='posterior samples')

    if res.save_plots:
        filename = 'kima-showresults-fig7.6.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def hist_nu(res, show_prior=False, **kwargs):
    """
    Plot the histogram of the posterior for the Student-t degrees of freedom
    """
    if not res.studentT:
        print('Model has Gaussian likelihood! hist_nu() doing nothing...')
        return

    estimate = percentile68_ranges_latex(res.nu)

    fig, ax = plt.subplots(1, 1)
    ax.hist(res.nu, **kwargs)
    title = 'Posterior distribution for degrees of freedom $\\nu$ \n '\
            '%s' % estimate
    if kwargs.get('density', False):
        ylabel = 'posterior'
    else:
        ylabel = 'posterior samples'
    ax.set(xlabel='$\\nu$', ylabel=ylabel, title=title)

    if show_prior:
        try:
            # low, upp = res.priors['Jprior'].interval(1)
            d = kwargs.get('density', False)
            ax.hist(res.priors['Jprior'].rvs(res.ESS), density=d, alpha=0.15,
                    color='k', zorder=-1)
            ax.legend(['posterior', 'prior'])

        except Exception as e:
            print(str(e))


def phase_plot(res, sample, highlight=None):
    """ Plot the phase curves given the solution in `sample` """
    # this is probably the most complicated function in the whole file!!

    if res.max_components == 0 and not res.KO:
        print('Model has no planets! phase_plot() doing nothing...')
        return

    # make copies to not change attributes
    t, y, e = res.t.copy(), res.y.copy(), res.e.copy()
    if t[0] > 24e5:
        t -= 24e5

    def kima_pars_to_keplerian_pars(p):
        # transforms kima planet pars (P,K,phi,ecc,w)
        # to pykima.keplerian.keplerian pars (P,K,ecc,w,T0,vsys=0)
        # assert p.size == res.n_dimensions
        P = p[0]
        phi = p[2]
        t0 = res.M0_epoch - (P * phi) / (2. * np.pi)
        return np.array([P, p[1], p[3], p[4], t0, 0.0])

    mc = res.max_components
    if res.KO:
        mc += res.nKO

    # get the planet parameters for this sample
    pars = sample[res.indices['planets']].copy()
    if res.KO:
        pars = np.r_[pars, sample[res.indices['KOpars']]]

    # sort by decreasing amplitude (arbitrary)
    ind = np.argsort(pars[1 * mc:2 * mc])[::-1]
    for i in range(res.n_dimensions):
        pars[i * mc:(i + 1) * mc] = pars[i * mc:(i + 1) * mc][ind]

    # (functions to) get parameters for individual planets
    def this_planet_pars(i):
        return pars[i::mc]

    def parsi(i):
        return kima_pars_to_keplerian_pars(this_planet_pars(i))

    # extract periods, phases and calculate times of periastron
    P = pars[0 * mc:1 * mc]
    phi = pars[2 * mc:3 * mc]
    T0 = res.M0_epoch - (P * phi) / (2. * np.pi)

    # how many planets in this sample?
    nplanets = (pars[:mc] != 0).sum()
    planetis = list(range(nplanets))

    if nplanets == 0:
        print('Sample has no planets! phase_plot() doing nothing...')
        return

    # get the model for this sample
    # (this adds in the instrument offsets and the systemic velocity)
    v = res.model(sample)

    # put all data around zero
    if res.GPmodel:
        # subtract the GP (already removes vsys, trend and instrument offsets)
        GPvel = res.GP.predict_with_hyperpars(res, sample, add_parts=False)
        y -= GPvel
    else:
        GPvel = np.zeros_like(t)

    y -= sample[-1]  # subtract this sample's systemic velocity

    if res.multi:
        # subtract each instrument's offset
        for i in range(res.n_instruments - 1):
            of = sample[res.indices['inst_offsets']][i]
            y[res.obs == i + 1] -= of

    if res.trend:
        # and subtract the trend
        trend_par = sample[res.indices['trend']]
        trend_par = np.r_[trend_par[::-1], 0.0]
        y -= np.polyval(trend_par, t - res.tmiddle)

    # if res.KO:  # subtract the known object
    #     allKOpars = sample[res.indices['KOpars']]
    #     for i in range(res.nKO):
    #         KOpars = kima_pars_to_keplerian_pars(allKOpars[i::res.nKO])
    #         KOvel = keplerian(t, *KOpars)
    #         y -= KOvel
    # else:
    KOvel = np.zeros_like(t)

    ekwargs = {
        'fmt': 'o',
        'mec': 'none',
        'ms': 5,
        'capsize': 0,
        'elinewidth': 0.8,
    }

    # very complicated logic just to make the figure the right size
    fs = [
        max(6.4, 6.4 + 1 * (nplanets - 2)),
        max(4.8, 4.8 + 1 * (nplanets - 3))
    ]
    if res.GPmodel:
        fs[1] += 3

    fig = plt.figure(constrained_layout=True, figsize=fs)
    fig.canvas.set_window_title('phase plot')
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

    if res.GPmodel:
        nrows += 1

    ncols = nplanets if nplanets <= 3 else 3
    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           height_ratios=[2] * (nrows - 1) + [1])
    gs_indices = {i: (i // 3, i % 3) for i in range(50)}

    # for each planet in this sample
    for i, letter in zip(range(nplanets), ascii_lowercase[1:]):
        ax = fig.add_subplot(gs[gs_indices[i]])

        p = P[i]
        t0 = T0[i]

        # plot the keplerian curve in phase (3 times)
        phase = np.linspace(0, 1, 200)
        tt = phase * p + t0
        vv = keplerian(tt, *parsi(i))
        for j in (-1, 0, 1):
            alpha = 0.3 if j in (-1, 1) else 1
            ax.plot(
                np.sort(phase) + j, vv[np.argsort(phase)], 'k', alpha=alpha)

        # the other planets which are not the ith
        other = copy(planetis)
        other.remove(i)

        # subtract the other planets from the data and plot it (the data)
        if res.multi:
            for k in range(1, res.n_instruments + 1):
                m = res.obs == k
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
                    if highlight:
                        if highlight in res.data_file[k - 1]:
                            alpha = 1
                        else:
                            alpha = 0.2
                    else:
                        alpha = 0.3 if j in (-1, 1) else 1

                    ax.errorbar(
                        np.sort(phase) + j, yy[np.argsort(phase)],
                        ee[np.argsort(phase)], color=color, alpha=alpha,
                        **ekwargs)

        else:
            phase = ((t - t0) / p) % 1.0
            other_planet_v = np.array([keplerian(t, *parsi(i)) for i in other])
            other_planet_v = other_planet_v.sum(axis=0)
            yy = y.copy()
            yy -= other_planet_v

            color = ax._get_lines.prop_cycler.__next__()['color']

            for j in (-1, 0, 1):
                alpha = 0.3 if j in (-1, 1) else 1
                ax.errorbar(
                    np.sort(phase) + j, yy[np.argsort(phase)],
                    e[np.argsort(phase)], color=color, alpha=alpha, **ekwargs)

        ax.set_xlim(-0.2, 1.2)
        ax.set(xlabel="phase", ylabel="RV [m/s]")
        ax.set_title('%s' % letter, loc='left')
        if nplanets == 1:
            k = parsi(i)[1]
            ecc = parsi(i)[2]
            title = f'P={p:.2f} days\n K={k:.2f} m/s  ecc={ecc:.2f}'
            ax.set_title(title, loc='right')
        else:
            ax.set_title('P=%.2f days' % p, loc='right')

    if res.GPmodel:
        axGP = fig.add_subplot(gs[1, :])
        if res.multi:
            for k in range(1, res.n_instruments + 1):
                m = res.obs == k
                if k < res.n_instruments:
                    of = sample[res.indices['inst_offsets']][k - 1]
                else:
                    of = 0.

                GPy = res.y[m] - sample[-1] - of
                axGP.errorbar(t[m], GPy, e[m], **ekwargs)
        else:
            axGP.errorbar(t, res.y - res.model(sample), e, **ekwargs)

        axGP.set(xlabel="Time [days]", ylabel="GP prediction [m/s]")

        # axGP.plot(t, GPvel, 'o-')
        # jitters = sample[res.indices['jitter']]
        tt = np.linspace(t[0], t[-1], 5000)
        pred, std = res.GP.predict_with_hyperpars(res, sample, tt,
                                                  return_std=True)
        axGP.plot(tt, pred, 'k')
        axGP.fill_between(tt, pred-2*std, pred+2*std, color='m', alpha=0.2)

        # names = ['GP'] + [get_instrument_name(d) for d in res.data_file]
        # leg = axGP.legend(names, loc="upper left", handletextpad=0.4,
        #                   bbox_to_anchor=(0, 1.17), ncol=6, fontsize=10)
        # for label, name in zip(leg.get_texts(), names):
        #     label.set_text(name)

    ax = fig.add_subplot(gs[-1, :])
    residuals = np.zeros_like(t)

    if res.multi:
        jitters = sample[res.indices['jitter']]
        print('residual rms per instrument')
        for k in range(1, res.n_instruments + 1):
            m = res.obs == k
            # label = res.data_file[k - 1]
            ax.errorbar(t[m], res.y[m] - v[m] - KOvel[m] - GPvel[m], e[m],
                        **ekwargs)
            ax.fill_between(t[m], -jitters[k - 1], jitters[k - 1], alpha=0.2)
            print(res.instruments[k - 1], end=': ')
            print(wrms(res.y[m] - v[m] - KOvel[m] - GPvel[m], 1 / e[m]**2))
            residuals[m] = res.y[m] - v[m] - KOvel[m] - GPvel[m]

    else:
        ax.errorbar(t, res.y - v - KOvel - GPvel, e, **ekwargs)
        residuals = res.y - v - KOvel - GPvel

    if res.studentT:
        outliers = find_outliers(res, sample)
        ax.errorbar(t[outliers], residuals[outliers], e[outliers], fmt='or',
                    ms=5)

    # ax.legend()
    ax.axhline(y=0, ls='--', alpha=0.5, color='k')
    ax.set_ylim(np.tile(np.abs(ax.get_ylim()).max(), 2) * [-1, 1])
    ax.set(xlabel='Time [days]', ylabel='residuals [m/s]')
    if res.studentT:
        rms1 = wrms(residuals, 1 / e**2)
        rms2 = wrms(residuals[~outliers], 1 / e[~outliers]**2)
        ax.set_title(f'rms={rms2:.2f} ({rms1:.2f}) m/s', loc='right')
    else:
        rms = wrms(residuals, 1 / e**2)
        ax.set_title(f'rms={rms:.2f} m/s', loc='right')

    if res.save_plots:
        filename = 'kima-showresults-fig6.1.png'
        print('saving in', filename)
        fig.savefig(filename)

    return residuals
