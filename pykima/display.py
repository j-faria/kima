from string import ascii_lowercase
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
from scipy.stats import gaussian_kde
from scipy.stats._continuous_distns import reciprocal_gen
from scipy.signal import find_peaks
from corner import corner

from .analysis import passes_threshold_np, find_outliers
from .GP import KERNEL
from .utils import (get_prior, hyperprior_samples, percentile68_ranges_latex,
                    wrms, lighten_color, get_instrument_name)

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
        '6': [res.plot_random_planets, {'show_vsys': True}],
        '6p': [
            'res.plot_random_planets(show_vsys=True);'\
            'res.phase_plot(res.maximum_likelihood_sample(Np=passes_threshold_np(res)))',
            {}
        ],
        '7': [
            (res.hist_vsys,
             res.hist_jitter,
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


def make_plot1(res, ax=None, errors=False):
    """ Plot the histogram of the posterior for Np """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    bins = np.arange(res.max_components + 2)
    nplanets = res.posterior_sample[:, res.index_component]
    n, _ = np.histogram(nplanets, bins=bins)
    ax.bar(bins[:-1], n, zorder=2)

    if errors:
        # from scipy.stats import multinomial
        prob = n / res.ESS
        # errors = multinomial(res.ESS, prob).rvs(1000).std(axis=0)
        errors = np.sqrt(res.ESS * prob * (1 - prob))
        ax.errorbar(bins[:-1], n, errors, fmt='.', ms=0, capsize=3, color='k')

    if res.removed_crossing:
        ic = res.index_component
        nn = (~np.isnan(res.posterior_sample[:, ic + 1:ic + 11])).sum(axis=1)
        nn, _ = np.histogram(nn, bins=bins)
        ax.bar(bins[:-1], nn, color='r', alpha=0.2, zorder=2)
        ax.legend(['all posterior samples', 'crossing orbits removed'])
    else:
        pt_Np = passes_threshold_np(res)
        ax.bar(pt_Np, n[pt_Np], color='C3', zorder=2)

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
        return ax.figure


def make_plot2(res, nbins=100, bins=None, plims=None, logx=True, density=False,
               kde=False, kde_bw=None, show_peaks=False, show_prior=False,
               show_year=True, show_timespan=True, **kwargs):
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

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        fig = ax.figure
    else:
        fig, ax = plt.subplots(1, 1)

    kwline = {'ls': '--', 'lw': 1.5, 'alpha': 0.3, 'zorder': -1}
    if show_year:  # mark 1 year
        year = 365.25
        ax.axvline(x=year, color='r', label='1 year', **kwline)
        # ax.axvline(x=year/2., ls='--', color='r', lw=3, alpha=0.6)
        # plt.axvline(x=year/3., ls='--', color='r', lw=3, alpha=0.6)
    if show_timespan:  # mark the timespan of the data
        ax.axvline(x=res.t.ptp(), color='k', label='time span', **kwline)

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
                prior_support = res.priors['Pprior'].support()
                if not np.isinf(prior_support).any():
                    start, end = prior_support
                else:
                    start, end = 1e-1, 1e7
                bins = 10**np.linspace(np.log10(start), np.log10(end), nbins)
            else:
                bins = 10**np.linspace(np.log10(plims[0]), np.log10(plims[1]),
                                       nbins)

        counts, bin_edges = np.histogram(T, bins=bins)
        ax.bar(x=bin_edges[:-1],
               height=counts / res.ESS,
               width=np.ediff1d(bin_edges),
               align='edge',
               alpha=0.8)
        # ax.hist(T, bins=bins, alpha=0.8, density=density)

        if show_prior and T.size > 100:
            kwprior = {
                'alpha': 0.15,
                'color': 'k',
                'zorder': -1,
                'label': 'prior',
            }

            if res.hyperpriors:
                P = hyperprior_samples(T.size)
            else:
                P = res.priors['Pprior'].rvs(T.size)

            counts, bin_edges = np.histogram(P, bins=bins)
            ax.bar(x=bin_edges[:-1],
                   height=counts / res.ESS,
                   width=np.ediff1d(bin_edges),
                   align='edge',
                   **kwprior)

    if kwargs.get('legend', True):
        ax.legend()

    if kde:
        ylabel = 'KDE density'
    else:
        ylabel = 'Number of posterior samples / ESS'
    ax.set(xscale='log' if logx else 'linear',
           xlabel=r'Period [days]',
           ylabel=ylabel)

    title = kwargs.get('title', True)
    if title:
        if isinstance(title, str):
            ax.set_title(title)
        else:
            ax.set_title('Posterior distribution for the orbital period(s)')

    ax.set_ylim(bottom=0)

    if plims is not None:
        ax.set_xlim(plims)

    if res.save_plots:
        filename = 'kima-showresults-fig2.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def make_plot3(res, points=True, gridsize=50, **kwargs):
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

    if 'ax1' in kwargs and 'ax2' in kwargs:
        ax1, ax2 = kwargs.pop('ax1'), kwargs.pop('ax2')
        fig = ax1.figure
    else:
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


def make_plot4(res, Np=None, ranges=None, show_prior=False, **hist_kwargs):
    """
    Plot histograms for the GP hyperparameters. If Np is not None, highlight
    the samples with Np Keplerians.
    """
    if not res.GPmodel:
        print('Model does not have GP! make_plot4() doing nothing...')
        return

    if res.model == 'RVFWHMmodel':
        return make_plot4_rvfwhm(res)

    n = res.etas.shape[1]
    available_etas = [f'eta{i}' for i in range(1, n + 1)]
    labels = [f'eta{i}' for i in range(1, n + 1)]
    if ranges is None:
        ranges = len(labels) * [None]

    if Np is not None:
        m = res.posterior_sample[:, res.index_component] == Np

    fig, axes = plt.subplots(2, int(np.ceil(len(available_etas) / 2)))

    if res.GPkernel is KERNEL.celerite:
        axes[-1, -1].axis('off')

    for i, eta in enumerate(available_etas):
        ax = np.ravel(axes)[i]
        ax.hist(getattr(res, eta), bins=40, range=ranges[i], **hist_kwargs)

        if show_prior:
            try:
                prior = res.priors[eta + '_prior']
                logprior = False
            except KeyError:
                prior = res.priors['log_' + eta + '_prior']
                logprior = True

            if logprior:
                ax.hist(np.exp(prior.rvs(res.ESS)), bins=40, color='k', alpha=0.2)
            else:
                ax.hist(prior.rvs(res.ESS), bins=40, color='k', alpha=0.2)

        if Np is not None:
            ax.hist(eta[m],
                    bins=40,
                    histtype='step',
                    alpha=0.5,
                    label='$N_p$=%d samples' % Np,
                    range=ranges[i])
            ax.legend()

        ax.set(xlabel=labels[i], ylabel='posterior samples')

    for j in range(i + 1, 2 * nplots):
        np.ravel(axes)[j].axis('off')

    if show_prior:
        axes[0, 0].legend(
            ['posterior', 'prior'],
            bbox_to_anchor=(-0.1, 1.25), ncol=2,
            loc='upper left',
        )

    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout()

    if res.save_plots:
        filename = 'kima-showresults-fig4.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def make_plot4_rvfwhm(res, Np=None, ranges=None):
    """
    Plot histograms for the GP hyperparameters. If Np is not None, highlight
    the samples with Np Keplerians. 
    """
    if not res.GPmodel:
        print('Model does not have GP! make_plot4() doing nothing...')
        return

    n = res.etas.shape[1]
    labels = [f'eta{i}' for i in range(1, n + 1)]
    if ranges is None:
        ranges = len(labels) * [None]

    if Np is not None:
        m = res.posterior_sample[:, res.index_component] == Np

    fig = plt.figure(constrained_layout=True)

    if res.GPkernel == 'standard':
        gs = fig.add_gridspec(6, 2)
    elif res.GPkernel == 'qpc':
        gs = fig.add_gridspec(6, 3)

    histkw = dict(density=True, color='C0')
    allkw = dict(yticks=[])

    if res.GPkernel == 'qpc':
        ax1 = fig.add_subplot(gs[0:3, 0])
        ax1.hist(res.etas[:, 0], **histkw)
        ax1.set(xlabel=r'$\eta_1$ RV [m/s]', ylabel='posterior', **allkw)
        ax1.set_xlim((0, None))

        ax2 = fig.add_subplot(gs[0:3, 1])
        ax2.hist(res.etas[:, 1], **histkw)
        ax2.set(xlabel=r'$\eta_1$ FWHM [m/s]', ylabel='posterior', **allkw)
        ax2.set_xlim((0, None))

        ax = fig.add_subplot(gs[3:6, 0], sharex=ax1)
        ax.hist(res.etas[:, -2], **histkw)
        ax.set(xlabel=r'$\eta_5$ RV [m/s]', ylabel='posterior', **allkw)

        ax = fig.add_subplot(gs[3:6, 1], sharex=ax2)
        ax.hist(res.etas[:, -1], **histkw)
        ax.set(xlabel=r'$\eta_5$ FWHM [m/s]', ylabel='posterior', **allkw)

        col = 2

    else:
        ax1 = fig.add_subplot(gs[0:3, 0])
        ax1.hist(res.etas[:, 0], **histkw)
        ax1.set(xlabel=r'$\eta_1$ RV [m/s]', ylabel='posterior', **allkw)
        ax1.set_xlim((0, None))

        ax2 = fig.add_subplot(gs[3:6, 0])
        ax2.hist(res.etas[:, 1], **histkw)
        ax2.set(xlabel=r'$\eta_1$ FWHM [m/s]', ylabel='posterior', **allkw)
        ax2.set_xlim((0, None))

        col = 1

    units = [' [days]', ' [days]', '']
    for i in range(3):
        ax = fig.add_subplot(gs[2*i:2*(i+1), col])
        ax.hist(res.etas[:, 2+i], **histkw)
        ax.set(xlabel=fr'$\eta_{2+i}$' + units[i], ylabel='posterior', **allkw)
    # fig, axes = plt.subplots(2, int(np.ceil(n / 2)))
    # axes = np.ravel(axes)
    # for i, eta in enumerate(res.etas.T):
    #     ax = axes[i]
    #     ax.hist(eta, bins=40, range=ranges[i])

    #     if Np is not None:
    #         ax.hist(eta[m],
    #                 bins=40,
    #                 histtype='step',
    #                 alpha=0.5,
    #                 label='$N_p$=%d samples' % Np,
    #                 range=ranges[i])
    #         ax.legend()

    #     ax.set(xlabel=labels[i], ylabel='posterior samples')

    # for ii in range(i+1, len(fig.axes)):
    #     axes[ii].axis('off')

    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # if res.save_plots:
    #     filename = 'kima-showresults-fig4.png'
    #     print('saving in', filename)
    #     fig.savefig(filename)

    # if res.return_figs:
    #     return fig


def make_plot5(res, include_jitters=False, show=True, ranges=None):
    """ Corner plot for the GP hyperparameters """

    if not res.GPmodel:
        print('Model does not have GP! make_plot5() doing nothing...')
        return

    data = []
    labels = []

    if include_jitters:
        if res.multi:
            instruments = [get_instrument_name(df) for df in res.data_file]
            if res.model == 'RVFWHMmodel':
                labels += [rf'$s_{{\rm {i}}}^{{\rm RV}}$' for i in instruments]
                labels += [rf'$s_{{\rm {i}}}^{{\rm FWHM}}$' for i in instruments]
            else:
                labels += [rf'$s_{{\rm {i}}}$' for i in instruments]

        data.append(res.jitter)

    if res.model == 'RVFWHMmodel':
        labels += [r'$\eta_1^{RV}$', r'$\eta_1^{FWHM}$']
        labels += [f'$\eta_{i}$' for i in range(2, 5)]
    else:
        labels += [f'$\eta_{i}$' for i in range(1, 5)]

    data.append(res.etas)

    data = np.hstack(data)

    if data.shape[0] < data.shape[1]:
        print('Not enough samples to make the corner plot')
        return

    rc = {
        'font.size': 6,
    }
    with plt.rc_context(rc):
        fig = corner(data, show_titles=True, labels=labels, titles=labels,
                     plot_datapoints=True, plot_contours=False,
                     plot_density=False)

    fig.subplots_adjust(top=0.95, bottom=0.1, wspace=0, hspace=0)
    # for ax in fig.axes:
    # ax.yaxis.set_label_coords(0.5, 0.5, fig.transFigure)

    # available_etas = ['eta1', 'eta2', 'eta3', 'eta4']
    # labels = [r'$s$'] * res.n_jitters
    # labels += [r'$\eta_%d$' % (i + 1) for i, _ in enumerate(available_etas)]
    # units = ['m/s'] * res.n_jitters + ['m/s', 'days', 'days', None]
    # xlabels = []
    # if res.multi:
    #     for i in range(res.n_jitters):
    #         label = r'%s$_{{\rm %s}}}$' % (labels[i], res.instruments[i])
    #         xlabels.append(label)
    # else:
    #     xlabels.append(labels[0])

    # for label, unit in zip(labels[res.n_jitters:], units[res.n_jitters:]):
    #     xlabels.append(label)
    #     #    ' (%s)' % unit if unit is not None else label)

    # # all Np together
    # res.post_samples = np.c_[res.extra_sigma, res.etas]
    # # if res.multi:
    # #     variables = list(res.extra_sigma.T)
    # # else:
    # #     variables = [res.extra_sigma]

    # # for eta in available_etas:
    # #     variables.append(getattr(res, eta))

    # # res.post_samples = np.vstack(variables).T
    # # ranges = [1.]*(len(available_etas) + res.extra_sigma.shape[1])

    # if ranges is None:
    #     ranges = [1.] * res.post_samples.shape[1]
    # # ranges[3] = (res.pmin, res.pmax)

    # try:
    #     res.corner1 = corner(
    #         res.post_samples,
    #         labels=xlabels,
    #         show_titles=True,
    #         plot_contours=False,
    #         plot_datapoints=False,
    #         plot_density=True,
    #         # fill_contours=True,
    #         smooth=True,
    #         contourf_kwargs={
    #             'cmap': plt.get_cmap('afmhot'),
    #             'colors': None
    #         },
    #         hexbin_kwargs={
    #             'cmap': plt.get_cmap('afmhot_r'),
    #             'bins': 'log'
    #         },
    #         hist_kwargs={'density': True},
    #         range=ranges,
    #         data_kwargs={'alpha': 1},
    #     )
    # except AssertionError as exc:
    #     print('AssertionError from corner in make_plot5()', end='')
    #     if "I don't believe" in str(exc):
    #         print(', you probably need to get more posterior samples')
    #     return

    # res.corner1.suptitle(
    #     'Joint and marginal posteriors for GP hyperparameters')

    # if show:
    #     res.corner1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # if res.save_plots:
    #     filename = 'kima-showresults-fig5.png'
    #     print('saving in', filename)
    #     res.corner1.savefig(filename)

    # if res.return_figs:
    #     return res.corner1


def corner_all(res):
    n = max(10000, res.ESS)

    values = res.posterior_sample[:, res.indices['jitter']]
    prior_rvs = []
    for p in res.parameter_priors[res.indices['jitter']]:
        prior_rvs.append(p.rvs(n))

    for par in ('planets', 'vsys'):
        values = np.c_[values, res.posterior_sample[:, res.indices[par]]]
        prior = res.parameter_priors[res.indices[par]]
        if isinstance(prior, list):
            for p in prior:
                prior_rvs.append(p.rvs(n))
        else:
            prior_rvs.append(prior.rvs(n))

    hkw = dict(density=True)
    fig = corner(values, color='C0', hist_kwargs=hkw,
                 plot_density=False, plot_contours=False, plot_datapoints=True)
    xlims = [ax.get_xlim() for ax in fig.axes]

    hkw = dict(density=True, alpha=0.5)
    fig = corner(np.array(prior_rvs).T, fig=fig, color='k', hist_kwargs=hkw,
                 plot_density=False, plot_contours=False, plot_datapoints=False)

    for xlim, ax in zip(xlims, fig.axes):
        ax.set_xlim(xlim)


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


def hist_jitter(res, show_prior=False, **kwargs):
    """
    Plot the histogram of the posterior for the additional white noise
    """
    if res.arbitrary_units:
        units = ' (arbitrary)'
    else:
        units = ' (m/s)'  # if res.units == 'ms' else ' (km/s)'

    kw = dict(constrained_layout=True)
    if res.model == 'RVFWHMmodel':
        fig, axs = plt.subplots(2, res.n_instruments, **kw)
    elif res.model in ['RVmodel','RV_binaries_model']:
        fig, axs = plt.subplots(1, res.n_instruments, **kw)
    fig.suptitle('Posterior distribution for extra white noise')
    axs = np.ravel(axs)

    for i, ax in enumerate(axs):
        estimate = percentile68_ranges_latex(res.jitter[:, i]) + ' m/s'
        ax.hist(res.jitter[:, i])
        leg = ax.legend([], title=estimate)
        leg._legend_box.sep = 0

    for ax in axs:
        ax.set(yticks=[], ylabel='posterior')

    insts = [get_instrument_name(i) for i in res.instruments]
    if res.model == 'RVFWHMmodel':
        labels = [f'RV jitter {i} [m/s]' for i in insts]
        labels += [f'FWHM jitter {i} [m/s]' for i in insts]
    else:
        labels = [f'jitter {i} [m/s]' for i in insts]

    for ax, label in zip(axs, labels):
        ax.set_xlabel(label)

    return
    if res.multi:  # there are n_instruments jitters
        # lambda substrs
        fig, axs = plt.subplots(1, res.n_instruments, sharey=True,
                                figsize=(res.n_instruments * 3, 5),
                                squeeze=True)
        for i, jit in enumerate(res.jitter.T):
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


def hist_trend(res, per_year=True, show_prior=False, ax=None):
    """
    Plot the histogram of the posterior for the coefficients of the trend
    """
    if not res.trend:
        print('Model has no trend! hist_trend() doing nothing...')
        return

    deg = res.trend_degree
    names = ['slope', 'quadr', 'cubic']
    if res.arbitrary_units:
        units = ['/yr', '/yr²', '/yr³']
    else:
        units = ['m/s/yr', 'm/s/yr²', 'm/s/yr³']

    trend = res.trendpars.copy()

    if per_year:  # transfrom from /day to /yr
        trend *= 365.25**np.arange(1, res.trend_degree + 1)

    if ax is not None:
        ax = np.atleast_1d(ax)
        assert len(ax) == deg, f'wrong length, need {deg} axes'
        fig = ax[0].figure
    else:
        fig, ax = plt.subplots(deg, 1, constrained_layout=True, squeeze=True)
    print(ax.shape)

    fig.suptitle('Posterior distribution for trend coefficients')
    for i in range(deg):
        estimate = percentile68_ranges_latex(trend[:, i]) + ' ' + units[i]

        ax[i].hist(trend[:, i].ravel(), label='posterior')
        if show_prior:
            prior = res.priors[names[i] + '_prior']
            f = 365.25**(i + 1) if per_year else 1.0
            ax[i].hist(
                prior.rvs(res.ESS) * f, alpha=0.15, color='k', zorder=-1,
                label='prior')

        ax[i].set(xlabel=f'{names[i]} ({units[i]})')

        if show_prior:
            ax[i].legend(title=estimate)
        else:
            leg = ax[i].legend([], [], title=estimate)
            leg._legend_box.sep = 0


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


def plot_data(res, ax=None, y=None, extract_offset=True, legend=True,
              show_rms=False, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if y is None:
        y = res.y.copy()

    assert y.size == res.t.size, 'wrong dimensions!'

    if extract_offset:
        y_offset = round(y.mean(), 0) if abs(y.mean()) > 100 else 0
    else:
        y_offset = 0

    if res.multi:
        for j in range(res.n_instruments):
            inst = res.instruments[j]
            m = res.obs == j + 1
            kw = dict(fmt='o', label=inst)
            kw.update(**kwargs)
            ax.errorbar(res.t[m], y[m] - y_offset, res.e[m], **kw)
    else:
        ax.errorbar(res.t, y - y_offset, res.e, fmt='o')

    if legend:
        ax.legend(loc='upper left')

    if res.multi:
        kw = dict(color='b', lw=2, alpha=0.1, zorder=-2)
        for ot in res._offset_times:
            ax.axvline(ot, **kw)

    if res.arbitrary_units:
        lab = dict(xlabel='Time [days]', ylabel='Q [arbitrary]')
    else:
        lab = dict(xlabel='Time [days]', ylabel='RV [m/s]')
    ax.set(**lab)

    if show_rms:
        # if res.studentT:
        #     outliers = find_outliers(res)
        #     rms1 = wrms(y, 1 / res.e**2)
        #     rms2 = wrms(y[~outliers], 1 / res.e[~outliers]**2)
        #     ax.set_title(f'rms: {rms2:.2f} ({rms1:.2f}) m/s', loc='right')
        # else:
        rms = wrms(y, 1 / res.e**2)
        ax.set_title(f'rms: {rms:.2f} m/s', loc='right')

    if y_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y_offset)] + str(int(abs(y_offset)))
        fs = ax.xaxis.get_label().get_fontsize()
        ax.set_title(offset, loc='left', fontsize=fs)

    return ax, y_offset


def phase_plot(res,
               sample,
               highlight=None,
               only=None,
               phase_axs=None,
               add_titles=True,
               highlight_points=None,
               sort_by_decreasing_K=True,
               show_gls_residuals=False):
    """ Plot the phase curves given the solution in `sample` """
    # this is probably the most complicated function in the whole package!!

    if res.max_components == 0 and not res.KO:
        print('Model has no planets! phase_plot() doing nothing...')
        return

    # make copies to not change attributes
    t, y, e = res.t.copy(), res.y.copy(), res.e.copy()
    M0_epoch = res.M0_epoch
    if t[0] > 24e5:
        t -= 24e5
        M0_epoch -= 24e5

    def kima_pars_to_keplerian_pars(p):
        # transforms kima planet pars (P,K,phi,ecc,w)
        # to pykima.keplerian.keplerian pars (P,K,ecc,w,T0,vsys=0)
        # assert p.size == res.n_dimensions
        P = p[0]
        phi = p[2]
        t0 = M0_epoch - (P * phi) / (2. * np.pi)
        try:
            return np.array([P, p[1], p[3], p[4], p[5], t0, 0.0])
        except IndexError:
            return np.array([P, p[1], p[3], p[4], 0, t0, 0.0])


    if highlight_points is not None:
        hlkw = dict(fmt='*', ms=6, color='y', zorder=2)
        hl = highlight_points
        highlight_points = True

    nd = res.n_dimensions
    mc = res.max_components
    if res.KO:
        mc += res.nKO

    # get the planet parameters for this sample
    pars = sample[res.indices['planets']].copy()

    if res.KO:
        k = pars.size // nd - 1
        for i in range(res.nKO):
            KOpars = sample[res.indices['KOpars']][i::res.nKO]
            if pars.size == 0:
                pars = KOpars
            else:
                pars = np.insert(pars, range(1 + k, (1 + k) * 6, 1 + k),
                                 KOpars)
                k += 1

    if sort_by_decreasing_K:
        # sort by decreasing amplitude (arbitrary)
        ind = np.argsort(pars[1 * mc:2 * mc])[::-1]
        for i in range(nd):
            pars[i * mc:(i + 1) * mc] = pars[i * mc:(i + 1) * mc][ind]

    # (functions to) get parameters for individual planets
    def this_planet_pars(i):
        return pars[i::mc]

    def parsi(i):
        return kima_pars_to_keplerian_pars(this_planet_pars(i))

    # extract periods, phases and calculate times of periastron
    P = pars[0 * mc:1 * mc]
    phi = pars[2 * mc:3 * mc]
    T0 = M0_epoch - (P * phi) / (2. * np.pi)

    # how many planets in this sample?
    nplanets = (pars[:mc] != 0).sum()
    planetis = list(range(nplanets))

    if nplanets == 0:
        print('Sample has no planets! phase_plot() doing nothing...')
        return

    # get the model for this sample
    # (this adds in the instrument offsets and the systemic velocity)
    # v = res.eval_model(sample)[0]

    # put all data around zero
    if res.GPmodel:
        # subtract the GP
        if res.model == 'RVFWHMmodel':
            GPvel = res.stochastic_model(sample)[0]
        else:
            GPvel = res.stochastic_model(sample)
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
    # KOvel = np.zeros_like(t)

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

    if show_gls_residuals:
        ncols += 1
        # fig.set_size_inches(fs[0], fs[1])

    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           height_ratios=[2] * (nrows - 1) + [1])
    gs_indices = {i: (i // 3, i % 3) for i in range(50)}

    # for each planet in this sample
    for i, letter in zip(range(nplanets), ascii_lowercase[1:]):
        if phase_axs is None:
            ax = fig.add_subplot(gs[gs_indices[i]])
        else:
            try:
                ax = phase_axs[i]
            except IndexError:
                continue

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
                    alpha = 0.2 if j in (-1, 1) else 1
                    if highlight:
                        if highlight not in res.data_file[k - 1]:
                            alpha = 0.2
                    elif only:
                        if only not in res.data_file[k - 1]:
                            alpha = 0

                    ax.errorbar(
                        np.sort(phase) + j, yy[np.argsort(phase)],
                        ee[np.argsort(phase)], color=color, alpha=alpha,
                        **ekwargs)

                    if highlight_points:
                        hlm = (m & hl)[m]
                        ax.errorbar(np.sort(phase[hlm]) + j,
                                    yy[np.argsort(phase[hlm])],
                                    ee[np.argsort(phase[hlm])],
                                    alpha=alpha, **hlkw)


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
        if add_titles:
            title_kwargs = dict(fontsize=12)
            ax.set_title('%s' % letter, loc='left', **title_kwargs)
            if nplanets == 1:
                k = parsi(i)[1]
                ecc = parsi(i)[2]
                title = f'P={p:.2f} days\n K={k:.2f} m/s  ecc={ecc:.2f}'
                ax.set_title(title, loc='right', **title_kwargs)
            else:
                ax.set_title('P=%.2f days' % p, loc='right', **title_kwargs)

    end = -1 if show_gls_residuals else None

    ## GP panel
    ###########
    if res.GPmodel:
        axGP = fig.add_subplot(gs[1, :end])
        _, y_offset = plot_data(res, ax=axGP, legend=False, **ekwargs)
        axGP.set(xlabel="Time [days]", ylabel="GP [m/s]")

        tt = np.linspace(t[0], t[-1], 3000)
        no_planets_model = res.eval_model(sample, tt, include_planets=False)
        no_planets_model = res.burst_model(sample, tt, no_planets_model)

        if res.model in ['RVmodel','RV_binaries_model']:
            pred, std = res.stochastic_model(sample, tt, return_std=True)
        elif res.model == 'RVFWHMmodel':
            (pred, _), (std, _) = res.stochastic_model(sample, tt,
                                                       return_std=True)
            no_planets_model = no_planets_model[0]

        pred += no_planets_model - y_offset
        axGP.plot(tt, pred, 'k')
        axGP.fill_between(tt,
                          pred - 2 * std,
                          pred + 2 * std,
                          color='m',
                          alpha=0.2)

    ## residuals
    ############
    ax = fig.add_subplot(gs[-1, :end])
    residuals = res.residuals(sample, full=True)
    plot_data(res, ax=ax, y=residuals, legend=False, show_rms=True, **ekwargs)

    # if highlight_points:
    #     ax.errorbar(t[hl], residuals[hl], e[hl], **hlkw)

    if res.studentT:
        outliers = find_outliers(res, sample)
        ax.errorbar(res.t[outliers], residuals[outliers], res.e[outliers],
                    fmt='xr', ms=7, lw=3)

    ax.axhline(y=0, ls='--', alpha=0.5, color='k')
    ax.set_ylim(np.tile(np.abs(ax.get_ylim()).max(), 2) * [-1, 1])
    ax.set(xlabel='Time [BJD]', ylabel='r [m/s]')
    title_kwargs = dict(loc='right', fontsize=12)


    if show_gls_residuals:
        axp = fig.add_subplot(gs[-2:, -1])
        from astropy.timeseries import LombScargle
        gls = LombScargle(res.t, residuals, res.e)
        freq, power = gls.autopower()
        axp.semilogy(power, 1/freq, 'k', alpha=0.6)

        kwl = dict(color='k', alpha=0.2, ls='--')
        kwt = dict(color='k', alpha=0.3, rotation=90, ha='left', va='top', fontsize=9)
        fap001 = gls.false_alarm_level(0.01)
        axp.axvline(fap001, **kwl)
        axp.text(0.98*fap001, 1/freq.min(), '1%', **kwt)

        fap01 = gls.false_alarm_level(0.1)
        axp.axvline(fap01, **kwl)
        axp.text(0.98*fap01, 1/freq.min(), '10%', **kwt)

        axp.set(xlabel='residual power', ylabel='Period [days]')
        axp.invert_xaxis()
        axp.yaxis.tick_right()
        axp.yaxis.set_label_position('right')


    if res.save_plots:
        filename = 'kima-showresults-fig6.1.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig

    return residuals


def plot_random_samples(res,
                        ncurves=50,
                        samples=None,
                        over=0.1,
                        show_vsys=False,
                        ntt=5000,
                        isolate_known_object=True,
                        full_plot=False,
                        **kwargs):

    if samples is None:
        samples = res.posterior_sample
        mask = np.ones(samples.shape[0], dtype=bool)
    else:
        samples = np.atleast_2d(samples)

    t = res.t.copy()
    M0_epoch = res.M0_epoch
    if t[0] > 24e5:
        t -= 24e5
        M0_epoch -= 24e5

    tt = res._get_tt(ntt, over)
    if res.GPmodel:
        ttGP = res._get_ttGP()

    if t.size > 100:
        ncurves = min(10, ncurves)

    ncurves = min(ncurves, samples.shape[0])

    if samples.shape[0] == 1:
        ii = np.zeros(1, dtype=int)
    elif ncurves == samples.shape[0]:
        ii = np.arange(ncurves)
    else:
        # select `ncurves` indices from the 70% highest likelihood samples
        lnlike = res.posterior_lnlike[:, 1]
        sorted_lnlike = np.sort(lnlike)[::-1]
        mask_lnlike = lnlike > np.percentile(sorted_lnlike, 70)
        ii = np.random.choice(np.where(mask & mask_lnlike)[0], ncurves)

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        fig = ax.figure
    else:
        fig, ax = plt.subplots(1, 1)

    _, y_offset = plot_data(res, ax)

    ## plot the Keplerian curves
    alpha = 0.1 if ncurves > 1 else 1

    cc = kwargs.get('curve_color', 'k')
    gpc = kwargs.get('gp_color', 'plum')

    for icurve, i in enumerate(ii):
        sample = samples[i]
        stoc_model = np.atleast_2d(res.stochastic_model(sample, tt))
        model = np.atleast_2d(res.eval_model(sample, tt))
        offset_model = res.eval_model(sample, tt, include_planets=False)

        if res.multi:
            model = res.burst_model(sample, tt, model)
            offset_model = res.burst_model(sample, tt, offset_model)

        ax.plot(tt, (stoc_model + model).T - y_offset, color=cc, alpha=alpha)
        if res.GPmodel:
            ax.plot(tt, (stoc_model + offset_model).T - y_offset, color=gpc,
                    alpha=alpha)

        if show_vsys:
            kw = dict(alpha=alpha, color='r', ls='--')
            if res.multi:
                for j in range(res.n_instruments):
                    instrument_mask = res.obs == j + 1
                    start = t[instrument_mask].min()
                    end = t[instrument_mask].max()
                    m = np.where((tt > start) & (tt < end))
                    ax.plot(tt[m], offset_model[m] - y_offset, **kw)
            else:
                ax.plot(tt, offset_model - y_offset, **kw)

        if res.KO and isolate_known_object:
            for k in range(1, res.nKO + 1):
                kepKO = res.eval_model(res.posterior_sample[i], tt,
                                       single_planet=-k)
                ax.plot(tt, kepKO - y_offset, 'g-', alpha=alpha)

    if res.save_plots:
        filename = 'kima-showresults-fig6.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def plot_random_samples_rvfwhm(res,
                               ncurves=50,
                               samples=None,
                               over=0.1,
                               show_vsys=False,
                               Np=None,
                               ntt=10000,
                               **kwargs):
    """
    Display the RV data together with curves from the posterior predictive.
    A total of `ncurves` random samples are chosen, and the Keplerian 
    curves are calculated covering 100 + `over`% of the data timespan.
    If the model has a GP component, the prediction is calculated using the
    GP hyperparameters for each of the random samples.
    """
    colors = [cc['color'] for cc in plt.rcParams["axes.prop_cycle"]]

    if samples is None:
        samples = res.posterior_sample
        mask = np.ones(samples.shape[0], dtype=bool)
    else:
        samples = np.atleast_2d(samples)

    t = res.t.copy()
    M0_epoch = res.M0_epoch
    if t[0] > 24e5:
        t -= 24e5
        M0_epoch -= 24e5

    tt = np.linspace(t.min() - over * t.ptp(), t.max() + over * t.ptp(),
                     ntt + int(100 * over))

    if res.GPmodel:
        # let's be more reasonable for the number of GP prediction points
        #! OLD: linearly spaced points (lots of useless points within gaps)
        #! ttGP = np.linspace(t[0], t[-1], 1000 + t.size*3)
        #! NEW: have more points near where there is data
        kde = gaussian_kde(t)
        ttGP = kde.resample(25000 + t.size * 3).reshape(-1)
        # constrain ttGP within observed times, to not waste
        ttGP = (ttGP + t[0]) % t.ptp() + t[0]
        ttGP = np.r_[ttGP, t]
        ttGP.sort()  # in-place

        if t.size > 100:
            ncurves = 10

    y = res.y.copy()
    yerr = res.e.copy()

    y2 = res.y2.copy()
    y2err = res.e2.copy()

    y_offset = round(y.mean(), 0) if abs(y.mean()) > 100 else 0
    y2_offset = round(y2.mean(), 0) if abs(y2.mean()) > 100 else 0

    ncurves = min(ncurves, samples.shape[0])

    if samples.shape[0] == 1:
        ii = np.zeros(1, dtype=int)
    elif ncurves == samples.shape[0]:
        ii = np.arange(ncurves)
    else:
        # select `ncurves` indices from the 70% highest likelihood samples
        lnlike = res.posterior_lnlike[:, 1]
        sorted_lnlike = np.sort(lnlike)[::-1]
        mask_lnlike = lnlike > np.percentile(sorted_lnlike, 70)
        ii = np.random.choice(np.where(mask & mask_lnlike)[0], ncurves)

    if 'ax1' in kwargs and 'ax2' in kwargs:
        ax1, ax2 = kwargs.pop('ax1'), kwargs.pop('ax2')
        fig = ax1.figure
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                       constrained_layout=True)

    ax1.set_title('Posterior samples in data space')

    ## plot the Keplerian curves
    alpha = 0.1 if ncurves > 1 else 1

    for icurve, i in enumerate(ii):
        stoc_model = res.stochastic_model(samples[i], tt)
        model = res.eval_model(samples[i], tt)

        if res.multi:
            model = res.burst_model(samples[i], tt, model)

        ax1.plot(tt, stoc_model[0] + model[0] - y_offset, 'k', alpha=alpha)
        # ax2.plot(tt, stoc_model[1] + model[1], 'k', alpha=alpha)

        offset_model = res.eval_model(samples[i], tt, include_planets=False)
        if res.multi:
            model = res.burst_model(samples[i], tt, offset_model)

        if res.GPmodel:
            kw = dict(color='plum', alpha=alpha)
            ax1.plot(tt, stoc_model[0] + offset_model[0] - y_offset, **kw)
            ax2.plot(tt, stoc_model[1] + offset_model[1] - y2_offset, **kw)

        if show_vsys:
            kw = dict(alpha=0.1, color='r', ls='--')
            if res.multi:
                for j in range(res.n_instruments):
                    instrument_mask = res.obs == j + 1
                    start = t[instrument_mask].min()
                    end = t[instrument_mask].max()
                    m = np.where( (tt > start) & (tt < end) )
                    ax1.plot(tt[m], offset_model[0][m] - y_offset, **kw)
                    ax2.plot(tt[m], offset_model[1][m] - y2_offset, **kw)
            else:
                ax1.plot(tt, offset_model[0] - y_offset, **kw)
                ax2.plot(tt, offset_model[1] - y2_offset, **kw)

        continue

    ## plot the data
    if res.multi:
        for j in range(res.inst_offsets.shape[1] // 2 + 1):
            inst = res.instruments[j]
            m = res.obs == j + 1

            kw = dict(fmt='o', ms=3, color=colors[j], label=inst)
            kw.update(**kwargs)
            ax1.errorbar(t[m], y[m] - y_offset, yerr[m], **kw)
            ax2.errorbar(t[m], res.y2[m] - y2_offset, res.e2[m], **kw)

        ax1.legend(loc='upper left', fontsize=8)

    else:
        ax1.errorbar(t, y - y_offset, yerr, fmt='o')
        ax2.errorbar(t, y2 - y2_offset, res.e2, fmt='o')

    if res.multi:
        kw = dict(color='b', lw=2, alpha=0.1, zorder=-2)
        ax1.vlines(res._offset_times, *ax1.get_ylim(), **kw)
        ax2.vlines(res._offset_times, *ax2.get_ylim(), **kw)

    if y_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y_offset)] + str(int(abs(y_offset)))
        ax1.set_title(offset, loc='left', fontsize=9)
    if y2_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y2_offset)] + str(int(abs(y2_offset)))
        ax2.set_title(offset, loc='left', fontsize=9)

    if res.arbitrary_units:
        ylabel = 'Q [arbitrary]'
    else:
        ylabel = 'RV [m/s]'

    ax1.set_ylabel(ylabel)
    ax2.set(xlabel='Time [days]', ylabel=f'FWHM [m/s]')

    if res.save_plots:
        filename = 'kima-showresults-fig6.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig

    return fig, ax1, ax2


def orbit(res, star_mass=1.0):
    from .analysis import get_planet_mass
    from .utils import mjup2msun
    import rebound

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, aspect="equal")

    for p in res.posterior_sample[:2]:
        sim = rebound.Simulation()
        sim.G = 0.00029591  # AU^3 / solMass / day^2
        sim.add(m=star_mass)

        nplanets = int(p[res.index_component])
        pars = p[res.indices['planets']]
        for i in range(nplanets):
            P, K, φ, ecc, ω = pars[i::res.max_components]
            # print(P, K, φ, ecc, ω)
            m = get_planet_mass(P, K, ecc, star_mass=star_mass)[0]
            m *= mjup2msun
            sim.add(P=P, m=m, e=ecc, omega=ω, M=φ, inc=0)
            # self.move_to_com()

        if res.KO:
            pars = p[res.indices['KOpars']]
            for i in range(res.nKO):
                P, K, φ, ecc, ω = pars[i::res.nKO]
                # print(P, K, φ, ecc, ω)
                m = get_planet_mass(P, K, ecc, star_mass=star_mass)[0]
                m *= mjup2msun
                sim.add(P=P, m=m, e=ecc, omega=ω, M=φ, inc=0)
                # self.move_to_com()

        kw = dict(
            fig=fig,
            color=True,
            show_particles=False,
        )
        rebound.plotting.OrbitPlot(sim, **kw)

    plt.show()