from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde
from scipy.stats._continuous_distns import reciprocal_gen
from scipy.signal import find_peaks
from corner import corner
from astropy.timeseries.periodograms.lombscargle.core import LombScargle

from .analysis import np_bayes_factor_threshold, find_outliers
from .utils import (get_prior, hyperprior_samples, percentile68_ranges_latex,
                    wrms, get_instrument_name)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


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
        '6': [res.plot_random_samples, {'show_vsys': True}],
        '6p': [
            'res.plot_random_samples(show_vsys=True);'\
            'res.phase_plot(res.maximum_likelihood_sample(Np=np_bayes_factor_threshold(res)))',
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


def make_plot1(res, ax=None, errors=False, show_ESS=True):
    """ Plot the histogram of the posterior for Np """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    bins = np.arange(res.max_components + 2)
    nplanets = res.posterior_sample[:, res.index_component]
    n, _ = np.histogram(nplanets, bins=bins)
    ax.bar(bins[:-1], n / res.ESS, zorder=2)
    if show_ESS:
        ax.text(0.05, 0.9, f'ESS: {res.ESS}', transform=ax.transAxes)

    if errors:
        # from scipy.stats import multinomial
        prob = n / res.ESS
        # errors = multinomial(res.ESS, prob).rvs(1000).std(axis=0)
        error_multinomial = np.sqrt(res.ESS * prob * (1 - prob)) / res.ESS
        ax.errorbar(bins[:-1],
                    n / res.ESS,
                    error_multinomial,
                    fmt='.',
                    ms=0,
                    capsize=3,
                    color='k')

    if res.removed_crossing:
        ic = res.index_component
        nn = (~np.isnan(res.posterior_sample[:, ic + 1:ic + 11])).sum(axis=1)
        nn, _ = np.histogram(nn, bins=bins)
        ax.bar(bins[:-1], nn, color='r', alpha=0.2, zorder=2)
        ax.legend(['all posterior samples', 'crossing orbits removed'])
    else:
        pt_Np = np_bayes_factor_threshold(res)
        ax.bar(pt_Np, n[pt_Np] / res.ESS, color='C1', zorder=2)

    xlim = (-0.5, res.max_components + 0.5)
    xticks = np.arange(res.max_components + 1)
    ax.set(xlabel='Number of Planets',
           ylabel='Number of Posterior Samples / ESS',
           xlim=xlim,
           xticks=xticks,
           title='Posterior distribution for $N_p$')

    nn = n[np.nonzero(n)]
    print('Np probability ratios: ', nn.flat[1:] / nn.flat[:-1])
    if errors:
        from scipy.stats import multinomial
        rs = multinomial(res.ESS, prob).rvs(10000)
        print(23*' ', np.divide(rs[:, 1:], rs[:, :-1]).std(axis=0).round(0))

    if res.save_plots:
        filename = 'kima-showresults-fig1.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return ax.figure


def make_plot2(res,
               nbins=100,
               bins=None,
               plims=None,
               logx=True,
               kde=False,
               kde_bw=None,
               show_peaks=False,
               show_prior=False,
               show_year=True,
               show_timespan=True,
               separate_colors=False,
               return_bins=False,
               **kwargs):
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

    # if res.log_period:
    #     T = np.exp(res.T)
    # else:
    #     T = res.T

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        fig = ax.figure
    else:
        fig, ax = plt.subplots(1, 1)

    kwline = {'ls': '--', 'lw': 1.5, 'alpha': 0.3, 'zorder': -1}

    if show_year:  # mark 1 year
        year = 365.25
        ax.axvline(x=year, color='r', label='1 year', **kwline)

    if show_timespan:  # mark the timespan of the data
        ax.axvline(x=res.data.t.ptp(), color='k', label='time span', **kwline)

    if kde:
        T = res.T
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
                # get bin limits from prior support
                prior_support = res.priors['Pprior'].support()
                if not np.isinf(prior_support).any():
                    start, end = prior_support
                else:  # or default to these
                    start, end = 1e-1, 1e7
            else:
                start, end = plims

            bins = 10**np.linspace(np.log10(start), np.log10(end), nbins)

        bottoms = np.zeros_like(bins)
        for i in range(res.max_components):
            m = res.posterior_sample[:, res.indices['np']] == i + 1
            T = res.posterior_sample[m, res.indices['planets.P']]
            T = T[:, :i + 1].ravel()

            counts, bin_edges = np.histogram(T, bins=bins)

            color = None

            ax.bar(x=bin_edges[:-1],
                   height=counts / res.ESS,
                   width=np.ediff1d(bin_edges),
                   bottom=bottoms[:-1],
                   align='edge',
                   alpha=0.8,
                   color=color)

            bottoms += np.append(counts / res.ESS, 0)

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

    ax.set_xscale('log' if logx else 'linear')

    if kwargs.get('labels', True):
        if kde:
            ylabel = 'KDE density'
        else:
            ylabel = 'Number of posterior samples / ESS'
        ax.set(xlabel=r'Period [days]', ylabel=ylabel)

    title = kwargs.get('title', True)
    if title:
        if isinstance(title, str):
            ax.set_title(title)
        else:
            ax.set_title('Posterior distribution for the orbital period(s)')

    # ax.set_ylim(0, 1)

    if plims is not None:
        ax.set_xlim(plims)

    if res.save_plots:
        filename = 'kima-showresults-fig2.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        if return_bins:
            return fig, bins
        else:
            return fig


def make_plot3(res,
               mask=None,
               include_known_object=False,
               points=True,
               colors_np=True,
               gridsize=50,
               **kwargs):
    """
    Plot the 2d histograms of the posteriors for semi-amplitude and orbital
    period and eccentricity and orbital period. If `points` is True, plot
    each posterior sample, else plot hexbins
    """

    if not res.KO or not include_known_object:
        if res.max_components == 0:
            print('Model has no planets! make_plot3() doing nothing...')
            return

        if res.posteriors.P.size == 0:
            print('None of the posterior samples have planets!', end=' ')
            print('make_plot3() doing nothing...')
            return

    if mask is None:
        # ip = res.indices['planets']
        # mc = res.max_components
        T = res.posteriors.P # res.posterior_sample[:, ip][:, 0 * mc:1 * mc]
        A = res.posteriors.K # res.posterior_sample[:, ip][:, 1 * mc:2 * mc]
        E = res.posteriors.e # res.posterior_sample[:, ip][:, 3 * mc:4 * mc]
        # T = res.T
        # A, E = res.A, res.E
    else:
        # pars = res.posterior_sample[mask, res.indices['planets']]
        T = res.posteriors.P[mask, :] # np.hstack(pars[:, 0 * res.max_components:1 * res.max_components])
        A = res.posteriors.K[mask, :] # np.hstack(pars[:, 1 * res.max_components:2 * res.max_components])
        E = res.posteriors.e[mask, :] # np.hstack(pars[:, 3 * res.max_components:4 * res.max_components])

    include_known_object = include_known_object and res.KO

    if include_known_object:
        if mask is None:
            KOpars = res.posterior_sample[:, res.indices['KOpars']]
        else:
            KOpars = res.posterior_sample[mask, res.indices['KOpars']]
        T_KO = np.hstack(KOpars[:, 0 * res.nKO:1 * res.nKO])
        A_KO = np.hstack(KOpars[:, 1 * res.nKO:2 * res.nKO])
        E_KO = np.hstack(KOpars[:, 3 * res.nKO:4 * res.nKO])

    if res.log_period:
        T = np.exp(res.T)

    E0 = np.all(E == 0.0)

    if E0:
        if 'ax1' in kwargs:
            ax1 = kwargs.pop('ax1')
            fig = ax1.figure
        else:
            fig, ax1 = plt.subplots(1, 1)
    else:
        if 'ax1' in kwargs and 'ax2' in kwargs:
            ax1, ax2 = kwargs.pop('ax1'), kwargs.pop('ax2')
            fig = ax1.figure
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # the y scale in loglog looks bad if the semi-amplitude doesn't have
    # high dynamic range; the threshold of 30 is arbitrary
    Khdr_threshold = 30

    if points:
        kw = dict(markersize=2, zorder=2, alpha=0.1)
        kw = {**kwargs, **kw}
        if colors_np:
            kw['color'] = None
        if A.size > 1 and A.ptp() > Khdr_threshold:
            ax1.loglog(T, A, '.', **kw)
        else:
            ax1.semilogx(T, A, '.', **kw)

        if not E0:
            ax2.semilogx(T, E, '.', **kw)

        if include_known_object:
            ax1.semilogx(T_KO, A_KO, '.', markersize=2, zorder=2)
            if not E0:
                ax2.semilogx(T_KO, E_KO, '.', markersize=2, zorder=2)

    else:
        if A.size > 1 and A.ptp() > 30:
            ax1.hexbin(T, A, gridsize=gridsize, bins='log', xscale='log',
                       yscale='log', cmap=plt.get_cmap('afmhot_r'))
        else:
            ax1.hexbin(T, A, gridsize=gridsize, bins='log', xscale='log',
                       yscale='linear', cmap=plt.get_cmap('afmhot_r'))

        if not E0:
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

            if not E0:
                ax2.semilogx(T, E, '.', markersize=1, alpha=0.05, color='r',
                             zorder=1)

    ax1.set(ylabel='Semi-amplitude [m/s]',
            title='Joint posterior semi-amplitude $-$ orbital period')

    if not E0:
        ax2.set(ylabel='Eccentricity', xlabel='Period [days]',
                title='Joint posterior eccentricity $-$ orbital period',
                ylim=[0, 1], xlim=[0.1, 1e7])

    try:
        ax2.set(xlim=res.priors['Pprior'].support())
    except (AttributeError, KeyError, ValueError, NameError):
        pass

    if res.save_plots:
        filename = 'kima-showresults-fig3.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def make_plot4(res, Np=None, ranges=None, show_prior=False, fig=None,
               **hist_kwargs):
    """
    Plot histograms for the GP hyperparameters. If Np is not None, highlight
    the samples with Np Keplerians.
    """
    if not res.GPmodel:
        print('Model does not have GP! make_plot4() doing nothing...')
        return

    # dispatch if RVFWHMmodel
    if res.model == 'RVFWHMmodel':
        return make_plot4_rvfwhm(res, Np, ranges, show_prior, fig,
                                 **hist_kwargs)

    n = res.etas.shape[1]
    available_etas = [f'eta{i}' for i in range(1, n + 1)]
    labels = [rf'$\eta_{i}$' for i in range(1, n + 1)]
    if ranges is None:
        ranges = len(labels) * [None]

    if Np is not None:
        m = res.posterior_sample[:, res.index_component] == Np

    nplots = int(np.ceil(n / 2))
    if fig is None:
        fig, axes = plt.subplots(2, nplots)
    else:
        axes = fig.axes
        assert len(axes) == 2 * nplots, 'figure has wrong number of axes!'

    hist_kwargs.setdefault('density', True)

    for i, eta in enumerate(available_etas):
        ax = np.ravel(axes)[i]
        ax.hist(getattr(res, eta), bins=40, range=ranges[i], **hist_kwargs)

        if show_prior:
            priors = [p for p in res.priors.keys() if 'eta' in p]
            print(eta)
            # ax.hist(prior.rvs(res.ESS), bins=40, color='k', alpha=0.2)

        if Np is not None:
            ax.hist(eta[m],
                    bins=40,
                    histtype='step',
                    alpha=0.5,
                    label='$N_p$=%d samples' % Np,
                    range=ranges[i])
            ax.legend()

        ax.set(xlabel=labels[i], ylabel='posterior')

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


def make_plot4_rvfwhm(res, Np=None, ranges=None, show_prior=False, fig=None,
                      **hist_kwargs):
    """
    Plot histograms for the GP hyperparameters. If Np is not None, highlight
    the samples with Np Keplerians. 
    """
    if not res.GPmodel:
        print('Model does not have GP! make_plot4() doing nothing...')
        return

    # n = res.etas.shape[1]
    labels = ('η1', 'η2', 'η3', 'η4')

    if ranges is None:
        ranges = len(labels) * [None]

    if Np is not None:
        m = res.posterior_sample[:, res.index_component] == Np

    fig = plt.figure()

    if res.GPkernel == 'standard':
        gs = fig.add_gridspec(6, 2)
    elif res.GPkernel == 'qpc':
        gs = fig.add_gridspec(6, 3)

    histkw = dict(density=True, bins='doane')
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
        ax2.hist(res.etas[:, 1], color='C5', **histkw)
        ax2.set(xlabel=r'$\eta_1$ FWHM [m/s]', ylabel='posterior', **allkw)
        ax2.set_xlim((0, None))

        if show_prior:
            kw = dict(color='k', alpha=0.2, density=True, zorder=-1,
                      bins='doane')
            #
            prior = res.priors['eta1_1_prior']
            ax1.hist(prior.rvs(10*res.ESS), **kw)
            ax1.set_xlim(*prior.support())
            #
            prior = res.priors['eta1_2_prior']
            ax2.hist(prior.rvs(10*res.ESS), **kw)
            ax2.set_xlim(*prior.support())

        col = 1

    units = [' [days]', ' [days]', '']

    for i in range(3):
        j = 2 * (i + 1)
        ax = fig.add_subplot(gs[2 * i:j, col])
        ax.hist(res.etas[:, res._GP_par_indices[j]], **histkw)
        if res._GP_par_indices[j] != res._GP_par_indices[j + 1]:
            ax.hist(res.etas[:, res._GP_par_indices[j + 1]], color='C5',
                    **histkw)
        ax.set(xlabel=fr'$\eta_{2+i}$' + units[i], ylabel='posterior', **allkw)

        if show_prior:
            kw = dict(color='k', alpha=0.2, density=True, zorder=-1,
                      bins='doane')
            prior = res.priors[f'eta{2+i}_1_prior']
            if prior is not None:
                ax.hist(prior.rvs(10*res.ESS), **kw)

    fig.tight_layout()
    return fig


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
        for i in range(2, res.n_hyperparameters):
            print(i)
            labels += [rf'$\eta_{i}$']
    else:
        labels += [rf'$\eta_{i}$' for i in range(1, 5)]

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


def corner_planet_parameters(res, fig=None, pmin=None, pmax=None):
    """ Corner plot of the posterior samples for the planet parameters """

    labels = [r'$P$', r'$K$', r'$\phi$', 'ecc', r'$\omega$']

    samples = res.posterior_sample[:, res.indices['planets']]
    mc = res.max_components

    # # wrap around phi and omega
    # samples[:, 2] = np.pi + (samples[:, 2] - np.pi) % (2 * np.pi)
    # samples[:, 4] = np.pi + (samples[:, 4] - np.pi) % (2 * np.pi)

    # # separate bins for each parameter
    # bins = []
    # for planetp in data:
    #     if hist_tools_available:
    #         bw = hist_tools.freedman_bin_width
    #         # bw = hist_tools.knuth_bin_width
    #         this_planet_bins = []
    #         for sample in planetp.T:
    #             this_planet_bins.append(
    #                 bw(sample, return_bins=True)[1].size)
    #         bins.append(this_planet_bins)
    #     else:
    #         bins.append(None)

    # set the parameter ranges to include everythinh
    def r(x, over=0.2):
        return x.min() - over * x.ptp(), x.max() + over * x.ptp()

    # ranges = []
    # for i in range(res.n_dimensions):
    #     i1, i2 = res.max_components * i, res.max_components * (i + 1)
    #     ranges.append(r(samples[:, i1:i2]))

    #
    c = corner.corner
    colors = plt.rcParams["axes.prop_cycle"]

    # # for i, (datum, colorcycle) in enumerate(zip(data, colors)):
    # fig = c(
    #     samples,
    #     fig=fig,
    #     # labels=labels,
    #     show_titles=len(samples) == 1,
    #     plot_contours=False,
    #     plot_datapoints=True,
    #     plot_density=False,
    #     # bins=bins[i],
    #     # range=ranges,
    #     # color=colorcycle['color'],
    #     # fill_contours=True, smooth=True,
    #     # contourf_kwargs={'cmap':plt.get_cmap('afmhot'), 'colors':None},
    #     #hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
    #     hist_kwargs={'density': True},
    #     # range=[1., 1., (0, 2*np.pi), (0., 1.), (0, 2*np.pi)],
    #     data_kwargs={
    #         'alpha': 1,
    #         'ms': 1,
    #         # 'color': colorcycle['color']
    #     },
    # )

    for i in range(mc):
        fig = c(
            samples[:, i::mc],
            labels=labels,
            show_titles=len(samples) == 1,
            plot_contours=False,
            plot_datapoints=True,
            plot_density=False,
            # bins=bins[i],
            # range=ranges,
            # color=colorcycle['color'],
            # fill_contours=True, smooth=True,
            # contourf_kwargs={'cmap':plt.get_cmap('afmhot'), 'colors':None},
            #hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
            hist_kwargs={'density': True},
            data_kwargs={
                'alpha': 1,
                'ms': 1,
            },
        )

    plt.show()


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

    kwargs.setdefault('bins', 'doane')

    ax.hist(vsys, **kwargs)

    title = 'Posterior distribution for $v_{\\rm sys}$ \n %s' % estimate
    if kwargs.get('density', False):
        ylabel = 'posterior'
    else:
        ylabel = 'posterior samples'
    ax.set(xlabel='vsys' + units, ylabel=ylabel, title=title)

    if show_prior:
        try:
            prior = res.priors['Cprior']
        except KeyError:
            prior = res.priors['Vprior']

        # low, upp = prior.interval(1)
        d = kwargs.get('density', False)
        ax.hist(prior.rvs(res.ESS),
                density=d,
                alpha=0.15,
                color='k',
                zorder=-1)
        ax.legend(['posterior', 'prior'])

        # except Exception as e:
        #     print(str(e))

    if res.save_plots:
        filename = 'kima-showresults-fig7.2.png'
        print('saving in', filename)
        fig.savefig(filename)

    if show_offsets and res.multi:
        n_inst_offsets = res.inst_offsets.shape[1]
        nrows = 2 if res.model == 'RVFWHMmodel' else 1
        fig, axs = plt.subplots(nrows, n_inst_offsets // nrows, sharey=True,
                                figsize=(2 + n_inst_offsets * 3, 5), squeeze=True,
                                constrained_layout=True)
        figures.append(fig)
        if n_inst_offsets == 1:
            axs = [axs,]

        prior = res.priors['offsets_prior']

        if res.model == 'RVFWHMmodel':
            k = 0
            wrt = res.instruments[-1]
            for j in range(2):
                for i in range(n_inst_offsets // 2):
                    this = res.instruments[i]
                    a = res.inst_offsets[:, k]
                    axs[j, i].hist(a)
                    label = 'offset\n%s rel. to %s' % (this, wrt)
                    estimate = percentile68_ranges_latex(a) + units
                    axs[j, i].set(xlabel=label, title=estimate,
                                  ylabel='posterior samples')
                    k += 1

                    if show_prior and j == 0:
                        d = kwargs.get('density', False)
                        kw = dict(density=d, alpha=0.15, color='k', zorder=-1)
                        axs[j, i].hist(prior.rvs(res.ESS), **kw)
                        axs[j, i].legend(['posterior', 'prior'])

        else:
            for i in range(n_inst_offsets):
                # wrt = get_instrument_name(res.data_file[-1])
                # this = get_instrument_name(res.data_file[i])
                wrt = res.instruments[-1]
                this = res.instruments[i]
                label = 'offset\n%s rel. to %s' % (this, wrt)
                a = res.inst_offsets[:, i]
                estimate = percentile68_ranges_latex(a) + units
                axs[i].hist(a)

                if show_prior:
                    d = kwargs.get('density', False)
                    kw = dict(density=d, alpha=0.15, color='k', zorder=-1)
                    axs[i].hist(prior.rvs(res.ESS), **kw)
                    axs[i].legend(['posterior', 'prior'])

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


def hist_jitter(res, show_prior=False, show_stats=False, **kwargs):
    """
    Plot the histogram of the posterior for the additional white noise
    """
    # if res.arbitrary_units:
    #     units = ' (arbitrary)'
    # else:
    #     units = ' (m/s)'  # if res.units == 'ms' else ' (km/s)'

    RVFWHM = res.model == 'RVFWHMmodel'

    if 'fig' in kwargs:
        fig = kwargs.pop('fig')
        axs = fig.axes
        axs = np.array(axs).reshape(-1, res.n_instruments)
        overplot = True
    else:
        kw = dict(figsize=(min(10, 5 + res.n_instruments * 2), 4),
                  constrained_layout=True)
        if RVFWHM:
            fig, axs = plt.subplots(2, res.n_instruments, **kw)
        else:
            fig, axs = plt.subplots(1, res.n_instruments, **kw)
        overplot = False

    if kwargs.pop('show_title', True):
        fig.suptitle('Posterior distribution for extra white noise')

    if isinstance(axs, np.ndarray) and res.multi:
        if RVFWHM:
            for row in axs:
                for ax in row:
                    ax.sharex(row[0])

    kwargs.setdefault('density', True)
    kwargs.setdefault('bins', 'doane')
    axs = np.ravel(axs)
    for i, ax in enumerate(axs):
        estimate = percentile68_ranges_latex(res.jitter[:, i]) + ' m/s'
        ax.hist(res.jitter[:, i], label=estimate, **kwargs)
        leg = ax.legend()
        leg._legend_box.sep = 0

        if show_prior:
            if RVFWHM and i >= res.n_instruments:
                prior = res.priors['J2prior'].rvs(res.ESS)
            else:
                prior = res.priors['Jprior'].rvs(res.ESS)
            ax.hist(prior, density=True, color='k', alpha=0.2, zorder=-1)

        if show_stats:
            from matplotlib import transforms
            transform = transforms.blended_transform_factory(ax.transData,
                                                             ax.transAxes)
            kw = dict(fontsize=8, transform=transform)

            if RVFWHM and i >= res.n_instruments:
                j = i - res.n_instruments
                m = res.data.e2[res.data.obs == j + 1].mean()
                ax.axvline(m, 0, 0.2, color='r')
                ax.text(m, 0.1, r'$\overline{\sigma}_{FWHM}$', color='r', **kw)
                s = res.data.y2[res.data.obs == j + 1].std()
                ax.axvline(s, 0, 0.2, color='g')
                ax.text(s, 0.2, r'SD FWHM', color='g', **kw)

            else:
                m = res.data.e[res.data.obs == i + 1].mean()
                ax.axvline(m, 0, 0.2, color='r')
                ax.text(m, 0.1, r'$\overline{\sigma}_{RV}$', color='r', **kw)
                s = res.data.y[res.data.obs == i + 1].std()
                ax.axvline(s, 0, 0.2, color='g')
                ax.text(s, 0.2, r'SD RV', color='g', **kw)

    for ax in axs:
        ax.set(yticks=[], ylabel='posterior')

    insts = [get_instrument_name(i) for i in res.instruments]
    if res.model == 'RVFWHMmodel':
        labels = [f'RV jitter {i} [m/s]' for i in insts]
        labels += [f'FWHM jitter {i} [m/s]' for i in insts]
    else:
        labels = [f'jitter {i} [m/s]' for i in insts]

    for ax, label in zip(axs, labels):
        ax.set_xlabel(label, fontsize=10)

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
        ax = np.atleast_1d(ax)

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


def plot_data(res, ax=None, axf=None, y=None, y2=None, extract_offset=True,
              ignore_y2=False, time_offset=0.0, legend=True, show_rms=False,
              outliers=None, **kwargs):

    fwhm_model = res.model == 'RVFWHMmodel' and not ignore_y2

    if ax is None:
        if fwhm_model:
            fig, (ax, axf) = plt.subplots(2, 1, sharex=True)
        else:
            fig, ax = plt.subplots(1, 1)

    t = res.data.t.copy()
    e = res.data.e.copy()

    if y is None:
        y = res.data.y.copy()

    if fwhm_model:
        y2 = res.data.y2.copy()
        e2 = res.data.e2.copy()

    assert y.size == res.data.N, 'wrong dimensions!'

    if extract_offset:
        y_offset = round(y.mean(), 0) if abs(y.mean()) > 100 else 0
        if fwhm_model:
            y2_offset = round(y2.mean(), 0) if abs(y2.mean()) > 100 else 0
    else:
        y_offset = 0
        if fwhm_model:
            y2_offset = 0

    kw = dict(fmt='o', ms=3)
    kw.update(**kwargs)

    if res.multi:
        for j in range(res.n_instruments):
            inst = res.instruments[j]
            m = res.data.obs == j + 1
            kw.update(label=inst)
            if outliers is None:
                ax.errorbar(t[m] - time_offset, y[m] - y_offset, e[m], **kw)
            else:
                ax.errorbar(t[m & ~outliers] - time_offset,
                            y[m & ~outliers] - y_offset, e[m & ~outliers],
                            **kw)
            if fwhm_model:
                axf.errorbar(t[m] - time_offset, y2[m] - y2_offset, e2[m],
                             **kw)
    else:
        kw.update(label=res.instruments)

        if outliers is None:
            ax.errorbar(t - time_offset, y - y_offset, e, **kw)
        else:
            ax.errorbar(t[~outliers] - time_offset, y[~outliers] - y_offset,
                        e[~outliers], **kw)
        if fwhm_model:
            axf.errorbar(t - time_offset, y2 - y2_offset, e2, **kw)

    if legend:
        ax.legend(loc='best')
        # ax.legend(loc='upper left')

    if res.multi:
        kw = dict(color='b', lw=2, alpha=0.1, zorder=-2)
        for ot in res._offset_times:
            ax.axvline(ot - time_offset, **kw)
            if fwhm_model:
                axf.axvline(ot, **kw)

    if res.arbitrary_units:
        lab = dict(xlabel='Time [days]', ylabel='Q [arbitrary]')
    else:
        lab = dict(xlabel='Time [days]', ylabel='RV [m/s]')

    ax.set(**lab)
    if fwhm_model:
        axf.set(xlabel='Time [days]', ylabel='FWHM [m/s]')

    if show_rms:
        # if res.studentT:
        #     outliers = find_outliers(res)
        #     rms1 = wrms(y, 1 / res.e**2)
        #     rms2 = wrms(y[~outliers], 1 / res.e[~outliers]**2)
        #     ax.set_title(f'rms: {rms2:.2f} ({rms1:.2f}) m/s', loc='right')
        # else:
        rms = wrms(y, 1 / e**2)
        if outliers is None or not np.any(outliers):
            title = f'rms: {rms:.2f} [m/s]'
        else:
            rms_out = wrms(y[~outliers], 1 / e[~outliers]**2)
            title = f'rms: {rms:.2f} ({rms_out:.2f} w/o outliers) [m/s]'

        ax.set_title(title, loc='right', fontsize=10)

    if y_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y_offset)] + str(int(abs(y_offset)))
        fs = ax.xaxis.get_label().get_fontsize()
        ax.set_title(offset, loc='left', fontsize=fs)

    if fwhm_model and y2_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y2_offset)] + str(int(abs(y2_offset)))
        fs = axf.xaxis.get_label().get_fontsize()
        axf.set_title(offset, loc='left', fontsize=fs)


    if fwhm_model:
        return ax, axf, y_offset, y2_offset
    else:
        return ax, y_offset


def gls_data(res, sample=None, ax=None):
    from gatspy.periodic import LombScargle, LombScargleMultiband
    from astropy.timeseries import LombScargle as GLS
    fwhm_model = res.model == 'RVFWHMmodel' #and not ignore_y2

    if ax is None:
        kw = dict(sharex=True, constrained_layout=True)
        if fwhm_model:
            fig, (axw, ax, axf) = plt.subplots(3, 1, **kw)
        else:
            fig, (axw, ax) = plt.subplots(2, 1, **kw)

    window_function = GLS(res.data.t, np.ones_like(res.data.t), res.data.e,
                          fit_mean=False, center_data=False)
    freq, power = window_function.autopower()
    axw.semilogx(1 / freq, power)

    if res.multi:
        model = LombScargleMultiband(Nterms_base=1, Nterms_band=0)
        model.fit(res.data.t, res.data.y, res.data.e, filts=res.data.obs)
        # power = model.periodogram(period)
    else:
        model = LombScargle()
        model.fit(res.data.t, res.data.y, res.data.e)

    period, power = model.periodogram_auto(oversampling=30)
    ax.semilogx(period, power)

    if fwhm_model:
        if res.multi:
            model = LombScargleMultiband(Nterms_base=1, Nterms_band=0)
            model.fit(res.data.t, res.data.y2, res.data.e2, filts=res.data.obs)
            # power = model.periodogram(period)
        else:
            model = LombScargle()
            model.fit(res.data.t, res.data.y2, res.data.e2)

        period, power = model.periodogram_auto(oversampling=30)
        axf.semilogx(period, power)


def plot_transit_data(res, ax=None, y=None, extract_offset=False,
                      time_offset=0.0, legend=True, show_rms=False, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if y is None:
        y = res.data.y.copy()

    assert y.size == res.data.t.size, 'wrong dimensions!'

    if extract_offset:
        y_offset = round(y.mean(), 0) if abs(y.mean()) > 100 else 0
    else:
        y_offset = 0

    kw = dict(fmt='o')
    kw.update(**kwargs)

    if res.multi:
        for j in range(res.n_instruments):
            inst = res.instruments[j]
            m = res.data.obs == j + 1
            kw.update(label=inst)
            ax.errorbar(res.data.t[m] - time_offset, y[m] - y_offset,
                        res.data.e[m], **kw)
    else:
        kw.update(label=res.instruments)

        ax.errorbar(res.data.t - time_offset, y - y_offset, res.data.e, **kw)

    if legend:
        ax.legend(loc='upper left')

    if res.multi:
        kw = dict(color='b', lw=2, alpha=0.1, zorder=-2)
        for ot in res._offset_times:
            ax.axvline(ot - time_offset, **kw)

    lab = dict(xlabel='Time [days]', ylabel='Flux')
    ax.set(**lab)

    if show_rms:
        # if res.studentT:
        #     outliers = find_outliers(res)
        #     rms1 = wrms(y, 1 / res.e**2)
        #     rms2 = wrms(y[~outliers], 1 / res.e[~outliers]**2)
        #     ax.set_title(f'rms: {rms2:.2f} ({rms1:.2f}) m/s', loc='right')
        # else:
        rms = wrms(y, 1 / res.data.e**2)
        ax.set_title(f'rms: {rms:.2f} m/s', loc='right', fontsize=10)

    if y_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y_offset)] + str(int(abs(y_offset)))
        fs = ax.xaxis.get_label().get_fontsize()
        ax.set_title(offset, loc='left', fontsize=fs)

    return ax, y_offset




def corner_known_object(res, star_mass=1.0, adda=False, **kwargs):
    if not res.KO:
        print('Model has no known object! '
              'corner_known_object() doing nothing...')
        return

    import pygtc

    labels = [r'$P$', r'$K$', r'$M_0$', 'ecc', r'$\omega$']
    for i in range(1):
        data = res.KOpars[:, i::res.nKO]
        fig = pygtc.plotGTC(
            chains=data,
            holdRC=True,
            # smoothingKernel=0,
            paramNames=labels,
            plotDensity=False,
            labelRotation=(True, False),
            filledPlots=False,
            colorsOrder=['blues_old'],
            figureSize='AandA_page',  # AandA_column
        )
        axs = fig.axes
        for ax in axs:
            ax.yaxis.set_label_coords(-0.3, 0.5, transform=None)
            ax.xaxis.set_label_coords(0.5, -0.4, transform=None)

        fs = axs[-1].xaxis.label.get_fontsize()
        start = len(labels)
        for i in range(-start, 0):
            val = percentile68_ranges_latex(data.T[i])
            # print(f'{names[i]} = {val}', fs)
            axs[i].set_title(f'{labels[i]} = {val}', fontsize=fs - 1)

        fig.tight_layout()

    # labels = [f'{n}{u}' for n, u in zip(names, units)]
    # print(data.shape, labels)


    # fs = axs[-1].xaxis.label.get_fontsize()
    # start = -3 if fixed_ecc else -4
    # start = start - 1 if adda else start


    # # fig = corner(np.c_[P, K, E, M], labels=labels, show_titles=True, **kwargs)
    fig.subplots_adjust(wspace=0.15, hspace=0)
    # return fig, inds, (M, A)


def phase_plot(res,
               sample,
               highlight=None,
               only=None,
               phase_axs=None,
               add_titles=True,
               sharey=False,
               highlight_points=None,
               sort_by_increasing_P=False,
               sort_by_decreasing_K=False,
               show_gls_residuals=False,
               **kwargs):
    """ Plot the phase curves given the solution in `sample` """
    # this is probably the most complicated function in the whole package!!

    if res.max_components == 0 and not res.KO:
        print('Model has no planets! phase_plot() doing nothing...')
        return

    # make copies to not change attributes
    t, y, e = res.data.t.copy(), res.data.y.copy(), res.data.e.copy()
    M0_epoch = res.M0_epoch
    if t[0] > 24e5:
        time_offset = 24e5
        time_label = 'Time [BJD - 2400000]'
    elif t[0] > 5e4:
        time_offset = 5e4
        time_label = 'Time [BJD - 50000]'
    else:
        time_offset = 0
        time_label = 'Time [BJD]'

    #     t -= 24e5
    #     M0_epoch -= 24e5

    if highlight_points is not None:
        hlkw = dict(fmt='*', ms=6, color='y', zorder=2)
        hl = highlight_points
        highlight_points = True

    nd = res.n_dimensions
    mc = res.max_components

    # get the planet parameters for this sample
    pars = sample[res.indices['planets']].copy()

    # how many planets in this sample?
    nplanets = (pars[:mc] != 0).sum()
    # if mc != nplanets:
    #     mc = nplanets

    planetis = list(range(mc))
    KO_planet = [False] * mc

    if res.KO:
        mc += res.nKO
        if nplanets == 0:
            k = 0
        else:
            k = planetis[-1]

        for i in range(res.nKO):
            KOpars = sample[res.indices['KOpars']][i::res.nKO]
            if pars.size == 0:
                pars = KOpars
            else:
                pars = np.insert(pars, range(1 + k, (1 + k) * 6, 1 + k),
                                 KOpars)
                k += 1
            nplanets += 1
            planetis.append(i)
            KO_planet.append(True)

    if sort_by_decreasing_K:
        # sort by decreasing amplitude (arbitrary)
        ind = np.argsort(pars[1 * mc:2 * mc])[::-1]
        for i in range(nd):
            pars[i * mc:(i + 1) * mc] = pars[i * mc:(i + 1) * mc][ind]
        planetis = list(ind)
        KO_planet = np.array(KO_planet)[ind]
    elif sort_by_increasing_P:
        # sort by increasing period
        ind = np.argsort(pars[0:mc])
        for i in range(nd):
            pars[i * mc:(i + 1) * mc] = pars[i * mc:(i + 1) * mc][ind]
        planetis = list(ind)
        KO_planet = np.array(KO_planet)[ind]
    else:
        planetis = list(range(nplanets))

    # extract periods, phases and calculate times of periastron
    P = pars[0 * mc:1 * mc][:nplanets]
    K = pars[1 * mc:2 * mc][:nplanets]

    phi = pars[2 * mc:3 * mc][:nplanets]
    T0 = M0_epoch - (P * phi) / (2. * np.pi)
    # semi-amplitudes
    # and eccentricities
    ECC = pars[3 * mc:4 * mc][:nplanets]

    if nplanets == 0:
        print('Sample has no planets! phase_plot() doing nothing...')
        return

    KO_planet = KO_planet[:nplanets]

    # print(nplanets)
    # print(KO_planet)
    # print(planetis)
    # print(P)
    # print(K)
    # print(phi)

    # subtract stochastic model and vsys / offsets from data
    v = res.full_model(sample, include_planets=False)
    if res.model == 'RVFWHMmodel':
        v = v[0]
    y = y - v

    ekwargs = {
        'fmt': 'o',
        'mec': 'none',
        'ms': 4,
        'capsize': 0,
        'elinewidth': 0.8,
    }
    for k in ekwargs:
        if k in kwargs:
            ekwargs[k] = kwargs[k]

    # very complicated logic just to make the figure the right size
    fs = [
        max(7, 7 + 1.5 * (nplanets - 2)),
        max(4.8, 4.8 + 1 * (nplanets - 3))
    ]
    if res.GPmodel:
        fs[1] += 3

    fig = plt.figure(constrained_layout=True, figsize=fs)
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
    hr = [2] * (nrows - 1) + [1]
    wr = None

    if show_gls_residuals:
        wr = ncols * [2] + [1]
        ncols += 1
        # fig.set_size_inches(fs[0], fs[1])

    gs = gridspec.GridSpec(nrows, ncols, figure=fig, height_ratios=hr,
                           width_ratios=wr)
    gs_indices = {i: (i // 3, i % 3) for i in range(50)}
    axs = []

    shown_planets = 0
    shown_KO_planets = 0

    # for each planet in this sample
    for i, letter in zip(range(nplanets), ascii_lowercase[1:]):
        if phase_axs is None:
            ax = fig.add_subplot(gs[gs_indices[i]])
        else:
            try:
                ax = phase_axs[i]
            except IndexError:
                continue

        axs.append(ax)

        ax.axvline(0.5, ls='--', color='k', alpha=0.2, zorder=-5)
        ax.axhline(0.0, ls='--', color='k', alpha=0.2, zorder=-5)

        p = P[i]
        t0 = T0[i]

        # plot the keplerian curve in phase (3 times)
        phase = np.linspace(0, 1, 200)
        tt = phase * p + t0

        if KO_planet[i]:
            shown_KO_planets += 1
            sign = -1
            planet_index = sign * shown_KO_planets
            # print('KO:', planet_index)
        else:
            shown_planets += 1
            sign = 1
            planet_index = sign * shown_planets
            planet_index = planetis[planet_index - 1] + 1
            # print('normal:', planet_index, p, t0)

        # keplerian for this planet
        vv = res.eval_model(sample, tt, single_planet=planet_index)
        # the background model at these times
        offset_model = res.eval_model(sample, tt, include_planets=False)

        if res.model == 'RVFWHMmodel':
            vv = vv[0]
            offset_model = offset_model[0]

        for j in (-1, 0, 1):
            alpha = 0.2 if j in (-1, 1) else 1
            ax.plot(np.sort(phase) + j,
                    vv[np.argsort(phase)] - offset_model,
                    color='k', alpha=alpha)

        # the other planets which are not the ith
        # other = copy(planetis)
        # other.remove(planeti)

        # subtract the other planets from the data and plot it (the data)
        vv = res.planet_model(sample, except_planet=planet_index)
        if res.model == 'RVFWHMmodel':
            vv = vv[0]

        if res.multi:
            for k in range(1, res.n_instruments + 1):
                m = res.data.obs == k
                phase = ((t[m] - t0) / p) % 1.0

                yy = (y - vv)[m]
                ee = e[m].copy()

                # one color for each instrument
                color = ax._get_lines.prop_cycler.__next__()['color']

                for j in (-1, 0, 1):
                    label = res.instruments[k - 1] if j == 0 else ''
                    alpha = 0.2 if j in (-1, 1) else 1
                    if highlight:
                        if highlight not in res.data_file[k - 1]:
                            alpha = 0.1
                    elif only:
                        if only not in res.data_file[k - 1]:
                            alpha = 0

                    ax.errorbar(np.sort(phase) + j,
                                yy[np.argsort(phase)],
                                ee[np.argsort(phase)],
                                label=label, color=color, alpha=alpha,
                                **ekwargs)

                    if highlight_points:
                        hlm = (m & hl)[m]
                        ax.errorbar(np.sort(phase[hlm]) + j,
                                    yy[np.argsort(phase[hlm])],
                                    ee[np.argsort(phase[hlm])],
                                    alpha=alpha, **hlkw)


        else:
            phase = ((t - t0) / p) % 1.0
            yy = y - vv

            color = ax._get_lines.prop_cycler.__next__()['color']

            for j in (-1, 0, 1):
                alpha = 0.3 if j in (-1, 1) else 1
                ax.errorbar(
                    np.sort(phase) + j, yy[np.argsort(phase)],
                    e[np.argsort(phase)], color=color, alpha=alpha, **ekwargs)

        ax.set(xlabel="phase", ylabel="RV [m/s]")
        ax.set_xlim(-0.1, 1.1)
        # ax.set_xticklabels(['', '0', '0.25', '0.5', '0.75', '1'])

        if add_titles:
            title_kwargs = dict(fontsize=12)
            ax.set_title('%s' % letter, loc='left', **title_kwargs)
            # if nplanets == 1:
            k = K[i]
            ecc = ECC[i]
            title = f'P={p:.2f} days\n K={k:.2f} m/s  ecc={ecc:.2f}'
            ax.set_title(title, loc='right', **title_kwargs)
            # else:
            #     ax.set_title('P=%.2f days' % p, loc='right', **title_kwargs)

    if sharey:
        for ax in axs:
            ax.sharey(axs[0])

    end = -1 if show_gls_residuals else None

    try:
        overlap = res._time_overlaps[0]
    except ValueError:
        overlap = False

    # print(fig)
    # ax = fig.axes[0]
    # ax.legend()


    ## GP panel
    ###########
    if res.GPmodel:
        axGP = fig.add_subplot(gs[1, :end])
        _, y_offset = plot_data(res, ax=axGP, ignore_y2=True, legend=False,
                                time_offset=time_offset, **ekwargs)
        axGP.set(xlabel=time_label, ylabel="GP [m/s]")

        tt = np.linspace(t[0], t[-1], 3000)
        no_planets_model = res.eval_model(sample, tt, include_planets=False)
        no_planets_model = res.burst_model(sample, tt, no_planets_model)

        if res.model == 'RVmodel':
            pred, std = res.stochastic_model(sample, tt, return_std=True)

        elif res.model == 'RVFWHMmodel':
            (pred, _), (std, _) = res.stochastic_model(sample, tt,
                                                       return_std=True)
            if overlap:
                no_planets_model = no_planets_model[::2]
            else:
                no_planets_model = no_planets_model[0]

        pred = pred + no_planets_model - y_offset
        pred = np.atleast_2d(pred)
        for p in pred:
            axGP.plot(tt - time_offset, p, 'k')
            axGP.fill_between(tt - time_offset, p - 2 * std, p + 2 * std,
                              color='m', alpha=0.2)

    ## residuals
    ############
    ax = fig.add_subplot(gs[-1, :end])
    residuals = res.residuals(sample, full=True)
    if res.model == 'RVFWHMmodel':
        residuals = residuals[0]

    outliers = None
    if res.studentT:
        outliers = find_outliers(res, sample)
        ax.errorbar(res.data.t[outliers] - time_offset, residuals[outliers],
                    res.data.e[outliers], fmt='xk', ms=7, lw=3)

    plot_data(res, ax=ax, y=residuals, ignore_y2=True, legend=True,
              show_rms=True, outliers=outliers, time_offset=time_offset,
              **ekwargs)

    # legend in the residual plot?
    hand, lab = ax.get_legend_handles_labels()
    leg = ax.legend(hand, lab, loc='upper left', ncol=4, borderaxespad=0.,
                    borderpad=0.3, bbox_to_anchor=(0.0, 1.3), handletextpad=0,
                    columnspacing=0.1)

    if highlight_points:
        ax.errorbar(t[hl], residuals[hl], e[hl], **hlkw)


    ax.axhline(y=0, ls='--', alpha=0.5, color='k')
    ax.set_ylim(np.tile(np.abs(ax.get_ylim()).max(), 2) * [-1, 1])
    ax.set(xlabel=time_label, ylabel='r [m/s]')
    title_kwargs = dict(loc='right', fontsize=12)


    if show_gls_residuals:
        axp = fig.add_subplot(gs[:, -1])
        from astropy.timeseries import LombScargle
        gls = LombScargle(res.data.t, residuals, res.data.e)
        freq, power = gls.autopower()
        axp.semilogy(power, 1 / freq, 'k', alpha=0.6)

        kwl = dict(color='k', alpha=0.2, ls='--')
        kwt = dict(color='k', alpha=0.3, rotation=90, ha='left', va='top', fontsize=9)
        fap001 = gls.false_alarm_level(0.01)
        axp.axvline(fap001, **kwl)
        axp.text(0.98 * fap001, 1 / freq.min(), '1%', **kwt)

        fap01 = gls.false_alarm_level(0.1)
        axp.axvline(fap01, **kwl)
        axp.text(0.98 * fap01, 1 / freq.min(), '10%', **kwt)

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


def plot_random_samples(res, ncurves=50, samples=None, over=0.1, ntt=5000,
                        pmin=None, pmax=None, show_vsys=False,
                        isolate_known_object=True, full_plot=False,
                        ignore_outliers=False, **kwargs):

    # dispatch
    if res.model == 'RVFWHMmodel':
        args = locals().copy()
        return plot_random_samples_rvfwhm(*args, **kwargs)

    if samples is None:
        samples = res._apply_cuts_period(pmin, pmax)
        # samples = res.posterior_sample
        samples_provided = False
    else:
        samples = np.atleast_2d(samples)
        samples_provided = True

    mask = np.ones(samples.shape[0], dtype=bool)

    t = res.data.t.copy()
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

    if full_plot and ncurves > 1:
        print('full_plot can only be used when ncurves=1')
        full_plot = False

    if samples.shape[0] == 1:
        ii = np.zeros(1, dtype=int)
    elif ncurves == samples.shape[0] or samples_provided:
        # ii = np.arange(ncurves)
        ii = np.random.choice(np.arange(samples.shape[0]), size=ncurves,
                              replace=False)
    else:
        try:
            # select `ncurves` indices from the 70% highest likelihood samples
            lnlike = res.posterior_lnlike[:, 1]
            sorted_lnlike = np.sort(lnlike)[::-1]
            mask_lnlike = lnlike > np.percentile(sorted_lnlike, 70)
            ii = np.random.choice(np.where(mask & mask_lnlike)[0], ncurves)
        except ValueError:
            ii = np.random.choice(np.arange(samples.shape[0]), size=ncurves,
                                  replace=False)

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        fig = ax.figure
    else:
        if full_plot:
            fig, axs = plt.subplot_mosaic('aac\naac\nbbc')
            ax = axs['a']
            axs['b'].sharex(ax)
        else:
            fig, ax = plt.subplots(1, 1)

    _, y_offset = plot_data(res, ax, **kwargs)

    if ignore_outliers:
        if res.studentT:
            outliers = find_outliers(
                res, res.maximum_likelihood_sample(printit=False))
            if outliers.any():
                mi = res.data.y[~outliers].min() - res.data.e.max()
                ma = res.data.y[~outliers].max() + res.data.e.max()
                yclip = np.clip(res.data.y, mi, ma)
                ax.plot(res.data.t[outliers], yclip[outliers], 'rs')
                ax.set_ylim(mi, ma)
        else:
            print('cannot identify outliers, likelihood is not Student t')


    # plot the Keplerian curves
    alpha = 0.1 if ncurves > 1 else 0.8

    # cc = kwargs.get('curve_color', 'k')
    gpc = kwargs.get('gp_color', 'plum')

    for icurve, i in enumerate(ii):
        sample = samples[i]
        stoc_model = np.atleast_2d(res.stochastic_model(sample, tt))
        model = np.atleast_2d(res.eval_model(sample, tt))
        offset_model = res.eval_model(sample, tt, include_planets=False)

        if res.multi:
            model = res.burst_model(sample, tt, model)
            offset_model = res.burst_model(sample, tt, offset_model)

        ax.set_prop_cycle(None)
        if model.shape[0] == 1:
            color = 'k'
        else:
            color = None
        ax.plot(tt, (stoc_model + model).T - y_offset, color=color,
                alpha=alpha, zorder=-1)

        if res.GPmodel:
            ax.plot(tt, (stoc_model + offset_model).T - y_offset, color=gpc,
                    alpha=alpha)

        if show_vsys:
            kw = dict(alpha=alpha, color='r', ls='--')
            if res.multi:
                for j in range(res.n_instruments):
                    instrument_mask = res.data.obs == j + 1
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

    if full_plot:
        r = res.residuals(sample, full=True)
        plot_data(res, ax=axs['b'], y=r, legend=False, show_rms=True)
        axs['b'].axhline(y=0, ls='--', color='k', alpha=0.5)
        gls = LombScargle(res.data.t, r, res.data.e)
        f, p = gls.autopower(samples_per_peak=15)
        axs['c'].semilogy(p, 1 / f, color='k', alpha=0.8)
        axs['c'].invert_xaxis()
        fap001 = gls.false_alarm_level(0.01)
        axs['c'].axvline(fap001, ls='--', alpha=0.2)

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
                               show_only_GP=False,
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
    full_plot = kwargs.pop('full_plot', False)

    if samples is None:
        samples = res.posterior_sample
        mask = np.ones(samples.shape[0], dtype=bool)
    else:
        samples = np.atleast_2d(samples)
    t = res.data.t.copy()
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

        # if t.size > 100:
        #     ncurves = min(10, ncurves)

    y = res.data.y.copy()
    yerr = res.data.e.copy()

    y2 = res.data.y2.copy()
    y2err = res.data.e2.copy()

    # y_offset = round(y.mean(), 0) if abs(y.mean()) > 100 else 0
    # y2_offset = round(y2.mean(), 0) if abs(y2.mean()) > 100 else 0

    # print(samples.shape)
    # print(ncurves)
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
        # ii = np.random.choice(np.where(mask)[0], ncurves)

    if 'ax1' in kwargs and 'ax2' in kwargs:
        ax1, ax2 = kwargs.pop('ax1'), kwargs.pop('ax2')
        fig = ax1.figure
    else:
        fig = plt.figure(constrained_layout=True, figsize=(10, 8))
        if full_plot:
            width_ratios = [3, 1, 1]
            gs = plt.GridSpec(2, 3, width_ratios=width_ratios, figure=fig)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax1p = fig.add_subplot(gs[0, 1])
            ax1r = fig.add_subplot(gs[0, 2])
            ax2p = fig.add_subplot(gs[1, 1])
            ax2r = fig.add_subplot(gs[1, 2])
            # ax1r, ax2r = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[3, 0])
        else:
            gs = plt.GridSpec(2, 1, figure=fig)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            # ax1r, ax2r = fig.add_subplot(gs[1]), fig.add_subplot(gs[3])

    _, _, y_offset, y2_offset = plot_data(res, ax=ax1, axf=ax2, ms=3,
                                          legend=False)

    ## plot the Keplerian curves
    alpha = 0.2 if ncurves > 1 else 1

    try:
        overlap = res._time_overlaps[0]
    except ValueError:
        overlap = False

    for icurve, i in enumerate(ii):
        # print(icurve, i)
        # print(res.log_posterior(samples[i], separate=True))

        # just the GP, centered around 0
        # for models without GP, stoc_model will be full of zeros
        stoc_model = res.stochastic_model(samples[i], tt)
        # the model, including planets, systemic RV/FWHM, and offsets
        model = res.eval_model(samples[i], tt)
        # burst the model if there are multiple instruments
        model = res.burst_model(samples[i], tt, model)

        if not show_only_GP:
            kw = dict(color='k', alpha=alpha, zorder=-2)
            if overlap:
                v = stoc_model[0] + model[::2] - y_offset
                ax1.plot(tt, v.T, **kw)
            else:
                ax1.plot(tt, stoc_model[0] + model[0] - y_offset, **kw)
                # ax2.plot(tt, stoc_model[1] + model[1], 'k', alpha=alpha)

        # the model without planets, just systemic RV/FWHM and offsets
        offset_model = res.eval_model(samples[i], tt, include_planets=False)
        # burst the model if there are multiple instruments
        offset_model = res.burst_model(samples[i], tt, offset_model)

        if res.KO:
            for iko in range(res.nKO):
                KOpl = res.eval_model(samples[i], tt,
                                      single_planet=-iko - 1)[0]
                ax1.plot(tt, KOpl - y_offset + (iko + 1) * res.data.y.ptp(),
                         color='g', alpha=alpha)

        if res.GPmodel:
            kw = dict(color='plum', alpha=alpha, zorder=1)
            if overlap:
                v = stoc_model[0] + offset_model[::2] - y_offset
                ax1.plot(tt, v.T, **kw)
                f = stoc_model[1] + offset_model[1::2] - y2_offset
                ax2.plot(tt, f.T, **kw)
            else:
                ax1.plot(tt, stoc_model[0] + offset_model[0] - y_offset, **kw)
                ax2.plot(tt, stoc_model[1] + offset_model[1] - y2_offset, **kw)

        if show_vsys:
            kw = dict(alpha=0.1, color='r', ls='--')
            if res.multi:
                for j in range(res.n_instruments):
                    instrument_mask = res.data.obs == j + 1
                    start = t[instrument_mask].min()
                    end = t[instrument_mask].max()
                    m = np.where( (tt > start) & (tt < end) )
                    ax1.plot(tt[m], offset_model[0][m] - y_offset, **kw)
                    ax2.plot(tt[m], offset_model[1][m] - y2_offset, **kw)
            else:
                ax1.plot(tt, offset_model[0] - y_offset, **kw)
                ax2.plot(tt, offset_model[1] - y2_offset, **kw)

        if full_plot:
            from gatspy.periodic import LombScargleMultiband
            from astropy.timeseries import LombScargle
            r = res.residuals(samples[i], full=True)
            freq = LombScargle(res.data.t, r[0], res.data.e).autofrequency()

            kwl = dict(color='k', alpha=0.2, ls='--')
            for i, ax in enumerate((ax1r, ax2r)):
                # option 1
                # gls = LombScargle(res.t, r[i], res.data.e)
                # freq, power = gls.autopower()
                # option 2
                gls = LombScargleMultiband(Nterms_base=1, Nterms_band=0)
                gls.fit(res.data.t, r[i], (res.data.e, res.data.e2)[i],
                        filts=res.data.obs)
                power = gls.periodogram(1 / freq)
                ax.semilogy(power, 1 / freq, 'k', alpha=alpha)

                gls = LombScargle(res.data.t, r[i] - gls.ymean_,
                                  (res.data.e, res.data.e2)[i])
                fap001 = gls.false_alarm_level(0.01)
                ax.axvline(fap001, **kwl)
                # kwt = dict(color='k', alpha=0.3, rotation=90, ha='left', va='top', fontsize=8)
                # fap001 = gls.false_alarm_level(0.01)
                # ax.axvline(fap001, **kwl)
                # ax.text(0.98*fap001, 1/freq.min(), '1%', **kwt)

                # fap01 = gls.false_alarm_level(0.1)
                # ax.axvline(fap01, **kwl)
                # ax.text(0.98*fap01, 1/freq.min(), '10%', **kwt)

                # ax.set(xlabel='residual power', ylabel='Period [days]')
                # ax.invert_xaxis()
                # ax.yaxis.tick_right()
                # ax.yaxis.set_label_position('right')

    # ## plot the data
    # if res.multi:
    #     for j in range(res.inst_offsets.shape[1] // 2 + 1):
    #         inst = res.instruments[j]
    #         m = res.data.obs == j + 1

    #         kw = dict(fmt='o', ms=3, color=colors[j], label=inst)
    #         kw.update(**kwargs)
    #         ax1.errorbar(t[m], y[m] - y_offset, yerr[m], **kw)
    #         ax2.errorbar(t[m], res.y2[m] - y2_offset, res.e2[m], **kw)

    #     ax1.legend(loc='upper left', fontsize=8)

    # else:
    #     ax1.errorbar(t, y - y_offset, yerr, fmt='o')
    #     ax2.errorbar(t, y2 - y2_offset, res.e2, fmt='o')

    if full_plot:
        kwl = dict(color='k', alpha=0.2, ls='--')
        gls = LombScargleMultiband(Nterms_base=1, Nterms_band=0)
        gls.fit(res.data.t, res.data.y, res.data.e, filts=res.data.obs)
        power = gls.periodogram(1 / freq)
        ax1p.semilogy(power, 1 / freq, 'r', alpha=1)
        gls = LombScargle(res.data.t, res.data.y - gls.ymean_, res.data.e)
        # kwt = dict(color='k', alpha=0.3, rotation=90, ha='left', va='top', fontsize=8)
        fap001 = gls.false_alarm_level(0.01)
        ax1p.axvline(fap001, **kwl)
        # ax.text(0.98*fap001, 1/freq.min(), '1%', **kwt)

        gls = LombScargleMultiband(Nterms_base=1, Nterms_band=0)
        gls.fit(res.data.t, res.data.y2, res.data.e2, filts=res.data.obs)
        power = gls.periodogram(1 / freq)
        ax2p.semilogy(power, 1 / freq, 'r', alpha=1)
        gls = LombScargle(res.data.t, res.data.y2 - gls.ymean_, res.data.e2)
        fap001 = gls.false_alarm_level(0.01)
        ax2p.axvline(fap001, **kwl)


    if res.arbitrary_units:
        ylabel = 'Q [arbitrary]'
    else:
        ylabel = 'RV [m/s]'

    ax1.set(ylabel=ylabel, xlabel='Time [days]')
    # ax1r.set(ylabel='', xlabel='Time [days]')
    ax2.set(ylabel='FWHM [m/s]', xlabel='Time [days]')
    # ax2r.set(ylabel='', xlabel='Time [days]')
    # if full_plot:
    # ax2.set(xlabel='Time [days]', ylabel=f'FWHM [m/s]')

    if full_plot:
        ax1p.set(xlabel='Power', ylabel='Period [days]')
        ax2p.set(xlabel='Power', ylabel='Period [days]')
        ax1r.set(xlabel='Residual Power')
        ax2r.set(xlabel='Residual Power')
        for ax in (ax1p, ax1r, ax2p, ax2r):
            ax.invert_xaxis()

    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position('right')



    # fig.tight_layout()

    if res.save_plots:
        filename = 'kima-showresults-fig6.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig

    return fig, ax1, ax2


def plot_random_samples_transit(res, ncurves=50, samples=None, over=0.1,
                                show_vsys=False, ntt=5000,
                                isolate_known_object=True, full_plot=False,
                                **kwargs):

    import batman

    if samples is None:
        samples = res.posterior_sample
    else:
        samples = np.atleast_2d(samples)
    mask = np.ones(samples.shape[0], dtype=bool)

    t = res.data.t.copy()
    M0_epoch = res.M0_epoch
    if t[0] > 24e5:
        t -= 24e5
        M0_epoch -= 24e5

    tt = res._get_tt(ntt, over)
    if res.GPmodel:
        ttGP = res._get_ttGP()

    # if t.size > 100:
    #     ncurves = min(10, ncurves)

    ncurves = min(ncurves, samples.shape[0])

    if samples.shape[0] == 1:
        ii = np.zeros(1, dtype=int)
    elif ncurves == samples.shape[0]:
        ii = np.arange(ncurves)
    else:
        try:
            # select `ncurves` indices from the 70% highest likelihood samples
            lnlike = res.posterior_lnlike[:, 1]
            sorted_lnlike = np.sort(lnlike)[::-1]
            mask_lnlike = lnlike > np.percentile(sorted_lnlike, 70)
            ii = np.random.choice(np.where(mask & mask_lnlike)[0], ncurves)
        except ValueError:
            ii = np.arange(ncurves)

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        fig = ax.figure
    else:
        fig, ax = plt.subplots(1, 1)

    _, y_offset = plot_transit_data(res, ax, **kwargs)

    ## plot the Keplerian curves
    alpha = 0.1 if ncurves > 1 else 1

    cc = kwargs.get('curve_color', 'k')
    gpc = kwargs.get('gp_color', 'plum')

    params = batman.TransitParams()
    params.inc = 90.                     #orbital inclination (in degrees)
    params.limb_dark = "quadratic"       #limb darkening model

    for icurve, i in enumerate(ii):
        sample = samples[i]
        params.u = sample[res.indices['u']]  # limb darkening coefficients [u1, u2]
        flux = np.full_like(tt, sample[-1])
        planet_pars = sample[res.indices['planets']]

        for j in range(res._mc):
            params.per = planet_pars[j::res._mc][0]                      #orbital period
            params.rp = planet_pars[j::res._mc][1]                       #planet radius (in units of stellar radii)
            params.a = planet_pars[j::res._mc][2]                        #semi-major axis (in units of stellar radii)
            params.ecc = planet_pars[j::res._mc][3]                      #eccentricity
            params.w = planet_pars[j::res._mc][4]            #eccentricity
            params.t0 = planet_pars[j::res._mc][5]                       #time of inferior conjunction

            m = batman.TransitModel(params, tt)    #initializes model
            flux += -1.0 + m.light_curve(params)          #calculates light curve

        #     stoc_model = np.atleast_2d(res.stochastic_model(sample, tt))
        #     model = np.atleast_2d(res.eval_model(sample, tt))
        #     offset_model = res.eval_model(sample, tt, include_planets=False)

        #     if res.multi:
        #         model = res.burst_model(sample, tt, model)
        #         offset_model = res.burst_model(sample, tt, offset_model)

        ax.plot(tt, flux - y_offset, color=cc, alpha=alpha, zorder=10)
        #     if res.GPmodel:
        #         ax.plot(tt, (stoc_model + offset_model).T - y_offset, color=gpc,
        #                 alpha=alpha)

        #     if show_vsys:
        #         kw = dict(alpha=alpha, color='r', ls='--')
        #         if res.multi:
        #             for j in range(res.n_instruments):
        #                 instrument_mask = res.data.obs == j + 1
        #                 start = t[instrument_mask].min()
        #                 end = t[instrument_mask].max()
        #                 m = np.where((tt > start) & (tt < end))
        #                 ax.plot(tt[m], offset_model[m] - y_offset, **kw)
        #         else:
        #             ax.plot(tt, offset_model - y_offset, **kw)

        if res.KO and isolate_known_object:
            # flux = np.full_like(tt, sample[-1])
            KO_pars = sample[res.indices['KOpars']]
            for j in range(0, res.nKO):
                print(KO_pars[j::res.nKO])
                params.per = KO_pars[j::res.nKO][0]                      #orbital period
                params.rp = KO_pars[j::res.nKO][1]                       #planet radius (in units of stellar radii)
                params.a = KO_pars[j::res.nKO][2]                        #semi-major axis (in units of stellar radii)
                params.ecc = KO_pars[j::res.nKO][3]                      #eccentricity
                params.w = KO_pars[j::res.nKO][4]        #longitude of periastron (in degrees)
                params.t0 = KO_pars[j::res.nKO][5]                       #time of inferior conjunction

                m = batman.TransitModel(params, tt)    #initializes model
                flux = m.light_curve(params)          #calculates light curve
                # kepKO = res.eval_model(res.posterior_sample[i], tt,
                #                        single_planet=-k)
                ax.plot(tt, flux - y_offset, color=f'C{j}', alpha=alpha)

    if res.save_plots:
        filename = 'kima-showresults-fig6.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def orbit(res, sample=None, n=10, star_mass=1.0, sortP=False):
    from .analysis import get_planet_mass
    from .utils import mjup2msun
    import rebound

    if sample is None:
        if sortP:
            sample = res.get_sorted_planet_samples(full=True)
            ind = np.random.choice(np.arange(sample.shape[0]), n)
            pp = sample[ind]
        else:
            pp = res.posterior_sample[:n]
    else:
        pp = np.atleast_2d(sample)

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, aspect="equal")

    for p in pp:
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
            # res.move_to_com()

        if res.KO:
            pars = p[res.indices['KOpars']]
            for i in range(res.nKO):
                P, K, φ, ecc, ω = pars[i::res.nKO]
                # print(P, K, φ, ecc, ω)
                m = get_planet_mass(P, K, ecc, star_mass=star_mass)[0]
                m *= mjup2msun
                sim.add(P=P, m=m, e=ecc, omega=ω, M=φ, inc=0)
                # res.move_to_com()

        kw = dict(
            fig=fig,
            color=True,
            show_particles=False,
        )
        rebound.plotting.OrbitPlot(sim, **kw)

    if len(pp) == 1:
        return sim


def simulation(results, sample):
    res, p = results, sample
    star_mass = 0.12
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.dt = 1
    sim.G = 0.00029591  # AU^3 / solMass / day^2
    sim.add(m=star_mass)

    # periods = []
    # eccentricities = []

    nplanets = int(p[res.index_component])
    pars = p[res.indices['planets']]
    for i in range(nplanets):
        P, K, φ, ecc, ω = pars[i::res.max_components]
        # print(P, K, φ, ecc, ω)
        m = get_planet_mass(P, K, ecc, star_mass=star_mass)[0]
        m *= mjup2msun
        sim.add(P=P, m=m, e=ecc, omega=ω, M=φ, inc=0)

        # periods.append(P)
        # eccentricities.append(ecc)

        # res.move_to_com()

    if res.KO:
        pars = p[res.indices['KOpars']]
        for i in range(res.nKO):
            P, K, φ, ecc, ω = pars[i::res.nKO]
            # print(P, K, φ, ecc, ω)
            m = get_planet_mass(P, K, ecc, star_mass=star_mass)[0]
            m *= mjup2msun
            sim.add(P=P, m=m, e=ecc, omega=ω, M=φ, inc=0)
            # periods.append(P)
            # eccentricities.append(ecc)

    sim.move_to_com()
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        # integrate for 100 years, integrating to the nearest timestep for
        # each output to keep the timestep constant and preserve WHFast's
        # symplectic nature
        sim.integrate(100 * 365, exact_finish_time=0)
        megno = sim.calculate_megno()
        return megno
        # ax.semilogx(periods, eccentricities, 'g.')
    except rebound.Escape:
        # At least one particle got ejected, returning large MEGNO.
        return 10
        # ax.semilogx(periods, eccentricities, 'rx')


from .analysis import get_planet_mass
from .utils import mjup2msun
import rebound


def megno(res, star_mass=1.0, samples=None):

    if samples is None:
        samples = res.posterior_sample

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111) #, aspect="equal")



    from rebound.interruptible_pool import InterruptiblePool
    pool = InterruptiblePool()
    results = pool.map(simulation, samples)
    return results
    # return np.array(M)


def plot_parameter_samples(res):
    sample = res.sample.copy()

    _i = res.indices['vsys'] - 1
    sample = np.delete(sample, _i, axis=1)
    _i = res.indices['np'] - 1, res.indices['np'] - 2
    sample = np.delete(sample, _i, axis=1)

    sample = np.c_[res.sample_info[:, 1], sample]

    s = np.argsort(res.sample_info[:, 1])
    sample = sample[s, :]

    fig, axs = plt.subplots(sample.shape[1], constrained_layout=True,
                            figsize=(6, 12), sharex=True)

    for p, ax in zip(sample.T, axs):
        ax.plot(p)
    # axs[0].set(yscale='symlog')
    axs[-1].set(xlabel='sample')
    return fig, axs
