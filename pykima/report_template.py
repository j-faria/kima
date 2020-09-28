import os
import zlib
import argparse
import numpy as np
from matplotlib import rc
# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{color}')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
old_subplots = plt.subplots
old_tl = plt.tight_layout

from colorful import blue
import pykima as pk


def long_substr(data):
    substrs = lambda x: {
        x[i:i + j]
        for i in range(len(x)) for j in range(len(x) - i + 1)
    }
    s = substrs(data[0])
    for val in data[1:]:
        s.intersection_update(substrs(val))
    return max(s, key=len)


def _parse_args():
    parser = argparse.ArgumentParser('kima-report')
    parser.add_argument('star', nargs='?', type=str)
    parser.add_argument('-s', '--save', action='store_true')
    return parser.parse_args()


def make_report(results=None, star=None, save=None, verbose=True, prot=None,
                known_planets=None, HZ=None, plot2_kw={}, plot3_kw={}):
    if results is None:
        res = pk.showresults(force_return=True)
    else:
        res = results

    res.data_file = [os.path.basename(d) for d in res.data_file]

    if star is None:
        star = ''

    with plt.style.context('seaborn-whitegrid'):
        size = 8.5, 10
        ncols = 6

        name = 'kima analysis' + (' ' + star if star else '')
        fig = plt.figure(name, figsize=size, constrained_layout=True)
        # fig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.1)
        # fig.suptitle(star)
        gs = gridspec.GridSpec(5, ncols, figure=fig,
                               height_ratios=[1, 1.3, 1, 1, 1])

        res.KO = False
        if res.KO:
            ax = plt.subplot(gs[0, :3])
            ax1 = plt.subplot(gs[0, 3:])

            plt.subplots = lambda *args, figsize=None, constrained_layout=False: (
                fig, (ax, ax1))
        else:
            ax = plt.subplot(gs[0, :4])
            plt.subplots = lambda *args: (fig, ax)

        res.plot_random_planets(ncurves=20, show_vsys=True, ms=2)
        ax.set(title='', )
        leg = ax.legend(loc="upper left", bbox_to_anchor=(0, 1.25), ncol=4,
                        fontsize=8)
        names = [
            d.replace(star, '').replace('_', '').replace('bin',
                                                        '').replace('.rdb', '')
            for d in res.data_file
        ]
        for label, name in zip(leg.get_texts(), names):
            label.set_text(name)
        # print(len(leg.legendHandles))
        # for handle in leg.legendHandles[len(names):]:
        #     handle.set_visible(False)

        axnp = plt.subplot(gs[0, 4:6])
        plt.subplots = lambda *args: (fig, axnp)
        res.make_plot1()
        axnp.set(ylabel='', yticks=[])
        axnp.title.set_fontsize(10)

        # orbital period posterior
        axP = plt.subplot(gs[1, :4])
        plt.subplots = lambda _1, _2: (fig, axP)

        if 'show_prior' not in plot2_kw:
            plot2_kw['show_prior'] = True

        res.make_plot2(**plot2_kw)

        if prot:
            kw = dict(color='c', alpha=0.5, lw=2, zorder=-1)
            axP.axvline(prot, label=r'$P_{\rm rot}$', **kw)
            axP.legend(fontsize=8)

        if known_planets:
            kw = dict(color='m', alpha=0.8, lw=2, zorder=-1)
            axP.vlines(known_planets, *axP.get_ylim(), label=r'planets', **kw)
            axP.legend(fontsize=8)

        if HZ is not None:
            axP.axvspan(*HZ, color='g', alpha=0.1, zorder=-1, label='HZ')
            axP.legend(fontsize=8)

        axP.title.set_fontsize(10)
        axP.set_ylabel('posterior')

        axPK = plt.subplot(gs[2, :4], sharex=axP)
        axPE = plt.subplot(gs[3, :4], sharex=axP)
        plt.subplots = lambda _1, _2, sharex=False: (fig, (axPK, axPE))
        res.make_plot3(points=True)
        axPK.title.set_fontsize(10)
        axPE.title.set_fontsize(10)
        # Plim = list(res.priors['Pprior'].support())
        # Plim[1] = max(Plim[1], 2*res.t.ptp())
        # axP.set_xlim(Plim)

        if HZ is not None:
            axPK.axvspan(*HZ, color='g', alpha=0.1, zorder=1, label='HZ')
            axPE.axvspan(*HZ, color='g', alpha=0.1, zorder=1, label='HZ')

        ax = plt.subplot(gs[4, 0:2])
        # plt.subplots = lambda _1,_2: (fig, ax)
        vsys = res.posterior_sample[:, -1].copy()
        print(vsys.mean())
        if abs(vsys.mean()) > 100:
            m = vsys.mean()
            vsys -= m
            sign = '+' if m > 0 else ''
            ax.set_title(f'{sign} {m/1e3:.2f} km/s', loc='left', fontsize=8)

        ax.hist(vsys)#, bins=50)
        # res.hist_vsys(show_offsets=False)
        # fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        # fmt.set_scientific(False)
        # ax.xaxis.set_major_formatter(fmt)
        xloc = plt.MaxNLocator(3, min_n_ticks=3)
        ax.xaxis.set_major_locator(xloc)
        ax.set(xlabel='vsys [m/s]', ylabel='posterior', yticks=[])
        ax.minorticks_on()

        ax = plt.subplot(gs[4, 2:4])
        if res.multi:
            labels = [os.path.splitext(d)[0] for d in res.data_file]
            subs = long_substr(labels)
            labels = [l.replace(subs, '') for l in labels]

            for i, s in enumerate(res.extra_sigma.T):
                ax.hist(s, alpha=0.9, histtype='step', label=labels[i], lw=2)
            ax.set_xlabel('extra sigma (m/s)')
            ax.legend(fontsize=6)
        else:
            plt.subplots = lambda _1, _2: (fig, ax)
            res.hist_extra_sigma()
        ax.set(title='', ylabel='posterior', yticks=[])

        if res.trend:
            ax = plt.subplot(gs[4, 4:])
            plt.subplots = lambda _1, _2: (fig, ax)
            res.hist_trend()
            ax.set(title='', ylabel='posterior', yticks=[])

        elif res.multi:
            ax = plt.subplot(gs[4, 4:])
            n_inst_offsets = res.inst_offsets.shape[1]

            labels = [os.path.splitext(d)[0] for d in res.data_file]
            subs = long_substr(labels)
            labels = [l.replace(subs, '') for l in labels]

            for i in range(n_inst_offsets):
                # label = '%s wrt %s' % (labels[-1], labels[i])
                ax.hist(res.inst_offsets[:, i], bins=50)  #, label=label)
            # ax.legend()
            ax.set(xlabel='instrument offsets', title='', ylabel='posterior',
                   yticks=[])

        # #
        axt = plt.subplot(gs[1, -2:])
        axt.axis('off')
        axt.text(0, 1, 'posterior samples: %d' % res.ESS)
        axt.text(0, 0.85, 'log evidence: %.2f' % res.evidence)
        axt.text(0, 0.7, 'information: %.2f nat' % res.information)

        NP = pk.analysis.passes_threshold_np(res)
        if NP >= 1:
            phase_axs = []
            p = res.maximum_likelihood_sample(printit=False)
            for ipl in range(NP):
                phase_axs.append(plt.subplot(gs[2 + ipl, 4:]))
            plt.ioff()
            fig_dummy = res.phase_plot(p, phase_axs=phase_axs,
                                       add_titles=False)
            plt.close(fig_dummy)
            plt.ion()

            letters = 'bcdefghij'
            for i, ax in enumerate(phase_axs):
                period = p[res.indices['planets']][i]
                ax.set_title(letters[i], loc='left', fontsize=8)
                ax.set_title(f'P={period:.2f} days', loc='right', fontsize=8)

        # #
        # s = b'x\x9csv\xd6u\x8aT8\xb4R\xc1+\xff\xf0\xe2|\x05\xb7\xc4\xa2\xccD\x05#\x03#\x03\x00r\xe4\x08o'
        # fig.text(0.99, 0.2,
        #          zlib.decompress(s).decode(), fontsize=10, color='gray',
        #          ha='right', va='bottom', alpha=0.5, rotation=90)

        if save is not None:
            if save is True:
                save = f'report_{"".join(star.split())}.pdf'

            with PdfPages(save) as pdf:
                if verbose:
                    print(blue | 'Saving report to', save)
                pdf.savefig(fig)

            plt.close('all')

        plt.subplots = old_subplots


def main():
    args = _parse_args()
    star = args.star
    res = pk.showresults(force_return=True)
    make_report(results=res, star=star, save=args.save)
    plt.show()
