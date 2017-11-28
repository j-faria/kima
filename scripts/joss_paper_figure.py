import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cbook as cbook
import matplotlib.image as image
import tqdm

import seaborn.apionly as sns
colors = sns.color_palette("colorblind")
sns.set_palette(sns.color_palette("colorblind"))

from keplerian import keplerian
from display import DisplayResults
# sys.path.append('/home/joao/Work/OPEN')
# from OPEN.ext.keplerian import keplerian


sys.path.append('.')
from styler import styler

res = DisplayResults('')
# res.sample = np.atleast_2d(np.loadtxt('sample.txt'))

# res = cPickle.load(open('BL2009_joss_figure1.pickle'))


@styler
def f(fig, *args, **kwargs):

	gs = gridspec.GridSpec(3, 6)
	# gs.update(hspace=0.4, wspace=0.3, bottom=0.08, top=0.97, left=0.08, right=0.98)
	ax1 = plt.subplot(gs[:2, :4])
	ax2 = plt.subplot(gs[1, 4:6])
	ax3 = plt.subplot(gs[2, 0:2])
	ax4 = plt.subplot(gs[2, 2:4])
	ax5 = plt.subplot(gs[2, 4:6])



	samples = res.get_sorted_planet_samples()
	samples, mask = \
	    res.apply_cuts_period(samples, 90, None, return_mask=True)


	t, y, yerr = res.data.T
	over = 0.1
	tt = np.linspace(t[0]-over*t.ptp(), t[-1]+over*t.ptp(), 
	                 10000+int(100*over))

	y = res.data[:,1].copy()
	yerr = res.data[:,2].copy()

	# select random `ncurves` indices 
	# from the (sorted, period-cut) posterior samples
	ncurves = 10
	ii = np.random.randint(samples.shape[0], size=ncurves)

	## plot the Keplerian curves
	for i in ii:
	    v = np.zeros_like(tt)
	    pars = samples[i, :].copy()
	    nplanets = pars.size / res.n_dimensions
	    for j in range(nplanets):
	        P = pars[j + 0*res.max_components]
	        K = pars[j + 1*res.max_components]
	        phi = pars[j + 2*res.max_components]
	        t0 = t[0] - (P*phi)/(2.*np.pi)
	        ecc = pars[j + 3*res.max_components]
	        w = pars[j + 4*res.max_components]
	        v += keplerian(tt, P, K, ecc, w, t0, 0.)

	    vsys = res.posterior_sample[mask][i, -1]
	    v += vsys
	    ax1.plot(tt, v, alpha=0.2, color='k')

	ax1.errorbar(*res.data.T, fmt='o', mec='none', capsize=0, ms=4,
		         color=sns.color_palette()[2])
	ax1.set(xlabel='Time [days]', ylabel='RV [m/s]',)




	n, bins, _ = ax2.hist(res.posterior_sample[:, res.index_component], 
		                  bins=np.arange(2, 4)-0.5, align='mid', rwidth=0.5, 
		                  color='k', alpha=0.5)
	ax2.set(#title=r'posterior for $N_p$',
		    xlabel=r'$N_p$', ylabel='Posterior',
		    yticks=[], xticks=range(3),)# xticklabels=map(str, range(3)))
	ax2.set_xlim([-0.5, res.max_components+.5])


	# bins = 10 ** np.linspace(np.log10(90), np.log10(1e3), 100)
	ax31 = plt.subplot(gs[2, 0])
	ax32 = plt.subplot(gs[2, 1])
	ax31.hist(samples[:,0], histtype='stepfilled', bins=np.linspace(90,110,50))
	ax32.hist(samples[:,1], bins=np.linspace(600,800,30),
		      histtype='stepfilled', color=colors[1])
	# ax3.set_xlim(70, 1000)
	for ax in [ax31, ax32]:
		ax.set_yticks([])
		ax.xaxis.tick_bottom()

	ax31.set_xlim(90, 112)
	ax31.set_xticks([90, 100, 110])
	ax32.set_xlim(590, 800)
	ax32.set_xticks([600, 700, 800])
	ax31.spines['right'].set_visible(False)
	ax32.spines['left'].set_visible(False)
	ax31.set_ylabel('Posterior')
	# ax2.spines['top'].set_visible(False)
	# ax.xaxis.tick_top()
	# ax.tick_params(labeltop='off')  # don't put tick labels at the top
	# xlabel='Period [days]',
	fig.text(0.215, 0.017, 'Period [days]', ha='center')

	ax41 = plt.subplot(gs[2, 2])
	ax42 = plt.subplot(gs[2, 3])
	# bins = 10 ** np.linspace(np.log10(1), np.log10(1e3), 40)
	ax41.hist(samples[:,2], histtype='stepfilled')
	ax42.hist(samples[:,3], histtype='stepfilled', color=colors[1])

	for ax in [ax41, ax42]:
		ax.set_yticks([])
		ax.xaxis.tick_bottom()

	ax41.set_xlim(5, 16)
	ax41.set_xticks([5, 10, 15])
	ax42.set_xlim(54, 65)
	ax42.set_xticks([55, 60, 65])
	ax41.spines['right'].set_visible(False)
	ax42.spines['left'].set_visible(False)
	ax41.set_ylabel('Posterior')
	fig.text(0.53, 0.017, 'Semi-amplitude [m/s]', ha='center')


	ax5.hist(samples[:,6], bins=np.linspace(0, 0.5, 30), histtype='stepfilled')
	ax5.hist(samples[:,7], bins=np.linspace(0, 0.5, 30), histtype='stepfilled')
	ax5.set(xlabel='Eccentricity', yticks=[], ylabel='Posterior')
	ax5.xaxis.labelpad = 6

	# im = image.imread('../../logo/logo_small.jpg')
	# aximg = plt.subplot(gs[0, 5])
	# aximg.axis('off')
	# aximg.imshow(im, alpha=1, extent=(0, 10, 0, 10))


f(type='main', save='joss_figure.png', png=True, formatx=False,
  tight=True, tight_kwargs={'w_pad':0.05, 'h_pad':0.5})
	# plt.show()