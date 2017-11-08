from pylab import *
import subprocess

parameters = ['s', #'alpha', 'Lambda_e', 'Lambda_p', 'tau',
              None, None, None,
              'P', 'K', 'w', 'ecc', 'phi',
              None,
              'C']

names = ['jitter',
         None, None, None,
         'Period', 'Semi-amplitude', 'Omega', 'Eccentricity', 'phi',
         None,
         'Systemic velocity']

import sys
sys.path.append('../../scripts')
from check_priors import do_plot, get_column


for i, par in enumerate(parameters):
    if par is None: continue
    data = get_column(i+1)
    fig, ax = do_plot(data, par, save=False)

    ax.set_yticks([])
    ax.set_title(names[i] + '(%.0e samples from prior)' % data.size)

    ax.legend(['min: %f, max: %f' % (data.min(), data.max())], loc='best')

    fig.savefig('prior_%s.png' % par)
