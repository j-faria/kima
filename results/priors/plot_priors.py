from pylab import *
import subprocess

parameters = ['s', 'alpha', 'Lambda_e', 'Lambda_p', 'tau',
              'P', 'K', 'ecc', 'w', 'phi',
              'C']

import sys
sys.path.append('../../scripts')
from check_priors import do_plot, get_column


for i, par in enumerate(parameters):
    data = get_column(i+1)
    do_plot(data, par, save='prior_%s.png' % par)
