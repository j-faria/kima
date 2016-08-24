# first, run DNest4 with max number of levels set to 1

import sys
import numpy as np 
import matplotlib.pyplot as plt
from astroML.plotting import hist

column = int(sys.argv[1])

try:
	log = sys.argv[2] == 'log'
except IndexError:
	log = False

with open('sample.txt') as f:
	firstline = f.readline()
firstline = firstline.strip().replace('#','')
names = firstline.split()

try:
	print 'Histogram of column %d: %s' % (column, names[column-1])
except IndexError:
	print 'Histogram of column %d' % column

data = np.loadtxt('sample.txt', unpack=True, usecols=(column-1,))


plt.figure()

if log:
	data = data[np.nonzero(data)[0]]
	data = np.log(data)

hist(data, bins='knuth', color='black', histtype='step', normed=True)
plt.show()