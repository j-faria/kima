# first, run DNest4 with max number of levels set to 1

import sys
import numpy as np 
import matplotlib.pyplot as plt
from astroML.plotting import hist

def do_plot(data, name, column=1, save=None):
    plt.figure()
    bins = 100 #np.linspace(data.min(), data.max(), 100)
    plt.hist(data, bins=bins, color='black', histtype='step', normed=True)
    # if log: plt.xscale('log')
    # hist(data, bins='knuth', color='black', histtype='step', normed=True)
    if save:
        plt.savefig(save)
    else:
        plt.show()


def get_column(column):
    return np.loadtxt('sample.txt', unpack=True, usecols=(column-1,))


if __name__ == '__main__':
    column = int(sys.argv[1])
    try:
        log = sys.argv[2] == 'log'
    except IndexError:
        log = False

    with open('sample.txt') as f:
    # with open('posterior_sample.txt') as f:
        firstline = f.readline()
    firstline = firstline.strip().replace('#','')
    names = firstline.split()

    try:
        print 'Histogram of column %d: %s' % (column, names[column-1])
    except IndexError:
        print 'Histogram of column %d' % column

    data = get_column(column)
    # data = np.loadtxt('sample.txt', unpack=True, usecols=(column-1,))

    if log:
        data = data[np.nonzero(data)[0]]
        data = np.log(data)

    # if log:
        # bins = np.logspace(np.log(data.min()), np.log(data.max()), 100)
    # else:

    do_plot(data, column)
