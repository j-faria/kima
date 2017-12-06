# first, run DNest4 with max number of levels set to 1

import sys
import numpy as np 
import matplotlib.pyplot as plt
# from astroML.plotting import hist

def do_plot(data, name, column=1, save=None, bins=None, normed=True, logxscale=False):
    fig, ax = plt.subplots(1,1)
    if bins is None:
        bins = 100 #np.linspace(data.min(), data.max(), 100)
    ax.hist(data, bins=bins, color='black', histtype='step', normed=normed,
            range=[data.min() - 0.2*data.ptp(), data.max() + 0.2*data.ptp()],
            align='mid')
    # hist(data, bins='knuth', color='black', histtype='step', normed=True)
    if logxscale: ax.set_xscale('log')
    if save:
        fig.savefig(save)
    else:
        return fig, ax


def get_column(column):
    return np.loadtxt('sample.txt', unpack=True, usecols=(column-1,))


if __name__ == '__main__':
    column = int(sys.argv[1])
    try:
        log = sys.argv[2] == 'log'
        exp = sys.argv[2] == 'exp'
    except IndexError:
        log = False
        exp = False


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

    print 'max:', data.max()
    print 'min:', data.min()
    bins = None
    normed = True
    logxscale = False

    if log:
        data = data[np.nonzero(data)[0]]
        data = np.log(data)

    if exp:
        data = np.exp(data)
        data = data[~np.isinf(data)]
        try:
            l1, l2 = sys.argv[3].split(',')
            print float(l1), float(l2)
        except IndexError:
            print 'provide (, separated) limits for bins after exp'
            sys.exit(0)

        bins = 10 ** np.linspace(np.log10(float(l1)), np.log10(float(l2)), 100)
        normed = False
        logxscale = True

    # if log:
        # bins = np.logspace(np.log(data.min()), np.log(data.max()), 100)
    # else:

    do_plot(data, column, bins=bins, normed=normed, logxscale=logxscale)
    plt.show()
