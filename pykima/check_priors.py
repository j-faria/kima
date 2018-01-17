import sys
import numpy as np 
import matplotlib.pyplot as plt
import argparse


def _parse_args():
    desc = """
    A small script to check correct sampling from the priors.
    Remember to run kima with the maximum number of levels
    set to 1 (and save interval = 1 to speed things up).
    Then, sample.txt contains samples from the prior.
    This script plots histograms of each column of that file.
    """
    parser = argparse.ArgumentParser(description=desc,
                                     prog='kima-checkpriors',
                                     # usage='%(prog)s [no,1,2,...,7]'
                                     )
    parser.add_argument('column', nargs=1, type=int,
                        help='which column to use for histogram')
    parser.add_argument('--log', action='store_true',
                        help='plot the logarithm of the samples')
    args = parser.parse_args()
    return args


def do_plot(data, save=None, xlabel=None, bins=None):
    
    fig, ax = plt.subplots(1,1)
    if bins is None:
        bins = 100

    ax.hist(data, bins=bins, color='k', histtype='step', align='mid',
            range=[data.min()-0.2*data.ptp(), data.max()+0.2*data.ptp()],
            )

    ax.set_xlabel(xlabel)
    if save:
        fig.savefig(save)
    else:
        return fig, ax

def main():
    args = _parse_args()
    column = args.column[0]
    log = args.log

    with open('sample.txt') as f:
        firstline = f.readline()
    firstline = firstline.strip().replace('#','')
    names = firstline.split()

    try:
        name = names[column-1]
        print 'Histogram of column %d: %s' % (column, name)
    except IndexError:
        print 'Histogram of column %d' % column

    data = np.loadtxt('sample.txt', usecols=(column-1,))
    data = data[np.nonzero(data)[0]]
    print ('  number of samples: %d' % data.size)
    print ('  max value: %f' % data.max())
    print ('  min value: %f' % data.min())
    
    xlabel = name
    if log:
        data = np.log(data)
        xlabel = 'log ' + name

    do_plot(data, xlabel=xlabel)
    plt.show()


if __name__ == '__main__':
    main()