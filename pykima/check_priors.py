# -*- coding: utf-8 -*-
import sys
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import argparse


def _parse_args():
    desc = """
    A small script to check correct sampling from the priors.
    Remember to run kima with the maximum number of levels
    set to 1 (and save interval = 1 to speed things up).
    Then, sample.txt contains samples from the prior.
    This script simply plots histograms of each column of that file.
    """
    parser = argparse.ArgumentParser(description=desc,
                                     prog='kima-checkpriors',
                                     # usage='%(prog)s [no,1,2,...,7]'
                                     )
    parser.add_argument('column', nargs='*',
                        help='column number or column name to use for histogram')
    parser.add_argument('--log', action='store_true',
                        help='plot the logarithm of the samples')
    parser.add_argument('--code', nargs=1, type=str,
                        help='code to generate "theoretical" samples '\
                             'to compare to the prior. \n'\
                             'Assign samples to an iterable called `samples`. '\
                             'Use numpy and scipy.stats as `np` and `st`, respectively. '\
                             'Number of prior samples in sample.txt is in variable `nsamples`. '\
                             'For example: samples=np.random.uniform(0,1,nsamples)')

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    # print(args)
    columns = args.column
    log = args.log
   
    with open('sample.txt') as f:
        firstline = f.readline()
    firstline = firstline.strip().replace('#', '')
    names = firstline.split()

    fig, ax = plt.subplots(1, 1)

    for column in columns:
        try:  # column number?
            column = int(column)
            try:
                name = names[column - 1]
                print('Histogram of column %d: %s' % (column, name))
            except IndexError:
                name = 'column %d' % column
                print('Histogram of column %d' % column)

            data = np.loadtxt('sample.txt', usecols=(column - 1,))
            data = data[np.nonzero(data)[0]]

        except ValueError:  # or column name?
            name = column
            data = np.genfromtxt('sample.txt', names=True)
            if column in data.dtype.names:
                data = data[column]
            elif column+'1' in data.dtype.names:
                col, columns = 1, []
                while column + str(col) in data.dtype.names:
                    columns.append(column + str(col))
                    col += 1
                data = np.array(data[columns].tolist()).ravel()

        data = data[np.nonzero(data)[0]]

        nsamples = data.size
        print('  number of samples: %d' % nsamples)
        # try:
        print('  max value: %f' % data.max())
        print('  min value: %f' % data.min())

        xlabel = name
        if log:
            data = np.log(data)
            xlabel = 'log ' + name

        # ax.set_xlabel(xlabel)

        ax.hist(
            data,
            density=True,
            bins=100,
            # color='k',
            histtype='step',
            align='mid',
            range=[
                data.min() - 0.2 * data.ptp(),
                data.max() + 0.2 * data.ptp()
            ],
            label=xlabel,
        )

        if args.code:
            namespace = locals()
            exec(args.code[0], globals(), namespace)
            samples = namespace['samples']

            ax.hist(
                samples,
                alpha=0.3,
                bins=100,
                align='mid',
                # range=[
                #     data.min() - 0.2 * data.ptp(),
                #     data.max() + 0.2 * data.ptp()
                # ],
            )

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
