from .dnest4 import postprocess
from .display import KimaResults
import sys
import argparse
from matplotlib.pyplot import isinteractive

def _parse_args():
    argshelp = """
    'no' skips the diagnostic DNest4 plots;
    1 plots the posterior for Np;
    2 plots the posterior for the orbital periods;
    3 plots the joint posterior for semi-amplitudes, eccentricities
    and orbital periods;
    4 and 5 plot the posteriors for the GP hyperparameters;
    """
    parser = argparse.ArgumentParser(prog='kima-showresults',
                                     usage='%(prog)s [no,1,2,...,5]')
    parser.add_argument('options', nargs='*', 
                        choices=['no','1','2','3','4','5',''],
                        help=argshelp, default='')
    parser.add_argument('--logz', action='store_true',
                        help='just print the value of the log evidence')
    parser.add_argument('--neff', action='store_true',
                        help='just print the effective sample size')
    args = parser.parse_args()
    return args


def showresults(logz=False, neff=False, options=''):
    if not isinteractive():
        # use argparse to force correct CLI arguments
        args = _parse_args()
        options, logz, neff = args.options, args.logz, args.neff

    if ('no' in options) or logz or neff:
        plot = False
    else:
        plot = True

    try:
        evidence, H, logx_samples, posterior = \
            postprocess(plot=plot, just_print_logz=logz, just_print_neff=neff)
    except IOError as e:
        print(e)
        sys.exit(1)

    if logz or neff:
        return

    if posterior.shape[0] > 5:
        res = KimaResults(options)
        if isinteractive(): 
            return res
    else:
        print('Too few samples, keep running the model')


if __name__ == '__main__':
    options = sys.argv
    res = showresults(options)
