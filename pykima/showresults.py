from .classic import postprocess
from .display import KimaResults
import sys
import argparse
from matplotlib.pyplot import isinteractive, show

def _parse_args():
    argshelp = """
    'no' skips the diagnostic DNest4 plots;
    1 plots the posterior for Np;
    2 plots the posterior for the orbital periods;
    3 plots the joint posterior for semi-amplitudes, eccentricities
    and orbital periods;
    4 and 5 plot the posteriors for the GP hyperparameters;
    6 plots random posterior samples in data-space, together with the data;
    7 plots posteriors for the fiber_offset and vsys;
    """
    parser = argparse.ArgumentParser(prog='kima-showresults',
                                     usage='%(prog)s [no,1,2,...,7]')
    parser.add_argument('options', nargs='*', 
                        choices=['no','1','2','3','4','5','6','7', ''],
                        help=argshelp, default='')
    # parser.add_argument('--logz', action='store_true',
                        # help='just print the value of the log evidence')
    # parser.add_argument('--neff', action='store_true',
                        # help='just print the effective sample size')
    args = parser.parse_args()
    return args


def showresults(options=''):
    if not isinteractive():
        # use argparse to force correct CLI arguments
        args = _parse_args()
        options = args.options

    if 'no' in options:
        plot = False
    else:
        plot = True

    try:
        evidence, H, logx_samples = postprocess(plot=plot)
    except IOError as e:
        print(e)
        sys.exit(1)

    res = KimaResults(options)
    if isinteractive(): 
        return res
    
    show() # render the plots

if __name__ == '__main__':
    options = sys.argv
    res = showresults(options)
