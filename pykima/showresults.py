from __future__ import print_function

import __main__
from .classic import postprocess
from .display import KimaResults
import sys, os
from collections import namedtuple
from matplotlib.pyplot import show

numbered_args_help = """optional numbered arguments:
  1    - plot the posterior for Np;
  2    - plot the posterior for the orbital periods;
  3    - plot the joint posterior for semi-amplitudes, eccentricities and orbital periods;
  4, 5 - plot the posteriors for the GP hyperparameters (marginal and joint);
  6    - plot random posterior samples in data-space, together with the RV data;
  7    - plot posteriors for the HARPS fiber offset and systematic velocity;
"""

def findpop(value, lst):
    """ Return whether `value` is in `lst` and remove all its occurrences """
    if value in lst:
        while True: # remove all instances `value` from lst
            try: lst.pop(lst.index(value))
            except ValueError: break
        return True # and return yes we found the value 
    else:
        return False # didn't find the value

def usage(full=True):
    u = "usage: kima-showresults "\
        "[rv] [planets] [orbital] [gp] [extra] [1, ..., 7]\n"\
        "                        [-h/--help] [--version]"
    u += '\n\n'
    if not full: return u

    pos = ["positional arguments:\n"]
    names = ['rv', 'planets', 'orbital', 'gp', 'extra']
    descriptions = \
        ["Plot posterior realizations of the model over the RV measurements",
         "Plot posterior for number of planets",
         "Plot posteriors for some of the orbital parameters",
         "Plot posteriors for GP hyperparameters",
         "Plot posteriors for fiber offset, systematic velocity, and extra white noise",
        ]
    for n, d in zip(names, descriptions):
        pos.append("  %-10s\t%s\n" % (n,d))
    u += ''.join(pos)
    
    u+= numbered_args_help
    return u


def _parse_args(options):
    if options == '':
        if 'kima-showresults' in sys.argv[0]:
            args = sys.argv[1:]
        else:
            args = options
    else:
        args = options.split()
    
    if '-h' in args or '--help' in args:
        print(usage())
        sys.exit(0)
    if '--version' in args:
        version_file = os.path.join(os.path.dirname(__file__), '../VERSION')
        print('kima', open(version_file).read().strip()) # same as kima
        sys.exit(0)

                         
    number_options = ['1','2','3','4','5','6','7']
    argstuple = namedtuple('Arguments', 
                                ['rv', 'planets', 'orbital', 'gp', 'extra'] \
                                + ['diagnostic'] \
                                + ['plot_number'])
    
    if 'all' in args:
        diag = findpop('diagnostic', args)
        return argstuple(rv=True, planets=True, orbital=True, 
                         gp=True, extra=True, diagnostic=diag, 
                         plot_number=[])

    rv = findpop('rv', args)
    gp = findpop('gp', args)
    extra = findpop('extra', args)
    planets = findpop('planets', args)
    orbital = findpop('orbital', args)
    diag = findpop('diagnostic', args)
    plots = list(set(args).intersection(number_options))
    for plot in plots:
        findpop(plot, args)

    if len(args)>=1:
        print(usage(full=False), end='')
        print('error: could not recognize argument:', "'%s'" % args[0])
        sys.exit(1)

    return argstuple(rv, planets, orbital, gp, extra, diag, plot_number=plots)


def showresults(options=''):
    """
    Generate and plot results from a kima run. 
    The argument `options` should be a string with the same options as for 
    the kima-showresults script.
    """

    # force correct CLI arguments
    args = _parse_args(options)

    plots = []
    if args.rv:
        plots.append('6')
    if args.planets:
        plots.append('1')
    if args.orbital:
        plots.append('2'); plots.append('3')
    if args.gp:
        plots.append('4'); plots.append('5')
    if args.extra:
        plots.append('7')
    for number in args.plot_number:
        plots.append(number)
    

    try:
        evidence, H, logx_samples = postprocess(plot=args.diagnostic)
    except IOError as e:
        print(e)
        sys.exit(1)

    res = KimaResults(list(set(plots)))
    show() # render the plots

    # __main__.__file__ doesn't exist in the interactive interpreter
    if not hasattr(__main__, '__file__'):
        return res

if __name__ == '__main__':
    options = sys.argv
    res = showresults(options)
