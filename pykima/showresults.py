from __future__ import print_function

import __main__
from .classic import postprocess
from .display import KimaResults
from .crossing_orbits import rem_crossing_orbits
from .utils import show_tips
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

other_args_help = """other optional arguments:
  --save-plots	      Save the plots as .png files (except diagnostic plots)
  --remove-roche      Remove orbits crossing the Roche limit of the star
                      (see --help-remove-roche for details)
  --remove-crossing   Remove crossing orbits
                      (see --help-remove-crossing for details)
"""

help_remove_crossing = """
The --remove-crossing optional argument will remove crossing orbits from the
posterior samples before doing any plotting. This assumes that the samples are
"sorted" in such a way that the first planet in the system (one row in the
posterior_sample.txt file) is the most "likely". This is often the case. Then,
the check for crossing orbits (based on periastron and apastron distances) is
done starting from the final Keplerian in a row and ending in the first. Running
with this option will affect many of the results, including the posteriors for
the number of planets, and orbital periods.
WARNING: This is an *a posteriori* analysis; neither the evidence nor any other
probability are renormalized. BE CAREFUL when taking conclusions from this!
"""

help_remove_roche = """
The --remove-roche option will...
"""


def findpop(value, lst):
    """ Return whether `value` is in `lst` and remove all its occurrences """
    if value in lst:
        while True:  # remove all instances `value` from lst
            try:
                lst.pop(lst.index(value))
            except ValueError:
                break
        return True  # and return yes we found the value
    else:
        return False  # didn't find the value


def usage(full=True):
    u = "usage: kima-showresults "\
        "[rv] [planets] [orbital] [gp] [extra] [1, ..., 7]\n"\
        "                        [all] [pickle] [--save-plots]\n"\
        "                        [-h/--help] [--version]"
    u += '\n\n'
    if not full: return u

    pos = ["positional arguments:\n"]
    names = [
        'rv',
        'planets',
        'orbital',
        'gp',
        'extra',
        'all',
        'pickle',
    ]
    descriptions = \
        ["Plot posterior realizations of the model over the RV measurements",
         "Plot posterior for number of planets",
         "Plot posteriors for some of the orbital parameters",
         "Plot posteriors for GP hyperparameters",
         "Plot posteriors for fiber offset, systematic velocity, and extra white noise",
         "Show all plots",
         "Save the model into a pickle file (filename will be prompted)",
        ]

    for n, d in zip(names, descriptions):
        pos.append("  %-10s\t%s\n" % (n, d))

    u += ''.join(pos)
    u += '\n' + numbered_args_help
    u += '\n' + other_args_help
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
        print('kima', open(version_file).read().strip())  # same as kima
        sys.exit(0)

    if '--help-remove-crossing' in args:
        print(help_remove_crossing)
        sys.exit(0)
    if '--help-remove-roche' in args:
        print(help_remove_roche)
        sys.exit(0)

    # save all plots?
    save_plots = findpop('--save-plots', args)

    # other options
    remove_roche = findpop('--remove-roche', args)
    remove_crossing = findpop('--remove-crossing', args)

    number_options = ['1', '2', '3', '4', '5', '6', '7']

    argstuple = namedtuple('Arguments',
                                ['rv', 'planets', 'orbital', 'gp', 'extra'] \
                                + ['diagnostic'] + ['pickle']
                                + ['plot_number'] + ['save_plots']
                                + ['remove_roche', 'remove_crossing'])

    pick = findpop('pickle', args)
    diag = findpop('diagnostic', args)

    if 'all' in args:
        return argstuple(rv=True, planets=True, orbital=True, gp=True,
                         extra=True, diagnostic=diag, pickle=pick,
                         plot_number=[], save_plots=save_plots,
                         remove_roche=remove_roche,
                         remove_crossing=remove_crossing)

    rv = findpop('rv', args)
    gp = findpop('gp', args)
    extra = findpop('extra', args)
    planets = findpop('planets', args)
    orbital = findpop('orbital', args)
    plots = list(set(args).intersection(number_options))
    for plot in plots:
        findpop(plot, args)

    if len(args) >= 1:
        print(usage(full=False), end='')
        print('error: could not recognize argument:', "'%s'" % args[0])
        sys.exit(1)

    return argstuple(rv, planets, orbital, gp, extra, diag, pick,
                     plot_number=plots, save_plots=save_plots,
                     remove_roche=remove_roche,
                     remove_crossing=remove_crossing)


def showresults(options='', force_return=False):
    """
    Generate and plot results from a kima run. 
    The argument `options` should be a string with the same options as for 
    the kima-showresults script.
    """

    # force correct CLI arguments
    args = _parse_args(options)
    # print(args)

    plots = []
    if args.rv:
        plots.append('6')
    if args.planets:
        plots.append('1')
    if args.orbital:
        plots.append('2')
        plots.append('3')
    if args.gp:
        plots.append('4')
        plots.append('5')
    if args.extra:
        plots.append('7')
    for number in args.plot_number:
        plots.append(number)

    try:
        evidence, H, logx_samples = postprocess(plot=args.diagnostic)
    except IOError as e:
        print(e)
        sys.exit(1)

    show_tips()

    res = KimaResults('')
    
    if args.remove_crossing:
        res = rem_crossing_orbits(res)
    
    res.make_plots(list(set(plots)), save_plots=args.save_plots)

    res.evidence = evidence
    res.information = H
    res.ESS = res.posterior_sample.shape[0]



    if args.pickle:
        getinput = input
        # if Python 2, use raw_input()
        if sys.version_info[:2] <= (2, 7):
            getinput = raw_input

        res.save(getinput('Filename to save pickle model: '))

    if not args.save_plots:
        show()  # render the plots

    # __main__.__file__ doesn't exist in the interactive interpreter
    if not hasattr(__main__, '__file__') or force_return:
        return res


if __name__ == '__main__':
    options = sys.argv
    res = showresults(options)
