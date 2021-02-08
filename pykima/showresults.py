""" This module defines the `kima-showresults` script """

# the reasoning to have both named and numbered arguments (that do mostly the
# same thing) is that the named arguments are easier to remember and more
# intuitive, but the numbered arguments are faster to type. Choosing to
# deprecate either option has disadvantages.

from __future__ import print_function

import __main__
from .classic import postprocess
from .results import KimaResults
from .crossing_orbits import rem_crossing_orbits
from .utils import show_tips
import sys, os, re
import argparse
from io import StringIO
from contextlib import redirect_stdout
import time
from collections import namedtuple
from matplotlib.pyplot import show

interactive_or_script = not hasattr(__main__, 'load_entry_point')

numbered_args_help = """optional numbered arguments:
  1    - plot the posterior for Np;
  2    - plot the posterior for the orbital periods;
  3    - plot the joint posterior for semi-amplitudes, eccentricities and orbital periods;
  4, 5 - plot the posteriors for the GP hyperparameters (marginal and joint);
  6    - plot samples from posterior in data-space together with the RV data;
  6p   - same as 6 plus the phase curves of the maximum likelihood solution;
  7    - plot the posteriors for other parameters (systemic velocity, extra sigma, etc);
  8    - plot the posteriors for the moving average parameters;
"""

other_args_help = """other optional arguments:
  --save-plots	      Save the plots as .png files (except diagnostic plots)
  --remove-roche      Remove orbits crossing the Roche limit of the star
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
No extra documentation yet...
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


def usage(full=True, add_usage=True):
    if add_usage:
        u = 'usage: '
    else:
        u = ''

    u += "kima-showresults "\
         "[rv] [phase] [planets] [orbital] [gp] [extra] [1, ..., 8]\n"\
         "                        [all] [pickle] [zip] [--save-plots] "\
                                 "[-h/--help] [--version]"
    u += '\n\n'
    if not full: return u

    pos = ["positional arguments:\n"]
    names = [
        'rv', 'phase', 'planets', 'orbital', 'gp', 'extra', 'all', 'pickle',
        'zip', '--save-plots'
    ]
    descriptions = \
        ["posterior realizations of the model over the RV measurements",
         "same as above plus phase curves of the maximum likelihood solution",
         "posterior for number of planets",
         "posteriors for some of the orbital parameters",
         "posteriors for GP hyperparameters",
         "posteriors for systemic velocity, extra white noise, etc",
         "show all plots",
         "save the model into a pickle file (filename will be prompted)",
         "save the model and files into a zip file (filename will be prompted)",
         "instead of showing, save the plots as .png files (does not work for diagnostic plots)",
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
        v = open(version_file).read().strip()  # same as kima
        print('kima (kima-showresults script)', v)
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

    number_options = ['1', '2', '3', '4', '5', '6', '6p', '7', '8']
    argstuple = namedtuple('Arguments', [
        'rv',
        'phase',
        'planets',
        'orbital',
        'gp',
        'extra',
        'diagnostic',
        'pickle',
        'zip',
        'plot_number',
        'save_plots',
        'remove_roche',
        'remove_crossing',
    ])

    pick = findpop('pickle', args)
    zipit = findpop('zip', args)
    diag = findpop('diagnostic', args)

    if 'all' in args:
        return argstuple(rv=False, phase=True, planets=True, orbital=True,
                         gp=True, extra=True, diagnostic=diag, pickle=pick,
                         zip=zipit, plot_number=[], save_plots=save_plots,
                         remove_roche=remove_roche,
                         remove_crossing=remove_crossing)

    rv = findpop('rv', args)
    phase = findpop('phase', args)
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

    return argstuple(rv, phase, planets, orbital, gp, extra, diag, pick, zipit,
                     plot_number=plots, save_plots=save_plots,
                     remove_roche=remove_roche,
                     remove_crossing=remove_crossing)


def showresults(options='', force_return=False, verbose=True, show_plots=True,
                kima_tips=True, numResampleLogX=1, moreSamples=1):
    """
    Generate and plot results from a kima run. The argument `options` should be 
    a string with the same options as for the kima-showresults script.
    """

    # force correct CLI arguments
    args = _parse_args(options)
    # print(args)

    plots = []
    if args.rv:
        plots.append('6')
    if args.phase:
        plots.append('6p')
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

    # don't repeat plot 6
    if '6p' in plots:
        try:
            plots.remove('6')
        except ValueError:
            pass

    hidden = StringIO()
    stdout = sys.stdout if verbose else hidden

    try:
        with redirect_stdout(stdout):
            evidence, H, logx_samples = postprocess(
                plot=args.diagnostic, numResampleLogX=1, moreSamples=1)
    except IOError as e:
        if interactive_or_script:
            raise e from None
        else:
            print(str(e))
            return

    # sometimes an IndexError is raised when the levels.txt file is being
    # updated too quickly, and the read operation is not atomic... we try one
    # more time and then give up
    except IndexError:
        try:
            with redirect_stdout(stdout):
                evidence, H, logx_samples = postprocess(
                    plot=args.diagnostic, numResampleLogX=1, moreSamples=1)
        except IndexError:
            msg = 'Something went wrong reading "levels.txt". Try again.'
            if interactive_or_script:
                raise IOError(msg) from None
            else:
                print(msg)
                return

    # show kima tips
    if verbose and kima_tips:
        show_tips()

    res = KimaResults('')

    if args.remove_crossing:
        res = rem_crossing_orbits(res)

    res.make_plots(list(set(plots)), save_plots=args.save_plots)

    res.evidence = evidence
    res.information = H
    res.ESS = res.posterior_sample.shape[0]

    # getinput = input
    # # if Python 2, use raw_input()
    # if sys.version_info[:2] <= (2, 7):
    #     getinput = raw_input

    if args.pickle:
        res.save_pickle(input('Filename to save pickle model: '))
    if args.zip:
        res.save_zip(input('Filename to save model (must end with .zip): '))

    if not args.save_plots and show_plots:
        show()  # render the plots

    #! old solution, but fails when running scripts (res ends up as None)
    # __main__.__file__ doesn't exist in the interactive interpreter
    # if not hasattr(__main__, '__file__') or force_return:
    #! this solution seems to always return, except with kima-showresults
    if not hasattr(__main__, 'load_entry_point') or force_return:
        return res


def make_wide(formatter, w=120, h=36):
    """Return a wider HelpFormatter, if possible."""
    try:
        # https://stackoverflow.com/a/5464440
        # beware: "Only the name of this class is considered a public API."
        kwargs = {'width': w, 'max_help_position': h}
        formatter(None, **kwargs)
        return lambda prog: formatter(prog, **kwargs)
    except TypeError:
        warnings.warn("argparse help formatter failed, falling back.")
        return formatter

class NoAction(argparse.Action):
    def __init__(self, **kwargs):
        kwargs.setdefault('default', argparse.SUPPRESS)
        kwargs.setdefault('nargs', 0)
        super(NoAction, self).__init__(**kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        pass

class ChoicesAction(argparse._StoreAction):
    def add_choice(self, choice, help=''):
        if self.choices is None:
            self.choices = []
        self.choices.append(choice)
        self.container.add_argument(choice, help=help, action='none')

    def add_undocumented(self, choice, help=''):
        if self.choices is None:
            self.choices = []
        self.choices.append(choice)
        # self.container.add_argument(choice, help=help, action='none')

def _parse_args2(options):
    if options == '':
        if 'kima-showresults' in sys.argv[0]:
            args_in = sys.argv[1:]
        else:
            args_in = options
    else:
        args_in = options.split()

    # create the top-level parser
    parser = argparse.ArgumentParser(
        'kima-showresults', description='Show the results from a kima run.',
        usage=usage(full=False, add_usage=False),
        # formatter_class=make_wide(argparse.HelpFormatter, w=140, h=20)
        formatter_class=make_wide(argparse.RawTextHelpFormatter, w=140, h=25)
    )
    parser.add_argument('--version', action='store_true',
                        help="show the version of kima and exit")

    parser.register('action', 'none', NoAction)
    parser.register('action', 'store_choice', ChoicesAction)

    group_opt = parser.add_argument_group(title='positional arguments')

    options = group_opt.add_argument('commands', metavar='', help='', nargs=argparse.REMAINDER,
                                     action='store_choice')

    # options.add_undocumented('rv', help="")
    options.add_choice('diagnostic', help="show diagnostics from DNest4")

    options.add_choice('rv', help="samples from posterior in data-space together with the RV data")
    options.add_choice('phase', help="phase curve(s) of the maximum likelihood solution")
    options.add_choice('planets', help="posterior for number of planets")
    options.add_choice('orbital', help="posteriors for some of the orbital parameters")
    options.add_choice('gp', help="posteriors for GP hyperparameters")
    options.add_choice('extra', help="posteriors for systemic velocity, extra white noise, etc")
    options.add_choice('all', help="show all plots\n\n")
    options.add_choice('pickle', help="save the model into a pickle file (filename will be prompted)")
    options.add_choice('zip', help="save output files into a zip file (filename will be prompted)\n\n")

    options.add_choice('1', help="posterior for number of planets")
    options.add_choice('2', help="posterior for the orbital periods")
    options.add_choice('3', help="joint posterior for semi-amplitudes, eccentricities and orbital periods")
    options.add_choice('4', help="posteriors for the GP hyperparameters (marginal)")
    options.add_choice('5', help="posteriors for the GP hyperparameters (joint)")
    options.add_choice('6', help="samples from posterior in data-space together with the RV data")
    options.add_choice('6p', help="same as 6, plus the phase curves of the maximum likelihood solution")
    options.add_choice('7', help="posteriors for systemic velocity, extra white noise, etc)")
    options.add_choice('8', help="posteriors for the moving average parameters")

    parser.add_argument('--save-plots', action='store_true',
        help="instead of showing, save the plots as .png files (except for diagnostic plots)")
    parser.add_argument('--remove-roche', action='store_true',
        help="remove orbits crossing the Roche limit of the star")
    parser.add_argument('--remove-crossing', action='store_true',
        help="remove crossing orbits (use --help-remove-crossing for details)")
    parser.add_argument('--help-remove-crossing', action='store_true',
                        help=argparse.SUPPRESS)

    args = parser.parse_args(args_in)

    unrecognized = []
    wrong_place = []
    for cmd in args.commands:
        # hack!
        # using argparse.REMAINDER above means the optional arguments cannot
        # come after the other arguments but this is not ideal
        # set them to True if they were given
        if cmd in ('--save-plots', '--remove-roche', '--remove-crossing'):
            args.__setattr__(cmd.replace('--', '').replace('-', '_'), True)
            wrong_place.append(cmd)
        # is there any argument not in the allowed choices?
        elif cmd not in options.choices:
            unrecognized.append(cmd)

    if len(unrecognized) > 0:
        print(f'kima-showresults: warning: unrecognized arguments: '\
              f'{", ".join(unrecognized)}')

    for cmd in unrecognized:
        args.commands.remove(cmd)
    for cmd in wrong_place:
        args.commands.remove(cmd)

    return args

def showresults2(options='', force_return=False, verbose=True, show_plots=True,
                 kima_tips=True, numResampleLogX=1):
    """
    Generate and plot results from a kima run. The argument `options` should be 
    a string with the same options as for the kima-showresults script.
    """

    # force correct CLI arguments
    args = _parse_args2(options)
    # print(args)

    if args.help_remove_crossing:
        print(help_remove_crossing)
        return

    if args.version:
        version_file = os.path.join(os.path.dirname(__file__), '../VERSION')
        v = open(version_file).read().strip()  # same as kima
        print('kima (kima-showresults script)', v)
        return

    args.diagnostic = 'diagnostic' in args.commands

    plots = []
    if 'rv' in args.commands:
        plots.append('6')
    if 'phase' in args.commands:
        plots.append('6p')
    if 'planets' in args.commands:
        plots.append('1')
    if 'orbital' in args.commands:
        plots.append('2')
        plots.append('3')
    if 'gp' in args.commands:
        plots.append('4')
        plots.append('5')
    if 'extra' in args.commands:
        plots.append('7')

    for cmd in args.commands:
        try:
            int(cmd)
            plots.append(cmd)
        except ValueError:
            pass

    if '6p' in args.commands:
        plots.append('6p')

    # don't repeat plot 6
    if '6p' in plots:
        try:
            plots.remove('6')
        except ValueError:
            pass
    
    if 'all' in args.commands:
        plots = '1 2 3 4 5 6 7 8'.split()

    hidden = StringIO()
    stdout = sys.stdout if verbose else hidden

    try:
        with redirect_stdout(stdout):
            evidence, H, logx_samples = postprocess(plot=args.diagnostic,
                                                    numResampleLogX=1,
                                                    moreSamples=1)
    except IOError as e:
        if interactive_or_script:
            raise e from None
        else:
            print(str(e))
            return

    # sometimes an IndexError is raised when the levels.txt file is being
    # updated too quickly, and the read operation is not atomic... we try one
    # more time and then give up
    except IndexError:
        try:
            with redirect_stdout(stdout):
                evidence, H, logx_samples = postprocess(plot=diagnostic,
                                                        numResampleLogX=1,
                                                        moreSamples=1)
        except IndexError:
            msg = 'Something went wrong reading "levels.txt". Try again.'
            if interactive_or_script:
                raise IOError(msg) from None
            else:
                print(msg)
                return

    # show kima tips
    if verbose and kima_tips:
        show_tips()

    res = KimaResults('')

    if args.remove_crossing:
        res = rem_crossing_orbits(res)

    res.make_plots(list(set(plots)), save_plots=args.save_plots)

    res.evidence = evidence
    res.information = H
    res.ESS = res.posterior_sample.shape[0]

    # getinput = input
    # # if Python 2, use raw_input()
    # if sys.version_info[:2] <= (2, 7):
    #     getinput = raw_input

    if 'pickle' in args.commands:
        res.save_pickle(input('Filename to save pickle model: '))
    if 'zip' in args.commands:
        res.save_zip(input('Filename to save model (must end with .zip): '))

    if not args.save_plots and show_plots:
        show()  # render the plots

    #! old solution, but fails when running scripts (res ends up as None)
    # __main__.__file__ doesn't exist in the interactive interpreter
    # if not hasattr(__main__, '__file__') or force_return:
    #! this solution seems to always return, except with kima-showresults
    if not hasattr(__main__, 'load_entry_point') or force_return:
        return res


if __name__ == '__main__':
    options = sys.argv
    res = showresults(options)
