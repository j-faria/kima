# -*- coding: utf-8 -*-
import sys
import os
import shutil
import fileinput
import subprocess
import argparse
import time
from datetime import datetime
import contextlib
import asyncio
import signal
import psutil
import shlex

from .version import kima_version
from .showresults import calculate_ESS


kimabold = "\033[1mkima\033[0m"
kimanormal = "kima"


# This is a custom signal handler for SIGTERM
def receiveSIGTERM(signalNumber, frame):
    print('kima job terminated, exiting gracefully.', flush=True)
    sys.exit(15)


signal.signal(signal.SIGTERM, receiveSIGTERM)


def _parse_args1(argstring=None):
    desc = """(compile and) Run kima jobs"""

    parser = argparse.ArgumentParser(description=desc, prog='kima-run')

    parser.add_argument('DIR', nargs='?', default=os.getcwd(),
                        help='change to this directory before running')

    parser.add_argument(
        '-t', '--threads', type=int, default=4,
        help='number of threads to use for the job (default 4)')

    parser.add_argument('-s', '--seed', type=int,
                        help='random number seed (default uses system time)')

    parser.add_argument('-b', '--background', action='store_true',
                        help='run in the background, capturing the output')

    parser.add_argument(
        '-o', '--output', type=str, default='kima.out', metavar='OUT',
        help='file where to write the output (default "kima.out")')

    parser.add_argument('-q', '--quiet', action='store_true',
                        help='no output to terminal')

    # parser.add_argument('-id', type=str, default='',
    #                     help='job ID, added to sample.txt, levels.txt, etc')

    parser.add_argument('--data-file', type=str, default='', metavar='FILE',
                        help="supply data file as first argument to kima")

    parser.add_argument('--save', nargs='?', type=str, const='',
                        help='Save output of run to a directory')

    parser.add_argument('--timeout', type=int,
                        help='stop the job after TIMEOUT seconds')

    parser.add_argument('--ess', type=int, help='effective sample size goal')

    parser.add_argument('-c', '--compile', action='store_true', default=False,
                        help="just compile, don't run")

    parser.add_argument('--force-compile', action='store_true', default=False,
                        help="force (re)compilation")

    parser.add_argument('--no-compile', action='store_true', default=False,
                        help="don't compile, just run")

    parser.add_argument('--vc', action='store_true', default=False,
                        help="verbose compilation")

    parser.add_argument('--no-notify', action='store_true', default=True,
                        help="do not send notification when job finished")
    parser.add_argument('--no-colors', action='store_true', default=False,
                        help=argparse.SUPPRESS)

    parser.add_argument('-d', '--debug', action='store_true',
                        help='run with valgrind')
    parser.add_argument('--version', action='store_true', help='show version')

    if argstring is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shlex.split(argstring))

    return args, parser


def _parse_args2():
    desc = """Kill running kima jobs"""
    parser = argparse.ArgumentParser(description=desc, prog='kima-kill')
    args = parser.parse_args()
    return args


@contextlib.contextmanager
def remember_cwd():
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)


# count lines in a file, fast! https://stackoverflow.com/a/27518377
def rawgencount(filename, sub=0):
    def _make_gen(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)

    with open(filename, 'rb') as f:
        f_gen = _make_gen(f.raw.read)
        return sum(buf.count(b'\n') for buf in f_gen) - sub


# check if we can send notifications
def can_send_notifications():
    import platform
    this_system = platform.system()
    if this_system == 'Linux':
        return bool(shutil.which('notify-send')), 'linux'
    elif this_system == 'Darwin':
        return bool(shutil.which('osascript')), 'macos'


def notify(summary, body):
    can, platform = can_send_notifications()
    if can:
        # logodir = os.path.join(
        #     # .../kima
        #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        #     'logo'
        # )
        # logo = os.path.join(logodir, 'logo_transparent_small.png')
        logo = 'kima_small_tr'

        if platform == 'linux':
            cmd = ['notify-send']
            cmd += ['-a', 'kima', '-i', f'{logo}', '-t', '3000']
            cmd += ['%s' % summary] + ['%s' % body]
            subprocess.check_call(cmd)

        elif platform == 'macos':
            cmd = ['osascript']
            cmd += ['-e', 'display notification']
            cmd += ['\"%s\"' % body, 'with title', '\"%s\"' % summary]
            print(' '.join(cmd))


def _change_OPTIONS(postfix):
    from numpy import loadtxt
    shutil.copy('OPTIONS', 'OPTIONS.bak')

    original_lines = [line.strip() for line in open('OPTIONS').readlines()]
    lines = [line for line in original_lines if not line.startswith('#')]
    option_values = loadtxt('OPTIONS', dtype=int)

    sample = 'sample_%s.txt' % postfix
    levels = 'levels_%s.txt' % postfix
    sample_info = 'sample_info_%s.txt' % postfix

    with open('OPTIONS', 'w') as f:
        for line in original_lines:
            if line.startswith('#'):
                print(line, file=f)
            else:
                break

        for v, l in zip(option_values, lines):
            print(v, '\t# ' + l.split('#')[1], file=f)

        print(sample, '\t# samples file', file=f)
        print(sample_info, '\t# sample_info file', file=f)
        print(levels, '\t# levels file', file=f)


def run_local(args=None, return_time=False):
    """ Run kima jobs """
    args, parser = _parse_args1(args)

    if args.version:
        # same as kima
        print(f'kima ({parser.prog} script)', kima_version)
        return

    # get back to current directory when finished
    with remember_cwd():

        if args.DIR != os.getcwd():
            print('Changing directory to', args.DIR)

        os.chdir(args.DIR)

        if not os.path.exists('kima_setup.cpp'):
            if os.path.isfile('kima') and os.access('kima', os.X_OK):
                if not args.quiet:
                    print(
                        'Found kima executable, assuming it can be re-compiled'
                    )
            else:
                print(
                    'Could not find "kima_setup.cpp" or a "kima" executable, '
                    'are you in the right directory?'
                )
                sys.exit(1)

        # compile
        if not args.no_compile:

            # if there is no Makefile, create one with the right paths
            if not os.path.exists('Makefile'):
                from .make_template import write_makefile
                write_makefile('.')

            try:
                if not args.quiet:
                    print('compiling...', end=' ', flush=True)

                if args.compile or args.force_compile:  # "re"-compile
                    subprocess.check_call('make clean'.split())

                makecmd = 'make -j %d' % args.threads
                start = time.time()
                make = subprocess.check_output(makecmd.split())
                end = time.time()

                if not args.quiet:
                    if args.vc:
                        print()
                        print(make.decode().strip())
                    print(f'done! (in {end-start:.1f} sec)', flush=True)

            except subprocess.CalledProcessError as e:
                print("{}: {}".format(type(e).__name__, e))
                sys.exit(1)

            if args.compile:  # only compile?
                sys.exit(0)

        # run
        if args.debug:
            cmd = 'valgrind --tool=callgrind '
        else:
            cmd = ''

        cmd += './kima %s -t %d' % (args.data_file, args.threads)

        if args.seed:
            cmd += ' -s %d' % args.seed

        if args.quiet:
            args.background = True

        if args.background:
            stdout = open(args.output, 'wb')
            stdout.write(b'starting kima at ')
            stdout.write(str(datetime.now()).encode())
            stdout.write(b'\n')
            stdout.flush()
        else:
            stdout = sys.stdout

        kimastr = kimanormal if args.no_colors else kimabold

        if not args.quiet:
            print('starting', kimastr, flush=True)

        start = time.time()
        took = 0

        # try:
        #     kima = subprocess.Popen(cmd.split(), stdout=stdout)

        #     with open('kima_running_pid', 'w') as f:
        #         f.write(str(kima.pid))

        #     kima.communicate(timeout=TO)

        try:
            if args.ess:
                asyncio.run(launch_kima_with_ess_watcher(cmd, args, stdout))
            else:
                asyncio.run(launch_kima(cmd, args, stdout,
                                        raise_exceptions=False))

        except asyncio.CancelledError as e:
            pass

        except KeyboardInterrupt:
            end = time.time()
            took = end - start
            msg1 = ' finishing the job, took %.2f seconds' % took
            msg2 = '(saved %d samples)' % rawgencount('sample.txt', sub=1)
            if not args.quiet:
                print(msg1, end=' ')
                print(msg2)
            if args.background:
                stdout.write(msg1[1:].encode())
                stdout.write(msg2.encode())
            if not args.no_notify:
                notify('kima job finished', 'took %.2f seconds' % took)

        finally:
            if args.background:
                stdout.close()
                if not args.quiet:
                    print('output saved to "%s"' % stdout.name)

        if args.save is not None:
            save = args.save
            if save == '':
                save = datetime.now().isoformat().split('.')[0]

            print(f'Saving results to "{save}"')

            if not os.path.isdir(save):
                os.mkdir(save)

            # shutil.copy('OPTIONS', save)
            # shutil.copy('kima_setup.cpp', save)
            shutil.copy('kima_model_setup.txt', save)
            shutil.copy('sample.txt', save)
            shutil.copy('sample_info.txt', save)
            shutil.copy('levels.txt', save)

            # the datafile paths need to be absolute, otherwise
            # kima-showresults will fail in the save directory
            from .utils import read_model_setup
            setup = read_model_setup()
            multi = setup['kima']['multi'] == 'true'

            if multi:
                model_file = os.path.join(save, 'kima_model_setup.txt')
                data_files = setup['kima']['files'].split(',')[:-1]
                for line in fileinput.FileInput(model_file, inplace=True):
                    if line.startswith('files:'):
                        newline = line
                        for data_file in data_files:
                            newline = newline.replace(
                                data_file, os.path.abspath(data_file))
                        print(newline, end='')
                    else:
                        print(line, end='')
            else:
                model_file = os.path.join(save, 'kima_model_setup.txt')
                data_file = setup['kima']['file']
                for line in fileinput.FileInput(model_file, inplace=True):
                    if line.startswith('file:'):
                        line = line.replace(data_file,
                                            os.path.abspath(data_file))
                        print(line, end='')
                    else:
                        print(line, end='')

            if os.path.exists('kima_running_pid'):
                os.remove('kima_running_pid')

    if return_time:
        return took


class Finished(Exception):
    pass


async def launch_kima(cmd, args, stdout=None, raise_exceptions=True):
    kimastr = kimanormal if args.no_colors else kimabold

    kima = await asyncio.create_subprocess_exec(*cmd.split(), stdout=stdout)
    start = time.time()

    try:
        await asyncio.wait_for(kima.communicate(), timeout=args.timeout)

    # the process timed out
    except asyncio.exceptions.TimeoutError as e:
        kima.terminate()
        end = time.time()
        took = end - start

        msg1 = f'job timed out after {took:.1f} seconds'
        msg2 = '(saved %d samples)' % rawgencount('sample.txt', sub=1)
        if not args.quiet:
            time.sleep(0.5)  # allow stdout flush before printing stuff
            print(kimastr, msg1, end=' ')
            print(msg2)
        if args.background:
            stdout.write(('kima ' + msg1 + ' ').encode())
            stdout.write(msg2.encode())

        if not args.no_notify:
            notify('kima job finished',
                   'after timeout of %.2f seconds' % took)

        if raise_exceptions:
            raise e

    else:
        if kima.returncode != 0:
            # the process finished but with an abnormal return code
            print(kimastr, f'terminated with error code {kima.returncode}')
            raise asyncio.CancelledError(kima.returncode) from None

        end = time.time()
        took = end - start

        msg1 = ' job finished, took %.2f seconds' % took
        if not args.quiet:
            print(kimastr + msg1)
        if args.background:
            stdout.write(('kima' + msg1).encode())
        if not args.no_notify:
            notify('kima job finished', 'took %.2f seconds' % took)

        if raise_exceptions:
            raise Finished


async def kima_ess_watcher(goal):
    await asyncio.sleep(2)  # initial wait
    while True:
        ESS = calculate_ESS()
        print(f'# {ESS=} ({goal=})')
        if ESS >= goal:
            raise Finished
        #     print('done!')
        #     raise ValueError('error!')
        # wait = min(60, max(10, int(100 / log(goal - ESS))))
        # print(wait)
        await asyncio.sleep(2)


async def launch_kima_with_ess_watcher(cmd, args, stdout=None):
    tasks = [
        asyncio.ensure_future(launch_kima(cmd, args, stdout)),
        asyncio.ensure_future(kima_ess_watcher(args.ess)),
    ]

    try:
        _ = await asyncio.gather(*tasks)
    except asyncio.CancelledError as e:
        print('one task cancelled')
        raise e
    except (asyncio.exceptions.TimeoutError, Finished):
        for task in tasks:
            task.cancel()


def kill():
    _parse_args2()

    if os.path.exists('kima_running_pid'):
        proc = psutil.Process(int(open('kima_running_pid').read()))
        time_user = proc.cpu_times().user
        time_user /= (proc.num_threads() - 1)
        if time_user > 60:
            if time_user > 3600:
                unit, time_user = 'hours', time_user / 3600
            else:
                unit, time_user = 'minutes', time_user / 60
        else:
            unit = 'seconds'

        time_user = f'{time_user:.1f} {unit}'
        print(f'Process {proc.pid} has been running for {time_user}')
        print(f'and already saved {rawgencount("sample.txt", sub=1)} samples')
        ans = input('Kill it? (Y/n) ')
        if ans.lower() in ('y', 'yes'):
            proc.kill()
    else:
        print('No process seems to be running...')
