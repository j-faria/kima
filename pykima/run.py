# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import argparse
import time

kimabold = "\033[1mkima\033[0m"


def _parse_args():
    desc = """(compile and) Run kima jobs"""
    
    parser = argparse.ArgumentParser(description=desc, prog='kima-run')

    parser.add_argument('-t', '--threads', type=int, default=4,
        help='number of threads to use for the job (default 4)')
    parser.add_argument('-s', '--seed', type=int,
        help='random number seed (default uses system time)')
    parser.add_argument('-b', '--background', action='store_true',
        help='run in the background, capturing the output')
    parser.add_argument('-o', '--output', type=str, default='kima.out',
        help='file where to write the output (default "kima.out")')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='no output to terminal')
    parser.add_argument('--timeout', type=int,
        help='stop the job after TIMEOUT seconds')
    parser.add_argument('-c', '--compile', action='store_true', default=False,
        help="just compile, don't run")
    parser.add_argument('--vc', action='store_true', default=False,
        help="verbose compilation")

    args = parser.parse_args()
    return args


# count lines in a file, fast! https://stackoverflow.com/a/27518377
def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)

def rawgencount(filename):
    with open(filename, 'rb') as f:
        f_gen = _make_gen(f.raw.read)
        return sum(buf.count(b'\n') for buf in f_gen)


def main():
    args = _parse_args()
    # print(args)

    if not os.path.exists('kima_setup.cpp'):
        print(
            'Could not find "kima_setup.cpp", are you in the right directory?')
        sys.exit(1)

    ## compile
    try:
        if not args.quiet:
            print('compiling...', end=' ', flush=True)
        make = subprocess.check_output('make')
        if not args.quiet:
            if args.vc:
                print()
                print(make.decode().strip())
            print('done!')

    except subprocess.CalledProcessError as e:
        print("{}: {}".format(type(e).__name__, e))
        sys.exit(1)
    
    if args.compile: # only compile?
        sys.exit(0)

    ## run 
    cmd = './kima -t %d' % args.threads
    if args.seed:
        cmd += ' -s %d' % args.seed

    if args.quiet:
        args.background = True

    if args.background:
        stdout = open(args.output, 'wb')
    else:
        stdout = sys.stdout

    TO = args.timeout

    if not args.quiet:
        print('starting', kimabold)

    start = time.time()
    try:
        kima = subprocess.check_call(cmd.split(), stdout=stdout, timeout=TO)

    except KeyboardInterrupt:
        end = time.time()
        took = end - start
        if not args.quiet:
            print(' finishing the job, took %.2f seconds' % took, end=' ')
            print('(saved %d samples)' % rawgencount('sample.txt'))

    except subprocess.TimeoutExpired:
        end = time.time()
        took = end - start
        if not args.quiet:
            print(kimabold, 'job timed out after %.1f seconds' % took, end=' ')
            print('(saved %d samples)' % rawgencount('sample.txt'))

    else:
        end = time.time()
        if not args.quiet:
            print(kimabold, 'job finished, took %.2f seconds' % (end - start))
    finally:
        if args.background:
            stdout.close()
            if not args.quiet:
                print('output saved to "%s"' % stdout.name)
