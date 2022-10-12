# -*- coding: utf-8 -*-
import sys
import os
import argparse
import subprocess
import pipes
from collections import namedtuple
from distutils.dir_util import copy_tree


def usage():
    u = "kima-template [DIRECTORY]\n"
    u += "Create a kima template in DIRECTORY (the current directory by default).\n"
    return u


def _parse_args():
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument('DIRECTORY', nargs='?', default=os.getcwd())
    parser.add_argument('--dace', type=str, metavar='STAR',
                        help='download .rdb files for STAR from DACE')
    parser.add_argument('--version', action='store_true',
                        help='show version and exit')
    parser.add_argument('--debug', action='store_true',
                        help='be more verbose, for debugging')
    return parser.parse_args()


def exists_remote(host, path, verbose=False):
    if verbose:
        print('[mkdir_remote] checking if "%s" exists in %s' % (path, host))
    return subprocess.call(['ssh', host, 'test -e ' + pipes.quote(path)]) == 0


def mkdir_remote(host, path, verbose=False):
    if verbose:
        print('[mkdir_remote] creating directory "%s" in %s' % (path, host))
    subprocess.check_call(['ssh', host, 'mkdir ' + pipes.quote(path)])


def write_makefile(directory):
    from os.path import dirname, join
    import sysconfig
    from kima.paths import kima_dir
    from textwrap import dedent

    kima_dir = dirname(kima_dir)

    lib = sysconfig.get_config_vars('EXT_SUFFIX')[0].replace('.so', '.a')
    kima_lib = f'libkima{lib}'
    # dnest4_lib = f'libdnest4{lib}'
    # print(dnest4_lib, kima_lib)

    make = f"""
    KIMA_DIR = {kima_dir}

    DNEST4_PATH = $(KIMA_DIR)/kima/vendor/DNest4/code
    EIGEN_PATH = $(KIMA_DIR)/kima/vendor/eigen
    LOADTXT_PATH = $(KIMA_DIR)/kima/vendor/cpp-loadtxt/src
    INCLUDES = -I$(KIMA_DIR)/kima -I$(DNEST4_PATH) -I$(EIGEN_PATH) -I$(LOADTXT_PATH)

    LIBS = -L$(DNEST4_PATH) -ldnest4

    CXXFLAGS = -pthread -std=c++17 -O3 -DNDEBUG -DEIGEN_MPL2_ONLY

    kima: kima_setup.cpp
    \t@$(CXX) -o $@ $< $(CXXFLAGS) $(KIMA_DIR)/kima/{kima_lib} $(INCLUDES) $(LIBS)

    clean:
    \trm -f kima kima_setup.o
    """
    with open(join(directory, 'Makefile'), 'w') as f:
        print(dedent(make), file=f)


def main(args=None, stopIfNoReplace=True):
    if args is None:
        args = _parse_args()
    if isinstance(args, str):
        Args = namedtuple('Args', 'version debug DIRECTORY')
        args = Args(version=False, debug=False, DIRECTORY=args)

    print(args)

    if args.version:
        version_file = os.path.join(os.path.dirname(__file__), '../VERSION')
        print('kima', open(version_file).read().strip())  # same as kima
        sys.exit(0)

    dst = args.DIRECTORY

    # kimadir = '{kimadir}'  # filled by setup.py

    thisdir = os.path.dirname(os.path.realpath(__file__))
    src = os.path.join(thisdir, 'template')
    templatefiles = ['kima_setup.cpp', 'OPTIONS']

    # by default, replace directory
    replace = True

    if ':' in dst and '@' in dst:  # directory is in the server
        server = True
        host, directory = dst.split(':')
        # user, host = server.split('@')
        dst = directory
        print('Populating directory %s (on %s)' % (dst, host))
        # print('with kima template files: '
        #       ', '.join([os.path.basename(f) for f in templatefiles]))

        answer = input('--> Continue? (y/N) ')
        if answer.lower() != 'y':
            print('doing nothing!')
            return
    else:
        server = False

        if os.path.exists(dst):
            # is there a kima_setup.cpp inside the directory?
            if os.path.exists(os.path.join(dst, 'kima_setup.cpp')):
                msg = 'Directory "%s" exists and contains a kima_setup file. ' % dst
                msg += 'Replace? (Y/n) '
                print(msg, end='')
                answer = input().lower()
                if answer == 'n':
                    replace = False
                    if stopIfNoReplace:
                        print('Doing nothing')
                        sys.exit(0)
                else:
                    replace = True

            # cwd obviously exists, but make sure not to overwrite other directories
            elif dst != os.getcwd():
                print('Directory "%s" exists. Continue? (Y/n) ' % dst, end='')
                answer = input().lower()
                if answer == 'n':
                    print('Doing nothing')
                    sys.exit(0)
                replace = True

        if replace:
            print(f'Populating directory "{dst}"')
            print('with kima template files:', end=' ')
            print(', '.join([os.path.basename(f) for f in templatefiles]))

    if server:
        pass
    #     if not exists_remote(host, dst, verbose=args.debug):
    #         mkdir_remote(host, dst, verbose=args.debug)
    #     for f in templatefiles:
    #         # -q -o LogLevel=QUIET
    #         cmd = 'scp %s %s' % (f, ':'.join([host, dst]))
    #         # print(cmd)
    #         subprocess.check_call(cmd.split())
    else:
        if replace:
            copy_tree(src, dst)
        write_makefile(dst)

    # if server:
    #     with open(os.path.join(src, 'Makefile')) as f:
    #         m = f.read().format(kimadir='/home/jfaria/disk1/kima')
    #     with open('tempMakefile1234', 'w') as f:
    #         f.write(m)
    #     cmd = 'scp -q -o LogLevel=QUIET %s %s' % ('tempMakefile1234', host +
    #                                               ':' + dst + '/Makefile')
    #     # print(cmd)
    #     subprocess.check_call(cmd.split())
    #     os.remove('tempMakefile1234')
    # else:
    #     with open(os.path.join(src, 'Makefile')) as f:
    #         m = f.read().format(kimadir=kimadir)
    #     with open(os.path.join(dst, 'Makefile'), 'w') as f:
    #         f.write(m)

    # if args.dace is not None:
    #     from vera import DACE
    #     from .utils import chdir
    #     with chdir(dst):
    #         s = getattr(DACE, args.dace)
    #         print(s)

    # return


if __name__ == '__main__':
    main()
