# -*- coding: utf-8 -*-
import sys
import os
import argparse
import glob
from distutils.dir_util import copy_tree

def usage(full=True):
    u = "usage: kima-template [DIRECTORY]\n"
    u += "Create a kima template in DIRECTORY (the current directory by default).\n"
    return u

def _parse_args():
    args = sys.argv[1:]
    if '-h' in args or '--help' in args:
        print(usage())
        sys.exit(0)
    if '--version' in args:
        version_file = os.path.join(os.path.dirname(__file__), '../VERSION')
        print('kima', open(version_file).read().strip()) # same as kima
        sys.exit(0)

    if len(args) == 1:
        return args[0]
    else:
        return

def main():
    args = _parse_args()

    kimadir = os.path.dirname(os.path.dirname(__file__))
    src = os.path.join(kimadir, 'template')
    
    if args is None:
        dst = os.getcwd()
    else:
        dst = args
    
    templatefiles = glob.glob(os.path.join(src, '*'))
    print('Populating directory %s' % dst)
    print('with kima template files: ' + \
          ', '.join([os.path.basename(f) for f in templatefiles]))
    copy_tree(src, dst)

    with open(os.path.join(dst, 'Makefile')) as f:
        m = f.read().format(kimadir=kimadir)
    with open(os.path.join(dst, 'Makefile'), 'w') as f:
        f.write(m)



if __name__ == '__main__':
    main()
