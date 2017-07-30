import sys
import os
import os.path as path
import re
import subprocess
import time
import datetime
import shutil
import fileinput

now = datetime.datetime.now
separator = '-'*5

## define some paths
path_to_this_file = path.abspath(__file__)
old_cwd = path.dirname(path_to_this_file)
print 'This file is "%s" and is in %s' % (path.basename(path_to_this_file), old_cwd)
top_level = path.dirname(path.dirname(path_to_this_file))
print 'Then, top-level directory is %s' % top_level


def dir_exists_or_create(directory):
    if path.exists(directory):
        print 'path %s exists' % directory
    else:
        print 'creating directory %s' % directory
        try:
            os.mkdir(directory)
        except OSError:
            os.makedirs(directory)


def change_main(data_file):
    main_template = open(path.join(top_level, 'src', 'main.cpp.template')).read()
    main = main_template % data_file

    with open(path.join(top_level, 'src', 'main.cpp'), 'w') as f:
        print >>f, main


def change_model(np):
    model_template = open(path.join(top_level, 'src', 'MyModel.cpp.template')).read()
    model = model_template % np
    print 'Writing model with Np=%d' % np

    with open(path.join(top_level, 'src', 'MyModel.cpp'), 'w') as f:
        print >>f, model


def compile():
    os.system('make clean')
    os.system('make')


def setup_analysis(results_path):
    # check data file in main.cpp
    with open(path.join(top_level, 'src', 'main.cpp')) as f:
        for line in f:
            if 'loadnew' in line:
                data_file = re.findall('"(.*?)"', line, re.DOTALL)[0]

    print 'Data file in "main.cpp" seems to be: %s' % data_file

    # new dirs
    date_dir = '{:%Y-%m-%d}'.format(now())
    time_dir = '{:%H%M}'.format(now())
    results_now_dir = path.join(results_path, date_dir, time_dir)
    print 'Creating directory %s to hold ins/outs.' % results_now_dir

    dir_exists_or_create(results_now_dir)


    print 'Copying OPTIONS file there.'
    options_file = path.join(results_now_dir, 'OPTIONS')
    shutil.copy(path.join(top_level, 'OPTIONS'), options_file)


    print 'Creating file data_file.txt with data file.'
    with open(path.join(results_now_dir, 'data_file.txt'), 'w') as f:
        print >>f, data_file


    print 'Setting the right paths in the OPTIONS file.'
    for line in fileinput.input(options_file, inplace=True, backup='.bak'):
        if 'samples file' in line:
            sample_file = path.join(results_now_dir, 'sample.txt')
            print sample_file + ' # (optional) samples file'
        elif 'sample_info file' in line:
            sample_info_file = path.join(results_now_dir, 'sample_info.txt')
            print sample_info_file + ' # (optional) sample_info file'
        elif 'levels file' in line:
            levels_file = path.join(results_now_dir, 'levels.txt')
            print levels_file + ' # (optional) levels file'

        else:
            print line.strip()

    return results_now_dir, options_file


if __name__ == '__main__':
    # sys.exit(0)
    # provide system number as first argument
    try:
        system = sys.argv[1]
    except IndexError:
        print 'Provide system number, same as in data file'
        sys.exit(1)

    results_dir = path.join(top_level, 'results', 'PlSy%s' % system)
    dir_exists_or_create(results_dir)

    data_file = 'rvs_%s.txt' % system
    print 'Using data file:', data_file
    r, opt = setup_analysis(results_dir)

    # for 0,1,2,3 planets
    for np in range(4):
        change_main(data_file)
        change_model(np)
        compile()

        nthreads = 4
        print 'Starting "main" with %d threads...' % nthreads
        os.chdir(top_level)
        cmd = ['./main', '-t', str(nthreads), '-o', opt]
        print ' '.join(cmd)
        print
        start = time.time()
        subprocess.check_call(cmd)
        end = time.time()
        print 'Took %f seconds' % (end-start)

        with open(path.join(r, 'times.txt'), 'a') as f:
            f.write('%d, %f\n' % (np, end-start))


        for f in ('sample.txt', 'sample_info.txt', 'levels.txt'):
            src, dst = path.join(r, f), \
                path.join(r, f[:-4]+'_%dplanets.txt' % np)
            shutil.move(src, dst)








