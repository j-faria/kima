# import sys
import os
import os.path as path
import re
import subprocess
# import time
import datetime
import shutil
import fileinput

now = datetime.datetime.now
separator = '-'*5


def dir_exists_or_create(directory):
    if path.exists(directory):
        print 'path %s exists' % directory
    else:
        print 'creating directory %s' % directory
        try:
            os.mkdir(directory)
        except OSError:
            os.makedirs(directory)


print separator, 'Today is',
print '{:%Y-%b-%d %H:%M:%S}'.format(now()),
print separator


## where are we?
hostname = subprocess.check_output('hostname').strip()
if hostname == 'joao-caup':
    print 'We are running in the desktop, hostname=%s' % hostname
    cluster = False
elif hostname == 'cnode0':
    print 'We are running in the cluster! hostname=%s' % hostname
    cluster = True

## define some paths
path_to_this_file = path.abspath(__file__)
old_cwd = path.dirname(path_to_this_file)
print 'This file is "%s" and is in %s' % (path.basename(path_to_this_file), old_cwd)
top_level = path.dirname(path.dirname(path_to_this_file))
print 'Then, top-level directory is %s' % top_level

## check data file in main.cpp
with open(path.join(top_level, 'src', 'main.cpp')) as f:
    for line in f:
        if 'loadnew' in line:
            data_file = re.findall('"(.*?)"', line, re.DOTALL)[0]

print 'Data file in "main.cpp" seems to be: %s' % data_file


system_number = int(re.findall('\d+', data_file)[0])
results_path = path.join('results', 'PlSy%d' % system_number)
print 'This is system number %d, so results will go to %s' % (system_number, results_path)


results_path = path.join(top_level, results_path)
dir_exists_or_create(results_path)


current_branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD".split()).strip()
current_commit = subprocess.check_output("git rev-parse --short HEAD".split()).strip()

print 'We are in branch %s, commit %s' % (current_branch, current_commit)
print


print "Let's do some work!"

date_dir = '{:%Y-%m-%d}'.format(now())
time_dir = '{:%H%M}'.format(now())
results_now_dir = path.join(results_path, date_dir, time_dir)
print 'Creating directory %s to hold ins/outs.' % results_now_dir

dir_exists_or_create(results_now_dir)

print 'Copying OPTIONS file there.'
options_file = path.join(results_now_dir, 'OPTIONS')
shutil.copy(path.join(top_level, 'OPTIONS'), options_file)


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


if cluster:
    pass
else:
    nthreads = 1
    print 'Starting "main" with %d threads...' % nthreads

    os.chdir(top_level)
    cmd = ['./main', '-t', str(nthreads), '-o', options_file]
    print ' '.join(cmd)
    print 
    subprocess.check_call(cmd)
    os.chdir(old_cwd)