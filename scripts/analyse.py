import sys
import os
import os.path as path
import re
import subprocess
# import time
import datetime
import shutil
import fileinput
import numpy
import matplotlib.pyplot as plt 

from dnest4 import postprocess
from display import DisplayResults

## define some paths
path_to_this_file = path.abspath(__file__)
old_cwd = path.dirname(path_to_this_file)
print 'This file is "%s" and is in %s' % (path.basename(path_to_this_file), old_cwd)
top_level = path.dirname(path.dirname(path_to_this_file))
print 'Then, top-level directory is %s' % top_level


def get_most_recent_path(system):
    r = path.join(top_level, 'results', 'PlSy%s' % system)
    date_dir = sorted(os.listdir(r))[-1]
    r = path.join(r, date_dir)
    time_dir = sorted(os.listdir(r))[-1]
    dirpath = path.join(r, time_dir)
    return dirpath
    # dirpath, dirnames, filenames = list(os.walk(r))[-1]
    # return dirpath, dirnames, filenames

def load_file(filename):
    return numpy.atleast_2d(numpy.loadtxt(filename))


if __name__ == '__main__':
    try:
        system = sys.argv[1]
    except IndexError:
        sys.exit(1)

    dirpath = get_most_recent_path(system)
    print 'Most recent path is:', dirpath

    data_file = open(path.join(dirpath, 'data_file.txt')).read().strip()
    evidence_file = open(path.join(dirpath, 'evidences_%s.txt' % system), 'w')


    # effort_metric = 0
    times = open(path.join(dirpath, 'times.txt')).readlines()
    times = [t.strip().split(',') for t in times]
    times = {int(t[0]): int(float(t[1])*1000) for t in times}


    fig1 = plt.figure('logZhist')
    ax = fig1.add_subplot(111)

    # for 0,1,2,3 planets
    for np in range(4):
        levels_orig = load_file(path.join(dirpath, "levels_%dplanets.txt" % np))
        sample_info = load_file(path.join(dirpath, "sample_info_%dplanets.txt" % np))
        sample = load_file(path.join(dirpath, "sample_%dplanets.txt" % np))


        logz_estimate, H_estimate, logx_samples, posterior_sample = \
            postprocess(plot=False, numResampleLogX=1,
                        loaded=(levels_orig, sample_info, sample))

        log10z_estimate = logz_estimate / numpy.log(10.)
        if posterior_sample.shape[0] > 5:
            res = DisplayResults('', data_file=data_file,)
                                 # posterior_samples_file=path.join(dirpath, 'posterior_sample.txt'))
        else:
            print 'Too few samples yet'

        assert res.max_components == np
        res.make_plot2()

        ax.plot([np]*log10z_estimate.size, log10z_estimate, 'o')
        # ax.hist(log10z_estimate, label=str(np), normed=True)


        effort_metric = times[np]
        result_string = '%d,' % effort_metric
        result_string += '%d,' % np
        result_string += '%f,' % numpy.mean(log10z_estimate)
        result_string += '%f,' % numpy.median(log10z_estimate)
        result_string += ','.join(numpy.percentile(log10z_estimate, [2, 16, 84, 98]).astype(str))

        print >>evidence_file, result_string


    evidence_file.close()

    plt.show()
    # fig1.savefig('log10Z_hist.png')
