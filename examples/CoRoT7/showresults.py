import os, sys
import subprocess

options = sys.argv

with open('Makefile') as f:
    kima_dir = [line for line in f.readlines() if 'KIMA_DIR =' in line]
    kima_dir = kima_dir[0].strip().split('=')[-1].strip()


sys.path.append(os.path.join(kima_dir, 'scripts'))
from dnest4 import postprocess  # , diffusion_plot

if 'no' in options:
    plot = False
else:
    plot = True

if 'pvc' in options:
    while True:
        logz, H, logx_samples, posterior = postprocess(plot=plot)
else:
    logz, H, logx_samples, posterior = postprocess(plot=plot)


if posterior.shape[0] > 5:
    from display import DisplayResults
    res = DisplayResults(options)
else:
    print('Too few samples, keep running the model')

