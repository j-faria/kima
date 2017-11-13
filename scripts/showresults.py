from dnest4 import postprocess
import sys
options = sys.argv

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
    print 'Too few samples, keep running the model'
