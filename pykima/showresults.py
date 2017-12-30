from .dnest4 import postprocess
from .display import KimaResults
import sys
from matplotlib.pyplot import isinteractive

def showresults(options=None):
    if options is None:
        options = sys.argv

    plot = not 'no' in options

    try:
        if 'pvc' in options:
            while True:
                logz, H, logx_samples, posterior = postprocess(plot=plot)
        else:
            logz, H, logx_samples, posterior = postprocess(plot=plot)
    except IOError as e:
        print(e)
        sys.exit(1)

    if posterior.shape[0] > 5:
        res = KimaResults(options)
        if isinteractive(): return res
    else:
        print('Too few samples, keep running the model')


if __name__ == '__main__':
    options = sys.argv
    res = showresults(options)
