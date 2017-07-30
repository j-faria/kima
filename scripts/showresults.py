import sys
try:
    options = sys.argv
except IndexError:
    options = ''


from dnest4 import postprocess  #, diffusion_plot

if 'no' in options:
    plot = False
else:
    plot = True

if 'pvc' in options:
    while True:
        logz_estimate, H_estimate, logx_samples, posterior_sample = postprocess(plot=plot)
else:
    logz_estimate, H_estimate, logx_samples, posterior_sample = postprocess(plot=plot)

# diffusion_plot()

if posterior_sample.shape[0] > 5:
    # import display
    from display import DisplayResults
    res = DisplayResults(options)
else:
    print 'Too few samples yet'

