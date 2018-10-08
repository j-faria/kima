import numpy as np


def most_probable_np(results):
    """ 
    Return the value of Np with the highest posterior probability.
    Arguments:
        results: a KimaResults instance. 
    """
    Nplanets = results.posterior_sample[:, results.index_component].astype(int)
    values, counts = np.unique(Nplanets, return_counts=True)
    return values[counts.argmax()]


def passes_threshold_np(results, threshold=150):
    """ 
    Return the value of Np supported by the data, considering a posterior ratio
    threshold (default 150).
    Arguments:
        results: a KimaResults instance. 
        threshold: posterior ratio threshold, default 150.
    """
    Nplanets = results.posterior_sample[:, results.index_component].astype(int)
    values, counts = np.unique(Nplanets, return_counts=True)
    ratios = counts[1:] / counts[:-1]
    if np.any(ratios > threshold):
        i = np.argmax(np.where(ratios > 150)[0]) + 1
        return values[i]
    else:
        return values[0]



def column_dynamic_ranges(results):
    """ Return the range of each column in the posterior file """
    return results.posterior_sample.ptp(axis=0)

def columns_with_dynamic_range(results):
    """ Return the columns in the posterior file which vary """
    dr = column_dynamic_ranges(results)
    return np.nonzero(dr)[0]