#include "DNest4.h"

/* hyper parameters */

// Cauchy prior centered on log(365 days), scale=1
// for log(muP), with muP in days
// -- truncated to (~-15.1, ~26.9) --
TruncatedCauchy *log_muP_prior = new TruncatedCauchy(log(365), 1., log(365)-21, log(365)+21);

// Uniform prior between 0.1 and 3 for wP
Uniform *wP_prior = new Uniform(0.1, 3.);
    
// Cauchy prior centered on log(1), scale=1
// for log(muK), with muK in m/s
// -- truncated to (-21, 21) --
// NOTE: we actually sample on muK itself, just the prior is for log(muK). 
// Confusing, I know...
TruncatedCauchy *log_muK_prior = new TruncatedCauchy(0., 1., 0.-21, 0.+21);

/*
Laplace Pprior;
Exponential Kprior;
*/
