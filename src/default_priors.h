#ifndef DEFAULT_PRIORS_H
#define DEFAULT_PRIORS_H

#include "DNest4.h"

Uniform *Cprior = new Uniform(-1000, 1000);
ModifiedLogUniform *Jprior = new ModifiedLogUniform(1.0, 99.); // additional white noise, m/s

// this default prior for the slope of a linear trend is awful!
// the prior probably should depend on the data,
// -> a good default would be Uniform( -data.topslope(), data.topslope() )
Uniform *slope_prior = new Uniform(-10, 10);


LogUniform *Pprior = new LogUniform(1.0, 1E5); // days
ModifiedLogUniform *Kprior = new ModifiedLogUniform(1.0, 2E3); // m/s

Uniform *eprior = new Uniform(0., 1.);
Uniform *phiprior = new Uniform(0.0, 2*M_PI);
Uniform *wprior = new Uniform(0.0, 2*M_PI);


/* GP parameters */
Uniform *log_eta1_prior = new Uniform(-5, 5);
Uniform *log_eta2_prior = new Uniform(0, 5);
Uniform *eta3_prior = new Uniform(10., 40.);
Uniform *log_eta4_prior = new Uniform(-1, 1);



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

#endif