#ifndef DEFAULT_PRIORS_H
#define DEFAULT_PRIORS_H

#include "DNest4.h"

/* GP parameters */
Uniform *log_eta1_prior = new Uniform(-5, 5);
Uniform *log_eta2_prior = new Uniform(0, 5);
Uniform *eta3_prior = new Uniform(10., 40.);
Uniform *log_eta4_prior = new Uniform(-5, 0);

#endif