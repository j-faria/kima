#ifndef DEFAULT_PRIORS_H
#define DEFAULT_PRIORS_H

#include "DNest4.h"

LogUniform *Pprior = new LogUniform(1.0, 1E5); // days
ModifiedLogUniform *Kprior = new ModifiedLogUniform(1.0, 2E3); // m/s

Uniform *eprior = new Uniform(0., 1.);
Uniform *phiprior = new Uniform(0.0, 2*M_PI);
Uniform *wprior = new Uniform(0.0, 2*M_PI);

#endif