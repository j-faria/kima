#ifndef DEFAULT_PRIORS_H
#define DEFAULT_PRIORS_H

#include "DNest4.h"

DNest4::Uniform *Cprior = new DNest4::Uniform(-1000, 1000);
DNest4::ModifiedJeffreys *Jprior = new DNest4::ModifiedJeffreys(1.0, 99.); // additional white noise, m/s

DNest4::Jeffreys *Pprior = new DNest4::Jeffreys(1.0, 1E7); // days
DNest4::ModifiedJeffreys *Kprior = new DNest4::ModifiedJeffreys(10.0, 1E4); // m/s

DNest4::Uniform *eprior = new DNest4::Uniform(0., 1.);
DNest4::Uniform *phiprior = new DNest4::Uniform(0.0, 2*M_PI);
DNest4::Uniform *wprior = new DNest4::Uniform(0.0, 2*M_PI);


#endif