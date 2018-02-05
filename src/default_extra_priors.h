#include "DNest4.h"

Uniform *Cprior = new Uniform(-1000, 1000);
ModifiedLogUniform *Jprior = new ModifiedLogUniform(1.0, 99.); // additional white noise, m/s

// this default prior for the slope of a linear trend is awful!
// the prior probably should depend on the data,
// -> a good default would be Uniform( -data.topslope(), data.topslope() )
Uniform *slope_prior = new Uniform(-10, 10);
