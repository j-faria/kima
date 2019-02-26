#include "DNest4.h"

Uniform *Cprior = new Uniform(-1000, 1000);
ModifiedLogUniform *Jprior = new ModifiedLogUniform(1.0, 99.); // additional white noise, m/s

// this default prior for the slope of a linear trend is awful!
// the prior probably should depend on the data,
// -> a good default would be Uniform( -data.topslope(), data.topslope() )
Uniform *slope_prior = new Uniform(-10, 10);


// this default prior for the instrument offsets is awful!
// the prior probably should depend on the data,
// -> a good default would be Uniform( -data.get_RV_span(), data.get_RV_span() )
Uniform *offsets_prior = new Uniform(-10, 10);

// HARPS fiber offset prior
// Gaussian fit to the offsets determined by Lo Curto et al. 2015
// (only for FGK stars)
// mean, std = 14.641789473684208, 2.7783035258938971
// Gaussian *fiber_offset_prior = new Gaussian(15., 3.);
// Note that M dwarfs show much smaller offsets, of ~0-1 m/s
Uniform *fiber_offset_prior = new Uniform(0., 50.);

