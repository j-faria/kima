#ifndef DEFAULT_GPRN_PRIORS_H
#define DEFAULT_GPRN_PRIORS_H

#include "DNest4.h"

/* Constant kernel*/
Uniform *constant_weight = new Uniform(0, 50);
Uniform *constant_prior = new Uniform(0, 50);

/* Squared Exponential kernel */
Uniform *se_weight = new Uniform(0,100);
Uniform *se_ell = new Uniform(0,100);

/* Periodic kernel */
Uniform *per_weight = new Uniform(0,150);
Uniform *per_ell = new Uniform(0,3);
Uniform *per_period = new Uniform(10,40);

/* Quasi-periodic kernel */
Uniform *quasi_weight = new Uniform(0, 50);;
Uniform *quasi_elle = new Uniform(1,10);
Uniform *quasi_period = new Uniform(5., 15.);
Uniform *quasi_ellp = new Uniform(0, 1);

/* Rational quadratic kernel */
Uniform *ratq_weight = new Uniform(0,100);
Uniform *ratq_alpha = new Uniform(0,100);
Uniform *ratq_ell = new Uniform(0,100);

/* Cosine kernel */
Uniform *cos_weight = new Uniform(0,100);
Uniform *cos_period = new Uniform(0,100);

/* Exponential kernel */
Uniform *exp_weight = new Uniform(0,100);
Uniform *exp_ell = new Uniform(0,100);

/* Matern 3/2 kernel */
Uniform *m32_weight = new Uniform(0,100);
Uniform *m32_ell = new Uniform(0,100);

/* Matern 5/2 kernel */
Uniform *m52_weight = new Uniform(0,100);
Uniform *m52_ell = new Uniform(0,100);

/* jitter terms */
Uniform *jitter_prior = new Uniform(0,1);

/* Constant mean */
Uniform *const_mean = new Uniform(0, 50);

/* Linear mean */
Uniform *linear_slope = new Uniform(0, 50);
Uniform *linear_intercept = new Uniform(0, 50);

/* Parabolic mean */
Uniform *parabolic_quadcoeff = new Uniform(0, 50);
Uniform *parabolic_lincoeff = new Uniform(0, 50);
Uniform *parabolic_free = new Uniform(0, 50);

/* Cubic mean */
Uniform *cubic_cubcoeff = new Uniform(0, 50);
Uniform *cubic_quadcoeff = new Uniform(0, 50);
Uniform *cubic_lincoeff = new Uniform(0, 50);
Uniform *cubic_free = new Uniform(0, 50);

/* Sinusoidal mean */
Uniform *sine_amp = new Uniform(0, 50);
Uniform *sine_freq = new Uniform(0, 50);
Uniform *sine_phase = new Uniform(0, 50);



#endif
