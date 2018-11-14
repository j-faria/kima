#ifndef DEFAULT_GPRN_PRIORS_H
#define DEFAULT_GPRN_PRIORS_H

#include "DNest4.h"

/* Constant kernel*/
Uniform *constant_weight = new Uniform(0, 100);
Uniform *constant_prior = new Uniform(0, 100);

/* Squared Exponential kernel */
Uniform *se_weight = new Uniform(0,100);
Uniform *se_ell = new Uniform(0,100);

/* Periodic kernel */
Uniform *per_weight = new Uniform(0,100);
Uniform *per_ell = new Uniform(0,1);
Uniform *per_period = new Uniform(0,100);

/* Quasi-periodic kernel */
Uniform *quasi_weight = new Uniform(0,100);
Uniform *quasi_elle = new Uniform(0,100);
Uniform *quasi_period = new Uniform(0,100);
Uniform *quasi_ellp = new Uniform(0,1);

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

/* Matern 3/2 */
Uniform *m32_weight = new Uniform(0,100);
Uniform *m32_ell = new Uniform(0,100);

/* Matern 5/2 */
Uniform *m52_weight = new Uniform(0,100);
Uniform *m52_ell = new Uniform(0,100);


#endif
