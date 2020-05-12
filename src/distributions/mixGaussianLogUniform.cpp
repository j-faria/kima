#include "mixGaussianLogUniform.h"

namespace DNest4
{

/**
 * Construct an mixGaussianLogUniform distribution
*/
mixGaussianLogUniform::mixGaussianLogUniform(double mean, double sigma, double lower, double upper)
{
    G = Gaussian(mean, sigma);
    LU = LogUniform(lower, upper);
}


/* mixGaussianLogUniform distribution function */
double mixGaussianLogUniform::cdf(double x) const
{
    return 0.5 * G.cdf(x) + 0.5 * LU.cdf(x);
}

/* Quantile estimate, midpoint interpolation */
double mixGaussianLogUniform::cdf_inverse(double p) const
{
    // not implemented
    throw std::runtime_error("cdf_inverse not implemented for `mixGaussianLogUniform`");
}

double mixGaussianLogUniform::log_pdf(double x) const
{
    return 0.5 * G.log_pdf(x) + 0.5 * LU.log_pdf(x);
}

double mixGaussianLogUniform::perturb(double& x, RNG& rng) const
{
    x = cdf(x);
    x += rng.randh();
    wrap(x, 0.0, 1.0);
    if (rng.rand_int(2) == 0)
        x = G.cdf_inverse(x);
    else
        x = LU.cdf_inverse(x);
    return 0.0;
}


} // namespace DNest4

