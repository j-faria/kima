#include "mixGaussianLogUniform.h"

namespace DNest4
{

/**
 * Construct a mixGaussianLogUniform distribution
*/
mixGaussianLogUniform::mixGaussianLogUniform(double mean, double sigma, double lower, double upper)
: mean(mean), sigma(sigma), lower(lower), upper(upper)
{
    // the two components
    G = Gaussian(mean, sigma);
    LU = LogUniform(lower, upper);
}


/* mixGaussianLogUniform cumulative distribution function */
double mixGaussianLogUniform::cdf(double x) const
{
    return 0.5 * G.cdf(x) + 0.5 * LU.cdf(x);
}

/* mixGaussianLogUniform quantile function */
double mixGaussianLogUniform::cdf_inverse(double p) const
{
    // not implemented
    // throw std::runtime_error("cdf_inverse not implemented for `mixGaussianLogUniform`");

    // estimate cdf_inverse with bisection
    double tol = 2e-12, a = lower, b = upper;
    double c = a;

    while ((b - a) >= tol)
    {
        c = (a + b) / 2;
        if (cdf(c) - p == 0.0)
            break;
        else if ((cdf(c) - p) * (cdf(a) - p) < 0)
            b = c;
        else
            a = c;
    }

    return c;
}

double mixGaussianLogUniform::log_pdf(double x) const
{
    return log(0.5 * exp(G.log_pdf(x)) + 0.5 * exp(LU.log_pdf(x)));
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

