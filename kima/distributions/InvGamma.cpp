#include "InvGamma.h"

namespace DNest4
{

InvGamma::InvGamma(double shape, double scale)
:shape(shape),scale(scale)
{
    if(shape <= 0.0)
        throw std::domain_error("InvGamma distribution must have positive shape.");
    if(scale <= 0.0)
        throw std::domain_error("InvGamma distribution must have positive scale.");
}

void InvGamma::setpars(double sh, double sc)
{
    if (sh <= 0.0)
        throw std::domain_error("InvGamma distribution must have positive shape.");
    if (sc <= 0.0)
        throw std::domain_error("InvGamma distribution must have positive scale.");

    shape = sh;
    scale = sc;
}

double InvGamma::cdf(double x) const
{
    return stats::pinvgamma(x, shape, scale);
}

double InvGamma::cdf_inverse(double p) const
{
    if(p < 0.0 || p > 1.0)
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");

    return stats::qinvgamma(p, shape, scale);
}

double InvGamma::log_pdf(double x) const
{
    if(x < 0.0)
        return -std::numeric_limits<double>::infinity();

    return stats::dinvgamma(x, shape, scale, true);
}


} // namespace DNest4

