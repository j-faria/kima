#include "Gaussian.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include "../Utils.h"
//#include <boost/math/special_functions/erf.hpp>

namespace DNest4
{

    Gaussian::Gaussian(double _center, double _width)
        : center(_center), width(_width)
    {
        if (width <= 0.0)
            throw std::domain_error("Gaussian distribution must have positive width");
    }

    void Gaussian::setpars(double c, double w)
    {
        if (w <= 0.0)
            throw std::domain_error("Gaussian distribution must have positive width");

        center = c;
        width = w;
    }

    double Gaussian::cdf(double x) const
    {
        return normal_cdf((x - center) / width);
    }

    double Gaussian::cdf_inverse(double x) const
    {
        if (x < 0.0 || x > 1.0)
            throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
        return center + width * normal_inverse_cdf(x);
        // return center + width * sqrt(2) * boost::math::erf_inv(2*x - 1);
    }

    double Gaussian::log_pdf(double x) const
    {
        double r = (x - center) / width;
        return -0.5 * r * r - _norm_pdf_logC;
    }

    TruncatedGaussian::TruncatedGaussian(double center, double width, double lower, double upper)
        : center(center), width(width), lower(lower), upper(upper)
    {
        if (width <= 0.0)
            throw std::domain_error("TruncatedGaussian distribution must have positive width");
        if (lower >= upper)
            throw std::domain_error("TruncatedGaussian: lower bound should be less than upper bound.");
        // the original, untruncated, Gaussian distribution
        unG = Gaussian(center, width);
        c = unG.cdf(upper) - unG.cdf(lower);
    }

    void TruncatedGaussian::setpars(double c, double w, double lo, double up)
    {
        if (w <= 0.0)
            throw std::domain_error("TruncatedGaussian distribution must have positive width");
        if (lo >= up)
            throw std::domain_error("TruncatedGaussian: lower bound should be less than upper bound");
        // the original, untruncated, Gaussian distribution

        center = c;
        width = w;

        unG.setpars(c, w);
        lower = lo;
        upper = up;
        c = unG.cdf(upper) - unG.cdf(lower);
    }

    void TruncatedGaussian::setpars(double c, double w)
    {
        if (w <= 0.0)
            throw std::domain_error("TruncatedGaussian distribution must have positive width");

        center = c;
        width = w;
        unG.setpars(c, w);
        c = unG.cdf(upper) - unG.cdf(lower);
    }

    double TruncatedGaussian::cdf(double x) const
    {
        double up = std::max(std::min(x, upper), lower);
        return (unG.cdf(up) - unG.cdf(lower)) / c;
    }

    double TruncatedGaussian::cdf_inverse(double x) const
    {
        if (x < 0.0 || x > 1.0)
            throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
        double xx = unG.cdf(lower) + x * c;
        return unG.cdf_inverse(xx);
    }

    double TruncatedGaussian::log_pdf(double x) const
    {
        if (x < lower or x > upper)
            return -std::numeric_limits<double>::infinity();
        return unG.log_pdf(x) - log(c);
    }

} // namespace DNest4