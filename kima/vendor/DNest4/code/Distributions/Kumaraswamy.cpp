#include "Kumaraswamy.h"
#include <stdexcept>
#include <cmath>
#include "../Utils.h"

namespace DNest4
{

    Kumaraswamy::Kumaraswamy(double _a, double _b)
        : a(_a), b(_b)
    {
        if (a <= 0.0 or b <= 0.0)
            throw std::domain_error("Kumaraswamy distribution must have positive a and b");
    }

    void Kumaraswamy::setpars(double _a, double _b)
    {
        if (_a <= 0.0 or _b <= 0.0)
            throw std::domain_error("Kumaraswamy distribution must have positive a and b");
        
        a = _a;
        b = _b;
    }

    double Kumaraswamy::cdf(double x) const
    {
        return 1 - pow(1 - pow(x, a), b);
    }

    double Kumaraswamy::cdf_inverse(double p) const
    {
        if (p < 0.0 || p > 1.0)
            throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
        return pow(1 - pow(1 - p, 1 / b), 1 / a);
    }

    double Kumaraswamy::log_pdf(double x) const
    {
        return log(a) + log(b) + (a - 1) * log(x) + (b - 1) * log(1 - pow(x, a));
    }

} // namespace DNest4
